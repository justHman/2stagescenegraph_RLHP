#!/usr/bin/env python3
"""Local human-preference collection UI for scene-graph predictions."""

import argparse
import colorsys
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import open_dict
from batch_session import BatchSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sgg_benchmark.config.hydra_config import load_config_from_file
from sgg_benchmark.data.transforms import build_transforms
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_checkpoint(root: Path) -> Path:
    """Prefer an explicit best checkpoint, otherwise use the most recently updated one."""
    if root.is_file() and root.suffix == ".pth":
        return root
    candidates = list(root.rglob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth SGG checkpoint found below {root}")
    best = [path for path in candidates if path.name in {"model_best.pth", "best_model.pth"}]
    return max(best or candidates, key=lambda path: path.stat().st_mtime)


def discover_yolo(root: Path | None) -> Path | None:
    if root is None:
        return None
    if root.is_file() and root.suffix == ".pt":
        return root
    candidates = list(root.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No YOLO .pt checkpoint found below {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_config(checkpoint: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for candidate in ("config.yaml", "config.yml", "hydra_config.yaml", ".hydra/config.yaml"):
        path = checkpoint.parent / candidate
        if path.is_file():
            return path
    raise FileNotFoundError("No checkpoint config found. Provide --config.")


def load_psg_labels(psg_dir: Path | None):
    """Load PSG labels from the dataset annotations or a pre-generated JSON file."""
    if psg_dir is None:
        raise ValueError("--psg-dir is required; provide PSG coco_format or a labels JSON file.")
    if psg_dir.is_file():
        with psg_dir.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data.get("objects"), list) or not isinstance(data.get("predicates"), list):
            raise ValueError("Labels JSON must contain list fields named 'objects' and 'predicates'.")
        return data["objects"], data["predicates"]
    annotation = psg_dir / "test" / "_annotations.coco.json"
    with annotation.open() as handle:
        data = json.load(handle)
    objects = ["__background__"] + [item["name"] for item in sorted(data["categories"], key=lambda x: x["id"])]
    predicates = ["__background__"] + [
        item["name"] if isinstance(item, dict) else str(item)
        for item in sorted(data["rel_categories"], key=lambda x: x.get("id", 0) if isinstance(x, dict) else 0)
    ]
    return objects, predicates


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def prepare_inference_statistics(cfg, object_names, predicate_names, cache_dir: Path):
    """Create neutral class statistics so an inference-only app never loads training data.

    Relation heads request dataset statistics while being constructed, even though
    a loaded inference checkpoint does not need training samples.  The label
    lists provide the required class cardinalities; the checkpoint supplies all
    learned model parameters.
    """
    object_count = int(cfg.model.roi_box_head.num_classes)
    predicate_count = int(cfg.model.roi_relation_head.num_classes)
    if len(object_names) != object_count or len(predicate_names) != predicate_count:
        raise ValueError(
            "The label file does not match this checkpoint configuration: "
            f"expected {object_count} objects and {predicate_count} predicates, got "
            f"{len(object_names)} and {len(predicate_names)}."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cfg.datasets.name}_statistics.cache"
    if cache_path.exists():
        return cache_path

    # These values are used only to construct training losses / optional priors.
    # The supplied REACT++ config has frequency bias and logit adjustment disabled.
    fg_matrix = torch.zeros((object_count, object_count, predicate_count), dtype=torch.float32)
    fg_matrix[:, :, 0] = 1.0
    uniform_probability = torch.full(
        (object_count, object_count, predicate_count), 1.0 / predicate_count, dtype=torch.float32
    )
    statistics = {
        "fg_matrix": fg_matrix,
        "pred_dist": uniform_probability.log(),
        "obj_classes": object_names,
        "rel_classes": predicate_names,
        "predicate_new_order": list(range(1, predicate_count)),
        "predicate_new_order_count": [1] * (predicate_count - 1),
        "pred_freq": torch.full((predicate_count,), 1.0 / predicate_count, dtype=torch.float32),
        "triplet_freq": {},
        "pred_weight": torch.ones(predicate_count, dtype=torch.float32),
    }
    statistics["pred_weight"][0] = 0.0
    torch.save(statistics, cache_path)
    return cache_path


def prepare_inference_word_vectors(cfg, cache_dir: Path):
    """Provide a local empty GloVe cache for checkpoint-only inference.

    REACT++ initializes embedding layers from GloVe during construction, then
    ``DetectronCheckpointer`` restores the trained embedding weights.  Supplying
    an empty local vocabulary avoids downloading GloVe solely for that temporary
    initialization step.
    """
    vector_type = str(cfg.model.text_embedding)
    if "glove" not in vector_type:
        return None
    dimension = int(cfg.model.roi_relation_head.embed_dim)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vector_path = cache_dir / f"{vector_type}.{dimension}d.pt"
    if not vector_path.exists():
        torch.save(({}, torch.empty((0, dimension), dtype=torch.float32), dimension), vector_path)
    return vector_path


class Predictor:
    def __init__(self, checkpoint: Path, config: Path, yolo_checkpoint: Path | None, psg_dir: Path, device: str):
        checkpoint, config, psg_dir = Path(checkpoint), Path(config), Path(psg_dir)
        self.cfg = load_config_from_file(str(config))
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but no GPU is available.")
        with open_dict(self.cfg):
            self.cfg.model.device = str(self.device)
            self.cfg.model.roi_relation_head.use_gt_box = False
            self.cfg.model.roi_relation_head.use_gt_object_label = False
            self.cfg.test.top_k = 0  # retain all detected directed pairs
            if yolo_checkpoint:
                self.cfg.model.pretrained_detector_ckpt = str(yolo_checkpoint)
        self.transforms = build_transforms(self.cfg, is_train=False)
        self.object_names, self.predicate_names = load_psg_labels(psg_dir)
        label_source = psg_dir if psg_dir.is_file() else psg_dir / "test" / "_annotations.coco.json"
        self.provenance = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "config_path": str(config.resolve()),
            "config_sha256": file_sha256(config),
            "label_map_path": str(label_source.resolve()),
            "label_map_sha256": file_sha256(label_source),
        }
        if yolo_checkpoint:
            self.provenance.update({
                "yolo_checkpoint_path": str(Path(yolo_checkpoint).resolve()),
                "yolo_checkpoint_sha256": file_sha256(yolo_checkpoint),
            })
        # Point the dataset-statistics lookup at a local inference-only cache.
        # This prevents the model constructor from downloading PSG training data.
        statistics_dir = PROJECT_ROOT / "scene_graph_tool" / ".cache" / checkpoint.stem
        with open_dict(self.cfg):
            self.cfg.output_dir = str(statistics_dir)
            self.cfg.glove_dir = str(statistics_dir / "word_vectors")
        statistics_cache = prepare_inference_statistics(
            self.cfg, self.object_names, self.predicate_names, statistics_dir
        )
        word_vectors = prepare_inference_word_vectors(self.cfg, Path(self.cfg.glove_dir))
        print(f"Inference statistics: {statistics_cache} (category-only; no dataset download)")
        if word_vectors:
            print(f"Inference word vectors: {word_vectors} (checkpoint embeddings; no GloVe download)")
        self.model = build_detection_model(self.cfg).to(self.device)
        DetectronCheckpointer(self.cfg, self.model).load(str(checkpoint))
        self.model.eval()

    def predict(self, rgb_image):
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
            "image_size": (width, height), "mode": "xyxy",
        }
        image, target = self.transforms(bgr, target)
        with torch.inference_mode():
            prediction = self.model([image.to(self.device)], [to_device(target, self.device)])[0]
        return self._normalize(prediction)

    def _normalize(self, prediction):
        boxes = prediction["boxes"].detach().cpu().numpy()
        labels = prediction["pred_labels"].detach().cpu().tolist()
        scores = prediction["pred_scores"].detach().cpu().tolist()
        pairs = prediction["rel_pair_idxs"].detach().cpu().tolist()
        rel_scores = prediction["pred_rel_scores"].detach().float().cpu().tolist()
        objects = [{
            "id": index, "label": self.object_names[int(label)], "label_id": int(label),
            "score": round(float(scores[index]), 5), "box_xyxy": [round(float(v), 2) for v in boxes[index]],
        } for index, label in enumerate(labels)]
        relations = []
        for index, ((subject, obj), distribution) in enumerate(zip(pairs, rel_scores)):
            foreground = distribution[1:]
            best_id = int(np.argmax(foreground)) + 1
            relations.append({
                "id": index, "subject_id": int(subject), "object_id": int(obj),
                "background_probability": round(float(distribution[0]), 6),
                "best_predicate": self.predicate_names[best_id],
                "best_predicate_id": best_id,
                "best_probability": round(float(distribution[best_id]), 6),
                "predicate_probabilities": {
                    self.predicate_names[predicate_id]: round(float(distribution[predicate_id]), 6)
                    for predicate_id in range(1, len(distribution))
                },
            })
        return {"objects": objects, "relations": relations}


def class_color(class_id):
    """Return a stable, visually distinct RGB color for an object class."""
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.95)
    return tuple(int(channel * 255) for channel in (red, green, blue))


def _place_badge(candidates, size, occupied, image_size, clearance):
    """Pick the first in-bounds badge position that does not cover another badge."""
    badge_width, badge_height = size
    image_width, image_height = image_size
    choices = []
    seen = set()
    for raw_x, raw_y in candidates:
        x = max(0, min(round(raw_x), max(0, image_width - badge_width)))
        y = max(0, min(round(raw_y), max(0, image_height - badge_height)))
        if (x, y) in seen:
            continue
        seen.add((x, y))
        rectangle = (x, y, x + badge_width, y + badge_height)
        overlap = 0
        for left, top, right, bottom in occupied:
            overlap += max(0, min(rectangle[2], right + clearance) - max(rectangle[0], left - clearance)) * max(
                0, min(rectangle[3], bottom + clearance) - max(rectangle[1], top - clearance)
            )
        choices.append((overlap, len(choices), rectangle))
        if overlap == 0:
            occupied.append(rectangle)
            return rectangle

    # Dense predictions may make every candidate collide. Choose the position
    # with the smallest overlap so a later badge never blindly erases an earlier one.
    rectangle = min(choices, default=(0, 0, (0, 0, badge_width, badge_height)))[2]
    occupied.append(rectangle)
    return rectangle


def draw_scene_graph(image, objects, relations, show_probability=True):
    """Render an annotation-style scene graph with class-colored nodes and labelled arrows."""
    source_height, source_width = image.shape[:2]
    # Most PSG samples are only 640 px wide but the UI displays them at around
    # 1000 px. Draw on a larger canvas so text and diagonal lines stay crisp.
    render_scale = min(2.0, max(1.0, 1280.0 / max(source_width, source_height, 1)))
    if render_scale > 1.05:
        result = cv2.resize(image, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_CUBIC)
    else:
        render_scale = 1.0
        result = image.copy()

    image_height, image_width = result.shape[:2]
    image_scale = max(0.85, min(2.4, min(result.shape[:2]) / 700.0))
    edge_width = max(2, round(3 * image_scale))
    box_width = max(2, round(2 * image_scale))
    font_scale = 0.56 * image_scale
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_thickness = max(1, round(1.4 * image_scale))
    badge_padding = max(4, round(5 * image_scale))
    badge_gap = max(4, round(6 * image_scale))

    centers = {}
    object_badges = []
    occupied_badges = []
    for obj in objects:
        x1, y1, x2, y2 = (round(float(value) * render_scale) for value in obj["box_xyxy"])
        x1, x2 = sorted((max(0, min(x1, image_width - 1)), max(0, min(x2, image_width - 1))))
        y1, y2 = sorted((max(0, min(y1, image_height - 1)), max(0, min(y2, image_height - 1))))
        color = class_color(obj["label_id"])
        centers[obj["id"]] = ((x1 + x2) // 2, (y1 + y2) // 2)
        # A thin full box plus heavier corner accents mirrors the annotation notebook.
        cv2.rectangle(result, (x1, y1), (x2, y2), color, box_width, cv2.LINE_AA)
        corner_length = max(8, min(round(18 * image_scale), (x2 - x1) // 3, (y2 - y1) // 3))
        for point, dx, dy in (((x1, y1), 1, 1), ((x2, y1), -1, 1), ((x1, y2), 1, -1), ((x2, y2), -1, -1)):
            cv2.line(result, point, (point[0] + dx * corner_length, point[1]), color, edge_width, cv2.LINE_AA)
            cv2.line(result, point, (point[0], point[1] + dy * corner_length), color, edge_width, cv2.LINE_AA)

        label_text = f'#{obj["id"]}  {obj["label"]}'
        score_text = f'{obj["score"]:.2f}'
        (label_width, label_height), label_baseline = cv2.getTextSize(
            label_text, font_face, font_scale, font_thickness
        )
        (score_width, score_height), score_baseline = cv2.getTextSize(
            score_text, font_face, font_scale, font_thickness
        )
        text_height = max(label_height + label_baseline, score_height + score_baseline)
        score_cell_width = score_width + badge_padding * 2
        badge_width = label_width + score_cell_width + badge_padding * 2
        badge_height = text_height + badge_padding * 2

        vertical_slots = [
            (x1 + badge_gap, y1 + badge_gap + slot * (badge_height + badge_gap))
            for slot in range(4)
        ]
        candidates = [
            (x1, y1 - badge_height - badge_gap),
            *vertical_slots,
            (x2 - badge_width, y1 - badge_height - badge_gap),
            (x2 - badge_width - badge_gap, y1 + badge_gap),
            (x1, y2 + badge_gap),
            (x2 - badge_width, y2 + badge_gap),
        ]
        badge_rect = _place_badge(
            candidates, (badge_width, badge_height), occupied_badges,
            (image_width, image_height), badge_gap,
        )
        object_badges.append({
            "rect": badge_rect, "color": color, "label": label_text, "score": score_text,
            "label_width": label_width, "score_cell_width": score_cell_width,
            "label_height": label_height, "label_baseline": label_baseline,
        })

    # Draw edges below all badges. A dark halo, saturated orange body, and pale
    # center line keep the direction visible on both bright and dark images.
    predicate_badges = []
    for relation in relations:
        subject_id, object_id = relation["subject_id"], relation["object_id"]
        if subject_id not in centers or object_id not in centers:
            continue
        start, end = centers[subject_id], centers[object_id]
        distance = float(np.hypot(start[0] - end[0], start[1] - end[1]))
        if distance < 6:
            continue
        padding = min(max(12 * image_scale, 0.12 * distance), 0.34 * distance)
        dx, dy = (end[0] - start[0]) / distance, (end[1] - start[1]) / distance
        normal_x, normal_y = -dy, dx
        predicate_rank = relation.get("predicate_rank", 0)
        predicate_count = relation.get("predicate_count", 1)
        parallel_offset = (predicate_rank - (predicate_count - 1) / 2.0) * round(16 * image_scale)
        edge_start = (round(start[0] + dx * padding + normal_x * parallel_offset),
                      round(start[1] + dy * padding + normal_y * parallel_offset))
        edge_end = (round(end[0] - dx * padding + normal_x * parallel_offset),
                    round(end[1] - dy * padding + normal_y * parallel_offset))
        outline_color, line_color, highlight_color = (10, 10, 10), (255, 104, 20), (255, 238, 120)
        cv2.arrowedLine(result, edge_start, edge_end, outline_color, edge_width + 5,
                        cv2.LINE_AA, tipLength=0.17)
        cv2.arrowedLine(result, edge_start, edge_end, line_color, edge_width + 1,
                        cv2.LINE_AA, tipLength=0.17)
        cv2.arrowedLine(result, edge_start, edge_end, highlight_color, max(1, edge_width - 2),
                        cv2.LINE_AA, tipLength=0.17)

        predicate = relation.get("display_predicate", relation["best_predicate"])
        probability = relation.get("display_probability", relation["best_probability"])
        text = f'{predicate} · {probability:.0%}' if show_probability else predicate
        predicate_font_scale = font_scale * 1.08
        predicate_thickness = font_thickness + 1
        (width, height), baseline = cv2.getTextSize(
            text, font_face, predicate_font_scale, predicate_thickness
        )
        badge_width = width + badge_padding * 2
        badge_height = height + baseline + badge_padding * 2
        # Move the badge completely to one side of the edge. The projected
        # half-size matters here: for a vertical edge the label width, rather
        # than its height, determines how far it must move.
        projected_half_size = abs(normal_x) * badge_width / 2 + abs(normal_y) * badge_height / 2
        normal_distance = projected_half_size + badge_gap * 1.5 + abs(parallel_offset)
        candidates = []
        for fraction in (0.5, 0.35, 0.65):
            point_x = start[0] * (1 - fraction) + end[0] * fraction
            point_y = start[1] * (1 - fraction) + end[1] * fraction
            for side in (1, -1):
                center_x = point_x + normal_x * normal_distance * side
                center_y = point_y + normal_y * normal_distance * side
                candidates.append((center_x - badge_width / 2, center_y - badge_height / 2))
        badge_rect = _place_badge(
            candidates, (badge_width, badge_height), occupied_badges,
            (image_width, image_height), badge_gap,
        )
        predicate_badges.append({
            "rect": badge_rect, "text": text, "height": height, "baseline": baseline,
            "font_scale": predicate_font_scale, "thickness": predicate_thickness,
        })

    # Object badges use a neutral dark label cell and a separate gold score cell.
    # This avoids low-contrast white-on-cyan text and makes confidence values distinct.
    for badge in object_badges:
        left, top, right, bottom = badge["rect"]
        score_left = right - badge["score_cell_width"]
        cv2.rectangle(result, (left, top), (right, bottom), (15, 23, 42), -1)
        cv2.rectangle(result, (score_left, top), (right, bottom), (255, 213, 79), -1)
        cv2.rectangle(result, (left, top), (right, bottom), badge["color"],
                      max(2, round(2 * image_scale)), cv2.LINE_AA)
        text_y = top + badge_padding + badge["label_height"]
        cv2.putText(result, badge["label"], (left + badge_padding, text_y), font_face,
                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        score_size, _ = cv2.getTextSize(badge["score"], font_face, font_scale, font_thickness)
        score_x = score_left + (badge["score_cell_width"] - score_size[0]) // 2
        cv2.putText(result, badge["score"], (score_x, text_y), font_face,
                    font_scale, (15, 23, 42), font_thickness, cv2.LINE_AA)

    for badge in predicate_badges:
        left, top, right, bottom = badge["rect"]
        cv2.rectangle(result, (left, top), (right, bottom), (10, 10, 10), -1)
        cv2.rectangle(result, (left, top), (right, bottom), (255, 104, 20),
                      max(2, round(2 * image_scale)), cv2.LINE_AA)
        origin = (left + badge_padding, top + badge_padding + badge["height"])
        cv2.putText(result, badge["text"], origin, font_face, badge["font_scale"],
                    (255, 255, 255), badge["thickness"], cv2.LINE_AA)
    return result


def filtered_relations(result, relation_threshold):
    """Return highest-confidence relations that pass the predicate threshold."""
    relations = [
        relation for relation in result["relations"]
        if relation["best_probability"] >= relation_threshold
    ]
    relations.sort(key=lambda relation: relation["best_probability"], reverse=True)
    return relations


def review_rows(result, relation_threshold, top_k, preferences):
    """Return one selectable row per displayed predicate edge and its human feedback."""
    rows = []
    objects = result["objects"]
    preferences_by_relation = {item["relation_id"]: item for item in preferences}
    relations = filtered_relations(result, relation_threshold)
    for relation in predicate_edges(relations, top_k):
        preference = preferences_by_relation.get(relation["id"])
        rows.append([
            relation["id"], f'{relation["subject_id"]}: {objects[relation["subject_id"]]["label"]}',
            relation["display_predicate"], f'{relation["object_id"]}: {objects[relation["object_id"]]["label"]}',
            relation["display_probability"], relation["background_probability"],
            preference["response"] if preference else "—",
            (", ".join(pred["name"] for pred in preference["presented_predicates"])
             if preference else "—"),
            preference["target_tensor"] if preference else "—",
        ])
    return rows


def top_two_predicates(relation):
    ranked = sorted(relation["predicate_probabilities"].items(), key=lambda item: item[1], reverse=True)
    return ranked[:2]


def predicate_edges(relations, top_k):
    """Expand each subject-object triplet into its top-K predicate display edges."""
    edges = []
    for relation in relations:
        ranked = sorted(relation["predicate_probabilities"].items(), key=lambda item: item[1], reverse=True)
        selected = ranked[:int(top_k)]
        for rank, (predicate, probability) in enumerate(selected):
            edge = dict(relation)
            edge.update({
                "display_predicate": predicate,
                "display_probability": probability,
                "predicate_rank": rank,
                "predicate_count": len(selected),
            })
            edges.append(edge)
    return edges


def relation_objects(result, relations):
    """Return only detector objects participating in at least one supplied triplet."""
    object_ids = {
        object_id for relation in relations
        for object_id in (relation["subject_id"], relation["object_id"])
    }
    return [obj for obj in result["objects"] if obj["id"] in object_ids]


def make_app(predictor: Predictor, output_dir: Path):
    """Create a single-annotator, disk-backed batch annotation workspace."""
    def image_rgb(path: Path):
        image = cv2.imread(str(path))
        if image is None:
            raise gr.Error(f"Cannot read image: {path.name}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_task(session: BatchSession):
        """Return the next unanswered eligible relation, computing predictions lazily."""
        threshold = session.data["settings"]["relation_threshold"]
        for index in range(session.data["current_image_index"], len(session.data["images"])):
            session.data["current_image_index"] = index
            image_id = session.data["images"][index]["id"]
            if not session.has_prediction(image_id):
                session.set_prediction(image_id, predictor.predict(image_rgb(session.image_path(image_id))))
            result = session.get_prediction(image_id)
            eligible = [item for item in filtered_relations(result, threshold) if len(top_two_predicates(item)) == 2]
            for relation in eligible:
                key = session.answer_key(image_id, relation["id"])
                if key not in session.data["answers"]:
                    session.data["current_relation_id"] = relation["id"]
                    session.save()
                    return image_id, result, relation, eligible
            session.data["images"][index]["status"] = "done" if eligible else "no_eligible_relations"
            session.save()
        session.data["current_relation_id"] = None
        session.save()
        return None

    def render_task(session_dir):
        if not session_dir:
            raise gr.Error("Upload a ZIP or images and start a session first.")
        session = BatchSession(Path(session_dir))
        task = get_task(session)
        completed = len(session.data["answers"])
        total = len(session.data["images"])
        if task is None:
            return (None, {}, "## Batch completed", "All eligible relations have been reviewed.",
                    gr.update(interactive=False), gr.update(interactive=False),
                    f"Completed {completed} labels across {total} images. Autosaved.",
                    "", [])
        image_id, result, relation, eligible = task
        subject, obj = result["objects"][relation["subject_id"]], result["objects"][relation["object_id"]]
        candidates = top_two_predicates(relation)
        random.shuffle(candidates)
        presented = [{"name": name, "probability": probability} for name, probability in candidates]
        presentation = {"image_id": image_id, "relation_id": relation["id"], "subject": subject,
                        "object": obj, "presented_predicates": presented}
        overlay = draw_scene_graph(
            image_rgb(session.image_path(image_id)), [subject, obj], [relation], show_probability=False
        )
        context = f"## {subject['label'].title()} → {obj['label'].title()}\nWhich relation is more accurate?"
        progress = (f"Image {session.data['current_image_index'] + 1}/{total} · "
                    f"Relation {eligible.index(relation) + 1}/{len(eligible)} · {completed} labels saved")
        qa_rows = [[item["id"], result["objects"][item["subject_id"]]["label"], item["best_predicate"],
                    result["objects"][item["object_id"]]["label"], item["best_probability"]] for item in eligible]
        return (overlay, presentation, context, progress,
                gr.update(value=candidates[0][0], interactive=True), gr.update(value=candidates[1][0], interactive=True),
                "Autosaved.", image_id, qa_rows)

    css = """
    .labeler-shell {max-width: 1500px; margin: auto} .choice button {min-height: 64px; font-size: 1.15rem}
    .muted {color: #6b7280} #annotation-image img {max-height: 72vh; object-fit: contain}
    """
    shortcuts = """
    () => {
      document.addEventListener('keydown', (event) => {
        if (event.target.matches('input, textarea, select')) return;
        const targets = {'1': 'choice-first', '2': 'choice-second', '0': 'choice-neutral',
                         'x': 'choice-wrong', 's': 'choice-skip', 'z': 'choice-undo'};
        const id = targets[event.key.toLowerCase()];
        const button = id && document.querySelector(`#${id} button`);
        if (button && !button.disabled) { event.preventDefault(); button.click(); }
      });
    }
    """
    with gr.Blocks(title="SceneGraph Labeler") as app:
        session_state = gr.State("")
        presentation_state = gr.State({})
        gr.Markdown("# Scene Graph Labeler\nLabel every image in sequence. Progress is saved automatically after each selection.")
        with gr.Group() as setup:
            archive = gr.File(
                label="Image set (.zip or one or more image files)",
                file_types=[".zip", ".jpg", ".jpeg", ".png", ".webp"], file_count="multiple", type="filepath",
            )
            with gr.Accordion("Advanced settings", open=False):
                threshold = gr.Slider(0, 1, value=0.10, step=0.01, label="Relation confidence threshold")
            start = gr.Button("Start labeling", variant="primary")
            setup_status = gr.Markdown("")
        with gr.Group(visible=False) as workspace:
            with gr.Row(elem_classes=["labeler-shell"]):
                with gr.Column(scale=7):
                    annotated = gr.Image(label="Current object pair", elem_id="annotation-image", interactive=False)
                with gr.Column(scale=3):
                    progress = gr.Markdown("")
                    context = gr.Markdown("")
                    gr.Markdown("### Choose an answer")
                    first = gr.Button("", variant="primary", elem_classes=["choice"], elem_id="choice-first")
                    second = gr.Button("", elem_classes=["choice"], elem_id="choice-second")
                    with gr.Row():
                        neutral = gr.Button("Unsure / equivalent", elem_id="choice-neutral")
                        wrong = gr.Button("No relation", variant="stop", elem_id="choice-wrong")
                    with gr.Row():
                        skip = gr.Button("Skip", elem_id="choice-skip")
                        undo = gr.Button("↶ Undo", elem_id="choice-undo")
                    status = gr.Markdown("Autosaved.")
            with gr.Accordion("Details & QA", open=False):
                qa_image = gr.Markdown("")
                qa_table = gr.Dataframe(headers=["id", "subject", "model predicate", "object", "confidence"], interactive=False)
                apply_settings = gr.Button("Apply threshold to remaining relations")
            export = gr.DownloadButton("Download labeled data", variant="secondary")

        def start_session(archive_path, relation_threshold):
            if not archive_path:
                raise gr.Error("Upload a ZIP or one/more images first.")
            session = BatchSession.create(archive_path, output_dir, relation_threshold)
            rendered = render_task(str(session.directory))
            return (str(session.directory), gr.update(visible=False), gr.update(visible=True),
                    f"Created a batch with {len(session.data['images'])} images.", *rendered)

        start.click(start_session, [archive, threshold],
                    [session_state, setup, workspace, setup_status, annotated, presentation_state, context, progress,
                     first, second, status, qa_image, qa_table])

        def respond(choice, session_dir, presentation):
            if not presentation:
                raise gr.Error("No relation is selected.")
            session = BatchSession(Path(session_dir))
            targets = {"first": [1.0, 0.0], "second": [0.0, 1.0], "neutral": [0.5, 0.5], "wrong": [0.0, 0.0], "skip": None}
            labels = {
                "first": f"strong preference: {presentation['presented_predicates'][0]['name']}",
                "second": f"strong preference: {presentation['presented_predicates'][1]['name']}",
                "neutral": "no preference", "wrong": "both wrong", "skip": "skipped",
            }
            entry = {"image_id": presentation["image_id"], "relation_id": presentation["relation_id"],
                     "subject": presentation["subject"], "object": presentation["object"],
                     "presented_predicates": presentation["presented_predicates"], "response": labels[choice],
                     "target_tensor": targets[choice]}
            session.record_answer(presentation["image_id"], presentation["relation_id"], entry)
            return render_task(session_dir)

        output_components = [annotated, presentation_state, context, progress, first, second, status, qa_image, qa_table]
        for choice, button in (("first", first), ("second", second), ("neutral", neutral), ("wrong", wrong), ("skip", skip)):
            button.click(lambda session_dir, presentation, selected=choice: respond(selected, session_dir, presentation),
                         [session_state, presentation_state], output_components)

        def undo_and_render(session_dir):
            session = BatchSession(Path(session_dir))
            changed = session.undo()
            if changed:
                # Resume from the image owning the restored answer.
                key = session.last_undone["key"]
                if key:
                    image_id = key.split("::", 1)[0]
                    session.data["current_image_index"] = next(i for i, image in enumerate(session.data["images"]) if image["id"] == image_id)
                    session.save()
            return render_task(session_dir)

        undo.click(undo_and_render, [session_state], output_components)

        def update_settings(session_dir, relation_threshold):
            session = BatchSession(Path(session_dir))
            session.set_settings(relation_threshold)
            return render_task(session_dir)

        apply_settings.click(update_settings, [session_state, threshold], output_components)
        export.click(lambda session_dir: str(BatchSession(Path(session_dir)).export()), [session_state], export)
    # Gradio 6 accepts custom styling and client JS at launch time.
    app.labeler_css = css
    app.labeler_js = shortcuts
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--checkpoint", type=Path, help="SGG .pth file; auto-discovered when omitted.")
    parser.add_argument("--config", help="Checkpoint config YAML; inferred when omitted.")
    parser.add_argument("--yolo-root", type=Path, help="Directory containing YOLO .pt files.")
    parser.add_argument("--yolo-checkpoint", type=Path, help="YOLO .pt file; newest under --yolo-root when omitted.")
    parser.add_argument("--psg-dir", type=Path, required=True, help="PSG coco_format directory.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "exports")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    checkpoint = args.checkpoint or discover_checkpoint(args.checkpoint_root)
    yolo_checkpoint = args.yolo_checkpoint or discover_yolo(args.yolo_root)
    config = find_config(checkpoint, args.config)
    print(f"SGG checkpoint: {checkpoint}")
    print(f"YOLO checkpoint: {yolo_checkpoint or 'from config'}")
    predictor = Predictor(checkpoint, config, yolo_checkpoint, args.psg_dir, args.device)
    app = make_app(predictor, args.output_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, css=app.labeler_css, js=app.labeler_js)


if __name__ == "__main__":
    main()
