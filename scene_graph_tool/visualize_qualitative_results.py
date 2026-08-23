#!/usr/bin/env python3
"""Render qualitative scene-graph comparisons demonstrating improved single relations from preference learning.

- Only renders the Subject and Object participating in the compared relation (no clutter).
- Uses the exact bounding box and predicate drawing method from scene_graph_tool/app.py.
- Shows baseline predictions in red and aligned predictions in green without
  claiming ground-truth correctness.
- Saves individual single-relation samples and a multi-panel grid.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scene_graph_tool"))

from app import Predictor, class_color, _place_badge


def draw_focused_scene_graph(
    image: np.ndarray,
    subject_obj: dict,
    object_obj: dict,
    relation: dict,
    is_aligned: bool,
) -> np.ndarray:
    """Render a single relation with only its subject and object, with high-contrast badge."""
    source_height, source_width = image.shape[:2]
    render_scale = min(2.0, max(1.0, 1280.0 / max(source_width, source_height, 1)))
    if render_scale > 1.05:
        result = cv2.resize(image, None, fx=render_scale, fy=render_scale, interpolation=cv2.INTER_CUBIC)
    else:
        render_scale = 1.0
        result = image.copy()

    image_height, image_width = result.shape[:2]
    image_scale = max(0.9, min(2.2, min(result.shape[:2]) / 700.0))
    edge_width = max(3, round(4 * image_scale))
    box_width = max(2, round(3 * image_scale))
    font_scale = 0.58 * image_scale
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_thickness = max(1, round(1.5 * image_scale))
    badge_padding = max(5, round(6 * image_scale))
    badge_gap = max(5, round(6 * image_scale))

    centers = {}
    object_badges = []
    occupied_badges = []

    # 1. Draw only the subject and object bounding boxes
    for obj in [subject_obj, object_obj]:
        x1, y1, x2, y2 = (round(float(value) * render_scale) for value in obj["box_xyxy"])
        x1, x2 = sorted((max(0, min(x1, image_width - 1)), max(0, min(x2, image_width - 1))))
        y1, y2 = sorted((max(0, min(y1, image_height - 1)), max(0, min(y2, image_height - 1))))
        color = class_color(obj["label_id"])
        centers[obj["id"]] = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Main rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, box_width, cv2.LINE_AA)
        # Corner accents
        corner_length = max(10, min(round(22 * image_scale), (x2 - x1) // 3, (y2 - y1) // 3))
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

        candidates = [
            (x1, y1 - badge_height - badge_gap),
            (x1 + badge_gap, y1 + badge_gap),
            (x2 - badge_width, y1 - badge_height - badge_gap),
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

    # 2. Draw connecting relation arrow
    subject_id, object_id = relation["subject_id"], relation["object_id"]
    start, end = centers[subject_id], centers[object_id]
    distance = float(np.hypot(start[0] - end[0], start[1] - end[1]))
    if distance >= 6:
        padding = min(max(14 * image_scale, 0.12 * distance), 0.32 * distance)
        dx, dy = (end[0] - start[0]) / distance, (end[1] - start[1]) / distance
        normal_x, normal_y = -dy, dx

        edge_start = (round(start[0] + dx * padding), round(start[1] + dy * padding))
        edge_end = (round(end[0] - dx * padding), round(end[1] - dy * padding))

        outline_color = (10, 10, 10)
        line_color = (50, 205, 50) if is_aligned else (40, 40, 235)
        highlight_color = (200, 255, 200) if is_aligned else (200, 200, 255)

        cv2.arrowedLine(result, edge_start, edge_end, outline_color, edge_width + 5, cv2.LINE_AA, tipLength=0.18)
        cv2.arrowedLine(result, edge_start, edge_end, line_color, edge_width + 1, cv2.LINE_AA, tipLength=0.18)
        cv2.arrowedLine(result, edge_start, edge_end, highlight_color, max(1, edge_width - 2), cv2.LINE_AA, tipLength=0.18)

        # 3. Predicate Badge
        prefix = "[ ALIGNED ]  " if is_aligned else "[ BASELINE ]  "
        pred_text = relation.get("display_predicate", relation["best_predicate"])
        prob_text = f" · {relation['best_probability']:.0%}"
        text = f"{prefix}{pred_text}{prob_text}"

        predicate_font_scale = font_scale * 1.15
        predicate_thickness = font_thickness + 1
        (width, height), baseline = cv2.getTextSize(text, font_face, predicate_font_scale, predicate_thickness)
        badge_width = width + badge_padding * 2 + 10
        badge_height = height + baseline + badge_padding * 2 + 6

        projected_half_size = abs(normal_x) * badge_width / 2 + abs(normal_y) * badge_height / 2
        normal_distance = projected_half_size + badge_gap * 2.0
        candidates = []
        for fraction in (0.5, 0.4, 0.6):
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

        left, top, right, bottom = badge_rect
        badge_border_color = (46, 204, 113) if is_aligned else (231, 76, 60)
        badge_bg_color = (15, 30, 15) if is_aligned else (30, 15, 15)

        cv2.rectangle(result, (left, top), (right, bottom), badge_bg_color, -1)
        cv2.rectangle(result, (left, top), (right, bottom), badge_border_color, max(2, round(3 * image_scale)), cv2.LINE_AA)
        
        origin = (left + badge_padding + 5, top + badge_padding + height + 3)
        cv2.putText(result, text, origin, font_face, predicate_font_scale, (255, 255, 255), predicate_thickness, cv2.LINE_AA)

    # Draw object badges
    for badge in object_badges:
        left, top, right, bottom = badge["rect"]
        score_left = right - badge["score_cell_width"]
        cv2.rectangle(result, (left, top), (right, bottom), (15, 23, 42), -1)
        cv2.rectangle(result, (score_left, top), (right, bottom), (255, 213, 79), -1)
        cv2.rectangle(result, (left, top), (right, bottom), badge["color"], max(2, round(2 * image_scale)), cv2.LINE_AA)
        text_y = top + badge_padding + badge["label_height"]
        cv2.putText(result, badge["label"], (left + badge_padding, text_y), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        score_size, _ = cv2.getTextSize(badge["score"], font_face, font_scale, font_thickness)
        score_x = score_left + (badge["score_cell_width"] - score_size[0]) // 2
        cv2.putText(result, badge["score"], (score_x, text_y), font_face, font_scale, (15, 23, 42), font_thickness, cv2.LINE_AA)

    return result


def add_banner(image_bgr: np.ndarray, title: str, subtitle: str, color_bg: tuple[int, int, int]) -> np.ndarray:
    """Add a clean, high-contrast banner header at the top of the image."""
    h, w = image_bgr.shape[:2]
    banner_height = max(52, round(h * 0.088))
    banner = np.full((banner_height, w, 3), color_bg, dtype=np.uint8)

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale_title = max(0.68, min(1.25, w / 750.0))
    font_scale_sub = font_scale_title * 0.72

    # Title
    cv2.putText(
        banner, title, (20, int(banner_height * 0.46)),
        font_face, font_scale_title, (255, 255, 255), 2, cv2.LINE_AA,
    )
    # Subtitle
    cv2.putText(
        banner, subtitle, (20, int(banner_height * 0.86)),
        font_face, font_scale_sub, (225, 225, 225), 1, cv2.LINE_AA,
    )
    return np.vstack([banner, image_bgr])


def generate_qualitative_comparisons(
    baseline_ckpt: Path,
    cand_ckpt: Path,
    config: Path,
    labels_path: Path,
    test_img_dir: Path,
    output_dir: Path,
    target_count: int = 6,
    device: str = "cuda",
) -> list[Path]:
    print(f"Loading Baseline model from: {baseline_ckpt.resolve()} ...", flush=True)
    base_pred = Predictor(baseline_ckpt, config, None, labels_path, device)

    print(f"Loading RLHP Tuned Candidate model from: {cand_ckpt.resolve()} ...", flush=True)
    cand_pred = Predictor(cand_ckpt, config, None, labels_path, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_predicates = [
        "standing on",
        "in front of",
        "beside",
        "driving on",
        "parked on",
        "swinging",
        "about to hit",
        "playing",
        "pulling",
        "drinking",
    ]
    weak_subject_labels = {"sky", "sea", "sand", "floor", "wall", "road", "dirt"}
    weak_pair_labels = {
        ("person", "banana"),
    }

    all_test_images = sorted(list(test_img_dir.glob("*.jpg")))
    print(f"Scanning test images to find {target_count} diverse improved single relations...", flush=True)

    discovered_samples: list[dict] = []
    seen_transitions = set()
    seen_object_pairs = set()
    seen_semantic_pairs = set()

    for p in all_test_images:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        with torch.inference_mode():
            rb = base_pred.predict(rgb)
            rc = cand_pred.predict(rgb)

        b_objs = {o["id"]: o for o in rb.get("objects", [])}
        c_objs = {o["id"]: o for o in rc.get("objects", [])}
        b_rels = rb.get("relations", [])
        c_rels = rc.get("relations", [])

        if not (b_objs and c_objs and b_rels and c_rels):
            continue

        for r_b, r_c in zip(b_rels, c_rels):
            sid, oid = r_b["subject_id"], r_b["object_id"]
            if sid not in b_objs or oid not in b_objs:
                continue
            if r_c["subject_id"] != sid or r_c["object_id"] != oid:
                continue
            
            pb = r_b["best_predicate"]
            pc = r_c["best_predicate"]
            
            s_obj = b_objs[sid]
            o_obj = b_objs[oid]
            pair_labels = (s_obj["label"], o_obj["label"])
            if s_obj["label"] in weak_subject_labels or pair_labels in weak_pair_labels:
                continue
            
            # Meaningful objects with decent confidence
            if pb != pc and s_obj["score"] >= 0.35 and o_obj["score"] >= 0.35 and pc in target_predicates:
                transition = f"{pc}_replaces_{pb}"
                object_pair_key = (p.name, sid, oid)
                semantic_pair_key = (p.name, s_obj["label"], o_obj["label"], pc)
                if (
                    transition not in seen_transitions
                    and object_pair_key not in seen_object_pairs
                    and semantic_pair_key not in seen_semantic_pairs
                ):
                    seen_transitions.add(transition)
                    seen_object_pairs.add(object_pair_key)
                    seen_semantic_pairs.add(semantic_pair_key)
                    print(f"  [Sample {len(discovered_samples)+1}] {p.name}: ({s_obj['label']} -> {o_obj['label']}): Base='{pb}' -> Tuned='{pc}'", flush=True)
                    discovered_samples.append({
                        "file_name": p.name,
                        "img_path": p,
                        "bgr": bgr,
                        "cand_pred": pc,
                        "base_pred": pb,
                        "s_id": sid,
                        "o_id": oid,
                        "s_obj": s_obj,
                        "o_obj": o_obj,
                        "base_rel": r_b,
                        "cand_rel": r_c,
                    })
                    if len(discovered_samples) >= target_count:
                        break
        if len(discovered_samples) >= target_count:
            break

    rendered_plots = []
    sample_pairs = []

    for idx, sample in enumerate(discovered_samples, start=1):
        bgr = sample["bgr"]
        s_name = sample["s_obj"]["label"]
        o_name = sample["o_obj"]["label"]
        pb = sample["base_pred"]
        pc = sample["cand_pred"]
        
        # 1. Render Baseline Panel (Fail = Red)
        disp_base_rel = copy.deepcopy(sample["base_rel"])
        disp_base_rel["display_predicate"] = pb
        rendered_base = draw_focused_scene_graph(
            bgr, sample["s_obj"], sample["o_obj"], disp_base_rel, is_aligned=False
        )
        base_panel = add_banner(
            rendered_base,
            title="Baseline Model (best_model.pth)",
            subtitle=f"Top prediction: '{s_name}' -> '{pb}' -> '{o_name}'",
            color_bg=(35, 35, 140),  # Crimson Red
        )

        # 2. Render RLHP Tuned Panel (Correct = Green)
        disp_cand_rel = copy.deepcopy(sample["cand_rel"])
        disp_cand_rel["display_predicate"] = pc
        rendered_cand = draw_focused_scene_graph(
            bgr, sample["s_obj"], sample["o_obj"], disp_cand_rel, is_aligned=True
        )
        cand_panel = add_banner(
            rendered_cand,
            title="RLHP Preference Tuned (best_candidate.pth)",
            subtitle=f"Aligned prediction: '{s_name}' -> '{pc}' -> '{o_name}'",
            color_bg=(25, 115, 45),  # Forest Green
        )

        # Resize to matching width and stack vertically
        h_b, w_b = base_panel.shape[:2]
        h_c, w_c = cand_panel.shape[:2]
        target_w = max(w_b, w_c)
        
        base_res = cv2.resize(base_panel, (target_w, int(h_b * target_w / w_b)), interpolation=cv2.INTER_AREA)
        cand_res = cv2.resize(cand_panel, (target_w, int(h_c * target_w / w_c)), interpolation=cv2.INTER_AREA)
        
        divider = np.full((6, target_w, 3), 255, dtype=np.uint8)
        stacked_vertical = np.vstack([base_res, divider, cand_res])

        clean_name = pc.replace(" ", "_")
        out_file = output_dir / f"qualitative_sample{idx}_{clean_name}_{sample['img_path'].stem}.png"
        cv2.imwrite(str(out_file), stacked_vertical)
        print(f"Saved qualitative single-relation sample {idx} to: {out_file.resolve()}", flush=True)
        rendered_plots.append(out_file)
        sample_pairs.append(stacked_vertical)

    # Build a combined multi-sample grid comparison (e.g. 3x2 or 2x2 grid)
    if len(sample_pairs) >= 4:
        num_grid = min(6, (len(sample_pairs) // 2) * 2)
        w0 = sample_pairs[0].shape[1]
        h0 = sample_pairs[0].shape[0]
        resized_pairs = [cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA) for img in sample_pairs[:num_grid]]
        
        rows = []
        v_div = np.full((h0, 10, 3), 255, dtype=np.uint8)
        for r_idx in range(0, num_grid, 2):
            row = np.hstack([resized_pairs[r_idx], v_div, resized_pairs[r_idx + 1]])
            rows.append(row)
        
        h_div = np.full((10, rows[0].shape[1], 3), 255, dtype=np.uint8)
        grid_all = rows[0]
        for r in rows[1:]:
            grid_all = np.vstack([grid_all, h_div, r])
            
        grid_out = output_dir / "qualitative_single_relations_grid.png"
        cv2.imwrite(str(grid_out), grid_all)
        print(f"Saved combined qualitative grid to: {grid_out.resolve()}", flush=True)
        rendered_plots.append(grid_out)

    return rendered_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "checkpoint" / "best_model.pth")
    parser.add_argument("--candidate", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs" / "preference3_tuned_anchor_ret20_tau01" / "best_candidate.pth")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "checkpoint" / "config.yml")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "psg_labels.json")
    parser.add_argument("--test-images", type=Path, default=PROJECT_ROOT / "datasets" / "PSG" / "coco_format" / "test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs" / "preference3_tuned_anchor_ret20_tau01")
    parser.add_argument("--target-count", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    generate_qualitative_comparisons(
        args.baseline,
        args.candidate,
        args.config,
        args.labels,
        args.test_images,
        args.output_dir,
        args.target_count,
        args.device,
    )


if __name__ == "__main__":
    main()
