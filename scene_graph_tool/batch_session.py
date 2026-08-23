"""Persistent, single-annotator batch sessions for the Scene Graph Labeler."""

from __future__ import annotations

import hashlib
import csv
import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_SETTINGS = {
    "top_k_relations": 3,
    "relation_candidate_top_k": 15,
    "object_max_count": 30,
    "object_min_probability": 0.50,
}

# Kept in the CSV schema only so exports from older sessions remain comparable.
LEGACY_SETTING_FIELDS = ("relation_shortlist_multiplier", "relation_min_probability")

CSV_FIELDS = [
    "schema_version", "session_id", "session_created_at", "answer_created_at", "source_archive",
    "source_name", "image_id", "image_checksum",
    "checkpoint_path", "checkpoint_sha256", "config_path", "config_sha256",
    "label_map_path", "label_map_sha256", "yolo_checkpoint_path", "yolo_checkpoint_sha256",
    "relation_id", "prediction_schema_version", "relation_prediction_index",
    "subject_prediction_index", "object_prediction_index", "relation_pair_key",
    "prediction_object_count", "prediction_relation_count", "relation_distribution_json",
    "response", "target_0", "target_1",
    "subject_id", "subject_label", "subject_label_id", "subject_score",
    "subject_x1", "subject_y1", "subject_x2", "subject_y2",
    "object_id", "object_label", "object_label_id", "object_score",
    "object_x1", "object_y1", "object_x2", "object_y2",
    "predicate_1_id", "predicate_1_name", "predicate_1_probability",
    "predicate_2_id", "predicate_2_name", "predicate_2_probability",
    "model_best_predicate", "model_best_probability", "background_probability",
    "relation_score", "triplet_score", "relative_margin", "uncertainty_score",
    "top_k_relations", "relation_candidate_top_k", "object_max_count", "object_min_probability",
    *LEGACY_SETTING_FIELDS,
]


class BatchSession:
    """A small disk-backed state store; all mutations are immediately saved."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / "session.json"
        self.data = self._read()
        self._migrate_settings()
        self._migrate_legacy_predictions()
        self.last_undone = None

    @classmethod
    def create(
        cls,
        uploads: str | Path | list[str] | list[Path],
        exports_root: Path,
        top_k_relations: int = 3,
        relation_candidate_top_k: int = 15,
        provenance: dict | None = None,
    ):
        """Create a session from one ZIP or one/more individual image files."""
        upload_paths = [Path(path) for path in (uploads if isinstance(uploads, list) else [uploads])]
        if not upload_paths:
            raise ValueError("Upload a ZIP or at least one image.")
        session_dir = Path(exports_root) / f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        source_names = {}
        is_zip = [path.suffix.lower() == ".zip" for path in upload_paths]
        if any(is_zip) and not all(is_zip):
            shutil.rmtree(session_dir, ignore_errors=True)
            raise ValueError("Upload either one ZIP or individual images, not both together.")
        try:
            if all(is_zip):
                if len(upload_paths) != 1:
                    raise ValueError("Upload one ZIP at a time.")
                with zipfile.ZipFile(upload_paths[0]) as bundle:
                    for member in bundle.infolist():
                        member_path = Path(member.filename)
                        if member.is_dir() or member_path.suffix.lower() not in IMAGE_SUFFIXES:
                            continue
                        # Never allow archive members to escape the session directory.
                        destination = (input_dir / member_path).resolve()
                        if input_dir.resolve() not in destination.parents:
                            raise ValueError("ZIP contains an unsafe file path.")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.open(member) as source, destination.open("wb") as target:
                            shutil.copyfileobj(source, target)
                        source_names[destination.relative_to(input_dir).as_posix()] = member_path.as_posix()
            else:
                for index, source in enumerate(upload_paths):
                    if source.suffix.lower() not in IMAGE_SUFFIXES:
                        raise ValueError("Supported image formats are jpg, jpeg, png, and webp.")
                    destination = input_dir / f"{index:04d}_{source.name}"
                    shutil.copy2(source, destination)
                    source_names[destination.name] = source.name
        except (zipfile.BadZipFile, ValueError) as exc:
            shutil.rmtree(session_dir, ignore_errors=True)
            if isinstance(exc, zipfile.BadZipFile):
                raise ValueError("The uploaded file is not a valid ZIP archive.") from exc
            raise

        images = sorted(
            path.relative_to(input_dir).as_posix()
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise ValueError("The ZIP does not contain a supported image (jpg, jpeg, png, webp).")
        settings = dict(DEFAULT_SETTINGS)
        settings["top_k_relations"] = int(top_k_relations)
        settings["relation_candidate_top_k"] = int(relation_candidate_top_k)
        data = {
            "schema_version": 6,
            "relation_selection_version": 3,
            "session_id": session_dir.name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source_archive": upload_paths[0].name if len(upload_paths) == 1 else f"{len(upload_paths)} uploaded images",
            "provenance": dict(provenance or {}),
            "settings": settings,
            "images": [
                {
                    "id": name,
                    "source_name": source_names.get(name, name),
                    "sha256": cls._sha256(input_dir / name),
                    "status": "pending",
                }
                for name in images
            ],
            "current_image_index": 0,
            "current_relation_id": None,
            "predictions": {},
            "prediction_index": {},
            "answers": {},
            "history": [],
        }
        instance = cls.__new__(cls)
        instance.directory, instance.path, instance.data = session_dir, session_dir / "session.json", data
        instance.last_undone = None
        instance.save()
        return instance

    @staticmethod
    def _sha256(path: Path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _read(self):
        with self.path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(self.path)

    @property
    def input_dir(self):
        return self.directory / "input"

    @property
    def predictions_dir(self):
        return self.directory / "predictions"

    def _prediction_path(self, image_id: str):
        return self.predictions_dir / f"{hashlib.sha256(image_id.encode()).hexdigest()}.json"

    def _migrate_settings(self):
        """Fill selector defaults and safely migrate sessions from older ranking."""
        settings = self.data.setdefault("settings", {})
        changed = False
        if "top_k_relations" not in settings:
            settings["top_k_relations"] = int(settings.get("top_k", DEFAULT_SETTINGS["top_k_relations"]))
            changed = True
        if "object_max_count" not in settings and "object_top_k" in settings:
            settings["object_max_count"] = int(settings.pop("object_top_k"))
            changed = True
        if "relation_candidate_top_k" not in settings:
            multiplier = int(settings.get("relation_shortlist_multiplier", 5))
            settings["relation_candidate_top_k"] = int(settings["top_k_relations"]) * multiplier
            changed = True
        for name, value in DEFAULT_SETTINGS.items():
            if name not in settings:
                settings[name] = value
                changed = True
        for name in LEGACY_SETTING_FIELDS:
            if name in settings:
                settings.pop(name)
                changed = True
        for image in self.data.get("images", []):
            if "source_name" not in image:
                image["source_name"] = image.get("id")
                changed = True
            if "sha256" not in image:
                image_path = self.input_dir / image.get("id", "")
                if image_path.is_file():
                    image["sha256"] = self._sha256(image_path)
                    changed = True
        if "session_id" not in self.data:
            self.data["session_id"] = self.directory.name
            changed = True
        if "provenance" not in self.data:
            self.data["provenance"] = {}
            changed = True
        if self.data.get("relation_selection_version", 1) < 3:
            previously_finished = []
            for index, image in enumerate(self.data.get("images", [])):
                if image.get("status") in {"done", "no_eligible_relations"}:
                    image["status"] = "pending"
                    previously_finished.append(index)
            if previously_finished:
                self.data["current_image_index"] = min(
                    int(self.data.get("current_image_index", 0)), min(previously_finished)
                )
            self.data["current_relation_id"] = None
            self.data["relation_selection_version"] = 3
            changed = True
        if self.data.get("schema_version", 0) < 6:
            self.data["schema_version"] = 6
            changed = True
        if changed:
            self.save()

    def _migrate_legacy_predictions(self):
        """Move predictions from pre-cache sessions out of the hot session file."""
        legacy = self.data.get("predictions", {})
        if not legacy or "prediction_index" in self.data:
            return
        self.data["prediction_index"] = {}
        for image_id, prediction in legacy.items():
            self._write_prediction(image_id, prediction)
        self.data["predictions"] = {}
        self.save()

    def _write_prediction(self, image_id: str, prediction: dict):
        destination = self._prediction_path(image_id)
        destination.parent.mkdir(exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(prediction, handle, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(destination)
        self.data.setdefault("prediction_index", {})[image_id] = destination.name

    def has_prediction(self, image_id: str):
        return image_id in self.data.get("prediction_index", {}) or image_id in self.data.get("predictions", {})

    def get_prediction(self, image_id: str):
        if image_id in self.data.get("predictions", {}):
            return self.data["predictions"][image_id]
        filename = self.data.get("prediction_index", {}).get(image_id)
        if filename is None:
            return None
        with (self.predictions_dir / filename).open(encoding="utf-8") as handle:
            return json.load(handle)

    def image_path(self, image_id: str):
        return self.input_dir / image_id

    def current_image(self):
        index = self.data["current_image_index"]
        return self.data["images"][index] if 0 <= index < len(self.data["images"]) else None

    def set_prediction(self, image_id: str, prediction: dict):
        self._write_prediction(image_id, prediction)
        self.data.setdefault("predictions", {}).pop(image_id, None)
        self.save()

    def set_settings(self, settings: dict):
        """Replace acquisition settings after validating their supported ranges."""
        values = {
            "top_k_relations": int(settings["top_k_relations"]),
            "relation_candidate_top_k": int(settings["relation_candidate_top_k"]),
            "object_max_count": int(settings["object_max_count"]),
            "object_min_probability": float(settings["object_min_probability"]),
        }
        if not 1 <= values["top_k_relations"] <= 10:
            raise ValueError("Relations to label (M) must be between 1 and 10.")
        if not 1 <= values["relation_candidate_top_k"] <= 1000:
            raise ValueError("Candidate relations (K) must be between 1 and 1000.")
        if values["relation_candidate_top_k"] < values["top_k_relations"]:
            raise ValueError("Candidate relations (K) must be at least relations to label (M).")
        if not 2 <= values["object_max_count"] <= 100:
            raise ValueError("Maximum objects must be between 2 and 100.")
        if not 0 <= values["object_min_probability"] <= 1:
            raise ValueError("Minimum object probability must be between 0 and 1.")
        self.data["settings"] = values
        self.data["current_relation_id"] = None
        self.save()

    def set_top_k_relations(self, top_k_relations: int):
        settings = dict(self.data["settings"])
        settings["top_k_relations"] = top_k_relations
        self.set_settings(settings)

    def answer_key(self, image_id: str, relation_id: int):
        return f"{image_id}::{relation_id}"

    def record_answer(self, image_id: str, relation_id: int, answer: dict):
        key = self.answer_key(image_id, relation_id)
        previous = self.data["answers"].get(key)
        self.data["history"].append({"key": key, "previous": previous})
        self.data["answers"][key] = answer
        self.save()

    def undo(self):
        if not self.data["history"]:
            self.last_undone = None
            return False
        action = self.data["history"].pop()
        self.last_undone = action
        if action["previous"] is None:
            self.data["answers"].pop(action["key"], None)
        else:
            self.data["answers"][action["key"]] = action["previous"]
        self.save()
        return True

    def export(self):
        destination = self.directory / "scene_graph_preferences.csv"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            prediction_cache = {}
            for answer in self.data["answers"].values():
                image_id = answer.get("image_id")
                if image_id not in prediction_cache:
                    prediction_cache[image_id] = self.get_prediction(image_id)
                writer.writerow(self._csv_row(answer, prediction_cache[image_id]))
        temporary.replace(destination)
        return destination

    def _csv_row(self, answer: dict, prediction: dict | None = None):
        subject = answer.get("subject", {})
        obj = answer.get("object", {})
        subject_box = list(subject.get("box_xyxy", [])) + [None] * 4
        object_box = list(obj.get("box_xyxy", [])) + [None] * 4
        predicates = list(answer.get("presented_predicates", [])) + [{}, {}]
        target = answer.get("target_tensor") or []
        selection = answer.get("selection", {})
        settings = dict(answer.get("settings", self.data["settings"]))
        image_metadata = next(
            (item for item in self.data.get("images", []) if item.get("id") == answer.get("image_id")),
            {},
        )
        provenance = self.data.get("provenance", {})
        reference = dict(answer.get("prediction_reference") or {})
        relation = next(
            (
                item for item in (prediction or {}).get("relations", [])
                if int(item.get("id", -1)) == int(answer.get("relation_id", -2))
            ),
            None,
        )
        if relation is not None:
            reference = {
                "schema_version": 1,
                "relation_index": int(relation["id"]),
                "subject_index": int(relation["subject_id"]),
                "object_index": int(relation["object_id"]),
                "object_count": len((prediction or {}).get("objects", [])),
                "relation_count": len((prediction or {}).get("relations", [])),
                "background_probability": relation.get("background_probability"),
                "predicate_probabilities": relation.get("predicate_probabilities", {}),
            }
        subject_index = reference.get("subject_index", subject.get("id"))
        object_index = reference.get("object_index", obj.get("id"))
        relation_index = reference.get("relation_index", answer.get("relation_id"))
        distribution = {
            "background_probability": reference.get(
                "background_probability", selection.get("background_probability")
            ),
            "predicate_probabilities": reference.get("predicate_probabilities", {
                str(item.get("name")): item.get("probability")
                for item in predicates[:2] if item.get("name")
            }),
        }
        if "object_max_count" not in settings and "object_top_k" in settings:
            settings["object_max_count"] = settings["object_top_k"]
        return {
            "schema_version": self.data.get("schema_version"),
            "session_id": answer.get("session_id", self.data.get("session_id", self.directory.name)),
            "session_created_at": self.data.get("created_at"),
            "answer_created_at": answer.get("created_at"),
            "source_archive": self.data.get("source_archive"),
            "source_name": answer.get("source_name", image_metadata.get("source_name")),
            "image_id": answer.get("image_id"),
            "image_checksum": answer.get("image_checksum", image_metadata.get("sha256")),
            **{name: provenance.get(name) for name in (
                "checkpoint_path", "checkpoint_sha256", "config_path", "config_sha256",
                "label_map_path", "label_map_sha256", "yolo_checkpoint_path", "yolo_checkpoint_sha256",
            )},
            "relation_id": answer.get("relation_id"),
            "prediction_schema_version": reference.get("schema_version", 1),
            "relation_prediction_index": relation_index,
            "subject_prediction_index": subject_index,
            "object_prediction_index": object_index,
            "relation_pair_key": (
                f"{subject_index}:{object_index}"
                if subject_index is not None and object_index is not None else None
            ),
            "prediction_object_count": reference.get("object_count"),
            "prediction_relation_count": reference.get("relation_count"),
            "relation_distribution_json": json.dumps(
                distribution, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            "response": answer.get("response"),
            "target_0": target[0] if len(target) > 0 else None,
            "target_1": target[1] if len(target) > 1 else None,
            "subject_id": subject.get("id"), "subject_label": subject.get("label"),
            "subject_label_id": subject.get("label_id"), "subject_score": subject.get("score"),
            "subject_x1": subject_box[0], "subject_y1": subject_box[1],
            "subject_x2": subject_box[2], "subject_y2": subject_box[3],
            "object_id": obj.get("id"), "object_label": obj.get("label"),
            "object_label_id": obj.get("label_id"), "object_score": obj.get("score"),
            "object_x1": object_box[0], "object_y1": object_box[1],
            "object_x2": object_box[2], "object_y2": object_box[3],
            "predicate_1_id": predicates[0].get("predicate_id"),
            "predicate_1_name": predicates[0].get("name"),
            "predicate_1_probability": predicates[0].get("probability"),
            "predicate_2_id": predicates[1].get("predicate_id"),
            "predicate_2_name": predicates[1].get("name"),
            "predicate_2_probability": predicates[1].get("probability"),
            "model_best_predicate": selection.get("model_best_predicate"),
            "model_best_probability": selection.get("model_best_probability"),
            "background_probability": selection.get("background_probability"),
            "relation_score": selection.get("relation_score", selection.get("model_best_probability")),
            "triplet_score": selection.get("triplet_score"),
            "relative_margin": selection.get("relative_margin"),
            "uncertainty_score": selection.get("uncertainty_score"),
            **{name: settings.get(name) for name in (*DEFAULT_SETTINGS, *LEGACY_SETTING_FIELDS)},
        }
