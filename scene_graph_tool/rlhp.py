#!/usr/bin/env python3
"""Offline RLHP pipeline for Scene Graph Labeler preference sessions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs"
DEFAULT_LABELS = PROJECT_ROOT / "scene_graph_tool" / "psg_labels.json"
DEFAULT_PSG_TEST = PROJECT_ROOT / "datasets" / "PSG" / "coco_format" / "test"
TRAINABLE_SUFFIXES = (
    "roi_heads.relation.predictor.proj_head",
    "roi_heads.relation.predictor.logit_scale",
)
REQUIRED_PROVENANCE = ("checkpoint_sha256", "config_sha256", "label_map_sha256")


class RLHPError(RuntimeError):
    """Expected pipeline validation or execution failure."""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _optional_number(value: str | None, cast=float):
    if value is None or str(value).strip() == "":
        return None
    return cast(value)


def _json_object(value: str | None) -> dict:
    if value is None or not str(value).strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_preference_source(path: str | Path) -> dict:
    """Load a native session JSON or a self-contained CSV export."""
    source = Path(path)
    if source.suffix.lower() != ".csv":
        return read_json(source)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RLHPError(f"Preference CSV is empty: {source}")
    first = rows[0]
    provenance = {
        name: first.get(name)
        for name in (
            "checkpoint_path", "checkpoint_sha256", "config_path", "config_sha256",
            "label_map_path", "label_map_sha256", "yolo_checkpoint_path", "yolo_checkpoint_sha256",
        )
        if first.get(name)
    }
    images, answers, cached_predictions = {}, {}, {}
    for index, row in enumerate(rows):
        image_id = row.get("image_id", "")
        source_name = row.get("source_name") or image_id
        checksum = row.get("image_checksum") or ""
        images.setdefault(image_id, {
            "id": image_id, "source_name": source_name, "sha256": checksum, "status": "done",
        })
        target_0, target_1 = _optional_number(row.get("target_0")), _optional_number(row.get("target_1"))
        target = None if target_0 is None or target_1 is None else [target_0, target_1]
        relation_id = _optional_number(row.get("relation_id"), int)
        subject_index = _optional_number(
            row.get("subject_prediction_index") or row.get("subject_id"), int
        )
        object_index = _optional_number(
            row.get("object_prediction_index") or row.get("object_id"), int
        )
        relation_index = _optional_number(
            row.get("relation_prediction_index") or row.get("relation_id"), int
        )
        distribution = _json_object(row.get("relation_distribution_json"))
        predicates = [
            {
                "predicate_id": _optional_number(row.get(f"predicate_{slot}_id"), int),
                "name": row.get(f"predicate_{slot}_name"),
                "probability": _optional_number(row.get(f"predicate_{slot}_probability")) or 0.0,
            }
            for slot in (1, 2)
        ]
        answer = {
            "session_id": row.get("session_id"),
            "created_at": row.get("answer_created_at"),
            "image_id": image_id,
            "source_name": source_name,
            "image_checksum": checksum,
            "relation_id": relation_id,
            "response": row.get("response"),
            "target_tensor": target,
            "subject": {
                "id": _optional_number(row.get("subject_id"), int),
                "label": row.get("subject_label"),
                "label_id": _optional_number(row.get("subject_label_id"), int),
                "score": _optional_number(row.get("subject_score")),
                "box_xyxy": [_optional_number(row.get(f"subject_{axis}")) for axis in ("x1", "y1", "x2", "y2")],
            },
            "object": {
                "id": _optional_number(row.get("object_id"), int),
                "label": row.get("object_label"),
                "label_id": _optional_number(row.get("object_label_id"), int),
                "score": _optional_number(row.get("object_score")),
                "box_xyxy": [_optional_number(row.get(f"object_{axis}")) for axis in ("x1", "y1", "x2", "y2")],
            },
            "presented_predicates": predicates,
            "prediction_reference": {
                "schema_version": _optional_number(row.get("prediction_schema_version"), int) or 0,
                "relation_index": relation_index,
                "subject_index": subject_index,
                "object_index": object_index,
                "pair_key": row.get("relation_pair_key"),
                "object_count": _optional_number(row.get("prediction_object_count"), int),
                "relation_count": _optional_number(row.get("prediction_relation_count"), int),
                "background_probability": distribution.get("background_probability"),
                "predicate_probabilities": distribution.get("predicate_probabilities", {}),
            },
        }
        answers[f"{image_id}::{relation_id}::{index}"] = answer
        if subject_index is not None and object_index is not None and relation_index is not None:
            cached = cached_predictions.setdefault(image_id, {"objects": {}, "relations": {}})
            for object_id, record in ((subject_index, answer["subject"]), (object_index, answer["object"])):
                cached["objects"].setdefault(object_id, {
                    "id": object_id,
                    "label": record.get("label"),
                    "label_id": record.get("label_id"),
                    "score": record.get("score"),
                    "box_xyxy": record.get("box_xyxy"),
                })
            foreground = distribution.get("predicate_probabilities") or {
                item["name"]: item["probability"] for item in predicates if item.get("name")
            }
            cached["relations"].setdefault(relation_index, {
                "id": relation_index,
                "subject_id": subject_index,
                "object_id": object_index,
                "background_probability": distribution.get("background_probability"),
                "predicate_probabilities": foreground,
            })
    predictions = {
        image_id: {
            "objects": [items[key] for key in sorted(items)] if (items := value["objects"]) else [],
            "relations": [items[key] for key in sorted(items)] if (items := value["relations"]) else [],
        }
        for image_id, value in cached_predictions.items()
    }
    return {
        "schema_version": _optional_number(first.get("schema_version"), int) or 0,
        "session_id": first.get("session_id") or source.stem,
        "created_at": first.get("session_created_at"),
        "source_archive": first.get("source_archive"),
        "provenance": provenance,
        "images": list(images.values()),
        "answers": answers,
        "predictions": predictions,
        "prediction_index": {},
    }


def normalized_image_names(value: str) -> set[str]:
    """Return stable filename/stem aliases, removing upload numbering prefixes."""
    name = Path(value).name
    name = re.sub(r"^\d{4}_", "", name)
    return {name, Path(name).stem}


def discover_test_image_names(source: str | Path | None) -> set[str]:
    if source is None:
        return set()
    path = Path(source)
    if not path.exists():
        raise RLHPError(f"PSG test source does not exist: {path}")
    names: set[str] = set()
    annotation = path
    if path.is_dir():
        candidate = path / "_annotations.coco.json"
        if candidate.is_file():
            annotation = candidate
        else:
            for image in path.rglob("*"):
                if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    names.update(normalized_image_names(image.name))
            return names
    if annotation.is_file():
        payload = read_json(annotation)
        if isinstance(payload, dict) and isinstance(payload.get("images"), list):
            for image in payload["images"]:
                names.update(normalized_image_names(str(image.get("file_name", image.get("id", "")))))
                if image.get("id") is not None:
                    names.add(str(image["id"]))
        elif isinstance(payload, list):
            for item in payload:
                names.update(normalized_image_names(str(item)))
        else:
            raise RLHPError(f"Unsupported PSG test manifest: {annotation}")
    return names


def discover_sessions(inputs: Iterable[str | Path]) -> list[Path]:
    sessions: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            sessions.add(path.resolve())
        elif path.is_dir():
            direct = path / "session.json"
            if direct.is_file():
                sessions.add(direct.resolve())
            else:
                sessions.update(item.resolve() for item in path.rglob("session.json"))
        else:
            raise RLHPError(f"Session input does not exist: {path}")
    if not sessions:
        raise RLHPError("No session.json files were found.")
    return sorted(sessions)


def load_predicate_map(labels_path: str | Path) -> dict[str, int]:
    labels = read_json(labels_path)
    predicates = labels.get("predicates") if isinstance(labels, dict) else None
    if not isinstance(predicates, list):
        raise RLHPError("Label file must contain a predicates list.")
    return {str(name): index for index, name in enumerate(predicates)}


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    a = [float(value) for value in first]
    b = [float(value) for value in second]
    if len(a) != 4 or len(b) != 4:
        return 0.0
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def _load_cached_prediction(session_path: Path, data: dict, image_id: str) -> dict | None:
    if image_id in data.get("predictions", {}):
        return data["predictions"][image_id]
    filename = data.get("prediction_index", {}).get(image_id)
    if filename:
        path = session_path.parent / "predictions" / filename
        if path.is_file():
            return read_json(path)
    return None


def _cached_pair_matches(prediction: dict | None, answer: dict, threshold: float) -> bool:
    if prediction is None:
        # Real model matching is mandatory before real training. Structural audit
        # remains useful when portable sessions omit their inference cache.
        return True
    objects = prediction.get("objects", [])

    def best(recorded: dict) -> int | None:
        candidates = [
            (box_iou(recorded.get("box_xyxy", []), item.get("box_xyxy", [])), int(item["id"]))
            for item in objects
            if item.get("label_id") == recorded.get("label_id")
        ]
        if not candidates:
            return None
        overlap, object_id = max(candidates)
        return object_id if overlap >= threshold else None

    subject_id, object_id = best(answer.get("subject", {})), best(answer.get("object", {}))
    if subject_id is None or object_id is None:
        return False
    relation = next(
        (
            item for item in prediction.get("relations", [])
            if int(item.get("subject_id", -1)) == subject_id and int(item.get("object_id", -1)) == object_id
        ),
        None,
    )
    if relation is None:
        return False
    available = set(relation.get("predicate_probabilities", {}))
    expected = {str(item.get("name")) for item in answer.get("presented_predicates", [])}
    return expected.issubset(available)


def _sample_signature(sample: dict) -> str:
    identity = {
        "image_checksum": sample["image_checksum"],
        "subject": [sample["subject"]["label_id"], [round(float(x), 2) for x in sample["subject"]["box_xyxy"]]],
        "object": [sample["object"]["label_id"], [round(float(x), 2) for x in sample["object"]["box_xyxy"]]],
        "predicates": [item["predicate_id"] for item in sample["predicates"]],
        # Both-wrong votes are audit-only and must never dilute a trainable
        # preference target for the same object/predicate pair.
        "audit_only": sum(sample["target"]) == 0,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _canonicalize_sample(sample: dict) -> dict:
    first, second = sample["predicates"]
    first_key = (int(first["predicate_id"]), str(first["name"]))
    second_key = (int(second["predicate_id"]), str(second["name"]))
    if second_key < first_key:
        sample["predicates"] = [second, first]
        sample["target"] = [sample["target"][1], sample["target"][0]]
    sample["sample_id"] = _sample_signature(sample)
    return sample


def _image_metadata(data: dict, image_id: str) -> dict:
    return next((item for item in data.get("images", []) if item.get("id") == image_id), {})


def audit_sessions(
    session_inputs: Iterable[str | Path],
    labels_path: str | Path,
    psg_test_source: str | Path | None,
    output_dir: str | Path,
    *,
    allow_legacy_provenance: bool = False,
    iou_threshold: float = 0.8,
    min_match_rate: float = 0.95,
    seed: int = 42,
    image_root: str | Path | None = None,
) -> tuple[dict, list[dict]]:
    """Load, validate, aggregate and split preference sessions."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predicate_map = load_predicate_map(labels_path)
    test_names = discover_test_image_names(psg_test_source)
    session_paths = discover_sessions(session_inputs)
    raw_samples: list[dict] = []
    dropped: list[dict] = []
    leakage: list[dict] = []
    provenance_groups: set[tuple[str, str, str]] = set()
    matched = 0
    match_total = 0
    deferred_matching = 0
    checksum_cache: dict[Path, str] = {}
    external_images = Path(image_root).resolve() if image_root else None

    for session_path in session_paths:
        data = load_preference_source(session_path)
        provenance = dict(data.get("provenance") or {})
        missing_provenance = [name for name in REQUIRED_PROVENANCE if not provenance.get(name)]
        if missing_provenance and not allow_legacy_provenance:
            raise RLHPError(
                f"{session_path} lacks provenance fields: {', '.join(missing_provenance)}. "
                "Use --allow-legacy-provenance only for an intentional legacy smoke run."
            )
        if not missing_provenance:
            provenance_groups.add(tuple(str(provenance[name]) for name in REQUIRED_PROVENANCE))

        for answer_key, answer in data.get("answers", {}).items():
            image_id = str(answer.get("image_id", answer_key.split("::", 1)[0]))
            image_meta = _image_metadata(data, image_id)
            source_name = str(answer.get("source_name") or image_meta.get("source_name") or image_id)
            if normalized_image_names(source_name) & test_names:
                leakage.append({"session": str(session_path), "image_id": image_id, "source_name": source_name})
                continue
            target = answer.get("target_tensor")
            if target is None:
                dropped.append({"session": str(session_path), "answer": answer_key, "reason": "skip"})
                continue
            if not isinstance(target, list) or len(target) != 2:
                dropped.append({"session": str(session_path), "answer": answer_key, "reason": "invalid_target"})
                continue
            predicates = answer.get("presented_predicates", [])
            subject, obj = answer.get("subject", {}), answer.get("object", {})
            if len(predicates) != 2 or subject.get("label_id") is None or obj.get("label_id") is None:
                dropped.append({"session": str(session_path), "answer": answer_key, "reason": "incomplete_relation"})
                continue
            normalized_predicates = []
            invalid_predicate = False
            for predicate in predicates:
                name = str(predicate.get("name", ""))
                predicate_id = predicate.get("predicate_id")
                if predicate_id is None:
                    predicate_id = predicate_map.get(name)
                if predicate_id is None or name not in predicate_map or int(predicate_id) != predicate_map[name]:
                    invalid_predicate = True
                    break
                normalized_predicates.append({
                    "predicate_id": int(predicate_id),
                    "name": name,
                    "probability": float(predicate.get("probability", 0.0)),
                })
            if invalid_predicate:
                dropped.append({"session": str(session_path), "answer": answer_key, "reason": "invalid_predicate"})
                continue
            image_path = session_path.parent / "input" / image_id
            if not image_path.is_file() and external_images is not None:
                candidates = (
                    external_images / Path(source_name).name,
                    external_images / Path(image_id).name,
                )
                image_path = next((candidate for candidate in candidates if candidate.is_file()), image_path)
            if not image_path.is_file():
                dropped.append({"session": str(session_path), "answer": answer_key, "reason": "missing_image_file"})
                continue
            checksum = str(answer.get("image_checksum") or image_meta.get("sha256") or "")
            if image_path not in checksum_cache:
                checksum_cache[image_path] = sha256_file(image_path)
            actual_checksum = checksum_cache[image_path]
            if checksum and checksum != actual_checksum:
                dropped.append({"session": str(session_path), "answer": answer_key, "reason": "image_checksum_mismatch"})
                continue
            checksum = actual_checksum
            if not checksum:
                dropped.append({"session": str(session_path), "answer": answer_key, "reason": "missing_image_checksum"})
                continue
            cached_prediction = _load_cached_prediction(session_path, data, image_id)
            match_total += 1
            if cached_prediction is None:
                deferred_matching += 1
            if _cached_pair_matches(cached_prediction, answer, iou_threshold):
                matched += 1
            else:
                dropped.append({"session": str(session_path), "answer": answer_key, "reason": "pair_not_matched"})
                continue
            raw_samples.append(_canonicalize_sample({
                "session_id": str(data.get("session_id") or session_path.parent.name),
                "session_path": str(session_path),
                "image_id": image_id,
                "source_name": source_name,
                "image_path": str(image_path),
                "image_checksum": checksum,
                "subject": {
                    "label_id": int(subject["label_id"]),
                    "label": subject.get("label"),
                    "box_xyxy": [float(value) for value in subject.get("box_xyxy", [])],
                },
                "object": {
                    "label_id": int(obj["label_id"]),
                    "label": obj.get("label"),
                    "box_xyxy": [float(value) for value in obj.get("box_xyxy", [])],
                },
                "prediction_reference": dict(answer.get("prediction_reference") or {}),
                "predicates": normalized_predicates,
                "target": [float(target[0]), float(target[1])],
                "provenance": provenance,
            }))

    if leakage:
        atomic_json(output / "audit_report.json", {"status": "failed", "test_leakage": leakage})
        raise RLHPError(f"Preference data contains {len(leakage)} PSG test image(s); audit stopped.")
    if len(provenance_groups) > 1:
        raise RLHPError("Sessions use incompatible checkpoint/config/label-map provenance.")

    aggregates: dict[str, dict] = {}
    for sample in raw_samples:
        current = aggregates.get(sample["sample_id"])
        if current is None:
            current = dict(sample)
            current["target_sum"] = list(sample["target"])
            current["vote_count"] = 1
            aggregates[sample["sample_id"]] = current
        else:
            current["target_sum"][0] += sample["target"][0]
            current["target_sum"][1] += sample["target"][1]
            current["vote_count"] += 1
    samples = []
    for sample in aggregates.values():
        count = sample.pop("vote_count")
        summed = sample.pop("target_sum")
        sample["vote_count"] = count
        sample["target"] = [summed[0] / count, summed[1] / count]
        sample["trainable"] = sum(sample["target"]) > 0
        samples.append(sample)
    samples.sort(key=lambda item: item["sample_id"])

    image_checksums = sorted({item["image_checksum"] for item in samples if item["trainable"]})
    random.Random(seed).shuffle(image_checksums)
    validation_count = max(1, round(len(image_checksums) * 0.2)) if len(image_checksums) > 1 else 0
    validation_images = set(image_checksums[:validation_count])
    for sample in samples:
        sample["split"] = "audit_only" if not sample["trainable"] else (
            "validation" if sample["image_checksum"] in validation_images else "train"
        )

    match_rate = matched / match_total if match_total else 0.0
    report = {
        "status": "passed" if match_total and match_rate >= min_match_rate else "failed",
        "created_at": utc_timestamp(),
        "sessions": [str(path) for path in session_paths],
        "raw_feedback": match_total,
        "aggregated_samples": len(samples),
        "train_samples": sum(item["split"] == "train" for item in samples),
        "validation_samples": sum(item["split"] == "validation" for item in samples),
        "audit_only_samples": sum(item["split"] == "audit_only" for item in samples),
        "unique_images": len({item["image_checksum"] for item in samples}),
        "pair_matched": matched,
        "pair_matching_rate": match_rate,
        "model_matching_deferred": deferred_matching,
        "dropped": dropped,
        "test_leakage": [],
        "seed": seed,
        "iou_threshold": iou_threshold,
        "minimum_match_rate": min_match_rate,
        "legacy_provenance_allowed": allow_legacy_provenance,
    }
    atomic_json(output / "audit_report.json", report)
    atomic_json(output / "dataset.json", samples)
    atomic_json(output / "split_manifest.json", {
        "seed": seed,
        "train_images": sorted({item["image_checksum"] for item in samples if item["split"] == "train"}),
        "validation_images": sorted(validation_images),
        "audit_only_sample_ids": [item["sample_id"] for item in samples if item["split"] == "audit_only"],
    })
    if report["status"] != "passed":
        raise RLHPError(
            f"Pair matching rate {match_rate:.1%} is below the required {min_match_rate:.1%}."
        )
    return report, samples


def pairwise_preference_loss(
    rewards: torch.Tensor,
    targets: torch.Tensor,
    *,
    tau: float = 1.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bradley-Terry cross entropy over two pre-softmax predicate rewards."""
    if rewards.shape != targets.shape or rewards.shape[-1] != 2:
        raise ValueError("rewards and targets must have matching [..., 2] shapes")
    per_sample = -(targets * F.log_softmax(float(tau) * rewards, dim=-1)).sum(dim=-1)
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)


def masked_retention_loss(
    student_distributions: torch.Tensor,
    teacher_distributions: torch.Tensor,
    predicate_ids: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Retain baseline behaviour outside each labelled predicate pair.

    The loss is the KL chain-rule decomposition for two groups: the 55
    non-preference predicates and the labelled pair. It preserves both the
    total probability mass assigned to the old predicates and their normalized
    distribution, while intentionally omitting KL inside the labelled pair so
    preference learning may reorder those two predicates.
    """
    if student_distributions.shape != teacher_distributions.shape or student_distributions.ndim != 2:
        raise ValueError("student and teacher distributions must have matching [N, C] shapes")
    if predicate_ids.shape != (student_distributions.shape[0], 2):
        raise ValueError("predicate_ids must have shape [N, 2]")
    if student_distributions.shape[1] <= 2:
        raise ValueError("retention loss requires at least one non-preference class")

    student = student_distributions.float().clamp(1e-8, 1.0 - 1e-8)
    teacher = teacher_distributions.detach().float().clamp(1e-8, 1.0 - 1e-8)
    old_mask = torch.ones_like(student, dtype=torch.bool)
    old_mask.scatter_(1, predicate_ids.long(), False)

    student_old = student.masked_fill(~old_mask, 0.0)
    teacher_old = teacher.masked_fill(~old_mask, 0.0)
    student_mass = student_old.sum(dim=-1).clamp(1e-8, 1.0 - 1e-8)
    teacher_mass = teacher_old.sum(dim=-1).clamp(1e-8, 1.0 - 1e-8)
    student_old = student_old / student_mass.unsqueeze(-1)
    teacher_old = teacher_old / teacher_mass.unsqueeze(-1)

    old_kl = (
        teacher_old
        * (teacher_old.clamp_min(1e-8).log() - student_old.clamp_min(1e-8).log())
    ).sum(dim=-1)
    mass_kl = (
        teacher_mass * (teacher_mass.log() - student_mass.log())
        + (1.0 - teacher_mass)
        * ((1.0 - teacher_mass).log() - (1.0 - student_mass).log())
    )
    per_sample = mass_kl + teacher_mass * old_kl
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights.float()).sum() / weights.float().sum().clamp_min(1e-12)


def prototype_anchor_loss(
    student_protos_norm: torch.Tensor,
    teacher_protos_norm: torch.Tensor,
) -> torch.Tensor:
    """Cosine distance anchor across all class prototypes to preserve baseline semantics."""
    if student_protos_norm.shape != teacher_protos_norm.shape:
        raise ValueError("student and teacher prototypes must have matching shapes")
    return (1.0 - (student_protos_norm.float() * teacher_protos_norm.detach().float()).sum(dim=-1)).mean()


def _preference_accuracy(probabilities: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    strong = (targets.max(dim=-1).values >= 0.999) & (targets.min(dim=-1).values <= 0.001)
    if not bool(strong.any()):
        return 0, 0
    correct = probabilities[strong].argmax(dim=-1).eq(targets[strong].argmax(dim=-1)).sum().item()
    return int(correct), int(strong.sum().item())


class MockPreferenceModel(torch.nn.Module):
    """Tiny differentiable stand-in used only by mock pipeline tests."""

    def __init__(self):
        super().__init__()
        self.proj_head = torch.nn.Linear(2, 2, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self.proj_head.weight.copy_(torch.eye(2))

    def get_protos_norm(self) -> torch.Tensor:
        return F.normalize(self.proj_head.weight.float(), dim=-1)

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        logits = torch.logit(probabilities.clamp(1e-4, 1 - 1e-4))
        return self.proj_head(logits) * self.logit_scale.exp()


class RealPreferenceModel:
    """Autograd-capable wrapper around the labeler's REACT++ Predictor."""

    def __init__(self, checkpoint: Path, config: Path, yolo_checkpoint: Path | None, labels: Path, device: str):
        try:
            import cv2
            from app import Predictor, to_device
        except ImportError as exc:  # pragma: no cover - exercised in the real environment
            raise RLHPError(f"Real training dependencies are unavailable: {exc}") from exc
        self.cv2 = cv2
        self.to_device = to_device
        self.wrapper = Predictor(checkpoint, config, yolo_checkpoint, labels, device)
        self.model = self.wrapper.model
        self.device = self.wrapper.device
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.trainable = []
        for name, parameter in self.model.named_parameters():
            if any(name == suffix or name.startswith(suffix + ".") for suffix in TRAINABLE_SUFFIXES):
                parameter.requires_grad_(True)
                self.trainable.append((name, parameter))
        if not self.trainable or not any(name.endswith("logit_scale") for name, _ in self.trainable):
            raise RLHPError("Could not locate REACT++ proj_head/logit_scale parameters.")
        predictor = self.model.roi_heads["relation"].predictor
        if not hasattr(predictor, "set_preference_forward"):
            raise RLHPError("REACT++ predictor does not support differentiable preference-forward.")
        # Keep eval semantics (no dropout or EMA update) while bypassing the
        # detached inference prototype cache. This makes proj_head receive the
        # preference gradient through both r_proj and prototypes_proj.
        predictor.set_preference_forward(True)

    def get_protos_norm(self) -> torch.Tensor:
        predictor = self.model.roi_heads["relation"].predictor
        _, _, protos_norm, _ = predictor._get_cached_protos()
        return protos_norm

    def frozen_hashes(self) -> dict[str, str]:
        hashes = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                hashes[name] = hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return hashes

    def forward_image(self, image_path: str | Path) -> dict:
        image = self.cv2.imread(str(image_path))
        if image is None:
            raise RLHPError(f"Cannot read preference image: {image_path}")
        height, width = image.shape[:2]
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
            "image_size": (width, height),
            "mode": "xyxy",
        }
        transformed, target = self.wrapper.transforms(image, target)
        predictor = self.model.roi_heads["relation"].predictor
        if hasattr(predictor, "_cache_valid"):
            predictor._cache_valid = False
        return self.model([transformed.to(self.device)], [self.to_device(target, self.device)])[0]

    def sample_relation_tensor(
        self,
        prediction: dict,
        sample: dict,
        field: str,
        iou_threshold: float = 0.8,
    ) -> torch.Tensor | None:
        """Return one matched relation row while preserving its autograd graph."""
        boxes = prediction["boxes"]
        labels = prediction["pred_labels"]

        reference = sample.get("prediction_reference") or {}

        def match(recorded: dict, preferred_index) -> int | None:
            if preferred_index is not None:
                index = int(preferred_index)
                if 0 <= index < len(boxes):
                    same_label = int(labels[index].item()) == int(recorded["label_id"])
                    overlap = box_iou(recorded["box_xyxy"], boxes[index].detach().cpu().tolist())
                    if same_label and overlap >= iou_threshold:
                        return index
            best_index, best_overlap = None, 0.0
            for index in range(len(boxes)):
                if int(labels[index].item()) != int(recorded["label_id"]):
                    continue
                overlap = box_iou(recorded["box_xyxy"], boxes[index].detach().cpu().tolist())
                if overlap > best_overlap:
                    best_index, best_overlap = index, overlap
            return best_index if best_overlap >= iou_threshold else None

        subject_index = match(sample["subject"], reference.get("subject_index"))
        object_index = match(sample["object"], reference.get("object_index"))
        if subject_index is None or object_index is None:
            return None
        pair_rows = prediction["rel_pair_idxs"]
        relation_row = reference.get("relation_index")
        if relation_row is not None:
            relation_row = int(relation_row)
            if not (0 <= relation_row < len(pair_rows)) or not (
                int(pair_rows[relation_row, 0]) == subject_index
                and int(pair_rows[relation_row, 1]) == object_index
            ):
                relation_row = None
        relation_row = relation_row if relation_row is not None else next(
            (
                index for index in range(len(pair_rows))
                if int(pair_rows[index, 0]) == subject_index and int(pair_rows[index, 1]) == object_index
            ),
            None,
        )
        if relation_row is None:
            return None
        if field not in prediction:
            raise RLHPError(f"Model prediction is missing required field: {field}")
        return prediction[field][relation_row]

    def sample_distribution(self, prediction: dict, sample: dict, iou_threshold: float = 0.8) -> torch.Tensor | None:
        return self.sample_relation_tensor(prediction, sample, "pred_rel_scores", iou_threshold)

    def sample_rewards(self, prediction: dict, sample: dict, iou_threshold: float = 0.8) -> torch.Tensor | None:
        logits = self.sample_relation_tensor(prediction, sample, "pred_rel_logits", iou_threshold)
        if logits is None:
            return None
        predicate_ids = torch.tensor(
            [item["predicate_id"] for item in sample["predicates"]],
            dtype=torch.long,
            device=logits.device,
        )
        return logits.index_select(0, predicate_ids)

    def sample_probabilities(self, prediction: dict, sample: dict, iou_threshold: float = 0.8) -> torch.Tensor | None:
        distribution = self.sample_distribution(prediction, sample, iou_threshold)
        if distribution is None:
            return None
        predicate_ids = torch.tensor(
            [item["predicate_id"] for item in sample["predicates"]],
            dtype=torch.long,
            device=distribution.device,
        )
        return distribution.index_select(0, predicate_ids)


def _group_by_image(samples: list[dict], split: str) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        if sample.get("split") == split:
            groups[sample["image_path"]].append(sample)
    return sorted(groups.items())


def _mock_epoch(model: MockPreferenceModel, samples: list[dict], tau: float, optimizer=None) -> dict:
    if not samples:
        return {"loss": math.nan, "strong_accuracy": math.nan, "matched": 0, "total": 0}
    base = torch.tensor(
        [[item["predicates"][0]["probability"], item["predicates"][1]["probability"]] for item in samples],
        dtype=torch.float32,
    )
    targets = torch.tensor([item["target"] for item in samples], dtype=torch.float32)
    weights = torch.tensor([item["vote_count"] for item in samples], dtype=torch.float32)
    if optimizer is not None:
        optimizer.zero_grad()
    rewards = model(base)
    loss = pairwise_preference_loss(rewards, targets, tau=tau, weights=weights)
    if optimizer is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    correct, total = _preference_accuracy(rewards.detach(), targets)
    return {"loss": float(loss.detach()), "strong_accuracy": correct / total if total else math.nan,
            "matched": len(samples), "total": len(samples)}


def _real_epoch(
    engine: RealPreferenceModel,
    groups: list[tuple[str, list[dict]]],
    tau: float,
    optimizer=None,
    accumulation: int = 4,
    clip_norm: float = 1.0,
    teacher_cache: dict[str, torch.Tensor] | None = None,
    retention_weight: float = 0.0,
    teacher_protos_norm: torch.Tensor | None = None,
    proto_anchor_weight: float = 0.0,
    capture_teacher: dict[str, torch.Tensor] | None = None,
    desc: str | None = None,
) -> dict:
    if accumulation < 1:
        raise ValueError("gradient accumulation must be at least 1")
    total_loss, total_preference, total_retention, total_proto_anchor = 0.0, 0.0, 0.0, 0.0
    total_weight, matched, total, correct, strong_total = 0.0, 0, 0, 0, 0
    optimizer_steps, clipped_steps = 0, 0
    grad_norm_sum, max_grad_norm, latest_grad_norm = 0.0, 0.0, math.nan
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    accumulation_weight = 0.0
    accumulated_groups = 0
    parameters = [parameter for _, parameter in engine.trainable]

    def flush_accumulated_gradients() -> None:
        nonlocal accumulation_weight, accumulated_groups
        nonlocal optimizer_steps, clipped_steps, grad_norm_sum, max_grad_norm, latest_grad_norm
        if optimizer is None or accumulation_weight <= 0.0:
            return
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(accumulation_weight)
        norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        latest_grad_norm = float(norm_tensor.detach())
        grad_norm_sum += latest_grad_norm
        max_grad_norm = max(max_grad_norm, latest_grad_norm)
        clipped_steps += int(latest_grad_norm > clip_norm)
        optimizer_steps += 1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accumulation_weight = 0.0
        accumulated_groups = 0

    iterator = tqdm(
        groups,
        desc=desc or ("Preference train" if optimizer is not None else "Preference eval"),
        dynamic_ncols=True,
        leave=True,
    )
    for group_index, (image_path, samples) in enumerate(iterator):
        prediction = engine.forward_image(image_path)
        rewards, distributions, teachers, pair_ids, targets, weights = [], [], [], [], [], []
        for sample in samples:
            total += 1
            distribution = engine.sample_distribution(prediction, sample)
            pair_rewards = engine.sample_rewards(prediction, sample)
            if distribution is None or pair_rewards is None:
                continue
            matched += 1
            ids = [item["predicate_id"] for item in sample["predicates"]]
            rewards.append(pair_rewards)
            distributions.append(distribution)
            pair_ids.append(ids)
            targets.append(sample["target"])
            weights.append(sample["vote_count"])
            if capture_teacher is not None:
                capture_teacher[sample["sample_id"]] = distribution.detach().cpu()
            if teacher_cache is not None:
                teacher = teacher_cache.get(sample["sample_id"])
                if teacher is None:
                    raise RLHPError(f"Missing baseline teacher distribution for {sample['sample_id']}")
                teachers.append(teacher)
        if not rewards:
            iterator.set_postfix(matched=f"{matched}/{total}")
            continue
        reward_tensor = torch.stack(rewards)
        distribution_tensor = torch.stack(distributions)
        target_tensor = torch.tensor(targets, dtype=reward_tensor.dtype, device=reward_tensor.device)
        weight_tensor = torch.tensor(weights, dtype=reward_tensor.dtype, device=reward_tensor.device)
        preference_loss = pairwise_preference_loss(reward_tensor, target_tensor, tau=tau, weights=weight_tensor)
        retention_loss = preference_loss.new_zeros(())
        if teacher_cache is not None and retention_weight > 0.0:
            teacher_tensor = torch.stack(teachers).to(
                device=distribution_tensor.device, dtype=distribution_tensor.dtype
            )
            pair_id_tensor = torch.tensor(pair_ids, dtype=torch.long, device=distribution_tensor.device)
            retention_loss = masked_retention_loss(
                distribution_tensor, teacher_tensor, pair_id_tensor, weights=weight_tensor
            )
        proto_anchor_loss = preference_loss.new_zeros(())
        if teacher_protos_norm is not None and proto_anchor_weight > 0.0:
            current_protos_norm = engine.get_protos_norm()
            proto_anchor_loss = prototype_anchor_loss(current_protos_norm, teacher_protos_norm)
        loss = (
            preference_loss
            + float(retention_weight) * retention_loss
            + float(proto_anchor_weight) * proto_anchor_loss
        )
        weight = float(weight_tensor.sum())
        if optimizer is not None:
            # Backpropagate an unnormalised sample-weighted sum. At the end of
            # the accumulation window, divide .grad by the exact matched sample
            # weight. This is memory-efficient and makes every labelled sample
            # contribute equally regardless of labels-per-image.
            (loss * weight).backward()
            accumulation_weight += weight
            accumulated_groups += 1
            if accumulated_groups >= accumulation:
                flush_accumulated_gradients()
        total_loss += float(loss.detach()) * weight
        total_preference += float(preference_loss.detach()) * weight
        total_retention += float(retention_loss.detach()) * weight
        total_proto_anchor += float(proto_anchor_loss.detach()) * weight
        total_weight += weight
        group_correct, group_strong = _preference_accuracy(reward_tensor.detach(), target_tensor)
        correct += group_correct
        strong_total += group_strong
        iterator.set_postfix(
            loss=f"{total_loss / max(total_weight, 1e-12):.4f}",
            pref=f"{total_preference / max(total_weight, 1e-12):.4f}",
            acc=f"{correct / max(strong_total, 1):.1%}",
            matched=f"{matched}/{total}",
            grad="-" if math.isnan(latest_grad_norm) else f"{latest_grad_norm:.2f}",
        )
    flush_accumulated_gradients()
    return {
        "loss": total_loss / total_weight if total_weight else math.nan,
        "preference_loss": total_preference / total_weight if total_weight else math.nan,
        "retention_loss": total_retention / total_weight if total_weight else math.nan,
        "proto_anchor_loss": total_proto_anchor / total_weight if total_weight else math.nan,
        "strong_accuracy": correct / strong_total if strong_total else math.nan,
        "matched": matched,
        "total": total,
        "optimizer_steps": optimizer_steps,
        "mean_grad_norm": grad_norm_sum / optimizer_steps if optimizer_steps else math.nan,
        "max_grad_norm": max_grad_norm if optimizer_steps else math.nan,
        "clip_fraction": clipped_steps / optimizer_steps if optimizer_steps else math.nan,
    }


def _new_run(runs_root: Path, requested: str | None = None) -> Path:
    run_id = requested or time.strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(3).hex()
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train_pipeline(args: argparse.Namespace) -> Path:
    runs_root = Path(args.runs_root)
    run_dir = _new_run(runs_root, args.run_id)
    run = {
        "run_id": run_dir.name,
        "status": "auditing",
        "created_at": utc_timestamp(),
        "mock": bool(args.mock),
        "baseline_checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "config": str(Path(args.config).resolve()) if args.config else None,
    }
    atomic_json(run_dir / "run.json", run)
    try:
        report, samples = audit_sessions(
            args.sessions, args.labels, args.psg_test, run_dir,
            allow_legacy_provenance=args.allow_legacy_provenance,
            iou_threshold=args.iou_threshold,
            min_match_rate=args.min_match_rate,
            seed=args.seed,
            image_root=args.image_root,
        )
        train_samples = [item for item in samples if item["split"] == "train"]
        validation_samples = [item for item in samples if item["split"] == "validation"]
        if not train_samples:
            raise RLHPError("Audit produced no trainable training samples.")
        run["status"] = "training"
        run["audit"] = report
        atomic_json(run_dir / "run.json", run)

        history = []
        if args.mock:
            model = MockPreferenceModel()
            baseline_train = _mock_epoch(model, train_samples, args.tau)
            baseline = _mock_epoch(model, validation_samples or train_samples, args.tau)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            best_loss, best_state, best_epoch, patience = math.inf, None, None, 0
            for epoch in range(1, args.epochs + 1):
                train_metrics = _mock_epoch(model, train_samples, args.tau, optimizer)
                validation_metrics = _mock_epoch(model, validation_samples or train_samples, args.tau)
                history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
                if validation_metrics["loss"] < best_loss:
                    best_loss = validation_metrics["loss"]
                    best_epoch = epoch
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if epoch == 1 or getattr(args, "save_every_epoch", False):
                    torch.save({"model": model.state_dict(), "mock": True, "epoch": epoch},
                               run_dir / f"epoch_{epoch}.pth")
                if patience >= args.early_stopping_patience:
                    break
            model.load_state_dict(best_state)
            candidate_path = run_dir / "best_candidate.pth"
            torch.save({"model": model.state_dict(), "mock": True, "rlhp_run_id": run_dir.name}, candidate_path)
            candidate = _mock_epoch(model, validation_samples or train_samples, args.tau)
            frozen_hashes = {}
        else:
            if not args.checkpoint or not args.config:
                raise RLHPError("Real training requires --checkpoint and --config.")
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RLHPError("CUDA training was requested but no GPU is available. Use --mock for flow tests.")
            expected_provenance = next((item.get("provenance", {}) for item in samples if item.get("provenance")), {})
            actual_provenance = {
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "config_sha256": sha256_file(args.config),
                "label_map_sha256": sha256_file(args.labels),
            }
            mismatched = [
                name for name in REQUIRED_PROVENANCE
                if expected_provenance.get(name) and expected_provenance[name] != actual_provenance[name]
            ]
            if mismatched:
                raise RLHPError(
                    "Training inputs do not match session provenance: " + ", ".join(mismatched)
                )
            engine = RealPreferenceModel(
                Path(args.checkpoint), Path(args.config), Path(args.yolo_checkpoint) if args.yolo_checkpoint else None,
                Path(args.labels), args.device,
            )
            frozen_before = engine.frozen_hashes()
            train_groups, validation_groups = _group_by_image(samples, "train"), _group_by_image(samples, "validation")
            # Baseline passes are also the model-backed preflight. Reusing them
            # avoids a duplicate full inference over every preference image and
            # captures frozen-teacher distributions for retention training.
            teacher_cache: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                baseline_train = _real_epoch(
                    engine, train_groups, args.tau, capture_teacher=teacher_cache,
                    desc="Baseline preference train",
                )
                baseline = _real_epoch(
                    engine, validation_groups or train_groups, args.tau, capture_teacher=teacher_cache,
                    desc="Baseline preference val",
                )
            match_metrics = {
                "matched": baseline_train["matched"] + baseline["matched"],
                "total": baseline_train["total"] + baseline["total"],
            }
            real_match_rate = match_metrics["matched"] / max(match_metrics["total"], 1)
            report["model_pair_matched"] = match_metrics["matched"]
            report["model_pair_total"] = match_metrics["total"]
            report["model_pair_matching_rate"] = real_match_rate
            atomic_json(run_dir / "audit_report.json", report)
            if real_match_rate < args.min_match_rate:
                raise RLHPError(
                    f"Model-backed pair matching rate {real_match_rate:.1%} is below {args.min_match_rate:.1%}."
                )
            optimizer = torch.optim.AdamW(
                [parameter for _, parameter in engine.trainable],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            best_loss, best_state, best_epoch, patience = math.inf, None, None, 0
            retention_weight = float(getattr(args, "retention_weight", 0.0))
            proto_anchor_weight = float(getattr(args, "proto_anchor_weight", 0.0))
            teacher_protos = engine.get_protos_norm().detach().clone() if proto_anchor_weight > 0.0 else None
            for epoch in range(1, args.epochs + 1):
                # Deterministic epoch-wise shuffle: preserve image grouping for
                # one-forward-per-image efficiency without repeating AdamW's
                # update order every epoch.
                epoch_train_groups = list(train_groups)
                random.Random(args.seed + epoch).shuffle(epoch_train_groups)
                train_metrics = _real_epoch(
                    engine, epoch_train_groups, args.tau, optimizer, args.gradient_accumulation, args.clip_norm,
                    teacher_cache=teacher_cache, retention_weight=retention_weight,
                    teacher_protos_norm=teacher_protos, proto_anchor_weight=proto_anchor_weight,
                    desc=f"Epoch {epoch}/{args.epochs} train",
                )
                with torch.no_grad():
                    validation_metrics = _real_epoch(
                        engine, validation_groups or train_groups, args.tau,
                        teacher_cache=teacher_cache, retention_weight=retention_weight,
                        teacher_protos_norm=teacher_protos, proto_anchor_weight=proto_anchor_weight,
                        desc=f"Epoch {epoch}/{args.epochs} val",
                    )
                epoch_record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
                history.append(epoch_record)
                with (run_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
                print(
                    f"Epoch {epoch}/{args.epochs}: "
                    f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['strong_accuracy']:.2%}; "
                    f"val loss={validation_metrics['loss']:.4f}, acc={validation_metrics['strong_accuracy']:.2%}",
                    flush=True,
                )
                if validation_metrics["loss"] < best_loss:
                    best_loss = validation_metrics["loss"]
                    best_epoch = epoch
                    best_state = {name: value.detach().cpu().clone() for name, value in engine.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if epoch == 1 or getattr(args, "save_every_epoch", False):
                    torch.save({"model": engine.model.state_dict(), "rlhp_run_id": run_dir.name, "epoch": epoch},
                               run_dir / f"epoch_{epoch}.pth")
                if patience >= args.early_stopping_patience:
                    break
            engine.model.load_state_dict(best_state)
            with torch.no_grad():
                candidate = _real_epoch(
                    engine, validation_groups or train_groups, args.tau,
                    teacher_cache=teacher_cache, retention_weight=retention_weight,
                    teacher_protos_norm=teacher_protos, proto_anchor_weight=proto_anchor_weight,
                    desc=f"Best epoch {best_epoch} preference val",
                )
            frozen_after = engine.frozen_hashes()
            changed_frozen = sorted(name for name in frozen_before if frozen_before[name] != frozen_after[name])
            if changed_frozen:
                raise RLHPError(f"Frozen parameters changed: {', '.join(changed_frozen[:5])}")
            frozen_hashes = frozen_after
            original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(original, dict) or "model" not in original:
                original = {"model": original}
            original.pop("optimizer", None)
            original.pop("scheduler", None)
            original["model"] = {name: value.detach().cpu() for name, value in engine.model.state_dict().items()}
            original["rlhp"] = {
                "run_id": run_dir.name,
                "created_at": utc_timestamp(),
                "best_epoch": best_epoch,
                "reward_mode": "pre_softmax_relation_logits",
                "trainable": [n for n, _ in engine.trainable],
            }
            candidate_path = run_dir / "best_candidate.pth"
            torch.save(original, candidate_path)

        metrics = {
            "baseline_train": baseline_train,
            "baseline_validation": baseline,
            "candidate_validation": candidate,
            "epochs": history,
            "best_epoch": best_epoch,
            "tau": args.tau,
            "reward_mode": "mock_pre_softmax_logits" if args.mock else "pre_softmax_relation_logits",
            "batching": "shuffled_image_groups_sample_weighted_accumulation",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_accumulation": args.gradient_accumulation,
            "clip_norm": args.clip_norm,
            "retention_weight": float(getattr(args, "retention_weight", 0.0)),
            "retention_mode": "masked_old_classes_with_mass" if not args.mock else None,
        }
        atomic_json(run_dir / "training_metrics.json", metrics)
        atomic_json(run_dir / "frozen_parameter_hashes.json", frozen_hashes)
        training_plot = run_dir / "training_curves.png"
        try:
            from plot_rlhp_training import plot_run
            plot_run(run_dir, training_plot)
            print(f"Training plot: {training_plot.resolve()}", flush=True)
        except Exception as plot_error:  # plotting must not invalidate a trained checkpoint
            print(f"Warning: could not create training plot: {plot_error}", file=sys.stderr, flush=True)
            training_plot = None
        run.update({
            "status": "trained",
            "completed_at": utc_timestamp(),
            "candidate_checkpoint": str(candidate_path.resolve()),
            "training_plot": str(training_plot.resolve()) if training_plot else None,
        })
        atomic_json(run_dir / "run.json", run)
        return run_dir
    except Exception as exc:
        run.update({"status": "failed", "failed_at": utc_timestamp(), "error": str(exc)})
        atomic_json(run_dir / "run.json", run)
        raise


def flatten_evaluation_metrics(summary: dict) -> dict[str, float]:
    # Specific names must precede the generic ``recall`` substring.
    aliases = (
        ("zeroshot_recall", "zR"), ("mean_recall", "mR"),
        ("f1_score", "F1"), ("recall", "R"),
    )
    flattened: dict[str, float] = {}
    for key, value in summary.items():
        lower = key.lower()
        if "bbox_map" in lower or lower.endswith("bbox_map"):
            flattened["bbox_mAP"] = float(value)
            continue
        if not isinstance(value, dict):
            continue
        # Evaluator also emits implementation details such as
        # ``*_mean_recall_collect`` (nested per-class samples) and
        # ``*_mean_recall_list``. Only consume the final public metric key.
        metric = next((label for token, label in aliases if lower.endswith(token)), None)
        if metric is None:
            continue
        for cutoff, score in value.items():
            cutoff_text = str(cutoff).split("@")[-1]
            if cutoff_text in {"20", "50", "100"}:
                if isinstance(score, list):
                    normalized_score = sum(float(item) for item in score) / len(score) if score else 0.0
                else:
                    normalized_score = float(score)
                flattened[f"{metric}@{cutoff_text}"] = normalized_score
    return flattened


def compare_metrics(
    baseline: dict[str, float], candidate: dict[str, float],
    baseline_preference_loss: float, candidate_preference_loss: float,
    pair_matching_rate: float = 1.0,
) -> dict:
    missing = [name for name in ("mR@50", "R@50", "F1@50", "bbox_mAP") if name not in baseline or name not in candidate]
    if missing:
        raise RLHPError(f"Evaluation metrics are missing gate fields: {', '.join(missing)}")
    deltas = {}
    for name in sorted(set(baseline) & set(candidate)):
        absolute = candidate[name] - baseline[name]
        relative = absolute / baseline[name] if baseline[name] else (math.inf if absolute > 0 else 0.0)
        deltas[name] = {"absolute": absolute, "relative": relative}
    checks = {
        "finite": all(
            math.isfinite(float(value))
            for value in list(baseline.values()) + list(candidate.values())
            + [baseline_preference_loss, candidate_preference_loss, pair_matching_rate]
        ),
        "pair_matching_rate": pair_matching_rate >= 0.95,
        "preference_loss_improved": candidate_preference_loss < baseline_preference_loss,
        "mR@50_improved": candidate["mR@50"] > baseline["mR@50"],
        "R@50_guardrail": candidate["R@50"] >= baseline["R@50"] * 0.99,
        "F1@50_guardrail": candidate["F1@50"] >= baseline["F1@50"] * 0.99,
        "bbox_mAP_guardrail": candidate["bbox_mAP"] >= baseline["bbox_mAP"] * 0.99,
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": deltas,
        "preference": {"baseline_loss": baseline_preference_loss, "candidate_loss": candidate_preference_loss},
        "gate": {"passed": all(checks.values()), "checks": checks},
    }


def _prepare_evaluation_model_dir(directory: Path, checkpoint: Path, config: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config, directory / "config.yml")
    (directory / "last_checkpoint").write_text(str(checkpoint.resolve()) + "\n", encoding="utf-8")


def _resolve_psg_data_dir(data_dir: str | Path) -> Path:
    """Accept either the PSG root or its coco_format directory, without downloading."""
    requested = Path(data_dir).expanduser().resolve()

    def is_complete(root: Path) -> bool:
        test_dir = root / "test"
        annotation = test_dir / "_annotations.coco.json"
        has_image = test_dir.is_dir() and any(
            path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            for path in test_dir.iterdir()
        )
        return annotation.is_file() and has_image

    for candidate in (requested, requested / "coco_format"):
        if is_complete(candidate):
            if candidate != requested:
                print(f"Resolved PSG data directory: {requested} -> {candidate}", flush=True)
            return candidate
    raise RLHPError(
        f"PSG data directory is incomplete: {requested}. Expected test/_annotations.coco.json "
        "and test images, optionally below a coco_format/ subdirectory."
    )


def _run_psg_evaluation(checkpoint: Path, config: Path, output: Path, device: str, data_dir: str | None) -> dict:
    model_dir = output / "model"
    _prepare_evaluation_model_dir(model_dir, checkpoint, config)
    command = [
        sys.executable, str(PROJECT_ROOT / "tools" / "evaluate.py"),
        "--model-dir", str(model_dir), "--device", device, "--output-dir", str(output / "results"),
    ]
    if data_dir:
        command.extend(["--data-dir", str(_resolve_psg_data_dir(data_dir))])
    log_path = output / "evaluate.log"
    print(f"PSG evaluation: {checkpoint.name}", flush=True)
    # Stream the evaluator's tqdm output to the terminal while keeping a full
    # log. subprocess.run(..., PIPE) used to hide progress until evaluation ended.
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            log_handle.write(chunk)
            log_handle.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        returncode = process.wait()
    if returncode:
        raise RLHPError(f"PSG evaluation failed; see {output / 'evaluate.log'}")
    return read_json(output / "results" / "eval_summary.json")


def evaluate_pipeline(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    run = read_json(run_dir / "run.json")
    training = read_json(run_dir / "training_metrics.json")
    audit = read_json(run_dir / "audit_report.json")
    if run.get("status") not in {"trained", "evaluated"}:
        raise RLHPError(f"Run is not ready for evaluation: {run.get('status')}")
    if args.mock or run.get("mock"):
        baseline = {
            "R@20": 0.20, "R@50": 0.30, "R@100": 0.35,
            "mR@20": 0.10, "mR@50": 0.20, "mR@100": 0.24,
            "F1@20": 0.133, "F1@50": 0.24, "F1@100": 0.285,
            "zR@20": 0.08, "zR@50": 0.12, "zR@100": 0.15, "bbox_mAP": 0.40,
        }
        candidate = dict(baseline)
        candidate.update({"mR@50": 0.21, "R@50": 0.299, "F1@50": 0.239, "bbox_mAP": 0.399})
    else:
        baseline_checkpoint = Path(args.baseline_checkpoint or run["baseline_checkpoint"])
        candidate_checkpoint = Path(run["candidate_checkpoint"])
        config = Path(args.config or run["config"])
        baseline_summary = _run_psg_evaluation(
            baseline_checkpoint, config, run_dir / "evaluation" / "baseline", args.device, args.data_dir
        )
        candidate_summary = _run_psg_evaluation(
            candidate_checkpoint, config, run_dir / "evaluation" / "candidate", args.device, args.data_dir
        )
        baseline, candidate = flatten_evaluation_metrics(baseline_summary), flatten_evaluation_metrics(candidate_summary)
    comparison = compare_metrics(
        baseline, candidate,
        float(training["baseline_validation"]["loss"]),
        float(training["candidate_validation"]["loss"]),
        float(audit.get("model_pair_matching_rate", audit["pair_matching_rate"])),
    )
    comparison["created_at"] = utc_timestamp()
    comparison_path = atomic_json(run_dir / "comparison_metrics.json", comparison)
    evaluation_plot = run_dir / "evaluation_comparison.png"
    try:
        from plot_rlhp_training import plot_evaluation_comparison
        plot_evaluation_comparison(comparison_path, evaluation_plot)
        print(f"Evaluation plot: {evaluation_plot.resolve()}", flush=True)
    except Exception as plot_error:  # metrics remain valid even if matplotlib fails
        print(f"Warning: could not create evaluation plot: {plot_error}", file=sys.stderr, flush=True)
        evaluation_plot = None
    run.update({
        "status": "evaluated",
        "evaluated_at": utc_timestamp(),
        "gate_passed": comparison["gate"]["passed"],
        "evaluation_plot": str(evaluation_plot.resolve()) if evaluation_plot else None,
    })
    atomic_json(run_dir / "run.json", run)
    return comparison


def promote_run(run_dir: str | Path, manifest_path: str | Path, *, allow_mock: bool = False) -> dict:
    run_path = Path(run_dir).resolve()
    run = read_json(run_path / "run.json")
    if run.get("mock") and not allow_mock:
        raise RLHPError("Mock checkpoints are test artifacts and cannot be promoted.")
    metrics = read_json(run_path / "comparison_metrics.json")
    if not metrics.get("gate", {}).get("passed"):
        raise RLHPError("Candidate did not pass the evaluation gate and cannot be promoted.")
    checkpoint = Path(run["candidate_checkpoint"])
    if not checkpoint.is_file():
        raise RLHPError(f"Candidate checkpoint is missing: {checkpoint}")
    manifest_file = Path(manifest_path)
    previous_manifest = read_json(manifest_file) if manifest_file.is_file() else {"active": None, "history": []}
    active = {
        "run_id": run["run_id"], "checkpoint": str(checkpoint.resolve()),
        "config": run.get("config"), "promoted_at": utc_timestamp(),
    }
    history = list(previous_manifest.get("history", []))
    if previous_manifest.get("active"):
        history.append(previous_manifest["active"])
    manifest = {"active": active, "history": history}
    atomic_json(manifest_file, manifest)
    run.update({"status": "promoted", "promoted_at": active["promoted_at"]})
    atomic_json(run_path / "run.json", run)
    return manifest


def rollback_manifest(manifest_path: str | Path) -> dict:
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise RLHPError("Active-model manifest does not exist.")
    manifest = read_json(manifest_file)
    history = list(manifest.get("history", []))
    if not history:
        raise RLHPError("No previous active checkpoint is available for rollback.")
    restored = history.pop()
    new_manifest = {"active": restored, "history": history, "rolled_back_at": utc_timestamp()}
    atomic_json(manifest_file, new_manifest)
    return new_manifest


def resolve_active_model(manifest_path: str | Path) -> tuple[Path, Path | None] | None:
    path = Path(manifest_path)
    if not path.is_file():
        return None
    active = read_json(path).get("active")
    if not active:
        return None
    checkpoint = Path(active["checkpoint"])
    config = Path(active["config"]) if active.get("config") else None
    if not checkpoint.is_file():
        raise RLHPError(f"Active checkpoint in manifest is missing: {checkpoint}")
    if config is not None and not config.is_file():
        raise RLHPError(f"Active config in manifest is missing: {config}")
    return checkpoint, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def audit_arguments(command):
        command.add_argument("--sessions", nargs="+", required=True, help="session.json files or directories")
        command.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
        command.add_argument("--psg-test", type=Path, default=DEFAULT_PSG_TEST)
        command.add_argument("--allow-legacy-provenance", action="store_true")
        command.add_argument("--iou-threshold", type=float, default=0.8)
        command.add_argument("--min-match-rate", type=float, default=0.95)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--image-root", type=Path, help="external PSG train/validation image directory")

    audit = subparsers.add_parser("audit", help="validate and normalize preference sessions")
    audit_arguments(audit)
    audit.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="audit and preference-fine-tune REACT++")
    audit_arguments(train)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--config", type=Path)
    train.add_argument("--yolo-checkpoint", type=Path)
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    train.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    train.add_argument("--run-id")
    train.add_argument("--mock", action="store_true")
    train.add_argument(
        "--tau", type=float, default=1.0,
        help="Bradley-Terry temperature applied to pre-softmax relation logits (default: 1.0).",
    )
    train.add_argument(
        "--retention-weight", type=float, default=0.0,
        help="Weight for frozen-baseline retention outside each labelled predicate pair.",
    )
    train.add_argument(
        "--proto-anchor-weight", type=float, default=0.0,
        help="Weight for prototype anchor loss preserving unlabelled predicate prototypes.",
    )
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--gradient-accumulation", type=int, default=4)
    train.add_argument("--clip-norm", type=float, default=1.0)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--early-stopping-patience", type=int, default=2)
    train.add_argument(
        "--save-every-epoch", action="store_true",
        help="Save epoch_N.pth after every epoch (required for per-epoch PSG curves).",
    )

    evaluate = subparsers.add_parser("evaluate", help="compare baseline and candidate on PSG test")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--baseline-checkpoint", type=Path)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--data-dir")
    evaluate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluate.add_argument("--mock", action="store_true")

    promote = subparsers.add_parser("promote", help="activate a gate-passing candidate")
    promote.add_argument("--run-dir", type=Path, required=True)
    promote.add_argument("--manifest", type=Path, default=DEFAULT_RUNS_ROOT / "active_model.json")

    rollback = subparsers.add_parser("rollback", help="restore the previous active checkpoint")
    rollback.add_argument("--manifest", type=Path, default=DEFAULT_RUNS_ROOT / "active_model.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            report, _ = audit_sessions(
                args.sessions, args.labels, args.psg_test, args.output,
                allow_legacy_provenance=args.allow_legacy_provenance,
                iou_threshold=args.iou_threshold, min_match_rate=args.min_match_rate, seed=args.seed,
                image_root=args.image_root,
            )
            print(json.dumps(report, indent=2))
        elif args.command == "train":
            print(train_pipeline(args))
        elif args.command == "evaluate":
            print(json.dumps(evaluate_pipeline(args), indent=2))
        elif args.command == "promote":
            print(json.dumps(promote_run(args.run_dir, args.manifest), indent=2))
        elif args.command == "rollback":
            print(json.dumps(rollback_manifest(args.manifest), indent=2))
        return 0
    except (RLHPError, ValueError, FileNotFoundError) as exc:
        print(f"RLHP error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
