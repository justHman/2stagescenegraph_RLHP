#!/usr/bin/env python3
"""Export baseline-vs-uncertainty relation selection scores from cached sessions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from batch_session import DEFAULT_SETTINGS
from server import relation_selection_comparison


FIELDS = [
    "session_id", "image_id", "method", "rank", "relation_id", "subject", "object",
    "predicate_1", "p1", "predicate_2", "p2", "relation_score", "triplet_score",
    "relative_margin", "uncertainty_score", "selected_by_both",
]


def cached_predictions(exports_root: Path):
    """Return the latest cached prediction for each image without mutating sessions."""
    predictions = {}
    for session_path in sorted(Path(exports_root).glob("session_*/session.json")):
        data = json.loads(session_path.read_text(encoding="utf-8"))
        for image_id, result in data.get("predictions", {}).items():
            predictions[image_id] = (session_path.parent.name, result)
        for image_id, filename in data.get("prediction_index", {}).items():
            prediction_path = session_path.parent / "predictions" / filename
            if prediction_path.is_file():
                predictions[image_id] = (
                    session_path.parent.name,
                    json.loads(prediction_path.read_text(encoding="utf-8")),
                )
    return predictions


def export_comparison(exports_root: Path, output: Path, settings: dict):
    rows = []
    for image_id, (session_id, result) in cached_predictions(exports_root).items():
        for row in relation_selection_comparison(result, settings):
            rows.append({"session_id": session_id, "image_id": image_id, **row})

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-root", type=Path, default=Path(__file__).parent / "exports")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "relation_selection_comparison.csv")
    parser.add_argument("--top-k", type=int, default=DEFAULT_SETTINGS["top_k_relations"])
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_SETTINGS["relation_candidate_top_k"])
    parser.add_argument("--object-max-count", type=int, default=DEFAULT_SETTINGS["object_max_count"])
    parser.add_argument("--object-min-probability", type=float, default=DEFAULT_SETTINGS["object_min_probability"])
    args = parser.parse_args()
    settings = {
        "top_k_relations": args.top_k,
        "relation_candidate_top_k": args.candidate_top_k,
        "object_max_count": args.object_max_count,
        "object_min_probability": args.object_min_probability,
    }
    rows = export_comparison(args.exports_root, args.output, settings)
    print(f"Wrote {len(rows)} rows to {args.output}")
    for method in ("highest_probability", "uncertainty_priority"):
        selected = [row for row in rows if row["method"] == method]
        if selected:
            mean_relation = sum(row["relation_score"] for row in selected) / len(selected)
            mean_uncertainty = sum(row["uncertainty_score"] for row in selected) / len(selected)
            print(f"{method}: mean relation={mean_relation:.6f}, mean uncertainty={mean_uncertainty:.6f}")


if __name__ == "__main__":
    main()
