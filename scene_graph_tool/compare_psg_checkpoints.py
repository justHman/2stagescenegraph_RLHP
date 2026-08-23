#!/usr/bin/env python3
"""Evaluate and visualize a baseline/candidate checkpoint pair on PSG test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_rlhp_epochs import evaluate_one
from plot_rlhp_training import plot_sgg_only
from rlhp import atomic_json


METRICS = ("mR@20", "mR@50", "mR@100")
MODES = (("sgdet", "SGDet"), ("predcls", "PredCLS"))


def plot_bars(payload: dict, output: Path) -> Path:
    entries = payload["epochs"]
    baseline, candidate = entries[0], entries[1]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    figure.subplots_adjust(top=0.78, bottom=0.13, wspace=0.18)
    x = np.arange(len(METRICS))
    width = 0.36
    for axis, (_, label) in zip(axes, MODES):
        base_values = np.array([baseline[label][metric] for metric in METRICS]) * 100.0
        candidate_values = np.array([candidate[label][metric] for metric in METRICS]) * 100.0
        base_bars = axis.bar(x - width / 2, base_values, width, label="Baseline", color="#4C78A8")
        candidate_bars = axis.bar(x + width / 2, candidate_values, width, label="Best candidate (epoch 3)", color="#E45756")
        axis.bar_label(base_bars, fmt="%.2f", padding=3, fontsize=8)
        axis.bar_label(candidate_bars, fmt="%.2f", padding=3, fontsize=8)
        axis.set_title(label)
        axis.set_xticks(x, METRICS)
        axis.set_ylabel("PSG test mR@K (%)")
        axis.set_ylim(0, max(base_values.max(), candidate_values.max()) * 1.16)
        axis.grid(axis="y", linestyle="--", alpha=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.90))
    figure.suptitle("PSG test: baseline vs RLHP best candidate", y=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    result_path = output_dir / "psg_best_comparison.json"
    payload = {
        "dataset": "PSG test",
        "epochs": [
            {"epoch": 0, "label": "baseline", "checkpoint": str(args.baseline.resolve())},
            {"epoch": 3, "label": "best candidate", "checkpoint": str(args.candidate.resolve())},
        ],
    }
    if result_path.is_file():
        with result_path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        for entry, old_entry in zip(payload["epochs"], saved.get("epochs", [])):
            if entry["checkpoint"] == old_entry.get("checkpoint"):
                entry.update({key: old_entry[key] for key in ("SGDet", "PredCLS") if key in old_entry})

    for entry in payload["epochs"]:
        checkpoint = Path(entry["checkpoint"])
        model_key = "baseline" if entry["epoch"] == 0 else "candidate"
        for mode, label in MODES:
            if label not in entry:
                entry[label] = evaluate_one(
                    checkpoint, args.config.resolve(), args.data_dir.resolve(),
                    output_dir / "evaluation" / model_key / mode, mode, args.device,
                )
                atomic_json(result_path, payload)

    atomic_json(result_path, payload)
    curve_path = plot_sgg_only(result_path, output_dir / "psg_best_mr_curve.png", "PSG test: baseline vs best epoch 3")
    bars_path = plot_bars(payload, output_dir / "psg_best_comparison.png")
    print(result_path)
    print(curve_path)
    print(bars_path)


if __name__ == "__main__":
    main()
