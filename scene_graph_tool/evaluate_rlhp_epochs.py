#!/usr/bin/env python3
"""Evaluate the RLHP baseline and every saved epoch on the PSG test split."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rlhp import atomic_json, flatten_evaluation_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPOCH_PATTERN = re.compile(r"epoch_(\d+)\.pth$")


def discover_checkpoints(run_dir: Path, baseline: Path, expected_epochs: int | None = None):
    checkpoints = [(0, baseline.resolve())]
    for path in run_dir.glob("epoch_*.pth"):
        match = EPOCH_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path.resolve()))
    checkpoints.sort(key=lambda item: item[0])
    if expected_epochs is not None:
        found = {epoch for epoch, _ in checkpoints}
        missing = [epoch for epoch in range(1, expected_epochs + 1) if epoch not in found]
        if missing:
            raise FileNotFoundError(f"Missing epoch checkpoints: {missing}")
        checkpoints = [item for item in checkpoints if item[0] <= expected_epochs]
    return checkpoints


def evaluate_one(
    checkpoint: Path,
    config: Path,
    data_dir: Path,
    output_dir: Path,
    mode: str,
    device: str,
) -> dict[str, float]:
    result_dir = output_dir / "results"
    summary_path = result_dir / "eval_summary.json"
    if not summary_path.is_file():
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config, model_dir / "config.yml")
        (model_dir / "last_checkpoint").write_text(str(checkpoint.resolve()) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "evaluate.py"),
            "--model-dir", str(model_dir),
            "--device", device,
            "--sgg-mode", mode,
            "--data-dir", str(data_dir),
            "--output-dir", str(result_dir),
        ]
        started = time.monotonic()
        print(f"[{mode}] evaluating {checkpoint.name} ...", flush=True)
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        (output_dir / "evaluate.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Evaluation failed for {checkpoint} ({mode}); see {output_dir / 'evaluate.log'}")
        print(f"[{mode}] {checkpoint.name} completed in {(time.monotonic() - started) / 60:.1f} min", flush=True)
    with summary_path.open(encoding="utf-8") as handle:
        summary_raw = json.load(handle)
        metrics = flatten_evaluation_metrics(summary_raw)
    required = ("mR@20", "mR@50", "mR@100")
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError(f"Missing {missing} in {summary_path}")
    res = {name: metrics[name] for name in required}
    for extra in ("R@20", "R@50", "R@100", "F1@20", "F1@50", "F1@100", "zR@20", "zR@50", "zR@100", "bbox_mAP"):
        if extra in metrics:
            res[extra] = metrics[extra]
    mr_list_key = f"{mode.lower()}_mean_recall_list"
    mr_lists = summary_raw.get(mr_list_key, summary_raw.get("mean_recall_list", {}))
    if isinstance(mr_lists, dict):
        import numpy as np
        bands = {}
        for k in ("20", "50", "100"):
            if k in mr_lists and mr_lists[k]:
                vals = [float(v) * 100.0 for v in mr_lists[k]]
                mean_val = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                bands[f"mR@{k}"] = {
                    "mean": mean_val,
                    "sem": sem,
                    "ci95_lower": max(0.0, mean_val - 1.96 * sem),
                    "ci95_upper": min(100.0, mean_val + 1.96 * sem),
                    "std": float(np.std(vals)),
                }
        if bands:
            res["bands"] = bands
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--modes", nargs="+", choices=("sgdet", "predcls"), default=("sgdet", "predcls"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    with (run_dir / "run.json").open(encoding="utf-8") as handle:
        run = json.load(handle)
    baseline = (args.baseline_checkpoint or Path(run["baseline_checkpoint"])).resolve()
    config = (args.config or Path(run["config"])).resolve()
    output = args.output or run_dir / "sgg_epoch_metrics.json"
    checkpoints = discover_checkpoints(run_dir, baseline, args.epochs)

    payload = {"dataset": "PSG test", "epochs": []}
    if output.is_file():
        with output.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    entries = {int(item["epoch"]): item for item in payload.get("epochs", [])}

    for epoch, checkpoint in checkpoints:
        entry = entries.setdefault(epoch, {"epoch": epoch, "checkpoint": str(checkpoint)})
        for mode in args.modes:
            label = "SGDet" if mode == "sgdet" else "PredCLS"
            if label not in entry:
                entry[label] = evaluate_one(
                    checkpoint, config, args.data_dir.resolve(),
                    run_dir / "psg_evaluation" / mode / f"epoch_{epoch}", mode, args.device,
                )
                payload["epochs"] = [entries[key] for key in sorted(entries)]
                atomic_json(output, payload)
    payload["epochs"] = [entries[key] for key in sorted(entries)]
    atomic_json(output, payload)
    from plot_rlhp_training import plot_sgg_only
    plot_path = plot_sgg_only(output, run_dir / "sgg_epochs_mR_curve.png")
    print(f"Metrics saved to: {output.resolve()}")
    print(f"Plot saved to: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
