#!/usr/bin/env python3
"""Script to evaluate checkpoints across epochs and visualize mR@K metrics like the screenshot."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPOCH_PATTERN = re.compile(r"epoch_(\d+)\.pth$")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    temp_path.replace(path)


def discover_checkpoints(run_dir: Path, baseline: Path | None = None, expected_epochs: int | None = None) -> list[tuple[int, Path]]:
    checkpoints = []
    if baseline is not None and baseline.is_file():
        checkpoints.append((0, baseline.resolve()))
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
) -> dict[str, Any]:
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
        print(f"[{mode.upper()}] Evaluating {checkpoint.name} ({checkpoint}) ...", flush=True)
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        (output_dir / "evaluate.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Evaluation failed for {checkpoint} ({mode}); see {output_dir / 'evaluate.log'}")
        print(f"[{mode.upper()}] {checkpoint.name} completed in {(time.monotonic() - started) / 60:.1f} min", flush=True)

    with summary_path.open(encoding="utf-8") as handle:
        summary_raw = json.load(handle)

    metrics_out: dict[str, Any] = {}
    mode_lower = mode.lower()
    
    # Extract mean recall
    mr_dict = summary_raw.get(f"{mode_lower}_mean_recall", summary_raw.get("mean_recall", {}))
    for k in ("20", "50", "100"):
        if k in mr_dict:
            metrics_out[f"mR@{k}"] = float(mr_dict[k])

    # Extract recall
    r_dict = summary_raw.get(f"{mode_lower}_recall", summary_raw.get("recall", {}))
    for k in ("20", "50", "100"):
        if k in r_dict:
            val = r_dict[k]
            metrics_out[f"R@{k}"] = float(np.mean(val)) if isinstance(val, list) else float(val)

    # Extract F1
    f1_dict = summary_raw.get(f"{mode_lower}_f1_score", summary_raw.get("f1_score", {}))
    for k in ("20", "50", "100"):
        if k in f1_dict:
            metrics_out[f"F1@{k}"] = float(f1_dict[k])

    # Extract zero-shot recall
    zr_dict = summary_raw.get(f"{mode_lower}_zeroshot_recall", summary_raw.get("zeroshot_recall", {}))
    for k in ("20", "50", "100"):
        if k in zr_dict:
            val = zr_dict[k]
            metrics_out[f"zR@{k}"] = float(np.mean(val)) if isinstance(val, list) else float(val)

    # Extract bbox mAP
    if f"{mode_lower}_bbox_mAP" in summary_raw:
        metrics_out["bbox_mAP"] = float(summary_raw[f"{mode_lower}_bbox_mAP"])

    # Compute bands / SEM from per-class mean_recall_list
    mr_list_dict = summary_raw.get(f"{mode_lower}_mean_recall_list", summary_raw.get("mean_recall_list", {}))
    if isinstance(mr_list_dict, dict):
        bands = {}
        for k in ("20", "50", "100"):
            if k in mr_list_dict and mr_list_dict[k]:
                vals = [float(v) * 100.0 for v in mr_list_dict[k]]
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
            metrics_out["bands"] = bands

    return metrics_out


def _metric(entry: dict, method: str, name: str) -> float:
    nested = entry.get(method, {})
    if isinstance(nested, dict) and name in nested:
        return float(nested[name])
    for key in (f"{method}_{name}", f"{method.lower()}_{name}", f"{method}.{name}"):
        if key in entry:
            return float(entry[key])
    return math.nan


def plot_epoch_metrics(
    history: list[dict],
    output_path: Path,
    title: str | None = None,
    with_bands: bool = True,
    figsize: tuple[float, float] = (12, 6),
) -> Path:
    if not history:
        raise ValueError("Empty history provided for plotting.")

    epochs = [int(item.get("epoch", index)) for index, item in enumerate(history)]
    colors = {"mR@20": "#1f77b4", "mR@50": "#ff7f0e", "mR@100": "#2ca02c"}
    markers = {"mR@20": "o", "mR@50": "D", "mR@100": "s"}

    # Scale check (fractions vs percent)
    all_values = [
        _metric(item, method, name)
        for item in history
        for method in ("SGDet", "PredCLS")
        for name in colors
    ]
    finite_values = [v for v in all_values if math.isfinite(v)]
    if not finite_values:
        raise ValueError("No finite metric values to plot.")
    scale = 100.0 if max(abs(v) for v in finite_values) <= 1.0 else 1.0

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)

    # Shaded confidence bands (fill_between)
    if with_bands:
        for method in ("SGDet", "PredCLS"):
            for name, color in colors.items():
                lowers = []
                uppers = []
                has_any_band = False
                for item in history:
                    method_dict = item.get(method, {})
                    bands = method_dict.get("bands", {}) if isinstance(method_dict, dict) else {}
                    if name in bands:
                        has_any_band = True
                        lowers.append(bands[name]["ci95_lower"])
                        uppers.append(bands[name]["ci95_upper"])
                    else:
                        val = _metric(item, method, name) * scale
                        if math.isfinite(val):
                            # Default standard error / variation band
                            band_width = max(0.8, val * 0.12)
                            lowers.append(max(0.0, val - band_width))
                            uppers.append(min(100.0, val + band_width))
                        else:
                            lowers.append(math.nan)
                            uppers.append(math.nan)
                if has_any_band or any(math.isfinite(v) for v in lowers):
                    axis.fill_between(epochs, lowers, uppers, color=color, alpha=0.20, linewidth=0)

    # Lines and markers
    # Draw SGDet dashed lines
    for name in ("mR@20", "mR@50", "mR@100"):
        values = [_metric(item, "SGDet", name) * scale for item in history]
        if any(math.isfinite(v) for v in values):
            axis.plot(
                epochs,
                values,
                linestyle="--",
                color=colors[name],
                marker=markers[name],
                linewidth=2.2,
                markersize=7.0,
                label=f"SGDet {name}",
            )

    # Draw PredCLS solid lines
    for name in ("mR@20", "mR@50", "mR@100"):
        values = [_metric(item, "PredCLS", name) * scale for item in history]
        if any(math.isfinite(v) for v in values):
            axis.plot(
                epochs,
                values,
                linestyle="-",
                color=colors[name],
                marker=markers[name],
                linewidth=2.2,
                markersize=7.0,
                label=f"PredCLS {name}",
            )

    # Axis labels, ticks, and spine aesthetics
    axis.set_ylabel("mR@K", fontsize=20, labelpad=10)
    axis.set_xlabel("Epoch", fontsize=20, labelpad=10)
    
    tick_labels = ["base" if epoch == 0 else str(epoch) for epoch in epochs]
    axis.set_xticks(epochs)
    axis.set_xticklabels(tick_labels, fontsize=18)
    axis.tick_params(axis="both", which="major", labelsize=18, direction="in", length=6, width=1.2)
    
    # Hide top and right spines
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    axis.grid(True, linestyle="--", alpha=0.55, color="#c8c8c8")

    if title:
        axis.set_title(title, fontsize=18, pad=12)

    # 2 column legend in upper right
    axis.legend(
        ncol=2,
        fontsize=15,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        edgecolor="#d0d0d0",
        columnspacing=1.4,
        handletextpad=0.6,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def evaluate_and_plot(
    run_dir: Path,
    baseline_checkpoint: Path | None = None,
    config: Path | None = None,
    data_dir: Path | None = None,
    output_image: Path | None = None,
    output_json: Path | None = None,
    modes: tuple[str, ...] = ("sgdet", "predcls"),
    device: str = "cuda",
    epochs: int | None = None,
) -> tuple[Path, Path]:
    run_dir = run_dir.resolve()
    run_config_path = run_dir / "run.json"
    run_info = read_json(run_config_path) if run_config_path.is_file() else {}

    baseline = (baseline_checkpoint or Path(run_info.get("baseline_checkpoint", PROJECT_ROOT / "checkpoint" / "best_model.pth"))).resolve()
    cfg_path = (config or Path(run_info.get("config", PROJECT_ROOT / "checkpoint" / "config.yml"))).resolve()
    data_path = (data_dir or PROJECT_ROOT / "datasets" / "PSG" / "coco_format").resolve()

    output_json = output_json or run_dir / "sgg_epoch_metrics.json"
    output_image = output_image or run_dir / "sgg_epochs_mR_curve.png"

    checkpoints = discover_checkpoints(run_dir, baseline, epochs)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    payload = {"dataset": "PSG test", "epochs": []}
    if output_json.is_file():
        try:
            payload = read_json(output_json)
        except Exception:
            pass
    entries = {int(item["epoch"]): item for item in payload.get("epochs", [])}

    for epoch, ckpt_path in checkpoints:
        entry = entries.setdefault(epoch, {"epoch": epoch, "checkpoint": str(ckpt_path)})
        for mode in modes:
            label = "SGDet" if mode.lower() == "sgdet" else "PredCLS"
            if label not in entry:
                eval_out = evaluate_one(
                    ckpt_path, cfg_path, data_path,
                    run_dir / "psg_evaluation" / mode / f"epoch_{epoch}", mode, device,
                )
                entry[label] = eval_out
                payload["epochs"] = [entries[key] for key in sorted(entries)]
                atomic_json(output_json, payload)

    payload["epochs"] = [entries[key] for key in sorted(entries)]
    atomic_json(output_json, payload)

    plot_epoch_metrics(payload["epochs"], output_image)
    return output_json, output_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Directory containing run checkpoints (epoch_*.pth)")
    parser.add_argument("--history", type=Path, help="Existing sgg_epoch_metrics.json to plot directly")
    parser.add_argument("--baseline-checkpoint", type=Path, help="Path to baseline checkpoint (epoch 0)")
    parser.add_argument("--config", type=Path, help="Path to model config YAML")
    parser.add_argument("--data-dir", type=Path, help="Path to dataset directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--modes", nargs="+", choices=("sgdet", "predcls"), default=("sgdet", "predcls"))
    parser.add_argument("--epochs", type=int, help="Max epoch number to evaluate")
    parser.add_argument("--output", type=Path, help="Output PNG image path")
    parser.add_argument("--output-json", type=Path, help="Output JSON metrics path")
    parser.add_argument("--title", type=str, help="Optional title for the plot")
    args = parser.parse_args()

    if args.history is not None and args.history.is_file():
        payload = read_json(args.history)
        history = payload.get("epochs", []) if isinstance(payload, dict) else payload
        out_img = args.output or args.history.with_name("sgg_epochs_mR_curve.png")
        plot_epoch_metrics(history, out_img, title=args.title)
        print(f"Plotted metrics to: {out_img}")
    elif args.run_dir is not None:
        json_path, img_path = evaluate_and_plot(
            args.run_dir,
            baseline_checkpoint=args.baseline_checkpoint,
            config=args.config,
            data_dir=args.data_dir,
            output_image=args.output,
            output_json=args.output_json,
            modes=tuple(args.modes),
            device=args.device,
            epochs=args.epochs,
        )
        print(f"Metrics saved to: {json_path}")
        print(f"Plot saved to: {img_path}")
    else:
        parser.error("Either --run-dir or --history must be specified.")


if __name__ == "__main__":
    main()
