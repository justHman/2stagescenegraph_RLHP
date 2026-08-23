#!/usr/bin/env python3
"""Plot RLHP preference training history and optional per-epoch SGG metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def accuracy_band(metrics: list[dict]) -> tuple[list[float], list[float]]:
    """Return a normal-approximation 95% binomial confidence band in percent."""
    lower, upper = [], []
    for item in metrics:
        accuracy, total = float(item.get("strong_accuracy", math.nan)), int(item.get("total", 0))
        if not math.isfinite(accuracy) or total <= 0:
            lower.append(math.nan)
            upper.append(math.nan)
            continue
        radius = 1.96 * math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / total)
        lower.append(max(0.0, accuracy - radius) * 100.0)
        upper.append(min(1.0, accuracy + radius) * 100.0)
    return lower, upper


def _metric(entry: dict, method: str, name: str):
    nested = entry.get(method, {})
    if isinstance(nested, dict) and name in nested:
        return float(nested[name])
    for key in (f"{method}_{name}", f"{method.lower()}_{name}", f"{method}.{name}"):
        if key in entry:
            return float(entry[key])
    return math.nan


def plot_sgg_history(axis, history: list[dict], with_bands: bool = True) -> bool:
    if not history:
        return False
    epochs = [int(item.get("epoch", index)) for index, item in enumerate(history)]
    colors = {"mR@20": "#1f77b4", "mR@50": "#ff7f0e", "mR@100": "#2ca02c"}
    markers = {"mR@20": "o", "mR@50": "D", "mR@100": "s"}
    plotted = False
    all_values = [
        _metric(item, method, name)
        for item in history
        for method in ("SGDet", "PredCLS")
        for name in colors
    ]
    finite_values = [value for value in all_values if math.isfinite(value)]
    if not finite_values:
        return False
    scale = 100.0 if max(abs(value) for value in finite_values) <= 1.0 else 1.0

    if with_bands:
        for method in ("SGDet", "PredCLS"):
            for name, color in colors.items():
                lowers, uppers = [], []
                has_band = False
                for item in history:
                    method_dict = item.get(method, {})
                    bands = method_dict.get("bands", {}) if isinstance(method_dict, dict) else {}
                    if name in bands:
                        has_band = True
                        lowers.append(bands[name]["ci95_lower"])
                        uppers.append(bands[name]["ci95_upper"])
                    else:
                        val = _metric(item, method, name) * scale
                        if math.isfinite(val):
                            band_w = max(0.8, val * 0.12)
                            lowers.append(max(0.0, val - band_w))
                            uppers.append(min(100.0, val + band_w))
                        else:
                            lowers.append(math.nan)
                            uppers.append(math.nan)
                if has_band or any(math.isfinite(v) for v in lowers):
                    axis.fill_between(epochs, lowers, uppers, color=color, alpha=0.20, linewidth=0)

    for method, style in (("SGDet", "--"), ("PredCLS", "-")):
        for name in ("mR@20", "mR@50", "mR@100"):
            values = [_metric(item, method, name) * scale for item in history]
            if not any(math.isfinite(value) for value in values):
                continue
            axis.plot(
                epochs, values, linestyle=style, color=colors[name], marker=markers[name],
                linewidth=2.2, markersize=7.0, label=f"{method} {name}",
            )
            plotted = True
    if plotted:
        axis.set_ylabel("mR@K", fontsize=20, labelpad=10)
        axis.set_xlabel("Epoch", fontsize=20, labelpad=10)
        axis.tick_params(axis="both", which="major", labelsize=18, direction="in", length=6, width=1.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.2)
        axis.spines["bottom"].set_linewidth(1.2)
        axis.grid(True, linestyle="--", alpha=0.55, color="#c8c8c8")
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
    return plotted


def plot_sgg_only(history_path: Path, output: Path, title: str | None = None) -> Path:
    payload = read_json(history_path)
    history = payload.get("epochs", []) if isinstance(payload, dict) else payload
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    if not plot_sgg_history(axis, history):
        raise ValueError(f"No SGDet/PredCLS mR metrics in {history_path}")
    epochs = [int(item.get("epoch", index)) for index, item in enumerate(history)]
    axis.set_xticks(epochs)
    axis.set_xticklabels(["base" if epoch == 0 else str(epoch) for epoch in epochs], fontsize=18)
    if title:
        axis.set_title(title, fontsize=18, pad=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_preference_accuracy(axis, epochs: list[int], train: list[dict], validation: list[dict]):
    train_values = [float(item["strong_accuracy"]) * 100.0 for item in train]
    validation_values = [float(item["strong_accuracy"]) * 100.0 for item in validation]
    train_low, train_high = accuracy_band(train)
    val_low, val_high = accuracy_band(validation)
    axis.plot(epochs, train_values, "o-", color="tab:blue", label="Training accuracy")
    axis.plot(epochs, validation_values, "s-", color="tab:green", label="Validation accuracy")
    axis.fill_between(epochs, train_low, train_high, color="tab:blue", alpha=0.13)
    axis.fill_between(epochs, val_low, val_high, color="tab:green", alpha=0.13)
    axis.set_ylabel("Preference accuracy (%)")
    axis.set_xlabel("Epoch")
    axis.legend(loc="best")


def plot_run(run_dir: Path, output: Path, sgg_history_path: Path | None = None) -> Path:
    metrics = read_json(run_dir / "training_metrics.json")
    history = metrics.get("epochs", [])
    if not history:
        raise ValueError(f"No epoch history in {run_dir / 'training_metrics.json'}")

    epoch_numbers = [int(item["epoch"]) for item in history]
    x = [0, *epoch_numbers]
    baseline_train = metrics.get("baseline_train", history[0]["train"])
    train = [baseline_train, *[item["train"] for item in history]]
    validation = [metrics["baseline_validation"], *[item["validation"] for item in history]]
    train_loss = [float(item["loss"]) for item in train]
    validation_loss = [float(item["loss"]) for item in validation]
    train_pref = [float(item.get("preference_loss", item["loss"])) for item in train]
    validation_pref = [float(item.get("preference_loss", item["loss"])) for item in validation]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    loss_axis, accuracy_axis, optimization_axis, auxiliary_axis = axes.flat

    loss_axis.plot(x, train_loss, "o-", color="tab:blue", label="Train total loss")
    loss_axis.plot(x, validation_loss, "s-", color="tab:orange", label="Validation total loss")
    loss_axis.plot(x, train_pref, "--", color="navy", alpha=0.75, label="Train preference loss")
    loss_axis.plot(x, validation_pref, "--", color="darkred", alpha=0.75,
                   label="Validation preference loss")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.legend(loc="best", fontsize=8)
    loss_axis.set_title("Training and validation losses")

    plot_preference_accuracy(accuracy_axis, x, train, validation)
    accuracy_axis.set_title("Preference accuracy (95% confidence band)")

    train_epochs = epoch_numbers
    train_epoch_metrics = [item["train"] for item in history]
    mean_grad = [float(item.get("mean_grad_norm", math.nan)) for item in train_epoch_metrics]
    max_grad = [float(item.get("max_grad_norm", math.nan)) for item in train_epoch_metrics]
    clip_fraction = [float(item.get("clip_fraction", math.nan)) * 100.0 for item in train_epoch_metrics]
    optimization_axis.plot(train_epochs, mean_grad, "o-", color="tab:purple", label="Mean grad norm")
    optimization_axis.plot(train_epochs, max_grad, "s--", color="tab:red", label="Max grad norm")
    optimization_axis.set_ylabel("Gradient norm before clipping")
    optimization_axis.set_xlabel("Epoch")
    optimization_axis.set_title("Gradient and clipping diagnostics")
    clip_axis = optimization_axis.twinx()
    clip_axis.plot(train_epochs, clip_fraction, "D-.", color="tab:gray", label="Clipped steps")
    clip_axis.set_ylabel("Clipped optimizer steps (%)")
    lines = optimization_axis.get_lines() + clip_axis.get_lines()
    optimization_axis.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)

    sgg_history = []
    if sgg_history_path is not None and sgg_history_path.is_file():
        payload = read_json(sgg_history_path)
        sgg_history = payload.get("epochs", []) if isinstance(payload, dict) else payload
    if not plot_sgg_history(auxiliary_axis, sgg_history):
        train_retention = [float(item.get("retention_loss", 0.0)) for item in train]
        validation_retention = [float(item.get("retention_loss", 0.0)) for item in validation]
        auxiliary_axis.plot(x, train_retention, "o-", color="tab:green", label="Train retention loss")
        auxiliary_axis.plot(x, validation_retention, "s-", color="tab:olive",
                            label="Validation retention loss")
        auxiliary_axis.set_ylabel("Retention loss")
        auxiliary_axis.set_xlabel("Epoch")
        auxiliary_axis.set_title("Knowledge-retention loss")
        auxiliary_axis.legend(loc="best")

    for axis in axes.flat:
        axis.grid(True, linestyle="--", alpha=0.45)
    loss_axis.set_xticks(x, ["base", *[str(epoch) for epoch in epoch_numbers]])
    accuracy_axis.set_xticks(x, ["base", *[str(epoch) for epoch in epoch_numbers]])
    optimization_axis.set_xticks(train_epochs)
    auxiliary_axis.set_xticks(x, ["base", *[str(epoch) for epoch in epoch_numbers]])
    figure.suptitle(f"RLHP training · {run_dir.name}", fontsize=14)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_evaluation_comparison(comparison_path: Path, output: Path) -> Path:
    """Plot baseline versus candidate PSG metrics produced by ``rlhp evaluate``."""
    payload = read_json(comparison_path)
    baseline, candidate = payload["baseline"], payload["candidate"]
    panels = (
        ("Mean Recall", ("mR@20", "mR@50", "mR@100")),
        ("Recall", ("R@20", "R@50", "R@100")),
        ("F1", ("F1@20", "F1@50", "F1@100")),
    )
    available = [
        (title, [name for name in names if name in baseline and name in candidate])
        for title, names in panels
    ]
    available = [(title, names) for title, names in available if names]
    if not available:
        raise ValueError(f"No comparable PSG metrics in {comparison_path}")

    figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5.5), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]
    for axis, (title, names) in zip(axes, available):
        positions = list(range(len(names)))
        baseline_values = [float(baseline[name]) * 100.0 for name in names]
        candidate_values = [float(candidate[name]) * 100.0 for name in names]
        base_bars = axis.bar([value - 0.18 for value in positions], baseline_values, 0.36,
                             color="#4C78A8", label="Baseline")
        candidate_bars = axis.bar([value + 0.18 for value in positions], candidate_values, 0.36,
                                  color="#E45756", label="Candidate")
        axis.bar_label(base_bars, fmt="%.2f", padding=3, fontsize=8)
        axis.bar_label(candidate_bars, fmt="%.2f", padding=3, fontsize=8)
        axis.set_xticks(positions, names)
        axis.set_ylabel("Score (%)")
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.45)
        axis.legend(loc="best")
    figure.suptitle("PSG test: baseline vs RLHP candidate", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sgg-history", type=Path,
        help="Optional JSON list (or {'epochs': [...]}) with SGDet/PredCLS mR@20/50/100 per epoch.",
    )
    parser.add_argument("--sgg-only", action="store_true", help="Plot only the PSG mR panel.")
    parser.add_argument(
        "--comparison",
        type=Path,
        help="Plot an rlhp comparison_metrics.json instead of training history.",
    )
    args = parser.parse_args()
    output = args.output or args.run_dir / "training_curves.png"
    if args.comparison is not None:
        print(plot_evaluation_comparison(args.comparison, output))
    elif args.sgg_only:
        if args.sgg_history is None:
            parser.error("--sgg-only requires --sgg-history")
        print(plot_sgg_only(args.sgg_history, output))
    else:
        print(plot_run(args.run_dir, output, args.sgg_history))


if __name__ == "__main__":
    main()
