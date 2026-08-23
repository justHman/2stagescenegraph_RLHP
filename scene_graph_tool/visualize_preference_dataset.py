#!/usr/bin/env python3
"""Visualize preference dataset characteristics: uncertainty pair frequencies and predicate class distribution."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_uncertainty_pairs(
    csv_path: Path,
    output_path: Path,
    top_n_pairs: int = 20,
    top_n_matrix: int = 15,
) -> Path:
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    
    pair_counter = Counter()
    predicate_counter = Counter()
    
    for r in rows:
        p1 = r["predicate_1_name"].strip()
        p2 = r["predicate_2_name"].strip()
        pair_key = tuple(sorted([p1, p2]))
        pair_counter[pair_key] += 1
        predicate_counter[p1] += 1
        predicate_counter[p2] += 1

    top_pairs = pair_counter.most_common(top_n_pairs)
    pair_labels = [f"{p[0]}  ⟷  {p[1]}" for p, _ in top_pairs]
    pair_counts = [count for _, count in top_pairs]

    # Matrix of top predicates co-occurrence in uncertainty pairs
    top_preds = [p for p, _ in predicate_counter.most_common(top_n_matrix)]
    matrix = np.zeros((len(top_preds), len(top_preds)), dtype=int)
    for i, p_a in enumerate(top_preds):
        for j, p_b in enumerate(top_preds):
            if i != j:
                key = tuple(sorted([p_a, p_b]))
                matrix[i, j] = pair_counter[key]

    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
    
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[0, 1])

    # Panel 1: Top N uncertainty pairs bar chart
    y_pos = np.arange(len(pair_labels))
    bars = ax_bar.barh(y_pos, pair_counts, color="#3b6998", height=0.72, alpha=0.9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(pair_labels, fontsize=10.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Frequency in Preference Feedback (Samples)", fontsize=12, labelpad=8)
    ax_bar.set_title(f"Top {top_n_pairs} Most Frequent Uncertainty / Ambiguity Pairs", fontsize=14, fontweight="bold")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.45)

    for bar, count in zip(bars, pair_counts):
        ax_bar.text(count + 0.8, bar.get_y() + bar.get_height() / 2, str(count),
                    va="center", fontsize=9.5, fontweight="bold", color="#1c3b5e")

    # Panel 2: Co-occurrence Heatmap Matrix
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=top_preds,
        yticklabels=top_preds,
        ax=ax_mat,
        cbar_kws={"label": "Pairwise Ambiguity Count", "shrink": 0.85},
        linewidths=0.5,
        linecolor="white",
    )
    ax_mat.set_title(f"Pairwise Ambiguity Co-occurrence Matrix (Top {top_n_matrix} Classes)", fontsize=14, fontweight="bold")
    ax_mat.set_xticklabels(ax_mat.get_xticklabels(), rotation=45, ha="right", fontsize=9.5)
    ax_mat.set_yticklabels(ax_mat.get_yticklabels(), rotation=0, fontsize=9.5)

    fig.suptitle(f"Analysis of Uncertainty Predicate Pairs in Preference Dataset (Total Samples = {len(rows)})",
                 fontsize=17, fontweight="bold", y=1.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Uncertainty pairs plot saved to: {output_path.resolve()}")
    return output_path


def plot_class_distribution(
    csv_path: Path,
    labels_path: Path,
    output_path: Path,
) -> Path:
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    
    labels_data = json.load(labels_path.open(encoding="utf-8"))
    all_56_predicates = labels_data["predicates"][1:]  # exclude __background__
    
    pred_counts = Counter()
    for r in rows:
        pred_counts[r["predicate_1_name"].strip()] += 1
        pred_counts[r["predicate_2_name"].strip()] += 1

    # Counts for all 56 classes
    class_stats = []
    for name in all_56_predicates:
        count = pred_counts.get(name, 0)
        class_stats.append({"name": name, "count": count})
    
    # Sort descending
    class_stats.sort(key=lambda x: x["count"], reverse=True)
    
    names = [c["name"] for c in class_stats]
    counts = [c["count"] for c in class_stats]
    
    # Split into active classes vs zero-frequency classes
    active_count = sum(1 for c in counts if c > 0)
    zero_count = len(counts) - active_count

    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
    
    x = np.arange(len(names))
    colors = ["#1f77b4" if c > 50 else ("#ff7f0e" if c > 10 else ("#2ca02c" if c > 0 else "#d3d3d3")) for c in counts]
    
    bars = ax.bar(x, counts, color=colors, width=0.75, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=9)
    ax.set_ylabel("Mention Count in Feedback Pairs", fontsize=12, labelpad=8)
    ax.set_title(
        f"Predicate Class Distribution in Preference Dataset (Coverage: {active_count}/{len(names)} Classes, {zero_count} Zero-Shot / Tail)",
        fontsize=14, fontweight="bold"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    
    # Annotations for head classes
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, count + 2.0, str(count),
                    ha="center", va="bottom", fontsize=7.5, rotation=90, fontweight="bold", color="#333333")
    
    # Legend for frequency tiers
    custom_lines = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        plt.Rectangle((0, 0), 1, 1, color="#ff7f0e"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c"),
        plt.Rectangle((0, 0), 1, 1, color="#d3d3d3"),
    ]
    ax.legend(
        custom_lines,
        ["Head (> 50 pairs)", "Body (10 - 50 pairs)", "Tail (1 - 9 pairs)", "Unseen in Preference (0 pairs)"],
        loc="upper right", fontsize=11, framealpha=0.95
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Class distribution plot saved to: {output_path.resolve()}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "scene_graph_preferences3.csv")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "psg_labels.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs" / "preference3_tuned_anchor_ret20_tau01")
    args = parser.parse_args()

    plot_uncertainty_pairs(
        args.sessions,
        args.output_dir / "uncertainty_pairs_distribution.png",
    )
    plot_class_distribution(
        args.sessions,
        args.labels,
        args.output_dir / "preference_class_distribution.png",
    )


if __name__ == "__main__":
    main()
