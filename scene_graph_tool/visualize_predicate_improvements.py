#!/usr/bin/env python3
"""Visualize per-predicate recall improvements between baseline and tuned candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_predicate_improvements(
    baseline_eval: Path,
    candidate_eval: Path,
    labels_path: Path,
    output_path: Path,
    top_k: str = "100",
) -> Path:
    data0 = json.load(baseline_eval.open(encoding="utf-8"))
    data_c = json.load(candidate_eval.open(encoding="utf-8"))
    labels_data = json.load(labels_path.open(encoding="utf-8"))
    
    predicate_names = labels_data["predicates"][1:]  # 56 foreground classes
    r0 = data0["sgdet_mean_recall_list"][str(top_k)]
    rc = data_c["sgdet_mean_recall_list"][str(top_k)]
    
    entries = []
    for idx, (name, v0, vc) in enumerate(zip(predicate_names, r0, rc)):
        delta = (vc - v0) * 100.0
        entries.append({
            "name": name,
            "base": v0 * 100.0,
            "cand": vc * 100.0,
            "delta": delta,
        })
    
    # Sort by delta descending
    entries.sort(key=lambda x: x["delta"], reverse=True)
    
    # Create figure: Left panel = All 56 classes delta, Right panel = Top Improved predicates comparison
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8])
    
    ax_all = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    
    # ── Panel 1: All 56 predicates delta ──────────────────────────────
    names = [e["name"] for e in entries]
    deltas = [e["delta"] for e in entries]
    y_pos = np.arange(len(names))
    
    colors = ["#2ca02c" if d > 0.05 else ("#d62728" if d < -0.05 else "#7f7f7f") for d in deltas]
    
    bars = ax_all.barh(y_pos, deltas, color=colors, height=0.75, alpha=0.88)
    ax_all.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_all.set_yticks(y_pos)
    ax_all.set_yticklabels(names, fontsize=8.5)
    ax_all.invert_yaxis()  # Top improved at the top
    ax_all.set_xlabel(f"Recall@{top_k} Absolute Improvement Δ (%)", fontsize=12, labelpad=8)
    ax_all.set_title(f"Per-Predicate Recall@{top_k} Change Across All 56 Classes (SGDet)", fontsize=13, fontweight="bold")
    ax_all.grid(axis="x", linestyle="--", alpha=0.45)
    
    # Add value labels
    for bar, d in zip(bars, deltas):
        if abs(d) >= 1.0:
            ha = "left" if d >= 0 else "right"
            offset = 0.5 if d >= 0 else -0.5
            ax_all.text(d + offset, bar.get_y() + bar.get_height() / 2,
                        f"{d:+.1f}%", va="center", ha=ha, fontsize=7.5, fontweight="bold",
                        color="#1b5e20" if d > 0 else "#b71c1c")

    # ── Panel 2: Top 15 Improved Predicates (Base vs Cand) ───────────
    top_entries = [e for e in entries if e["delta"] > 0.0][:15]
    top_names = [e["name"] for e in top_entries]
    base_vals = [e["base"] for e in top_entries]
    cand_vals = [e["cand"] for e in top_entries]
    
    y_top = np.arange(len(top_names))
    width = 0.38
    
    b_bars = ax_top.barh(y_top - width / 2, base_vals, width, label="Baseline", color="#4C78A8", alpha=0.9)
    c_bars = ax_top.barh(y_top + width / 2, cand_vals, width, label="RLHP (Tuned Anchor + Ret20)", color="#E45756", alpha=0.9)
    
    ax_top.set_yticks(y_top)
    ax_top.set_yticklabels(top_names, fontsize=10, fontweight="bold")
    ax_top.invert_yaxis()
    ax_top.set_xlabel(f"Recall@{top_k} (%)", fontsize=12, labelpad=8)
    ax_top.set_title("Top 15 Most Improved Predicates", fontsize=13, fontweight="bold")
    ax_top.legend(loc="lower right", fontsize=10)
    ax_top.grid(axis="x", linestyle="--", alpha=0.45)
    
    # Bar value labels
    for bar in b_bars:
        w = bar.get_width()
        if w > 0:
            ax_top.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=7.5, color="#2b4c7e")
    for bar in c_bars:
        w = bar.get_width()
        if w > 0:
            ax_top.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=7.5, color="#a72d2d", fontweight="bold")

    fig.suptitle("PSG Test Predicate-Level Performance Shift after Preference Learning", fontsize=16, fontweight="bold", y=1.02)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Predicate improvement plot saved to: {output_path.resolve()}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs" / "preference3_tuned_anchor_ret20_tau01" / "psg_evaluation" / "sgdet" / "epoch_0" / "results" / "eval_summary.json")
    parser.add_argument("--candidate", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs" / "preference3_tuned_anchor_ret20_tau01" / "psg_evaluation" / "sgdet" / "epoch_5" / "results" / "eval_summary.json")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "psg_labels.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs" / "preference3_tuned_anchor_ret20_tau01" / "predicate_improvements.png")
    parser.add_argument("--top-k", type=str, default="100")
    args = parser.parse_args()

    plot_predicate_improvements(args.baseline, args.candidate, args.labels, args.output, args.top_k)


if __name__ == "__main__":
    main()
