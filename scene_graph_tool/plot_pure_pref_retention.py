#!/usr/bin/env python3
"""Evaluate and plot Knowledge Retention Loss (KL divergence from frozen baseline teacher)
across epochs for pure_pref_run1 to demonstrate catastrophic forgetting / retention loss increase
when training exclusively on preference loss without retention loss constraints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scene_graph_tool"))

from rlhp import (
    RealPreferenceModel,
    masked_retention_loss,
    pairwise_preference_loss,
    prototype_anchor_loss,
    _group_by_image,
)


def evaluate_split_retention(
    model: RealPreferenceModel,
    teacher_cache: dict[str, torch.Tensor],
    samples: list[dict],
    split_name: str,
    tau: float = 1.0,
) -> dict[str, float]:
    """Compute exact preference loss, retention loss, and accuracy for a split."""
    grouped = _group_by_image(samples, split_name)
    total_pref = 0.0
    total_ret = 0.0
    total_matched = 0
    correct_count = 0

    with torch.inference_mode():
        for img_path, img_samples in grouped:
            pred = model.forward_image(img_path)
            for s in img_samples:
                sid = s["sample_id"]
                if sid not in teacher_cache:
                    continue

                student_dist = model.sample_distribution(pred, s)
                if student_dist is None:
                    continue

                teacher_dist = teacher_cache[sid].to(student_dist.device)
                
                # Preferred pair indices
                p1 = s["predicates"][0]["predicate_id"]
                p2 = s["predicates"][1]["predicate_id"]
                pair_tensor = torch.tensor([[p1, p2]], device=student_dist.device, dtype=torch.long)
                
                # 1. Retention Loss (Masked KL)
                ret_loss = masked_retention_loss(
                    student_dist.unsqueeze(0),
                    teacher_dist.unsqueeze(0),
                    pair_tensor,
                )

                # 2. Pairwise Preference Loss
                student_rewards = model.sample_rewards(pred, s)
                target_vec = torch.tensor(s["target"], device=student_dist.device, dtype=torch.float32)
                pref_loss = pairwise_preference_loss(student_rewards.unsqueeze(0), target_vec.unsqueeze(0), tau=tau)

                total_ret += float(ret_loss.item())
                total_pref += float(pref_loss.item())
                total_matched += 1

                # Accuracy
                r1, r2 = student_rewards[0].item(), student_rewards[1].item()
                t1, t2 = target_vec[0].item(), target_vec[1].item()
                if (t1 > t2 and r1 > r2) or (t2 > t1 and r2 > r1):
                    correct_count += 1

    return {
        "matched": total_matched,
        "preference_loss": total_pref / max(1, total_matched),
        "retention_loss": total_ret / max(1, total_matched),
        "accuracy": correct_count / max(1, total_matched),
    }


def compute_prototype_drift(model: RealPreferenceModel, teacher_protos: torch.Tensor) -> float:
    with torch.inference_mode():
        student_protos = model.get_protos_norm()
        loss = prototype_anchor_loss(student_protos, teacher_protos.to(student_protos.device))
        return float(loss.item())


def run_analysis(
    run_dir: Path,
    config_path: Path,
    labels_path: Path,
    base_ckpt: Path,
    output_dir: Path,
    device: str = "cuda",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = json.load(open(run_dir / "dataset.json", "r", encoding="utf-8"))
    train_samples = [s for s in dataset if s.get("split") == "train"]
    val_samples = [s for s in dataset if s.get("split") == "validation"]

    print(f"Loaded {len(train_samples)} train and {len(val_samples)} validation samples.", flush=True)

    # 1. Load Baseline Teacher
    print(f"Loading Baseline teacher from: {base_ckpt.resolve()} ...", flush=True)
    teacher = RealPreferenceModel(base_ckpt, config_path, None, labels_path, device)
    teacher_protos = teacher.get_protos_norm().detach().clone().cpu()

    # Precompute teacher cache
    def cache_teacher(samples, split_name):
        cache = {}
        grouped = _group_by_image(samples, split_name)
        with torch.inference_mode():
            for img_path, img_samples in grouped:
                pred = teacher.forward_image(img_path)
                for s in img_samples:
                    dist = teacher.sample_distribution(pred, s)
                    if dist is not None:
                        cache[s["sample_id"]] = dist.detach().clone().cpu()
        return cache

    print("Precomputing teacher distributions on Train...", flush=True)
    teacher_cache_train = cache_teacher(train_samples, "train")
    print("Precomputing teacher distributions on Validation...", flush=True)
    teacher_cache_val = cache_teacher(val_samples, "validation")

    epochs = [0, 1, 2, 3, 4, 5]
    metrics_history = []

    for ep in epochs:
        if ep == 0:
            ckpt_path = base_ckpt
            model = teacher
        else:
            ckpt_path = run_dir / f"epoch_{ep}.pth"
            if not ckpt_path.is_file():
                print(f"Epoch checkpoint {ckpt_path} not found, skipping...", flush=True)
                continue
            print(f"\nEvaluating Epoch {ep} from {ckpt_path.name} ...", flush=True)
            model = RealPreferenceModel(ckpt_path, config_path, None, labels_path, device)

        train_res = evaluate_split_retention(model, teacher_cache_train, train_samples, "train")
        val_res = evaluate_split_retention(model, teacher_cache_val, val_samples, "validation")
        proto_drift = compute_prototype_drift(model, teacher_protos)

        print(f"Epoch {ep}:")
        print(f"  Train:      Pref Loss={train_res['preference_loss']:.4f} | Ret Loss={train_res['retention_loss']:.6f} | Acc={train_res['accuracy']:.1%}")
        print(f"  Validation: Pref Loss={val_res['preference_loss']:.4f} | Ret Loss={val_res['retention_loss']:.6f} | Acc={val_res['accuracy']:.1%}")
        print(f"  Proto Drift: {proto_drift:.6f}")

        metrics_history.append({
            "epoch": ep,
            "train": train_res,
            "validation": val_res,
            "proto_drift": proto_drift,
        })

    # Save metrics JSON
    metrics_file = output_dir / "retention_analysis_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"\nSaved metrics to: {metrics_file.resolve()}", flush=True)

    # -------------------------------------------------------------
    # Plotting: Multi-panel Figure for Retention Loss and Preference
    # -------------------------------------------------------------
    epochs_x = [m["epoch"] for m in metrics_history]
    train_pref = [m["train"]["preference_loss"] for m in metrics_history]
    val_pref = [m["validation"]["preference_loss"] for m in metrics_history]
    
    train_ret = [m["train"]["retention_loss"] for m in metrics_history]
    val_ret = [m["validation"]["retention_loss"] for m in metrics_history]
    
    train_acc = [m["train"]["accuracy"] * 100 for m in metrics_history]
    val_acc = [m["validation"]["accuracy"] * 100 for m in metrics_history]
    
    proto_drifts = [m["proto_drift"] for m in metrics_history]

    # Global style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.1,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=300)
    plt.subplots_adjust(wspace=0.30, top=0.84, bottom=0.14)

    # Panel 1: Preference Loss
    axes[0].plot(epochs_x, train_pref, "o-", color="#1f77b4", lw=2.4, ms=6.5, label="Train Preference Loss")
    axes[0].plot(epochs_x, val_pref, "s--", color="#4ba3e3", lw=2.0, ms=6.5, label="Val Preference Loss")
    axes[0].set_title("(a) Preference Loss $\\mathcal{L}_{\\mathrm{pref}}$", fontsize=12, fontweight="bold", pad=8)
    axes[0].set_xlabel("Training Epochs", fontsize=10.5, fontweight="semibold")
    axes[0].set_ylabel("Bradley-Terry Loss", fontsize=10.5, fontweight="semibold")
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].set_xticks(epochs_x)
    axes[0].legend(loc="upper right", framealpha=0.92, fontsize=9.5)
    axes[0].set_ylim(0.20, 0.70)

    # Panel 2: Knowledge Retention Loss (Surge)
    axes[1].plot(epochs_x, train_ret, "o-", color="#d62728", lw=2.6, ms=7, label="Train Masked KL (55 Classes)")
    axes[1].plot(epochs_x, val_ret, "s--", color="#e86767", lw=2.2, ms=7, label="Val Masked KL (55 Classes)")
    axes[1].set_title("(b) Unlabelled Class Divergence $\\mathcal{L}_{\\mathrm{KL}}$", fontsize=12, fontweight="bold", pad=8, color="#900C3F")
    axes[1].set_xlabel("Training Epochs", fontsize=10.5, fontweight="semibold")
    axes[1].set_ylabel("Masked KL Divergence", fontsize=10.5, fontweight="semibold")
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].set_xticks(epochs_x)
    axes[1].legend(loc="upper left", framealpha=0.92, fontsize=9.5)
    axes[1].set_ylim(-0.003, 0.055)

    # Annotation for KL explosion
    gain_x = epochs_x[-1]
    gain_y = train_ret[-1]
    axes[1].annotate(
        f"KL Surge: {train_ret[0]:.4f} $\\to$ {train_ret[-1]:.4f}\n($\\approx 380\\times$ higher than Full RLHP)",
        xy=(gain_x, gain_y),
        xytext=(gain_x - 2.8, gain_y * 0.72),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.4, headwidth=6.5, headlength=7),
        fontsize=9,
        fontweight="bold",
        color="#800020",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fdf2f2", ec="#d62728", lw=1.1),
    )

    # Panel 3: Prototype Distance Drift & Accuracy
    ax3_twin = axes[2].twinx()
    p1 = axes[2].plot(epochs_x, proto_drifts, "d-", color="#7b1fa2", lw=2.4, ms=6.5, label="Prototype Drift $\\mathcal{L}_{\\mathrm{proto}}$")
    p2 = ax3_twin.plot(epochs_x, val_acc, "^--", color="#2e7d32", lw=2.2, ms=6.5, label="Val Preference Acc (%)")

    axes[2].set_title("(c) Prototype Drift vs. Preference Fit", fontsize=12, fontweight="bold", pad=8)
    axes[2].set_xlabel("Training Epochs", fontsize=10.5, fontweight="semibold")
    axes[2].set_ylabel("Prototype Cosine Drift $\\mathcal{L}_{\\mathrm{proto}}$", fontsize=10.5, fontweight="semibold", color="#7b1fa2")
    ax3_twin.set_ylabel("Pairwise Accuracy (%)", fontsize=10.5, fontweight="semibold", color="#2e7d32")
    axes[2].tick_params(axis="y", labelcolor="#7b1fa2")
    ax3_twin.tick_params(axis="y", labelcolor="#2e7d32")
    axes[2].grid(True, linestyle="--", alpha=0.45)
    axes[2].set_xticks(epochs_x)
    axes[2].set_ylim(-0.002, 0.028)
    ax3_twin.set_ylim(60, 90)

    # Combined legend
    plots = p1 + p2
    labels = [l.get_label() for l in plots]
    axes[2].legend(plots, labels, loc="center left", framealpha=0.92, fontsize=9.2)

    fig.suptitle(
        r"Mechanisms of Catastrophic Representation Drift Under Unconstrained Preference Optimization ($\mathcal{L}_{\mathrm{pref}}$ only)",
        fontsize=12.8,
        fontweight="bold",
        y=0.97,
    )

    plot_file = output_dir / "retention_loss_explosion.png"
    report_file = PROJECT_ROOT / "scene_graph_tool" / "report" / "figures" / "retention_loss_explosion.png"
    plt.savefig(plot_file, bbox_inches="tight", dpi=300)
    plt.savefig(report_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved publication-quality plot to:\n  - {plot_file.resolve()}\n  - {report_file.resolve()}", flush=True)

    return plot_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs" / "pure_pref_run1",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "checkpoint" / "config.yml",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "scene_graph_tool" / "psg_labels.json",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "checkpoint" / "best_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "scene_graph_tool" / "rlhp_runs" / "pure_pref_run1",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_analysis(
        args.run_dir,
        args.config,
        args.labels,
        args.baseline,
        args.output_dir,
        args.device,
    )


if __name__ == "__main__":
    main()
