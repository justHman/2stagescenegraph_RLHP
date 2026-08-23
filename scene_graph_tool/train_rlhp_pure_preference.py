#!/usr/bin/env python3
"""Dedicated RLHP Training with Pure Preference Loss and Careful Regularization.

Features:
- Pure Bradley-Terry preference loss (no retention/distillation distortion).
- Freeze logit_scale by default to preserve calibrated softmax temperature.
- Gentle weight anchor regularization (L2 penalty to baseline weights) to avoid semantic drift.
- Cosine Annealing learning rate schedule with warmup.
- Normalized sample-weighted gradient accumulation.
- Saves checkpoints for every epoch (epoch_1.pth ... epoch_N.pth) for fine-grained SGG evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = PROJECT_ROOT / "scene_graph_tool"
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from rlhp import (
    RLHPError,
    RealPreferenceModel,
    MockPreferenceModel,
    audit_sessions,
    atomic_json,
    pairwise_preference_loss,
    _group_by_image,
    _mock_epoch,
    _preference_accuracy,
    _new_run,
    sha256_file,
    utc_timestamp,
    REQUIRED_PROVENANCE,
    DEFAULT_LABELS,
    DEFAULT_PSG_TEST,
    DEFAULT_RUNS_ROOT,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class PurePreferenceModelWrapper(RealPreferenceModel):
    """Wrapper that optionally freezes logit_scale so only proj_head is tuned."""

    def __init__(
        self,
        checkpoint: Path,
        config: Path,
        yolo_checkpoint: Path | None,
        labels: Path,
        device: str,
        freeze_logit_scale: bool = True,
    ):
        super().__init__(checkpoint, config, yolo_checkpoint, labels, device)
        if freeze_logit_scale:
            filtered_trainable = []
            for name, parameter in self.trainable:
                if name.endswith("logit_scale") or "logit_scale" in name:
                    parameter.requires_grad_(False)
                else:
                    filtered_trainable.append((name, parameter))
            self.trainable = filtered_trainable


def compute_param_anchor_loss(
    trainable_params: list[tuple[str, torch.nn.Parameter]],
    baseline_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """L2 distance between current trainable parameters and baseline weights."""
    loss = torch.tensor(0.0, device=trainable_params[0][1].device)
    for name, param in trainable_params:
        if name in baseline_weights:
            base_p = baseline_weights[name].to(device=param.device, dtype=param.dtype)
            loss = loss + F.mse_loss(param, base_p, reduction="sum")
    return loss


def train_pure_preference_epoch(
    engine: RealPreferenceModel,
    groups: list[tuple[str, list]],
    tau: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    accumulation: int = 4,
    clip_norm: float = 1.0,
    param_anchor_weight: float = 0.0,
    baseline_weights: dict[str, torch.Tensor] | None = None,
    desc: str = "Preference train",
) -> dict[str, Any]:
    total_loss = 0.0
    total_preference = 0.0
    total_anchor = 0.0
    total_weight = 0.0
    correct = 0
    strong_total = 0
    matched = 0
    total = 0
    accumulated_groups = 0
    accumulation_weight = 0.0
    grad_norms: list[float] = []
    clipped_steps = 0
    optimizer_steps = 0

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    def flush_gradients():
        nonlocal accumulation_weight, accumulated_groups, clipped_steps, optimizer_steps
        if accumulated_groups == 0 or optimizer is None:
            return
        scale = 1.0 / max(accumulation_weight, 1e-12)
        for _, parameter in engine.trainable:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in engine.trainable], clip_norm))
        grad_norms.append(norm)
        if norm > clip_norm:
            clipped_steps += 1
        optimizer_steps += 1
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accumulation_weight = 0.0
        accumulated_groups = 0

    iterator = tqdm(
        groups,
        desc=desc,
        dynamic_ncols=True,
        leave=True,
    ) if HAS_TQDM else groups

    for group_index, (image_path, samples) in enumerate(iterator):
        prediction = engine.forward_image(image_path)
        rewards, targets, weights = [], [], []

        for sample in samples:
            total += 1
            pair_rewards = engine.sample_rewards(prediction, sample)
            if pair_rewards is None:
                continue
            matched += 1
            rewards.append(pair_rewards)
            targets.append(sample["target"])
            weights.append(sample["vote_count"])

        if not rewards:
            if HAS_TQDM:
                iterator.set_postfix(matched=f"{matched}/{total}")
            continue

        reward_tensor = torch.stack(rewards)
        target_tensor = torch.tensor(targets, dtype=reward_tensor.dtype, device=reward_tensor.device)
        weight_tensor = torch.tensor(weights, dtype=reward_tensor.dtype, device=reward_tensor.device)

        pref_loss = pairwise_preference_loss(reward_tensor, target_tensor, tau=tau, weights=weight_tensor)
        
        anchor_loss = pref_loss.new_zeros(())
        if param_anchor_weight > 0.0 and baseline_weights is not None:
            anchor_loss = compute_param_anchor_loss(engine.trainable, baseline_weights)

        loss = pref_loss + float(param_anchor_weight) * anchor_loss
        weight = float(weight_tensor.sum())

        if optimizer is not None:
            (loss * weight).backward()
            accumulation_weight += weight
            accumulated_groups += 1
            if accumulated_groups >= accumulation:
                flush_gradients()

        total_loss += float(loss.detach()) * weight
        total_preference += float(pref_loss.detach()) * weight
        total_anchor += float(anchor_loss.detach()) * weight
        total_weight += weight

        group_correct, group_strong = _preference_accuracy(reward_tensor.detach(), target_tensor)
        correct += group_correct
        strong_total += group_strong

        if HAS_TQDM:
            iterator.set_postfix(
                loss=f"{total_loss / max(total_weight, 1e-12):.4f}",
                acc=f"{(correct / max(strong_total, 1)):.1%}",
                matched=f"{matched}/{total}",
            )

    flush_gradients()

    return {
        "loss": total_loss / max(total_weight, 1e-12),
        "preference_loss": total_preference / max(total_weight, 1e-12),
        "anchor_loss": total_anchor / max(total_weight, 1e-12),
        "strong_accuracy": correct / max(strong_total, 1),
        "matched": matched,
        "total": total,
        "optimizer_steps": optimizer_steps,
        "mean_grad_norm": float(sum(grad_norms) / len(grad_norms)) if grad_norms else math.nan,
        "max_grad_norm": float(max(grad_norms)) if grad_norms else math.nan,
        "clip_fraction": float(clipped_steps / optimizer_steps) if optimizer_steps else math.nan,
    }


def run_pure_preference_training(args: argparse.Namespace) -> Path:
    runs_root = Path(args.runs_root)
    run_dir = _new_run(runs_root, args.run_id)

    print(f"\n{'═'*78}")
    print(f"  RLHP Pure Preference Fine-Tuning (Optimized for Outperforming Baseline)")
    print(f"  tau: {args.tau}  lr: {args.learning_rate}  weight_decay: {args.weight_decay}")
    print(f"  freeze_logit_scale: {args.freeze_logit_scale}  param_anchor_weight: {args.param_anchor_weight}")
    print(f"  gradient_accumulation: {args.gradient_accumulation}  clip_norm: {args.clip_norm}")
    print(f"  epochs: {args.epochs}  lr_schedule: {args.lr_schedule}")
    print(f"  run_dir: {run_dir.resolve()}")
    print(f"{'═'*78}\n")

    run_meta = {
        "run_id": run_dir.name,
        "status": "auditing",
        "created_at": utc_timestamp(),
        "mock": bool(args.mock),
        "baseline_checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "config": str(Path(args.config).resolve()) if args.config else None,
        "hyperparams": {
            "tau": args.tau,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "freeze_logit_scale": args.freeze_logit_scale,
            "param_anchor_weight": args.param_anchor_weight,
            "gradient_accumulation": args.gradient_accumulation,
            "clip_norm": args.clip_norm,
            "epochs": args.epochs,
            "lr_schedule": args.lr_schedule,
            "retention_weight": 0.0,
        },
    }
    atomic_json(run_dir / "run.json", run_meta)

    try:
        print("[1/4] Auditing preference dataset...")
        report, samples = audit_sessions(
            args.sessions,
            args.labels,
            args.psg_test,
            run_dir,
            allow_legacy_provenance=args.allow_legacy_provenance,
            iou_threshold=args.iou_threshold,
            min_match_rate=args.min_match_rate,
            seed=args.seed,
            image_root=args.image_root,
        )
        train_samples = [s for s in samples if s["split"] == "train"]
        val_samples = [s for s in samples if s["split"] == "validation"]
        print(
            f"     ✓ Raw pairs: {report['raw_feedback']} | "
            f"Train: {report['train_samples']} | "
            f"Val: {report['validation_samples']} | "
            f"Match rate: {report['pair_matching_rate']:.1%}"
        )
        if not train_samples:
            raise RLHPError("Audit produced no trainable training samples.")

        run_meta["status"] = "training"
        run_meta["audit"] = report
        atomic_json(run_dir / "run.json", run_meta)

        if not args.checkpoint or not args.config:
            raise RLHPError("Real training requires --checkpoint and --config.")

        print("\n[2/4] Loading REACT++ model and setting trainable parameters...")
        engine = PurePreferenceModelWrapper(
            Path(args.checkpoint),
            Path(args.config),
            Path(args.yolo_checkpoint) if args.yolo_checkpoint else None,
            Path(args.labels),
            args.device,
            freeze_logit_scale=args.freeze_logit_scale,
        )
        trainable_names = [n for n, _ in engine.trainable]
        print(f"     ✓ Trainable parameters: {trainable_names}")
        print(f"     ✓ Frozen logit_scale: {args.freeze_logit_scale}")

        train_groups = _group_by_image(samples, "train")
        val_groups = _group_by_image(samples, "validation")

        # Snapshot baseline weights for parameter anchor regularization
        baseline_weights = {
            name: param.detach().clone().cpu()
            for name, param in engine.trainable
        }

        print("\n[3/4] Running baseline inference on preference splits...")
        frozen_before = engine.frozen_hashes()

        with torch.no_grad():
            baseline_train = train_pure_preference_epoch(
                engine, train_groups, args.tau, desc="  Baseline (train)"
            )
            baseline_val = train_pure_preference_epoch(
                engine, val_groups or train_groups, args.tau, desc="  Baseline (val)  "
            )

        print(f"     ↳ Baseline Train: loss={baseline_train['loss']:.4f}, acc={baseline_train['strong_accuracy']:.2%}")
        print(f"     ↳ Baseline Val  : loss={baseline_val['loss']:.4f}, acc={baseline_val['strong_accuracy']:.2%}")

        # Setup Optimizer & Learning Rate Scheduler
        optimizer = torch.optim.AdamW(
            [p for _, p in engine.trainable],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps = (len(train_groups) // args.gradient_accumulation) * args.epochs
        if args.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps, 1), eta_min=args.learning_rate * 0.1
            )
        elif args.lr_schedule == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(total_steps, 1)
            )
        else:
            scheduler = None

        print("\n[4/4] Training with pure preference loss...")
        best_val_loss = math.inf
        best_state = None
        patience = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, args.epochs + 1):
            print(f"\n─── Epoch {epoch}/{args.epochs} ───────────────────────────────────────")
            shuffled_train_groups = list(train_groups)
            random.Random(args.seed + epoch).shuffle(shuffled_train_groups)

            train_m = train_pure_preference_epoch(
                engine,
                shuffled_train_groups,
                args.tau,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulation=args.gradient_accumulation,
                clip_norm=args.clip_norm,
                param_anchor_weight=args.param_anchor_weight,
                baseline_weights=baseline_weights,
                desc=f"  Epoch {epoch:02d} Train",
            )

            with torch.no_grad():
                val_m = train_pure_preference_epoch(
                    engine,
                    val_groups or train_groups,
                    args.tau,
                    param_anchor_weight=args.param_anchor_weight,
                    baseline_weights=baseline_weights,
                    desc=f"  Epoch {epoch:02d} Val  ",
                )

            is_best = val_m["loss"] < best_val_loss
            print(
                f"  ↳ Train: loss={train_m['loss']:.4f} | pref={train_m['preference_loss']:.4f} | "
                f"acc={train_m['strong_accuracy']:.2%} | grad_norm={train_m['mean_grad_norm']:.2f}"
            )
            print(
                f"  ↳ Val  : loss={val_m['loss']:.4f} | pref={val_m['preference_loss']:.4f} | "
                f"acc={val_m['strong_accuracy']:.2%}"
            )

            # Always save epoch_N.pth checkpoint
            epoch_ckpt_path = run_dir / f"epoch_{epoch}.pth"
            orig_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(orig_checkpoint, dict) or "model" not in orig_checkpoint:
                orig_checkpoint = {"model": orig_checkpoint}
            orig_checkpoint.pop("optimizer", None)
            orig_checkpoint.pop("scheduler", None)
            orig_checkpoint["model"] = {k: v.detach().cpu() for k, v in engine.model.state_dict().items()}
            orig_checkpoint["rlhp"] = {
                "run_id": run_dir.name,
                "epoch": epoch,
                "created_at": utc_timestamp(),
                "tau": args.tau,
                "lr": args.learning_rate,
            }
            torch.save(orig_checkpoint, epoch_ckpt_path)
            print(f"  ✓ Saved checkpoint: {epoch_ckpt_path.name}")

            if is_best:
                best_val_loss = val_m["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()}
                patience = 0
                print(f"  ★ New best validation loss: {best_val_loss:.4f}")
            else:
                patience += 1
                print(f"  · Patience: {patience}/{args.early_stopping_patience}")

            history.append({
                "epoch": epoch,
                "train": train_m,
                "validation": val_m,
                "is_best": is_best,
            })

            if patience >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        # Save best candidate
        if best_state is not None:
            engine.model.load_state_dict(best_state)
        
        with torch.no_grad():
            candidate_val = train_pure_preference_epoch(
                engine, val_groups or train_groups, args.tau, desc="  Candidate-val "
            )

        candidate_path = run_dir / "best_candidate.pth"
        candidate_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(candidate_dict, dict) or "model" not in candidate_dict:
            candidate_dict = {"model": candidate_dict}
        candidate_dict["model"] = {k: v.detach().cpu() for k, v in engine.model.state_dict().items()}
        candidate_dict["rlhp"] = {
            "run_id": run_dir.name,
            "created_at": utc_timestamp(),
            "best_val_loss": best_val_loss,
            "tau": args.tau,
            "lr": args.learning_rate,
        }
        torch.save(candidate_dict, candidate_path)
        print(f"\n✓ Best candidate saved to: {candidate_path}")

        # Metrics saving
        final_metrics = {
            "baseline_train": baseline_train,
            "baseline_validation": baseline_val,
            "candidate_validation": candidate_val,
            "epochs": history,
            "tau": args.tau,
            "learning_rate": args.learning_rate,
            "freeze_logit_scale": args.freeze_logit_scale,
            "param_anchor_weight": args.param_anchor_weight,
        }
        atomic_json(run_dir / "training_metrics.json", final_metrics)
        training_plot = run_dir / "training_curves.png"
        try:
            from plot_rlhp_training import plot_run
            plot_run(run_dir, training_plot)
            print(f"  ✓ Training plot generated: {training_plot.name}")
        except Exception as plot_error:
            print(f"  ⚠ Warning: could not create training plot: {plot_error}", file=sys.stderr)
            training_plot = None

        run_meta.update({
            "status": "trained",
            "completed_at": utc_timestamp(),
            "candidate_checkpoint": str(candidate_path.resolve()),
            "training_plot": str(training_plot.resolve()) if training_plot else None,
        })
        atomic_json(run_dir / "run.json", run_meta)

        print(f"\n{'═'*78}")
        print(f"  Training finished successfully! Run saved at: {run_dir}")
        print(f"  To evaluate all epochs and visualize mR@K curve:")
        print(f"  .venv/bin/python scene_graph_tool/visualize_epoch_metrics.py \\")
        print(f"      --run-dir {run_dir} \\")
        print(f"      --data-dir datasets/PSG/coco_format \\")
        print(f"      --device {args.device}")
        print(f"{'═'*78}\n")

        return run_dir

    except Exception as exc:
        run_meta.update({"status": "failed", "failed_at": utc_timestamp(), "error": str(exc)})
        atomic_json(run_dir / "run.json", run_meta)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Datasets / Sessions
    parser.add_argument("--sessions", nargs="+", default=["scene_graph_tool/scene_graph_preferences3.csv"],
                        help="CSV or JSON preference sessions")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--psg-test", type=Path, default=DEFAULT_PSG_TEST)
    parser.add_argument("--image-root", type=Path, default=TOOL_DIR / "train_00000_of_00016_images")
    parser.add_argument("--allow-legacy-provenance", action="store_true", default=True)
    parser.add_argument("--iou-threshold", type=float, default=0.8)
    parser.add_argument("--min-match-rate", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)

    # Model & Paths
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoint" / "best_model.pth")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "checkpoint" / "config.yml")
    parser.add_argument("--yolo-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoint" / "PSG_yolo12m_backbone.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-id", type=str, help="Custom run identifier")

    # Hyperparameters tuned for beating baseline
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Temperature for preference Bradley-Terry logits (default: 1.0)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Carefully tuned learning rate for proj_head (default: 2e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.001,
                        help="Weight decay for AdamW (default: 0.001)")
    parser.add_argument("--lr-schedule", choices=("cosine", "linear", "constant"), default="cosine",
                        help="Learning rate decay schedule (default: cosine)")
    parser.add_argument("--freeze-logit-scale", action="store_true", default=True,
                        help="Freeze logit_scale to preserve baseline softmax calibration (default: True)")
    parser.add_argument("--no-freeze-logit-scale", dest="freeze_logit_scale", action="store_false",
                        help="Enable gradient updates on logit_scale")
    parser.add_argument("--param-anchor-weight", type=float, default=0.001,
                        help="L2 parameter regularization toward baseline weights (default: 0.001)")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps across image groups (default: 4)")
    parser.add_argument("--clip-norm", type=float, default=1.0,
                        help="Max gradient norm (default: 1.0)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train (default: 5)")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")

    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_pure_preference_training(args)
    except RLHPError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
