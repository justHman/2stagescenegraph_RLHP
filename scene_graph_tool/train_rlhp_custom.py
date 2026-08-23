#!/usr/bin/env python3
"""Custom RLHP training script with tau=1, lr=1e-4 và progress bar đầy đủ ra terminal.

Chạy từ thư mục gốc SGG-Benchmark:
    python scene_graph_tool/train_rlhp_custom.py [OPTIONS]

Ví dụ nhanh (real GPU):
    python scene_graph_tool/train_rlhp_custom.py \\
        --sessions  scene_graph_tool/exports/ \\
        --checkpoint checkpoint/best_model.pth \\
        --config    checkpoint/config.yml \\
        --run-id    tau1_lr1e4_run1

Ví dụ mock (không cần GPU, test nhanh pipeline):
    python scene_graph_tool/train_rlhp_custom.py --mock \\
        --sessions scene_graph_tool/exports/
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ── Thêm scene_graph_tool vào path để import rlhp ──────────────────────────
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
    _real_epoch,
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

# ── Cố gắng import tqdm; nếu không có thì dùng fallback ────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[WARN] tqdm chưa cài. Chạy: pip install tqdm  để có progress bar đẹp hơn.")


# ═══════════════════════════════════════════════════════════════════════════
# Fallback progress bar đơn giản khi không có tqdm
# ═══════════════════════════════════════════════════════════════════════════
class _SimplePbar:
    """Progress bar đơn giản dùng khi tqdm không có sẵn."""
    def __init__(self, total: int, desc: str = "", **kwargs):
        self.total = total
        self.desc = desc
        self.n = 0
        self._last_print = 0.0
        self._start = time.time()
        self._postfix: dict[str, Any] = {}

    def update(self, n: int = 1):
        self.n += n
        now = time.time()
        if now - self._last_print > 1.0 or self.n >= self.total:
            elapsed = now - self._start
            pct = self.n / max(self.total, 1) * 100
            rate = self.n / max(elapsed, 1e-6)
            eta = (self.total - self.n) / max(rate, 1e-6)
            suffix = "  ".join(f"{k}={v}" for k, v in self._postfix.items())
            print(
                f"\r{self.desc}: {self.n}/{self.total} [{pct:5.1f}%] "
                f"[{elapsed:.0f}s<{eta:.0f}s] {suffix}",
                end="", flush=True
            )
            self._last_print = now

    def set_postfix(self, ordered_dict=None, **kwargs):
        if ordered_dict:
            self._postfix.update(ordered_dict)
        self._postfix.update(kwargs)

    def set_description(self, desc: str):
        self.desc = desc

    def close(self):
        print()  # newline sau khi xong

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make_pbar(total: int, desc: str = "", **kwargs):
    """Tạo tqdm nếu có, ngược lại dùng _SimplePbar."""
    if HAS_TQDM:
        return tqdm(total=total, desc=desc, ncols=100, **kwargs)
    return _SimplePbar(total=total, desc=desc)


# ═══════════════════════════════════════════════════════════════════════════
# Epoch training với progress bar
# ═══════════════════════════════════════════════════════════════════════════
def _real_epoch_with_pbar(
    engine: RealPreferenceModel,
    groups: list[tuple[str, list]],
    tau: float,
    optimizer=None,
    accumulation: int = 4,
    clip_norm: float = 1.0,
    teacher_cache: dict | None = None,
    retention_weight: float = 0.0,
    capture_teacher: dict | None = None,
    desc: str = "Train",
) -> dict:
    """Compatibility wrapper around the canonical progress-enabled epoch."""
    return _real_epoch(
        engine,
        groups,
        tau,
        optimizer=optimizer,
        accumulation=accumulation,
        clip_norm=clip_norm,
        teacher_cache=teacher_cache,
        retention_weight=retention_weight,
        capture_teacher=capture_teacher,
        desc=desc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pretty print helpers
# ═══════════════════════════════════════════════════════════════════════════
SEP = "─" * 72

def _print_epoch_header(epoch: int, max_epoch: int, split: str):
    print(f"\n{SEP}")
    print(f"  Epoch {epoch}/{max_epoch}  │  {split}")
    print(SEP)


def _print_metrics(label: str, m: dict):
    loss      = m.get("loss", math.nan)
    pref_loss = m.get("preference_loss", loss)
    ret_loss  = m.get("retention_loss", 0.0)
    acc       = m.get("strong_accuracy", math.nan)
    matched   = m.get("matched", "?")
    total_s   = m.get("total", "?")
    print(
        f"  {label:12s}│  loss={loss:.4f}  pref={pref_loss:.4f}  "
        f"ret={ret_loss:.4f}  acc={acc:.2%}  matched={matched}/{total_s}"
    )


def _print_summary_table(history: list[dict]):
    print(f"\n{SEP}")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Train Acc':>10}  {'Val Acc':>8}")
    print(SEP)
    for entry in history:
        ep  = entry["epoch"]
        tl  = entry["train"].get("loss", math.nan)
        vl  = entry["validation"].get("loss", math.nan)
        ta  = entry["train"].get("strong_accuracy", math.nan)
        va  = entry["validation"].get("strong_accuracy", math.nan)
        best = " ◀ best" if entry.get("is_best") else ""
        print(f"  {ep:>5}  {tl:>10.4f}  {vl:>10.4f}  {ta:>9.2%}  {va:>7.2%}{best}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline chính
# ═══════════════════════════════════════════════════════════════════════════
def train_with_progress(args: argparse.Namespace) -> Path:
    runs_root = Path(args.runs_root)
    run_dir   = _new_run(runs_root, args.run_id)

    # ── Banner ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  RLHP Custom Training")
    print(f"  tau={args.tau}  lr={args.learning_rate}  epochs={args.epochs}")
    print(f"  gradient_accumulation={args.gradient_accumulation}  clip_norm={args.clip_norm}")
    print(f"  run_id: {run_dir.name}")
    print(f"  device: {args.device}  mock: {args.mock}")
    print(f"{'═'*72}\n")

    run: dict = {
        "run_id":              run_dir.name,
        "status":              "auditing",
        "created_at":          utc_timestamp(),
        "mock":                bool(args.mock),
        "baseline_checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "config":              str(Path(args.config).resolve()) if args.config else None,
        "hyperparams": {
            "tau":                  args.tau,
            "learning_rate":        args.learning_rate,
            "weight_decay":         args.weight_decay,
            "gradient_accumulation": args.gradient_accumulation,
            "clip_norm":            args.clip_norm,
            "epochs":               args.epochs,
            "retention_weight":     getattr(args, "retention_weight", 0.0),
        },
    }
    atomic_json(run_dir / "run.json", run)

    try:
        # ── BƯỚC 1: Audit ─────────────────────────────────────────────────
        print(f"[1/4] Auditing preference sessions …")
        report, samples = audit_sessions(
            args.sessions, args.labels, args.psg_test, run_dir,
            allow_legacy_provenance=args.allow_legacy_provenance,
            iou_threshold=args.iou_threshold,
            min_match_rate=args.min_match_rate,
            seed=args.seed,
            image_root=args.image_root,
        )
        train_samples = [s for s in samples if s["split"] == "train"]
        val_samples   = [s for s in samples if s["split"] == "validation"]
        print(
            f"     ✓ raw={report['raw_feedback']}  "
            f"train={report['train_samples']}  "
            f"val={report['validation_samples']}  "
            f"audit_only={report['audit_only_samples']}  "
            f"match_rate={report['pair_matching_rate']:.1%}"
        )
        if not train_samples:
            raise RLHPError("Audit produced no trainable training samples.")

        run["status"] = "training"
        run["audit"]  = report
        atomic_json(run_dir / "run.json", run)

        history: list[dict] = []

        # ── MOCK branch ─────────────────────────────────────────────────────
        if args.mock:
            print(f"\n[2/4] Mock training (tau={args.tau}, lr={args.learning_rate}) …")
            model = MockPreferenceModel()
            baseline_train = _mock_epoch(model, train_samples, args.tau)
            baseline       = _mock_epoch(model, val_samples or train_samples, args.tau)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            best_loss, best_state, patience = math.inf, None, 0

            for epoch in range(1, args.epochs + 1):
                _print_epoch_header(epoch, args.epochs, "Mock")
                train_m = _mock_epoch(model, train_samples, args.tau, optimizer)
                val_m   = _mock_epoch(model, val_samples or train_samples, args.tau)
                _print_metrics("train", train_m)
                _print_metrics("val",   val_m)

                is_best = val_m["loss"] < best_loss
                entry = {"epoch": epoch, "train": train_m, "validation": val_m, "is_best": is_best}
                history.append(entry)

                if is_best:
                    best_loss  = val_m["loss"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience   = 0
                    print(f"  ★  New best val loss: {best_loss:.4f}")
                else:
                    patience += 1
                    print(f"  ·  No improvement ({patience}/{args.early_stopping_patience})")

                torch.save(
                    {"model": model.state_dict(), "mock": True, "epoch": epoch},
                    run_dir / f"epoch_{epoch}.pth"
                )
                if patience >= args.early_stopping_patience:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

            model.load_state_dict(best_state)
            candidate_path = run_dir / "best_candidate.pth"
            torch.save({"model": model.state_dict(), "mock": True, "rlhp_run_id": run_dir.name}, candidate_path)
            candidate    = _mock_epoch(model, val_samples or train_samples, args.tau)
            frozen_hashes = {}

        # ── REAL branch ─────────────────────────────────────────────────────
        else:
            if not args.checkpoint or not args.config:
                raise RLHPError("Real training requires --checkpoint and --config.")
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RLHPError("CUDA requested but no GPU found. Use --mock for CPU testing.")

            # ── Provenance check ──────────────────────────────────────────
            expected_prov = next(
                (s.get("provenance", {}) for s in samples if s.get("provenance")), {}
            )
            actual_prov = {
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "config_sha256":     sha256_file(args.config),
                "label_map_sha256":  sha256_file(args.labels),
            }
            mismatched = [
                k for k in REQUIRED_PROVENANCE
                if expected_prov.get(k) and expected_prov[k] != actual_prov[k]
            ]
            if mismatched:
                raise RLHPError("Provenance mismatch: " + ", ".join(mismatched))

            # ── Load model ───────────────────────────────────────────────
            print(f"\n[2/4] Loading REACT++ model …")
            engine = RealPreferenceModel(
                Path(args.checkpoint),
                Path(args.config),
                Path(args.yolo_checkpoint) if args.yolo_checkpoint else None,
                Path(args.labels),
                args.device,
            )
            trainable_names = [n for n, _ in engine.trainable]
            print(f"     ✓ Trainable params: {trainable_names}")

            train_groups = _group_by_image(samples, "train")
            val_groups   = _group_by_image(samples, "validation")
            retention_weight = float(getattr(args, "retention_weight", 0.0))

            # ── BƯỚC 3: Baseline pass (capture teacher) ──────────────────
            print(f"\n[3/4] Baseline inference (capture teacher distributions) …")
            frozen_before = engine.frozen_hashes()
            teacher_cache: dict[str, torch.Tensor] = {}

            with torch.no_grad():
                print(f"  Train baseline:")
                baseline_train = _real_epoch_with_pbar(
                    engine, train_groups, args.tau,
                    desc="  Baseline-train",
                    teacher_cache=None,
                    capture_teacher=teacher_cache,
                )
                print(f"  Val baseline:")
                baseline = _real_epoch_with_pbar(
                    engine, val_groups or train_groups, args.tau,
                    desc="  Baseline-val  ",
                    capture_teacher=teacher_cache,
                )
            _print_metrics("baseline-train", baseline_train)
            _print_metrics("baseline-val  ", baseline)

            # ── Match rate check ─────────────────────────────────────────
            total_m  = baseline_train["total"] + baseline["total"]
            matched_m = baseline_train["matched"] + baseline["matched"]
            real_match_rate = matched_m / max(total_m, 1)
            report["model_pair_matched"]      = matched_m
            report["model_pair_total"]        = total_m
            report["model_pair_matching_rate"] = real_match_rate
            atomic_json(run_dir / "audit_report.json", report)
            print(f"\n     Model-backed pair matching rate: {real_match_rate:.1%}")
            if real_match_rate < args.min_match_rate:
                raise RLHPError(
                    f"Pair matching rate {real_match_rate:.1%} < {args.min_match_rate:.1%}"
                )

            # ── Optimizer ────────────────────────────────────────────────
            optimizer = torch.optim.AdamW(
                [p for _, p in engine.trainable],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

            # ── BƯỚC 4: Training loop ────────────────────────────────────
            print(f"\n[4/4] Fine-tuning (tau={args.tau}, lr={args.learning_rate}) …")
            best_loss, best_state, patience = math.inf, None, 0

            for epoch in range(1, args.epochs + 1):
                _print_epoch_header(epoch, args.epochs, "Real GPU")

                epoch_train_groups = list(train_groups)
                random.Random(args.seed + epoch).shuffle(epoch_train_groups)

                train_m = _real_epoch_with_pbar(
                    engine, epoch_train_groups, args.tau,
                    optimizer=optimizer,
                    accumulation=args.gradient_accumulation,
                    clip_norm=args.clip_norm,
                    teacher_cache=teacher_cache,
                    retention_weight=retention_weight,
                    desc=f"  Epoch {epoch:02d} Train",
                )
                with torch.no_grad():
                    val_m = _real_epoch_with_pbar(
                        engine, val_groups or train_groups, args.tau,
                        teacher_cache=teacher_cache,
                        retention_weight=retention_weight,
                        desc=f"  Epoch {epoch:02d} Val  ",
                    )

                _print_metrics("↳ train", train_m)
                _print_metrics("↳ val  ", val_m)

                is_best = val_m["loss"] < best_loss
                history.append({
                    "epoch":      epoch,
                    "train":      train_m,
                    "validation": val_m,
                    "is_best":    is_best,
                })

                if is_best:
                    best_loss  = val_m["loss"]
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in engine.model.state_dict().items()
                    }
                    patience = 0
                    print(f"  ★  New best val loss: {best_loss:.4f}  → saving epoch_{epoch}.pth")
                    torch.save(
                        {"model": engine.model.state_dict(), "rlhp_run_id": run_dir.name, "epoch": epoch},
                        run_dir / f"epoch_{epoch}.pth"
                    )
                else:
                    patience += 1
                    print(f"  ·  No improvement ({patience}/{args.early_stopping_patience})")
                    if args.save_every_epoch:
                        torch.save(
                            {"model": engine.model.state_dict(), "rlhp_run_id": run_dir.name, "epoch": epoch},
                            run_dir / f"epoch_{epoch}.pth"
                        )

                if patience >= args.early_stopping_patience:
                    print(f"\n  Early stopping triggered at epoch {epoch}.")
                    break

            # ── Save best candidate ───────────────────────────────────────
            engine.model.load_state_dict(best_state)
            with torch.no_grad():
                candidate = _real_epoch_with_pbar(
                    engine, val_groups or train_groups, args.tau,
                    desc="  Candidate-val ",
                )

            frozen_after    = engine.frozen_hashes()
            changed_frozen  = [k for k in frozen_before if frozen_before[k] != frozen_after.get(k)]
            if changed_frozen:
                raise RLHPError(f"Frozen params changed: {changed_frozen[:5]}")
            frozen_hashes = frozen_after

            original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(original, dict) or "model" not in original:
                original = {"model": original}
            original.pop("optimizer", None)
            original.pop("scheduler", None)
            original["model"] = {k: v.detach().cpu() for k, v in engine.model.state_dict().items()}
            original["rlhp"]  = {
                "run_id":    run_dir.name,
                "created_at": utc_timestamp(),
                "trainable":  [n for n, _ in engine.trainable],
                "tau":        args.tau,
                "lr":         args.learning_rate,
            }
            candidate_path = run_dir / "best_candidate.pth"
            torch.save(original, candidate_path)

        # ── Lưu kết quả cuối ────────────────────────────────────────────────
        metrics = {
            "baseline_train":      baseline_train,
            "baseline_validation": baseline,
            "candidate_validation": candidate,
            "epochs": history,
            "tau":    args.tau,
            "learning_rate": args.learning_rate,
        }
        atomic_json(run_dir / "training_metrics.json", metrics)
        atomic_json(run_dir / "frozen_parameter_hashes.json", frozen_hashes)
        run.update({
            "status":               "trained",
            "completed_at":         utc_timestamp(),
            "candidate_checkpoint": str(candidate_path.resolve()),
        })
        atomic_json(run_dir / "run.json", run)

        # ── Summary table ────────────────────────────────────────────────
        _print_summary_table(history)
        print(f"\n  Candidate checkpoint: {candidate_path}")
        _print_metrics("  Final candidate val", candidate)
        print(f"\n{'═'*72}")
        print(f"  Run saved to: {run_dir}")
        print(f"{'═'*72}\n")

        return run_dir

    except Exception as exc:
        run.update({"status": "failed", "failed_at": utc_timestamp(), "error": str(exc)})
        atomic_json(run_dir / "run.json", run)
        raise


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Sessions / labels
    p.add_argument("--sessions", nargs="+", required=True,
                   help="session.json files, directories, hoặc file CSV export")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                   help="psg_labels.json (mặc định: scene_graph_tool/psg_labels.json)")
    p.add_argument("--psg-test", type=Path, default=DEFAULT_PSG_TEST,
                   help="PSG test split để kiểm tra leakage")
    p.add_argument("--image-root", type=Path,
                   help="Thư mục chứa ảnh PSG train/val (nếu ảnh không nằm trong session)")
    p.add_argument("--allow-legacy-provenance", action="store_true",
                   help="Bỏ qua kiểm tra SHA256 provenance (session cũ)")

    # Audit
    p.add_argument("--iou-threshold", type=float, default=0.8)
    p.add_argument("--min-match-rate", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--checkpoint", type=Path,
                   help="Đường dẫn baseline checkpoint .pth")
    p.add_argument("--config", type=Path,
                   help="Đường dẫn config.yml của model")
    p.add_argument("--yolo-checkpoint", type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--mock", action="store_true",
                   help="Dùng MockPreferenceModel (không cần GPU, test pipeline nhanh)")

    # Run metadata
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--run-id", help="Tên run tuỳ chọn; mặc định: timestamp")

    # ── Hyperparameters (default: tau=1, lr=1e-4 theo yêu cầu) ──────────
    p.add_argument("--tau", type=float, default=1.0,
                   help="Temperature scaling cho preference loss (default: 1.0)")
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="Learning rate AdamW (default: 1e-4)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument("--retention-weight", type=float, default=0.0,
                   help="Trọng số KL-retention loss (0 = tắt)")
    p.add_argument("--save-every-epoch", action="store_true",
                   help="Lưu epoch_N.pth sau mỗi epoch (cần cho evaluate_rlhp_epochs.py)")

    return p


def main():
    args = build_parser().parse_args()
    try:
        run_dir = train_with_progress(args)
        print(f"\nDone! Tiếp theo chạy evaluate:")
        print(f"  python scene_graph_tool/rlhp.py evaluate \\")
        print(f"      --run-dir {run_dir} \\")
        print(f"      --config  {args.config or 'checkpoint/config.yml'} \\")
        print(f"      --device  {args.device}")
        return 0
    except RLHPError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
