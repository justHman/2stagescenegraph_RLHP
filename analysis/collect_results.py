"""Aggregate RLHP evaluation results from downloaded runs into a single comparison JSON.

Reads every run directory under scene_graph_tool/rlhp_runs/extracted/, pulls
baseline vs candidate metrics (mR/R/F1 @20/50/100 + bbox mAP) and per-predicate
recall lists, and writes rlhp_results.json + rlhp_results.csv for the demo UI.
"""
import json
import csv
from pathlib import Path

REPO = Path(r"D:\Studying\University\Ki_8\IMP\2stagescenegraph_RLHP\scene_graph_tool")
RUNS = REPO / "rlhp_runs" / "extracted"
OUT_DIR = Path(r"D:\Studying\University\Ki_8\IMP\rlhp_analysis")
KS = ("20", "50", "100")

# Display metadata for each run (paper Table 2 & 3 labels)
RUN_LABELS = {
    "pure_pref_run1": ("Ablation 1 — Pure Preference", "preference3_pure_pref_setting_tau01"),
    "preference3_pure_pref_setting_tau01": ("Ablation 1 — Pure Preference (final)", None),
    "preference3_proto_anchor_only_tau01": ("Ablation 2 — Preference + Proto Anchor", None),
    "preference3_retention_only_tau01": ("Ablation 3 — Preference + KL Retention", None),
    "preference3_tuned_anchor_ret20_tau01": ("Full RLHP (Tuned) — Main Model", None),
}


def load_summary(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    mode = d.get("sgg_mode", "sgdet")
    return {
        "mode": mode,
        "bbox_mAP": d.get(f"{mode}_bbox_mAP"),
        "mR": {k: d.get(f"{mode}_mean_recall", {}).get(k) for k in KS},
        "R": {k: _mean(d.get(f"{mode}_recall", {}).get(k)) for k in KS},
        "F1": {k: d.get(f"{mode}_f1_score", {}).get(k) for k in KS},
        "per_predicate_recall": d.get(f"{mode}_mean_recall_list"),
        "mean_forward_ms": d.get("eval_timing", {}).get("median_forward_ms"),
    }


def _mean(v):
    if isinstance(v, list) and v:
        return sum(v) / len(v)
    return None


def main():
    results = {"runs": [], "dataset_audit": {}, "selection_comparison": []}

    # Preference dataset audit stats
    for audit_path in sorted(RUNS.glob("*/audit_report.json")):
        a = json.loads(audit_path.read_text(encoding="utf-8"))
        if a.get("status") == "passed":
            results["dataset_audit"] = {
                "raw_feedback": a.get("raw_feedback"),
                "aggregated_samples": a.get("aggregated_samples"),
                "unique_images": a.get("unique_images"),
                "train_samples": a.get("train_samples"),
                "validation_samples": a.get("validation_samples"),
                "audit_only_samples": a.get("audit_only_samples"),
                "pair_matching_rate": a.get("pair_matching_rate"),
                "test_leakage": len(a.get("test_leakage", [])),
                "iou_threshold": a.get("iou_threshold"),
            }
            break

    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        label, _ = RUN_LABELS.get(run_dir.name, (run_dir.name, None))
        # Prefer per-run evaluation/{baseline,candidate}/results/eval_summary.json
        base = load_summary(run_dir / "evaluation" / "baseline" / "results" / "eval_summary.json")
        cand = load_summary(run_dir / "evaluation" / "candidate" / "results" / "eval_summary.json")
        # Fallback: comparison_modes layout (pure_pref_run1)
        if base is None:
            base = load_summary(run_dir / "comparison_modes" / "evaluation" / "baseline" / "sgdet" / "results" / "eval_summary.json")
            cand = load_summary(run_dir / "comparison_modes" / "evaluation" / "candidate" / "sgdet" / "results" / "eval_summary.json")
        if base is None and cand is None:
            continue
        # comparison_metrics.json may hold the canonical deltas
        cm_path = run_dir / "comparison_metrics.json"
        comparison = json.loads(cm_path.read_text(encoding="utf-8")) if cm_path.exists() else None
        # training info
        run_json = run_dir / "run.json"
        run_info = json.loads(run_json.read_text(encoding="utf-8")) if run_json.exists() else {}

        entry = {
            "run_dir": run_dir.name,
            "label": label,
            "run_info": run_info if isinstance(run_info, dict) else {},
            "baseline": base,
            "candidate": cand,
            "comparison_metrics": comparison,
        }
        results["runs"].append(entry)

    # Uncertainty vs top-prob selection benchmark (Table 1)
    sel_csv = REPO / "relation_selection_comparison.csv"
    if sel_csv.exists():
        with open(sel_csv, encoding="utf-8") as f:
            results["selection_comparison"] = list(csv.DictReader(f))

    out_json = OUT_DIR / "rlhp_results.json"
    out_json.write_text(json.dumps(results, indent=1), encoding="utf-8")

    # Flat CSV for quick viewing
    with open(OUT_DIR / "rlhp_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "variant", "bbox_mAP"] + [f"{m}@{k}" for m in ("mR", "R", "F1") for k in KS])
        for r in results["runs"]:
            for variant in ("baseline", "candidate"):
                v = r[variant]
                if not v:
                    continue
                row = [r["run_dir"], variant, f"{v['bbox_mAP']:.4f}" if v["bbox_mAP"] else ""]
                row += [f"{v[m][k]:.4f}" if v[m][k] is not None else "" for m in ("mR", "R", "F1") for k in KS]
                w.writerow(row)

    print(f"Wrote {out_json}")
    for r in results["runs"]:
        print("\n==", r["run_dir"], f"[{r['label']}]")
        for variant in ("baseline", "candidate"):
            v = r[variant]
            if not v:
                print(f"  {variant}: missing")
                continue
            fmt = lambda d: " ".join(f"@{k}={100*d[k]:.2f}" for k in KS if d[k] is not None)
            print(f"  {variant:9s} mR {fmt(v['mR'])} | F1 {fmt(v['F1'])} | mAP {100*v['bbox_mAP']:.2f}")


if __name__ == "__main__":
    main()
