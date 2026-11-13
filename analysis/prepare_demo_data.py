"""Prepare demo_data.json + assets for the RLHP demo UI.

Bundles: benchmark comparison, epoch trajectories, per-predicate recall deltas,
preference dataset stats, uncertainty-selection examples, and copies the
qualitative comparison images + report figures into the demo folder.
"""
import json
import csv
import shutil
import math
from pathlib import Path

REPO_TOOL = Path(r"D:\Studying\University\Ki_8\IMP\2stagescenegraph_RLHP\scene_graph_tool")
RUNS = REPO_TOOL / "rlhp_runs" / "extracted"
TUNED = RUNS / "preference3_tuned_anchor_ret20_tau01"
OUT = Path(r"D:\Studying\University\Ki_8\IMP\rlhp_analysis\demo")
ASSETS = OUT / "assets"
KS = ("20", "50", "100")

ORDER = [
    "preference3_pure_pref_setting_tau01",
    "preference3_proto_anchor_only_tau01",
    "preference3_retention_only_tau01",
    "preference3_tuned_anchor_ret20_tau01",
]
LABELS = {
    "preference3_pure_pref_setting_tau01": "Pure Preference",
    "preference3_proto_anchor_only_tau01": "+ Proto Anchor",
    "preference3_retention_only_tau01": "+ KL Retention",
    "preference3_tuned_anchor_ret20_tau01": "Full RLHP (Tuned)",
    "preference3_pure_pref_setting_tau01_short": "Pure Preference",
}


def pct(x):
    return None if x is None else round(100 * x, 2)


def load_summary(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    mode = d.get("sgg_mode", "sgdet")
    recall = d.get(f"{mode}_recall", {})
    return {
        "bbox_mAP": pct(d.get(f"{mode}_bbox_mAP")),
        "mR": {k: pct(d.get(f"{mode}_mean_recall", {}).get(k)) for k in KS},
        "R": {k: pct(_mean(recall.get(k))) for k in KS},
        "F1": {k: pct(d.get(f"{mode}_f1_score", {}).get(k)) for k in KS},
        "per_predicate_recall": d.get(f"{mode}_mean_recall_list"),
    }


def _mean(v):
    return sum(v) / len(v) if isinstance(v, list) and v else None


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)

    # ---- 1. Benchmark comparison (each ablation run has baseline+candidate) ----
    models = []
    for name in ORDER:
        base = load_summary(RUNS / name / "evaluation" / "baseline" / "results" / "eval_summary.json")
        cand = load_summary(RUNS / name / "evaluation" / "candidate" / "results" / "eval_summary.json")
        models.append({"name": LABELS[name], "run": name, "baseline": base, "rlhp": cand})

    benchmark = []
    for m in models:
        b, c = m["baseline"], m["rlhp"]
        benchmark.append({
            "name": m["name"],
            "mR": {"baseline": b["mR"], "rlhp": c["mR"]},
            "R": {"baseline": b["R"], "rlhp": c["R"]},
            "F1": {"baseline": b["F1"], "rlhp": c["F1"]},
            "bbox_mAP": b["bbox_mAP"],
            "delta_mR50": round(c["mR"]["50"] - b["mR"]["50"], 2),
            "delta_mR100": round(c["mR"]["100"] - b["mR"]["100"], 2),
            "delta_F150": round(c["F1"]["50"] - b["F1"]["50"], 2),
        })

    full = benchmark[-1]

    # ---- 2. Epoch trajectories (Full RLHP) ----
    ep = json.loads((TUNED / "sgg_epoch_metrics.json").read_text(encoding="utf-8"))
    trajectories = {"epochs": [], "sgdet": [], "predcls": []}
    for i, e in enumerate(ep["epochs"]):
        trajectories["epochs"].append(i)
        for mode in ("SGDet", "PredCLS"):
            d = e[mode]
            row = {
                "mR@20": pct(d["mR@20"]), "mR@50": pct(d["mR@50"]), "mR@100": pct(d["mR@100"]),
                "F1@50": pct(d["F1@50"]), "R@50": pct(d["R@50"]),
            }
            trajectories[mode.lower()].append(row)
    # epoch 0 of the tuned trajectory is the baseline (before fine-tuning)
    baseline_point = {
        "mR@20": full["mR"]["baseline"]["20"], "mR@50": full["mR"]["baseline"]["50"],
        "mR@100": full["mR"]["baseline"]["100"], "F1@50": full["F1"]["baseline"]["50"],
        "R@50": full["R"]["baseline"]["50"],
    }
    trajectories["sgdet"][0] = baseline_point

    # ---- 3. Per-predicate recall deltas (baseline -> Full RLHP, SGDet @50) ----
    labels_json = json.loads((REPO_TOOL / "psg_labels.json").read_text(encoding="utf-8"))
    preds = labels_json["predicates"]
    base_list = full.get("_ppr_base")  # placeholder, filled below
    b_sum = load_summary(RUNS / "preference3_tuned_anchor_ret20_tau01" / "evaluation" / "baseline" / "results" / "eval_summary.json")
    c_sum = load_summary(RUNS / "preference3_tuned_anchor_ret20_tau01" / "evaluation" / "candidate" / "results" / "eval_summary.json")
    per_pred = []
    for k in ("50", "100"):
        bl = b_sum["per_predicate_recall"][k]
        cl = c_sum["per_predicate_recall"][k]
        # list index i -> class id i+1 (skips __background__)
        deltas = []
        for i, (bv, cv) in enumerate(zip(bl, cl)):
            deltas.append({
                "id": i + 1,
                "name": preds[i + 1] if i + 1 < len(preds) else str(i + 1),
                "baseline": pct(bv), "rlhp": pct(cv),
                "delta": round(100 * (cv - bv), 2),
            })
        per_pred.append({"k": k, "classes": deltas})

    # ---- 4. Preference dataset stats ----
    audit = json.loads((TUNED / "audit_report.json").read_text(encoding="utf-8"))
    pref_rows = list(csv.DictReader(open(REPO_TOOL / "scene_graph_preferences3.csv", encoding="utf-8")))
    resp_counts = {}
    pred_freq = {}
    margins = []
    for r in pref_rows:
        resp = r["relation_id"]  # response text like "strong preference: playing"
        key = r["response"].split(":")[0].strip() if r["response"] else "?"
        resp_counts[key] = resp_counts.get(key, 0) + 1
        best = r["model_best_predicate"]
        if best:
            pred_freq[best] = pred_freq.get(best, 0) + 1
        try:
            margins.append(float(r["uncertainty_score"]))
        except ValueError:
            pass
    margins.sort()
    pref_stats = {
        "total_rows": len(pref_rows),
        "response_distribution": resp_counts,
        "top_predicates": sorted(pred_freq.items(), key=lambda x: -x[1])[:12],
        "uncertainty": {
            "mean": round(sum(margins) / len(margins), 3),
            "median": round(margins[len(margins) // 2], 3),
            "min": round(margins[0], 3),
            "max": round(margins[-1], 3),
        },
        "audit": {
            "raw_feedback": audit.get("raw_feedback"),
            "aggregated": audit.get("aggregated_samples"),
            "unique_images": audit.get("unique_images"),
            "train": audit.get("train_samples"),
            "validation": audit.get("validation_samples"),
            "audit_only_both_wrong": audit.get("audit_only_samples"),
            "pair_matching_rate": audit.get("pair_matching_rate"),
            "test_leakage": len(audit.get("test_leakage", [])),
            "iou_threshold": audit.get("iou_threshold"),
        },
    }

    # ---- 5. Uncertainty vs top-prob selection examples (Table 1) ----
    sel_rows = list(csv.DictReader(open(REPO_TOOL / "relation_selection_comparison.csv", encoding="utf-8")))
    by_img = {}
    for r in sel_rows:
        by_img.setdefault(r["image_id"], []).append(r)
    selection_examples = []
    for img, rows in list(by_img.items())[:4]:
        selection_examples.append({
            "image_id": img,
            "top_prob": [
                {"rank": r["rank"], "subject": r["subject"], "object": r["object"],
                 "pair": f"{r['predicate_1']} / {r['predicate_2']}",
                 "p1": round(100 * float(r["p1"]), 2), "p2": round(100 * float(r["p2"]), 2),
                 "uncertainty": round(100 * float(r["uncertainty_score"]), 1)}
                for r in rows if r["method"] == "highest_probability"
            ],
            "uncertainty": [
                {"rank": r["rank"], "subject": r["subject"], "object": r["object"],
                 "pair": f"{r['predicate_1']} / {r['predicate_2']}",
                 "p1": round(100 * float(r["p1"]), 2), "p2": round(100 * float(r["p2"]), 2),
                 "uncertainty": round(100 * float(r["uncertainty_score"]), 1)}
                for r in rows if r["method"] == "uncertainty_priority"
            ],
        })

    # ---- 6. Annotation demo samples (real rows from preference CSV) ----
    demo_samples = []
    for r in pref_rows:
        try:
            u = float(r["uncertainty_score"])
            p1 = float(r["predicate_1_probability"])
            p2 = float(r["predicate_2_probability"])
        except ValueError:
            continue
        if p1 <= 0:  # skip degenerate all-zero candidates (both-wrong audit rows)
            continue
        resp = r["response"]
        if "both wrong" in resp or "no relation" in resp:
            continue
        demo_samples.append({
            "image_id": r["image_id"],
            "subject": {"label": r["subject_label"], "score": round(100 * float(r["subject_score"]), 1),
                        "box": [round(float(r[f"subject_{c}"]), 1) for c in ("x1", "y1", "x2", "y2")]},
            "object": {"label": r["object_label"], "score": round(100 * float(r["object_score"]), 1),
                       "box": [round(float(r[f"object_{c}"]), 1) for c in ("x1", "y1", "x2", "y2")]},
            "candidates": [
                {"name": r["predicate_1_name"], "p": round(100 * p1, 2)},
                {"name": r["predicate_2_name"], "p": round(100 * p2, 2)},
            ],
            "human_choice": resp,
            "uncertainty": round(u, 3),
            "relative_margin": round(1 - u, 3),
        })
    # diverse picks: most uncertain, mid-band, most confident
    demo_samples.sort(key=lambda s: -s["uncertainty"])
    mid = [s for s in demo_samples if 0.3 <= s["uncertainty"] <= 0.85]
    picked = demo_samples[:8] + mid[:8] + demo_samples[-8:]

    # ---- 7. Copy assets ----
    copied = []
    for f in sorted(TUNED.glob("qualitative_*.png")) + [TUNED / "evaluation_comparison.png", TUNED / "training_curves.png", TUNED / "preference_class_distribution.png", TUNED / "predicate_improvements.png", TUNED / "sgg_epochs_mR_curve.png"]:
        if f.exists() and f.stat().st_size < 15 * 1024 * 1024:
            shutil.copy2(f, ASSETS / f.name)
            copied.append(f.name)
    figs = REPO_TOOL / "report" / "figures"
    for name in ("GUI_image.png", "retention_loss_explosion.png", "uncertainty_pairs_distribution.png"):
        if (figs / name).exists():
            shutil.copy2(figs / name, ASSETS / name)
            copied.append(name)

    data = {
        "meta": {"paper": "Human Preference Alignment for Two-Stage Scene Graph Generation (YOLO12m-REACT++ on PSG)",
                 "benchmark": "PSG test (2,177 images)", "protocol": "SGDet", "lambda_kl": 20, "lambda_proto": 20, "tau": 0.1},
        "benchmark": benchmark,
        "headline": {
            "baseline": {"mR": full["mR"]["baseline"], "F1": full["F1"]["baseline"], "R": full["R"]["baseline"], "bbox_mAP": full["bbox_mAP"]},
            "rlhp": {"mR": full["mR"]["rlhp"], "F1": full["F1"]["rlhp"], "R": full["R"]["rlhp"], "bbox_mAP": c_sum["bbox_mAP"]},
        },
        "trajectories": trajectories,
        "per_predicate": per_pred,
        "pref_stats": pref_stats,
        "selection_examples": selection_examples,
        "demo_samples": picked,
        "assets": copied,
    }
    (OUT / "demo_data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("Wrote", OUT / "demo_data.json")
    print("Assets:", len(copied), "files")


if __name__ == "__main__":
    main()
