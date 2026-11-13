"""Smoke test: run baseline & RLHP ONNX on a real PSG test image, compare."""
import sys
import time
import numpy as np
import cv2
import onnxruntime as ort

sys.path.insert(0, r"D:\Studying\University\Ki_8\IMP\rlhp_analysis\live_demo")
from server import SGG_ONNX  # noqa: E402

IMG = r"D:\Studying\University\Ki_8\IMP\rlhp_analysis\live_demo\images\107910.jpg"
CKPT = r"D:\Studying\University\Ki_8\IMP\rlhp_analysis\live_demo\checkpoint"

frame = cv2.imread(IMG)
print("image:", frame.shape)

results = {}
for name, path in [("baseline", f"{CKPT}/react_pp_yolo12m.onnx"),
                   ("rlhp", f"{CKPT}/react_pp_yolo12m_rlhp.onnx")]:
    m = SGG_ONNX(path, rel_conf=0.03, box_conf=0.35)
    boxes_raw, rels_raw, ms = m.predict_raw(frame)
    boxes, rels = m.to_json(boxes_raw, rels_raw, rel_conf=0.03, box_conf=0.35)
    results[name] = {(r["subject"], r["object"], r["predicate"]): r["triplet_score"] for r in rels}
    print(f"\n[{name}] provider={m.provider} forward={ms:.0f}ms boxes={len(boxes)} rels={len(rels)}")
    for r in sorted(rels, key=lambda x: -x["triplet_score"])[:8]:
        s = boxes[r["subject"]]["label"]; o = boxes[r["object"]]["label"]
        print(f"   {s:12s} -> {r['predicate']:14s} -> {o:12s} {r['triplet_score']:.3f}")

# compare
common = set(results["baseline"]) & set(results["rlhp"])
only_b = set(results["baseline"]) - common
only_r = set(results["rlhp"]) - common
print(f"\ncommon triplets: {len(common)}, baseline-only: {len(only_b)}, rlhp-only: {len(only_r)}")
for t in sorted(only_r)[:6]:
    print("  rlhp-only:", t, round(results["rlhp"][t], 3))
for t in sorted(only_b)[:6]:
    print("  baseline-only:", t, round(results["baseline"][t], 3))
score_diffs = [(t, results["baseline"][t], results["rlhp"][t]) for t in common
               if abs(results["baseline"][t] - results["rlhp"][t]) > 0.01]
print(f"score-changed common triplets: {len(score_diffs)}")
for t, b, r in score_diffs[:6]:
    print(f"   {t}: {b:.3f} -> {r:.3f}")
