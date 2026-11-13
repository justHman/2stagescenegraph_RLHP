"""Live SGG inference + annotation demo server.

Serves the demo UI statically and runs the REAL REACT++ ONNX models
(baseline + RLHP-aligned) on real PSG images with onnxruntime.

Endpoints:
  POST /api/predict            — baseline vs RLHP side-by-side scene graphs
  POST /api/annotate/candidates — uncertainty-based pair selection (real model)
  POST /api/annotate/save      — append annotation to CSV (same spirit as the
                                 toolkit's scene_graph_preferences3.csv)
  GET  /api/annotations        — saved annotation stats

Run:
    D:/Code/shared_venvs/dl/Scripts/python.exe live_demo/server.py
Then open http://127.0.0.1:8081
"""
import base64
import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

HERE = Path(__file__).parent
DEMO_DIR = HERE.parent / "demo"
CKPT = HERE / "checkpoint"
MODELS = {
    "baseline": CKPT / "react_pp_yolo12m.onnx",
    "rlhp": CKPT / "react_pp_yolo12m_rlhp.onnx",
}
IMAGES_DIR = HERE / "images"
ANNOTATIONS_CSV = HERE / "annotations.csv"


class SGG_ONNX:
    """Same preprocessing/postprocessing as demo/standalone_onnx_demo.py."""

    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        obj_list = json.loads(meta["obj_classes"])
        rel_list = json.loads(meta["rel_classes"])
        self.obj_classes = {i: v for i, v in enumerate(obj_list, start=1)}
        self.rel_classes = {i: v for i, v in enumerate(rel_list)}

    def preprocess(self, image, size=640):
        """Letterbox + BGR→RGB + CHW + Normalize, same as standalone_onnx_demo."""
        h, w = image.shape[:2]
        r = min(size / h, size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (size - nh) // 2
        bottom = size - nh - top
        left = (size - nw) // 2
        right = size - nw - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = padded[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return img[None, ...], r, (left, top)

    def run(self, frame):
        """Return (boxes_raw, rels_raw, probs, pmap, infer_ms).

        probs: (num_pairs, 57) predicate distribution per raw pair row
        pmap:  (num_pairs, 2)  (subject_box_idx, object_box_idx) per pair row
        """
        img_input, ratio, (pad_x, pad_y) = self.preprocess(frame)
        t0 = time.perf_counter()
        out_names = [o.name for o in self.session.get_outputs()]
        outs = self.session.run(None, {self.input_name: img_input})
        infer_ms = (time.perf_counter() - t0) * 1000
        d = dict(zip(out_names, outs))
        boxes_raw = d["boxes"].copy()
        rels_raw = d["rels"].copy()
        probs = d["rel_probs"]
        pmap = d["pair_map"]
        boxes_raw[:, [0, 2]] = (boxes_raw[:, [0, 2]] - pad_x) / ratio
        boxes_raw[:, [1, 3]] = (boxes_raw[:, [1, 3]] - pad_y) / ratio
        return boxes_raw, rels_raw, probs, pmap, infer_ms

    def to_json(self, boxes_raw, rels_raw, rel_conf=0.05, box_conf=0.4):
        keep_rels = rels_raw[rels_raw[:, 3] >= rel_conf]
        rel_box_idx = set()
        if len(keep_rels):
            rel_box_idx = set(keep_rels[:, 0].astype(int)) | set(keep_rels[:, 1].astype(int))
        keep_box_mask = boxes_raw[:, 5] >= box_conf
        for idx in rel_box_idx:
            keep_box_mask[idx] = True
        keep = np.where(keep_box_mask)[0]
        old2new = {old: new for new, old in enumerate(keep)}
        boxes = []
        for old in keep:
            x1, y1, x2, y2, cls_id, score = boxes_raw[old]
            boxes.append({
                "index": int(old2new[old]),
                "box": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                "label": self.obj_classes.get(int(cls_id), "?"), "score": round(float(score), 3),
            })
        rels = []
        for s, o, rel_id, trip, rel_sc in keep_rels:
            s, o = int(s), int(o)
            if s in old2new and o in old2new:
                rels.append({
                    "subject": old2new[s], "object": old2new[o],
                    "predicate": self.rel_classes.get(int(rel_id), "?"),
                    "triplet_score": round(float(trip), 3), "rel_score": round(float(rel_sc), 3),
                })
        return boxes, rels

    def select_candidates(self, boxes_raw, probs, pmap, top_k=40, top_m=6):
        """Uncertainty selection — same spirit as the toolkit:
        shortlist Top-K pairs by p1, then Top-M by smallest relative margin."""
        n_boxes = len(boxes_raw)
        pair2row = {}
        for r in range(len(pmap)):
            a, b = int(pmap[r, 0]), int(pmap[r, 1])
            if a != b and a < n_boxes and b < n_boxes:
                pair2row[(a, b)] = r
        fg = probs[:, 1:]  # foreground predicate probs (56 classes)
        cands = []
        for (a, b), row in pair2row.items():
            p = fg[row]
            order = np.argsort(p)[::-1]
            p1i, p2i = order[0], order[1]
            p1, p2 = float(p[p1i]), float(p[p2i])
            if p1 <= 0:
                continue
            margin = (p1 - p2) / (p1 + p2)
            cands.append({
                "subject": a, "object": b, "row": row,
                "p1_id": int(p1i) + 1, "p2_id": int(p2i) + 1,
                "p1": p1, "p2": p2, "margin": margin,
                "uncertainty": 1.0 - margin,
            })
        if not cands:
            return []
        # 1) Top-K shortlist by confidence p1
        cands.sort(key=lambda c: -c["p1"])
        shortlist = cands[:top_k]
        # 2) Top-M by smallest relative margin (highest uncertainty)
        shortlist.sort(key=lambda c: c["margin"])
        return shortlist[:top_m]


SESSIONS = {}


def get_model(which):
    if which not in SESSIONS:
        SESSIONS[which] = SGG_ONNX(MODELS[which])
    return SESSIONS[which]


def _color(i):
    hue = int((i * 47) % 180)
    hsv = np.array([[[hue, 210, 235]]], np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def make_overlay(image, boxes, rels, accent=(60, 160, 255)):
    img = image.copy()
    centers = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        color = _color(i)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{b['label']} {b['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        yt = y1 - th - 5 if y1 - th - 5 > 0 else y1
        cv2.rectangle(img, (x1, yt - th - 4), (x1 + tw + 4, yt + th + 4), color, -1)
        cv2.putText(img, label, (x1 + 2, yt + th), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
    for r in rels:
        p1 = centers[r["subject"]]
        p2 = centers[r["object"]]
        cv2.arrowedLine(img, p1, p2, accent, 2, tipLength=0.08)
        mid = (int(p1[0] * 0.6 + p2[0] * 0.4), int(p1[1] * 0.6 + p2[1] * 0.4))
        lab = f"{r['predicate']} {r['triplet_score']:.2f}"
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (mid[0] - 2, mid[1] - th - 3), (mid[0] + tw + 3, mid[1] + 3), (20, 20, 20), -1)
        cv2.putText(img, lab, (mid[0], mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def make_annotation_overlay(image, all_boxes, cand):
    """Faint boxes for everything, bold highlight for the active pair."""
    img = image.copy()
    dim = img.copy()
    for b in all_boxes:
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        cv2.rectangle(dim, (x1, y1), (x2, y2), (90, 90, 90), 1)
    img = cv2.addWeighted(dim, 0.55, img, 0.45, 0)
    centers = []
    for i, b in enumerate(all_boxes):
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
    for idx, color, tag in [(cand["subject"], (255, 170, 60), "SUBJ"),
                            (cand["object"], (60, 180, 90), "OBJ")]:
        b = all_boxes[idx]
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        yt = y1 - th - 6 if y1 - th - 6 > 0 else y1
        cv2.rectangle(img, (x1, yt - th - 5), (x1 + tw + 6, yt + th + 5), color, -1)
        cv2.putText(img, tag, (x1 + 3, yt + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
    p1 = centers[cand["subject"]]
    p2 = centers[cand["object"]]
    cv2.arrowedLine(img, p1, p2, (255, 90, 40), 3, tipLength=0.09)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


from fastapi import FastAPI, UploadFile, File, Form, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.responses import JSONResponse, FileResponse  # noqa: E402

app = FastAPI(title="RLHP Live SGG Demo")


@app.on_event("startup")
def _load():
    for k in MODELS:
        get_model(k)


@app.get("/api/samples")
def samples():
    files = sorted(p.name for p in IMAGES_DIR.glob("*.jpg")) if IMAGES_DIR.exists() else []
    return {"samples": files, "provider": "CPUExecutionProvider"}


def _load_image(file, image_name):
    if file is not None and file.filename:
        data = file.file.read()
        arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return image, file.filename
    if image_name:
        p = IMAGES_DIR / Path(image_name).name
        if not p.exists():
            raise HTTPException(404, "image not found")
        return cv2.imread(str(p)), p.name
    raise HTTPException(400, "no image provided")


@app.post("/api/predict")
async def predict(
    file: UploadFile | None = File(default=None),
    image_name: str | None = Form(default=None),
    rel_conf: float = Form(default=0.05),
    box_conf: float = Form(default=0.4),
    compare: bool = Form(default=False),
):
    image, name = _load_image(file, image_name)
    if image is None:
        raise HTTPException(400, "cannot decode image")
    out = {"image_name": name, "width": image.shape[1], "height": image.shape[0]}
    which = ["baseline", "rlhp"] if compare else ["baseline"]
    for w in which:
        m = get_model(w)
        boxes_raw, rels_raw, probs, pmap, infer_ms = m.run(image)
        boxes, rels = m.to_json(boxes_raw, rels_raw, rel_conf, box_conf)
        out[w] = {
            "infer_ms": round(infer_ms, 1),
            "boxes": boxes, "relations": rels,
            "overlay": "data:image/jpeg;base64," + base64.b64encode(
                make_overlay(image, boxes, rels)).decode(),
        }
    if compare:
        b_set = {(r["subject"], r["object"], r["predicate"]): r["triplet_score"]
                 for r in out["baseline"]["relations"]}
        r_set = {(r["subject"], r["object"], r["predicate"]): r["triplet_score"]
                 for r in out["rlhp"]["relations"]}
        out["diff"] = {
            "common": len(set(b_set) & set(r_set)),
            "baseline_only": len(set(b_set) - set(r_set)),
            "rlhp_only": len(set(r_set) - set(b_set)),
        }
    return JSONResponse(out)


@app.post("/api/annotate/candidates")
async def annotate_candidates(
    file: UploadFile | None = File(default=None),
    image_name: str | None = Form(default=None),
    model: str = Form(default="baseline"),
    top_k: int = Form(default=40),
    top_m: int = Form(default=6),
):
    m = get_model(model if model in MODELS else "baseline")
    image, name = _load_image(file, image_name)
    if image is None:
        raise HTTPException(400, "cannot decode image")
    boxes_raw, rels_raw, probs, pmap, infer_ms = m.run(image)
    cands = m.select_candidates(boxes_raw, probs, pmap, top_k=top_k, top_m=top_m)
    # boxes json in raw space (all detections)
    all_boxes = []
    for i, (x1, y1, x2, y2, cls_id, score) in enumerate(boxes_raw):
        all_boxes.append({
            "index": i,
            "box": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
            "label": m.obj_classes.get(int(cls_id), "?"), "score": round(float(score), 3),
        })
    out_cands = []
    for c in cands:
        sb, ob = all_boxes[c["subject"]], all_boxes[c["object"]]
        out_cands.append({
            "subject": {"label": sb["label"], "score": sb["score"], "box": sb["box"]},
            "object": {"label": ob["label"], "score": ob["score"], "box": ob["box"]},
            "cand_a": {"id": c["p1_id"], "name": m.rel_classes.get(c["p1_id"], "?"), "p": round(100 * c["p1"], 2)},
            "cand_b": {"id": c["p2_id"], "name": m.rel_classes.get(c["p2_id"], "?"), "p": round(100 * c["p2"], 2)},
            "uncertainty": round(c["uncertainty"], 3),
            "relative_margin": round(c["margin"], 3),
            "overlay": "data:image/jpeg;base64," + base64.b64encode(
                make_annotation_overlay(image, all_boxes, c)).decode(),
        })
    return JSONResponse({
        "image_name": name, "model": model if model in MODELS else "baseline",
        "width": image.shape[1], "height": image.shape[0],
        "infer_ms": round(infer_ms, 1),
        "n_boxes": len(all_boxes),
        "candidates": out_cands,
    })


CSV_FIELDS = [
    "schema_version", "session_id", "answer_created_at", "image_name", "model",
    "subject_label", "subject_score", "subject_box",
    "object_label", "object_score", "object_box",
    "predicate_1_id", "predicate_1_name", "predicate_1_probability",
    "predicate_2_id", "predicate_2_name", "predicate_2_probability",
    "display_order", "choice", "uncertainty_score", "relative_margin",
]


@app.post("/api/annotate/save")
async def annotate_save(payload: dict):
    row = {k: payload.get(k) for k in CSV_FIELDS}
    if not row.get("schema_version"):
        row["schema_version"] = 1
    if not row.get("session_id"):
        row["session_id"] = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    row["answer_created_at"] = datetime.now(timezone.utc).isoformat()
    exists = ANNOTATIONS_CSV.exists()
    with open(ANNOTATIONS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    n = sum(1 for _ in open(ANNOTATIONS_CSV, encoding="utf-8")) - 1
    return {"saved": True, "total": max(n, 0)}


@app.get("/api/annotations")
def annotations():
    if not ANNOTATIONS_CSV.exists():
        return {"total": 0, "rows": []}
    with open(ANNOTATIONS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    choices = {}
    for r in rows:
        choices[r.get("choice", "?")] = choices.get(r.get("choice", "?"), 0) + 1
    return {"total": len(rows), "choices": choices, "rows": rows[-50:]}


app.mount("/", StaticFiles(directory=str(DEMO_DIR), html=True), name="demo")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="warning")
