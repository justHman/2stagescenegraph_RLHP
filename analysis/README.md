# RLHP Analysis & Demo

Kết quả phân tích + UI demo cho repo `2stagescenegraph_RLHP`
(Human Preference Alignment for Two-Stage SGG — YOLO12m-REACT++ trên PSG).

## Chạy UI demo

```bash
# UI tĩnh (benchmark/ablation/qualitative/annotation demo) — port 8080
cd demo
py -3.10 -m http.server 8080

# HOẶC full demo kèm live inference (khuyến nghị) — port 8081
D:/Code/shared_venvs/dl/Scripts/python.exe live_demo/server.py
# mở http://127.0.0.1:8081
```

Không cần cài thêm gì — dùng venv chia sẻ `D:\Code\shared_venvs\dl`
(torch 2.6.0+cu124, onnxruntime, cv2, fastapi, pyarrow).

## Tab 🔴 Live Model (inference thật)

Chạy model thật bằng onnxruntime trên ảnh thật:

- **2 model song song**: `baseline` = ONNX chính thức từ HF (`maelic/REACTPlusPlus_PSG`),
  `rlhp` = ONNX đã **patch 3 tensor projection head** từ
  `preference3_tuned_anchor_ret20_tau01/best_candidate.pth`
  (script: `make_rlhp_onnx.py` — verify baseline khớp 100% trước khi patch).
- **40 ảnh PSG test thật** extract từ `maelic/PSG-coco-format` parquet
  (`extract_images.py`) — bao gồm đúng các ảnh dùng trong figure qualitative của paper
  (107910, 107912, 1159300, ...).
- Upload ảnh bất kỳ để test model trên ảnh của bạn.
- CPU ~350ms/forward (onnxruntime trong `dl` là bản CPU-only; GPU RTX 3050 sẵn sàng
  nếu cài `onnxruntime-gpu`).

## Tab 🖊️ Live Annotation (active annotation thật)

Đúng pipeline active-learning của toolkit, chạy trên model + ảnh thật:

1. Model chạy inference → ONNX được **graph surgery** để xuất thêm
   `rel_probs` (distribution 57 predicate của MỌI candidate pair) + `pair_map`
   (`add_probs_output.py` — verify khớp 90/90 label với output gốc).
2. **Uncertainty selector** như paper: shortlist Top-K cặp theo p1, rồi chọn
   Top-M cặp có margin nhỏ nhất `Δ=(p1−p2)/(p1+p2)` → U=1−Δ cao nhất.
3. UI hiển thị ảnh với SUBJ/OBJ box highlight + 2 candidate predicate
   (xáo trộn thứ tự để tránh position bias) — chọn bằng click hoặc phím
   `1`/`2`/`0`(tie)/`X`(both wrong)/`S`(skip)/`Z`(undo).
4. Mỗi lựa chọn ghi vào `live_demo/annotations.csv` — cùng cấu trúc trường với
   `scene_graph_preferences3.csv` của toolkit, dùng được cho pipeline
   `train_rlhp_custom.py` (chỉ cần map tên cột).
5. Chọn được model: baseline HOẶC rlhp-aligned.

Script: `live_demo/server.py` (FastAPI, port 8081, mount luôn UI tĩnh).

## Các tab trong demo

| Tab | Nội dung |
|---|---|
| 📊 Benchmark | So sánh Baseline vs Full RLHP trên PSG test SGDet (2,177 ảnh) — bảng + chart mR/F1 @20/50/100 |
| 🧪 Ablation | 4 cấu hình loss (Pure Preference / +Proto / +KL / Full RLHP), Δ so với baseline riêng từng run |
| 🏷️ Theo Predicate | Recall từng class (56 predicates) baseline→RLHP, chọn K=50/100, top-N thay đổi lớn nhất |
| 📈 Epochs | Trajectory mR/F1 theo epoch 0→5 (SGDet + PredCls) |
| 🗂️ Dữ liệu Preference | Thống kê 510 feedback / 376 cặp train / 168 ảnh, phân bố lựa chọn, uncertainty |
| 🖼️ Qualitative | 15 ảnh so sánh Baseline (trên) vs RLHP (dưới) — click để phóng to |
| 🎮 Annotation Demo | Mô phỏng GUI thu thập preference với record thật từ CSV (phím 1/2/0/N/Z) |
| 🔴 Live Model | Inference thật: Baseline vs RLHP side-by-side trên ảnh PSG test / ảnh upload |

## Script

| File | Chức năng |
|---|---|
| `collect_results.py` | Gom `eval_summary.json` từ các run tải về → `rlhp_results.json` / `.csv` |
| `prepare_demo_data.py` | Tạo `demo/demo_data.json` + copy ảnh qualitative/figures vào `demo/assets/` |

## Nguồn dữ liệu

- Checkpoint & runs (~5.7 GB): Google Drive của repo → `2stagescenegraph_RLHP/scene_graph_tool/rlhp_runs/packaged_archives/` (đã giải nén vào `extracted/`)
- Preference CSV + taxonomy: sẵn trong repo (`scene_graph_preferences3.csv`, `psg_labels.json`)

## Kết quả chính (SGDet, PSG test, xác minh từ checkpoint tải về)

| Model | mR@20 | mR@50 | mR@100 | F1@50 | bbox mAP |
|---|---|---|---|---|---|
| Baseline YOLO12m-REACT++ | 23.28 | 26.37 | 28.28 | 31.22 | 56.57 |
| Pure Preference (ablation) | 18.92 | 21.21 | 22.57 | 26.64 | 56.57 |
| + Proto Anchor (ablation) | 21.24 | 24.51 | 26.18 | 29.58 | 56.57 |
| + KL Retention (ablation) | 22.93 | 26.21 | 27.88 | 30.87 | 56.57 |
| **Full RLHP (Tuned)** | **23.57** | **26.95** | **28.69** | **31.58** | **56.57** |

Khớp chính xác Table 2 & 3 trong paper. Cải thiện: mR@50 **+0.58** điểm (+2.2%),
F1@50 **+0.36**, bbox mAP bất biến do Stage 1 frozen.
