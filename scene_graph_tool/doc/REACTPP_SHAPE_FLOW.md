# REACT++: Tensor Shape Flow từ Input đến Output

Tài liệu này mô tả luồng tensor của đúng mô hình đang được `scene_graph_tool` sử dụng:

- Dataset/checkpoint: PSG
- Detector: YOLO12m
- Input model: `640 x 640`
- Số lớp object, bao gồm background: `C_obj = 134`
- Số lớp relation, bao gồm background: `C_rel = 57`
- Object feature dimension: `D_obj = 256`
- REACT++ hidden dimension: `D = 512`
- GloVe embedding dimension: `E = 200`

Cấu hình tương ứng nằm trong [`checkpoint/config.yml`](../../checkpoint/config.yml).

## 1. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| `B` | Batch size |
| `N_i` | Số object sau NMS của ảnh thứ `i`, tối đa 100 |
| `N = sum(N_i)` | Tổng số object trong batch |
| `R_i` | Số directed object pairs của ảnh thứ `i` |
| `R = sum(R_i)` | Tổng số relation candidates trong batch |
| `M_i` | Số relation còn lại sau ranking/top-K |
| `C_obj = 134` | Số object classes, gồm background |
| `C_rel = 57` | Số relation classes, gồm background |
| `D_obj = 256` | Output dimension của DAMP object feature extractor |
| `D = 512` | Hidden dimension của REACT++ |
| `E = 200` | Word embedding dimension |

## 2. Tổng quan pipeline

```text
Ảnh RGB [H, W, 3]
    │ Letterbox + normalize + HWC -> CHW
    ▼
Input [B, 3, 640, 640]
    │
    ▼
YOLO12m backbone + neck
    ├── P3 [B, 256, 80, 80]
    ├── P4 [B, 512, 40, 40]
    └── P5 [B, 512, 20, 20]
    │
    ├── YOLO Detect + NMS
    │      └── N_i object proposals/ảnh
    │
    ├── DAMP object feature extractor
    │      └── roi_features [N, 256]
    │
    ├── Directed pair generator
    │      └── rel_pair_idxs[i] [R_i, 2]
    │
    └── P5 Scene Context Extractor
           └── scene_features [B*16, 256]
                   │
                   ▼
           REACT++ Predictor
                   ├── object distributions [N_i, 134]
                   └── relation logits [R_i, 57]
                              │
                              ▼
                     Softmax + ranking
                              │
                              ▼
                     Scene graph output
```

## 3. Tiền xử lý input

Tool nhận ảnh OpenCV/RGB với shape:

```text
[H, W, 3]
```

Ảnh được letterbox về `640 x 640`, đổi từ HWC sang CHW và chuẩn hóa về khoảng `[0, 1]`:

```text
[H, W, 3]
-> [3, 640, 640]
-> ImageList.tensors [B, 3, 640, 640]
```

Ví dụ trace với `scene_graph_tool/test_images/2366635.jpg`:

```text
[502, 640, 3]
-> [1, 3, 640, 640]
```

Forward cấp cao được thực hiện trong [`GeneralizedYOLO.forward()`](../../sgg_benchmark/modeling/detector/generalized_yolo.py#L67).

## 4. YOLO12m backbone và neck

Với input `x = [B, 3, 640, 640]`, shape qua các block của checkpoint hiện tại là:

| Layer | Module | Output shape |
|---:|---|---|
| 0 | Conv, stride 2 | `[B, 64, 320, 320]` |
| 1 | Conv, stride 2 | `[B, 128, 160, 160]` |
| 2 | C3k2 | `[B, 256, 160, 160]` |
| 3 | Conv, stride 2 | `[B, 256, 80, 80]` |
| 4 | C3k2 | `[B, 512, 80, 80]` |
| 5 | Conv, stride 2 | `[B, 512, 40, 40]` |
| 6 | A2C2f | `[B, 512, 40, 40]` |
| 7 | Conv, stride 2 | `[B, 512, 20, 20]` |
| 8 | A2C2f | `[B, 512, 20, 20]` |
| 9 | Upsample | `[B, 512, 40, 40]` |
| 10 | Concat layer 9 + 6 | `[B, 1024, 40, 40]` |
| 11 | A2C2f | `[B, 512, 40, 40]` |
| 12 | Upsample | `[B, 512, 80, 80]` |
| 13 | Concat layer 12 + 4 | `[B, 1024, 80, 80]` |
| 14 | A2C2f, P3 | `[B, 256, 80, 80]` |
| 15 | Conv, stride 2 | `[B, 256, 40, 40]` |
| 16 | Concat layer 15 + 11 | `[B, 768, 40, 40]` |
| 17 | A2C2f, P4 | `[B, 512, 40, 40]` |
| 18 | Conv, stride 2 | `[B, 512, 20, 20]` |
| 19 | Concat layer 18 + 8 | `[B, 1024, 20, 20]` |
| 20 | C3k2, P5 | `[B, 512, 20, 20]` |

REACT++ lấy ba feature maps ở layer 14, 17 và 20:

```text
features = [
    P3: [B, 256, 80, 80],
    P4: [B, 512, 40, 40],
    P5: [B, 512, 20, 20]
]
```

Xem [`YoloModel.forward()`](../../sgg_benchmark/modeling/backbone/yolo.py#L103).

## 5. YOLO Detect head và NMS

Tổng số grid locations của ba mức feature pyramid:

```text
80*80 + 40*40 + 20*20
= 6400 + 1600 + 400
= 8400
```

Detect head trả decoded predictions:

```text
[B, 4 + 133, 8400]
= [B, 137, 8400]
```

Trong đó:

- `4`: box coordinates.
- `133`: PSG foreground object classes.

Raw detection maps:

```text
P3 detect: [B, 197, 80, 80]
P4 detect: [B, 197, 40, 40]
P5 detect: [B, 197, 20, 20]
```

`197 = 64` box-regression channels `+ 133` class channels.

Sau confidence filtering và NMS, mỗi ảnh có proposal dictionary:

```text
boxes       [N_i, 4]
lb_boxes    [N_i, 4]
pred_labels [N_i]
pred_scores [N_i]
labels      [N_i]
feat_idx    [N_i]
```

Theo config hiện tại:

```text
N_i <= detections_per_img = 100
```

`boxes` nằm trong hệ tọa độ ảnh gốc. `lb_boxes` nằm trong hệ tọa độ letterbox `640 x 640`, cùng hệ với P3/P4/P5. `feat_idx` là flat grid index đã sinh ra detection và được DAMP dùng ở bước tiếp theo.

Xem [`YoloModel.postprocess()`](../../sgg_benchmark/modeling/backbone/yolo.py#L143).

## 6. DAMP object feature extractor

DAMP nhận:

```text
P3       [B, 256, 80, 80]
P4       [B, 512, 40, 40]
P5       [B, 512, 20, 20]
feat_idx [N]
```

Config hiện tại:

```yaml
feat_idx_multiscale: true
feat_idx_neighbors: 1
```

Với mỗi object, `feat_idx` được giải mã thành tọa độ tương đối `(cx, cy)`. DAMP gather lân cận `3 x 3` ở cả ba mức:

```text
P3 gather: [N, 9, 256] --Gaussian weighted sum-> [N, 256]
P4 gather: [N, 9, 512] --Gaussian weighted sum-> [N, 512]
P5 gather: [N, 9, 512] --Gaussian weighted sum-> [N, 512]
```

Trong implementation, chiều chín lân cận được cộng có trọng số trực tiếp nên tensor tạm cuối cùng của mỗi level là `[N, C_i]`.

Projection từng level:

```text
P3: [N, 256] -> LayerNorm -> Linear(256, 256) -> [N, 256]
P4: [N, 512] -> LayerNorm -> Linear(512, 256) -> [N, 256]
P5: [N, 512] -> LayerNorm -> Linear(512, 256) -> [N, 256]
```

Fuse đa tỉ lệ:

```text
[N, 256] || [N, 256] || [N, 256]
-> [N, 768]
-> LayerNorm
-> Linear(768, 256)
-> roi_features [N, 256]
```

Xem [`DAMPBoxFeatureExtractor.forward()`](../../sgg_benchmark/modeling/roi_heads/box_head/roi_box_feature_extractors.py#L358).

## 7. Tạo candidate relation pairs

Ở inference, tool tạo mọi cặp có hướng và loại bỏ self-pair:

```text
candidate matrix  [N_i, N_i]
rel_pair_idxs[i]  [R_i, 2]
```

Vì `require_overlap = false`:

```text
R_i = N_i * (N_i - 1)
```

Hai cột của pair tensor:

```text
rel_pair_idxs[:, 0] = subject index
rel_pair_idxs[:, 1] = object index
```

Ví dụ:

```text
N_i = 47
R_i = 47 * 46 = 2162
rel_pair_idxs[i] = [2162, 2]
```

Trong training, relation sampler giới hạn:

```text
R_i <= batch_size_per_image = 512
```

với `positive_fraction = 0.35`.

Xem [`RelationSampling.prepare_test_pairs()`](../../sgg_benchmark/modeling/roi_heads/relation_head/sampling.py#L27).

## 8. P5 Scene Context Extractor

Checkpoint hiện tại dùng `P5SceneContextExtractor`; context là image-level thay vì RoIAlign riêng cho từng relation.

Đầu vào:

```text
P5 [B, 512, 20, 20]
```

Luồng shape:

```text
AdaptiveAvgPool2d(4, 4)
[B, 512, 20, 20]
-> [B, 512, 4, 4]

Flatten spatial + permute
-> [B, 16, 512]

Normalize + Linear(512, 256)
-> [B, 16, 256]

AIFI TransformerEncoderLayer
-> [B, 16, 256]

Flatten batch và token
-> union_features [B*16, 256]
```

Xem [`P5SceneContextExtractor.forward()`](../../sgg_benchmark/modeling/roi_heads/relation_head/roi_relation_feature_extractors.py#L347).

## 9. REACT++ predictor

Các input chính:

```text
roi_features   [N, 256]
union_features [B*16, 256]
rel_pair_idxs  list([R_i, 2])
object labels  [N]
```

### 9.1. Visual subject/object slots

```text
roi_features [N, 256]
-> Linear(256, 1024)
-> RMSNorm
-> vis [N, 1024]
```

Split thành hai role-specific slots:

```text
sub_vis = vis[:, :512]  -> [N, 512]
obj_vis = vis[:, 512:]  -> [N, 512]
```

### 9.2. Object semantics

Object label và one-hot distribution:

```text
entity_preds [N]
entity_dists [N, 134]
```

Frozen GloVe lookup và SwiGLU lifting:

```text
[N]
-> Embedding(134, 200)
-> [N, 200]
-> SwiGLU(200, 512)
-> sem [N, 512]
```

Trong SwiGLU này:

```text
gate/value [N, 400]
output     [N, 512]
```

Object prototype bank dùng cùng embedding/lifting path:

```text
obj_embed.weight [134, 200]
-> obj_proto_kv [134, 512]
```

### 9.3. Geometry RoPE

Mỗi box được encode thành chín đặc trưng chuẩn hóa:

```text
box_info [N, 9]
-> Linear(9, 16)
-> SiLU
-> Linear(16, 64)
-> geo_bias [N, 64]
```

### 9.4. Subject/object cross-attention

Subject branch:

```text
Query: sub_vis      [N, 512]
KV:    obj_proto_kv [134, 512]
```

Với bốn heads và head dimension 128:

```text
Q [N, 512]   -> [N, 4, 128]
K [134, 512] -> [134, 4, 128]
V [134, 512] -> [134, 4, 128]

attention logits/probabilities [4, N, 134]
attention output               [N, 4, 128]
merged output                  [N, 512]
```

Sau residual, RMSNorm và SwiGLU:

```text
sub [N, 512]
```

Object branch có cùng shape nhưng dùng bộ attention/FFN weights riêng:

```text
obj [N, 512]
```

Cộng trực tiếp semantic signal:

```text
sub = sub + sem -> [N, 512]
obj = obj + sem -> [N, 512]
```

### 9.5. Relation composition

Gather subject/object representation theo từng pair:

```text
si = sub[subject_idx] -> [R, 512]
oi = obj[object_idx]  -> [R, 512]
```

Asymmetric bilinear composition:

```text
r = si * oi + si - oi
-> [R, 512]
```

Qua relation SwiGLU và residual:

```text
[R, 512]
-> gate/value [R, 688]
-> output [R, 512]
-> residual + RMSNorm
-> r [R, 512]
```

### 9.6. Cross-attention với scene tokens

Scene features được nâng lên hidden dimension:

```text
[B*16, 256]
-> Linear(256, 512)
-> [B*16, 512]
-> reshape [B, 16, 512]
```

Cho từng ảnh `i`:

```text
Query: relation representations [R_i, 512]
KV:    scene tokens             [16, 512]
```

Attention shapes:

```text
Q [R_i, 512] -> [R_i, 4, 128]
K [16, 512]  -> [16, 4, 128]
V [16, 512]  -> [16, 4, 128]

attention matrix [4, R_i, 16]
context output   [R_i, 512]
```

Gating:

```text
r_ctx [R_i, 512]
gate  [R_i, 512]

r = r + sigmoid(gate) * r_ctx
-> [R_i, 512]
```

### 9.7. Projection và relation classification

Relation projection:

```text
r [R, 512]
-> Linear(512, 1024)
-> RMSNorm + SiLU
-> Linear(1024, 512)
-> r_proj [R, 512]
-> L2 normalize
-> r_norm [R, 512]
```

Relation prototype bank:

```text
proto_weight [57, 512]
-> Linear(512, 1024)
-> RMSNorm + SiLU
-> Linear(1024, 512)
-> L2 normalize
-> protos_norm [57, 512]
```

Cosine-similarity classifier:

```text
r_norm [R, 512]
@ protos_norm.T [512, 57]
-> relation logits [R, 57]
```

Sau đó tensor được split lại theo ảnh:

```text
entity_dists:
    list([N_1, 134], ..., [N_B, 134])

relation_logits:
    list([R_1, 57], ..., [R_B, 57])
```

Forward chính của predictor nằm tại [`REACTPlusPlusPredictor.forward()`](../../sgg_benchmark/modeling/roi_heads/relation_head/predictors/new_react_predictor.py#L553).

## 10. Relation postprocessing

Với mỗi ảnh:

```text
relation logits [R_i, 57]
-> softmax
-> relation probabilities [R_i, 57]
```

Bỏ background column khi chọn relation label:

```text
foreground probabilities [R_i, 56]
-> max trên dimension class
-> rel_scores [R_i]
-> rel_labels [R_i]
```

Triple score trong PyTorch postprocessor:

```text
triple_score = relation_score * subject_score * object_score
-> [R_i]
```

Sau sort/ranking:

```text
rel_pair_idxs    [M_i, 2]
pred_rel_scores  [M_i, 57]
pred_rel_labels  [M_i]
```

Config đánh giá mặc định có `test.top_k = 100`, do đó `M_i <= 100`. Tuy nhiên `scene_graph_tool` override thành:

```python
cfg.test.top_k = 0
```

nên tool giữ toàn bộ relation candidates:

```text
M_i = R_i
```

Xem [`PostProcessor.forward()`](../../sgg_benchmark/modeling/roi_heads/relation_head/inference.py#L35).

## 11. PyTorch output và JSON output của tool

Model trả một dictionary cho mỗi ảnh:

```python
{
    "boxes":             Tensor[N_i, 4],
    "pred_labels":       Tensor[N_i],
    "pred_scores":       Tensor[N_i],
    "rel_pair_idxs":     Tensor[M_i, 2],
    "pred_rel_scores":   Tensor[M_i, 57],
    "pred_rel_labels":   Tensor[M_i],
}
```

`scene_graph_tool` chuẩn hóa output thành:

```text
objects:   List length N_i
relations: List length M_i
```

Mỗi object:

```json
{
  "id": 0,
  "label": "person",
  "label_id": 77,
  "score": 0.96,
  "box_xyxy": [12.3, 45.6, 210.4, 390.1]
}
```

Mỗi relation:

```json
{
  "id": 0,
  "subject_id": 0,
  "object_id": 1,
  "background_probability": 0.12,
  "best_predicate": "holding",
  "best_predicate_id": 22,
  "best_probability": 0.76,
  "predicate_probabilities": {
    "holding": 0.76,
    "carrying": 0.13
  }
}
```

Phần chuyển đổi sang JSON nằm trong [`Predictor._normalize()`](../app.py#L183).

## 12. Trace thực tế

Trace trực tiếp với ảnh `scene_graph_tool/test_images/2366635.jpg` cho kết quả:

```text
Ảnh gốc:
[502, 640, 3]

Input model:
[1, 3, 640, 640]

YOLO feature maps:
P3 [1, 256, 80, 80]
P4 [1, 512, 40, 40]
P5 [1, 512, 20, 20]

YOLO NMS:
N = 47 objects

DAMP object features:
[47, 256]

Directed relation pairs:
47 * 46 = 2162
[2162, 2]

P5 scene context:
[16, 256]

REACT++ outputs:
object distributions [47, 134]
relation logits       [2162, 57]

Final scene_graph_tool output:
47 objects
2162 directed relations
```

## 13. Training output

Trong training, model không trả scene graph dictionary mà trả dictionary các scalar losses:

```text
loss_rel    []
l21_loss   []
dist_loss2 []
loss_dis   []
```

Trong đó relation logits trước loss vẫn có shape:

```text
relation_logits[i] [R_i, 57]
relation_labels[i] [R_i]
R_i <= 512
```

REACT++ đồng thời cập nhật EMA relation prototypes:

```text
representations [R, 512]
labels          [R]
prototype EMA   [57, 512]
```

## 14. ONNX output

Khi `model.export = True`, model đổi output dictionary thành hai tensor.

Object tensor:

```text
boxes [N_keep, 6]
```

Các cột:

```text
[x1, y1, x2, y2, object_label, object_score]
```

Relation tensor:

```text
rels [R_keep, 5]
```

Các cột:

```text
[subject_idx, object_idx, relation_label, triplet_score, relation_score]
```

ONNX export dùng geometric-mean triplet score:

```text
triplet_score = (subject_score * object_score * relation_score) ** (1/3)
```

Phần tạo hai output tensor nằm trong [`GeneralizedYOLO.generate_detect_sg()`](../../sgg_benchmark/modeling/detector/generalized_yolo.py#L250).

Lưu ý: export path hiện lấy `result[0]`, vì vậy output ONNX thực tế được thiết kế chủ yếu cho inference `B = 1`, dù input có khai báo dynamic batch axis.

## 15. Shape summary

| Stage | Input | Output |
|---|---|---|
| Preprocess | `[H, W, 3]` | `[B, 3, 640, 640]` |
| YOLO P3 | `[B, 3, 640, 640]` | `[B, 256, 80, 80]` |
| YOLO P4 | — | `[B, 512, 40, 40]` |
| YOLO P5 | — | `[B, 512, 20, 20]` |
| Detect | P3/P4/P5 | `[B, 137, 8400]` |
| NMS | `[B, 137, 8400]` | proposals `[N_i, 4]` |
| DAMP | P3/P4/P5 + `feat_idx` | `[N, 256]` |
| Pair generator | proposals | `[R_i, 2]` |
| P5 scene extractor | `[B, 512, 20, 20]` | `[B*16, 256]` |
| Visual projection | `[N, 256]` | `[N, 1024]` |
| Subject slot | `[N, 1024]` | `[N, 512]` |
| Object slot | `[N, 1024]` | `[N, 512]` |
| Object semantic embedding | `[N]` | `[N, 512]` |
| Geometry encoder | `[N, 9]` | `[N, 64]` |
| Object-prototype attention | `[N, 512]`, `[134, 512]` | `[N, 512]` |
| Pair composition | subject/object `[R, 512]` | `[R, 512]` |
| Scene attention | `[R_i, 512]`, `[16, 512]` | `[R_i, 512]` |
| Relation projection | `[R, 512]` | `[R, 512]` |
| Cosine classifier | `[R, 512]`, `[57, 512]` | `[R, 57]` |
| PyTorch postprocess | `[R_i, 57]` | `[M_i, 57]` |
| Tool JSON | model dictionary | `N_i` objects, `M_i` relations |
| ONNX export | model dictionary | boxes `[N_keep, 6]`, rels `[R_keep, 5]` |
