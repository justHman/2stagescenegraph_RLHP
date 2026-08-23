# Pipeline RLHP cho Scene Graph Labeler + REACT++

## Tóm tắt

Triển khai pipeline Reinforcement Learning from Human Preferences (RLHP) cho
`GeneralizedYOLO + REACT++` theo luồng:

```text
Human preference trên PSG train/validation
    -> audit và chuẩn hóa dữ liệu
    -> fine-tune predicate classifier
    -> đánh giá baseline/candidate trên PSG test
    -> duyệt và kích hoạt checkpoint
```

Scene Graph Labeler lấy hai predicate có xác suất cao nhất trên từng cặp object
có detection đáng tin cậy, giữ Top-K cặp theo xác suất predicate cao nhất `p1`,
sau đó chọn Top-M trong shortlist có relative margin nhỏ nhất để người dùng đánh
giá. Không dùng ngưỡng xác suất predicate cố định. Selector này chỉ là nguồn thu
thập feedback; pipeline RLHP không bổ sung hoặc so sánh các phương pháp giảm
uncertainty.

RLHP chạy offline bằng CLI. Checkpoint baseline luôn được giữ nguyên; candidate
chỉ trở thành model active sau khi đạt evaluation gate và được người vận hành
promote thủ công.

## 1. Thu thập human preference

- Chỉ dùng ảnh thuộc PSG train/validation để tạo preference training data. PSG
  test được giữ độc lập và chỉ dùng cho đánh giá cuối cùng.
- Giữ nguyên UI và cách chọn relation hiện tại của labeler.
- Mỗi feedback trong `session.json` phải lưu đủ:
  - checksum của ảnh;
  - SHA256 của checkpoint, config và label map dùng khi inference;
  - subject/object `label_id`, tên lớp và bbox;
  - hai predicate được hiển thị, gồm `predicate_id`, tên và xác suất;
  - thứ tự hiển thị, lựa chọn của người dùng và target tensor;
  - session ID và thời điểm tạo feedback.
- CSV schema v6 lưu thêm reference tường minh tới output đã label:
  - index subject/object/relation, `relation_pair_key` và số object/relation;
  - toàn bộ phân phối 56 foreground predicate cùng background probability trong
    `relation_distribution_json`;
  - loader dựng lại cache prediction từ CSV để structural audit không phải hoãn
    matching đến lúc load model.
- Các lựa chọn được giữ theo schema hiện tại:
  - strong preference cho predicate thứ nhất: `[1, 0]`;
  - strong preference cho predicate thứ hai: `[0, 1]`;
  - no preference: `[0.5, 0.5]`;
  - both wrong: `[0, 0]`;
  - skip: không có training target.

## 2. Audit và chuẩn hóa dữ liệu

Thêm loader đọc trực tiếp các file `session.json`; file CSV export không phải
nguồn dữ liệu chính của pipeline.

Preflight thực hiện các bước sau:

1. Xác minh provenance của checkpoint, config, label map và image checksum.
2. Đối chiếu image ID với PSG split và dừng ngay nếu có ảnh PSG test trong
   preference dataset.
3. Dùng prediction reference trong CSV/cache để structural audit, sau đó chạy
   model gốc để khớp lại subject/object bằng `label_id` và IoU bbox
   `>= 0.8`; không phụ thuộc `relation_id` vì thứ tự relation có thể thay đổi.
4. Chuẩn hóa thứ tự hai predicate và target tương ứng.
5. Gộp feedback trùng cho cùng ảnh, subject/object và predicate pair; trung bình
   target và giữ `vote_count` làm sample weight.
6. Loại `skip` khỏi dataset. Giữ `both wrong` trong audit/report nhưng không đưa
   vào objective vì pairwise loss gốc không tạo gradient cho target `[0, 0]`.
7. Split train/validation `80/20` theo image checksum với seed `42`, bảo đảm một
   ảnh không xuất hiện ở cả hai tập.

Preflight chỉ cho phép train khi:

- không có PSG test leakage;
- toàn bộ sample có provenance hợp lệ, trừ khi người vận hành chủ động dùng cờ
  `--allow-legacy-provenance`;
- ít nhất `95%` feedback có thể khớp lại với output của baseline;
- các session trong cùng run dùng checkpoint/config/label map tương thích.

Nếu validation thất bại, CLI dừng trước khi train và xuất báo cáo các sample bị
loại cùng lý do cụ thể.

## 3. Fine-tuning REACT++

### Objective RLHP

Reward của một predicate là xác suất của predicate đó trong full softmax 57 lớp
của REACT++. Với hai predicate ứng viên, đặt reward là `P1` và `P2`, rồi tính:

$$
\boldsymbol{\pi}
= \operatorname{softmax}\!\left(\tau [P_1,P_2]\right),
\qquad
\mathcal{L}_{\mathrm{pref}}
= -\sum_{i=1}^{2}\mu_i\log \pi_i.
$$

Objective preference chỉ nhận strong preference và no preference; không thêm
rejection hoặc uncertainty loss. Mặc định `tau=10` và cho phép override qua CLI
để tái lập thí nghiệm. Baseline retention bên dưới là regularizer tùy chọn.

### Baseline retention tùy chọn

Khi `--retention-weight` lớn hơn `0`, baseline frozen cung cấp full-softmax
teacher distribution cho đúng relation pair trước khi fine-tuning. Với hai
predicate được label, retention loss giữ tổng probability mass và phân phối
tương đối của 55 predicate còn lại, nhưng không giữ thứ tự bên trong cặp được
label. Objective khi đó là:

$$
\mathcal{L}
= \mathcal{L}_{\mathrm{pref}}
+ \lambda_{\mathrm{retain}}\mathcal{L}_{\mathrm{retain}}.
$$

Teacher distributions được cache từ baseline pass, không cần giữ một bản sao
model thứ hai trên GPU. Checkpoint tốt nhất được chọn theo validation objective
kết hợp này; preference loss và retention loss vẫn được ghi riêng.

### Phạm vi parameter

- Khởi tạo từ baseline `checkpoint/best_model.pth`.
- Giữ model ở evaluation mode trong preference-forward để tắt dropout và không
  cập nhật EMA prototype. Riêng inference prototype cache bị vô hiệu hóa trong
  forward này: `proj_head` nhận gradient đối xứng qua cả relation embedding và
  projected prototype, tránh trạng thái prototype bị `detach` nhưng vẫn dịch
  chuyển sau mỗi lần cập nhật `proj_head`.
- Đóng băng YOLO backbone/detector, object path và relation encoder.
- Chỉ bật gradient cho:
  - `roi_heads.relation.predictor.proj_head`;
  - `roi_heads.relation.predictor.logit_scale`.
- Trước khi lưu candidate, xác minh hash của mọi frozen parameter không thay đổi.

### Training mặc định

- Optimizer: AdamW.
- Learning rate: `1e-5`.
- Weight decay: `0.01`.
- Batch theo một ảnh, gom toàn bộ preference của ảnh vào một forward.
- Gradient accumulation: `4`.
- Gradient clipping: norm `1.0`.
- Tối đa `5` epoch.
- Early stopping patience `2` theo validation preference loss.
- Lưu checkpoint epoch đầu và checkpoint có validation loss tốt nhất.

## 4. Đánh giá baseline và candidate

Lệnh evaluation chạy baseline và candidate tuần tự trên cùng toàn bộ PSG test,
cùng config, seed, preprocessing và evaluator. Không dùng ảnh PSG test trong
bất kỳ bước fine-tuning hoặc chọn checkpoint nào.

### Preference validation

- Validation preference loss.
- Strong-preference accuracy.
- Số sample usable/dropped và pair-matching rate.

### PSG test

Tái sử dụng evaluator SGDet hiện có trong `tools/evaluate.py` và báo cáo:

- `R@20`, `R@50`, `R@100`;
- `mR@20`, `mR@50`, `mR@100`;
- `F1@20`, `F1@50`, `F1@100`;
- `zR@20`, `zR@50`, `zR@100`;
- bbox `mAP`.

Report phải chứa giá trị baseline, candidate, chênh lệch tuyệt đối và chênh lệch
tương đối cho từng metric.

Candidate đạt evaluation gate khi:

- không có NaN/Inf và pair-matching rate `>= 95%`;
- validation preference loss tốt hơn baseline;
- SGDet `mR@50` cao hơn baseline;
- `R@50`, `F1@50` và bbox `mAP` không giảm quá `1%` tương đối so với baseline.

Các metric còn lại luôn được báo cáo nhưng không trực tiếp quyết định gate.

## 5. CLI, artifact và kích hoạt model

Thêm CLI `scene_graph_tool/rlhp.py` với các lệnh:

- `audit`: kiểm tra provenance, split, phân bố feedback và pair matching;
- `train`: chạy audit bắt buộc rồi fine-tune từ baseline;
- `evaluate`: so sánh baseline/candidate trên preference validation và PSG test;
- `promote`: chỉ chấp nhận candidate đã pass gate và cập nhật active-model
  manifest theo cách atomic;
- `rollback`: đưa manifest về checkpoint active trước đó.

Mỗi run lưu vào `scene_graph_tool/rlhp_runs/<run_id>/`:

- `run.json` và resolved config;
- audit report và danh sách sample bị loại;
- train/validation split manifest;
- training log và epoch checkpoints;
- best candidate checkpoint;
- baseline/candidate metrics và gate result;
- frozen-parameter hashes và provenance của artifact.

`promote` không ghi đè baseline. Scene Graph Labeler ưu tiên checkpoint truyền
trực tiếp bằng CLI; nếu không có thì đọc active-model manifest. Việc promote hoặc
rollback chỉ có hiệu lực sau khi restart server để model không bị hot-swap giữa
một annotation session.

Pipeline không thêm RLHP dashboard, API training status hoặc GPU job vào web
server.

## 6. Kiểm thử và acceptance criteria

- Unit test schema, provenance, predicate-order normalization, deduplication và
  vote aggregation.
- Test split theo image checksum không leakage giữa train/validation.
- Test bắt buộc dừng khi preference data chứa image ID thuộc PSG test.
- Test matcher với object cùng class, bbox gần nhau và IoU dưới ngưỡng.
- Unit test pairwise loss cho `[1,0]`, `[0,1]`, `[0.5,0.5]`; xác nhận `both wrong`
  và `skip` không tham gia optimization.
- Gradient test xác nhận chỉ `proj_head` và `logit_scale` thay đổi.
- Checkpoint round-trip phải load được bằng Predictor hiện tại và giữ nguyên
  output schema.
- Evaluation test xác nhận baseline/candidate dùng cùng PSG test và report đủ
  `R`, `mR`, `F1`, `zR` tại `20/50/100` cùng bbox `mAP`.
- Test gate tại các boundary, promote/rollback manifest và restart behavior.
- Smoke end-to-end với model/dataset giả; CUDA training phải báo lỗi rõ ràng nếu
  không có GPU.

Pipeline được xem là hoàn tất khi có thể chạy liên tục từ một tập session hợp lệ
đến candidate checkpoint, sinh báo cáo so sánh với baseline trên PSG test, và
chỉ activate candidate đã pass gate sau thao tác promote thủ công.

## Giả định đã chốt

- Đây là supervised preference fine-tuning theo RLHP, không triển khai PPO,
  policy rollout hoặc reward model riêng.
- Labeler lọc object, lấy Top-K relation theo `p1`, rồi chọn Top-M theo relative
  margin; phần selector nằm ngoài logic fine-tuning của RLHP.
- Preference data được thu từ PSG train/validation; PSG test hoàn toàn độc lập.
- Reward dùng predicate probability, không dùng raw logits.
- `Both wrong` được giữ để audit nhưng không dùng cho training.
- Baseline luôn được bảo toàn; candidate chỉ trở thành active sau khi pass gate
  và được người vận hành promote thủ công.
