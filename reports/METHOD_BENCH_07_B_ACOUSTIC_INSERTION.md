# METHOD-BENCH-07 Agent B — Acoustic insertion reconstruction

## Kết quả

Đã huấn luyện mô hình temporal Conv1d cho exact-gap insertion, đánh giá speaker-grouped TRAIN OOF bốn fold, seed 101. AP-I pooled **0.012828**, gần prevalence 0.009126 nhưng thấp hơn static-local control 0.011473 (chênh +0.00136) và thấp xa MB03 neural direct baseline **0.162125**. Đây là kết quả âm cho rank-conditioned temporal CNN đã thử, không phải bằng chứng bác bỏ mọi acoustic localization.

## Khôi phục baseline và kiểm tra alignment

Artifact MB03 `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_predictions.npz`, SHA-256 `c1814a70eb53b14859bb42450f6efc1f8edb5a0631200ba9e73c195dfee6464d`, có đủ keys `gap_PLAIN_0..3`; cột 1 là xác suất insertion. Tổng 61,362 exact gaps, 560 positive. NPZ không lưu event IDs theo hàng; alignment được kiểm tra qua producer `get_arrays`, manifest utterance order và gap index 0..N, fold lengths/supports. Đây là giới hạn provenance còn lại; không tuyên bố artifact tự chứa ID-level join.

| Fold | Speakers giữ lại | Gaps / positives | MB03 AP-I | MB07 temporal AP-I | Static-local AP-I |
|---|---|---:|---:|---:|---:|
| F0 | ABA/ASI/BWC | 15,266 / 134 | .188792 | .011852 | .017313 |
| F1 | EBVS/HJK/HKK | 15,298 / 127 | .102708 | .010431 | .009209 |
| F2 | HQTV/LXC/NJS | 15,439 / 150 | .109449 | .015074 | .011194 |
| F3 | PNV/SKA/SVBI | 15,359 / 149 | .251541 | .015844 | .012924 |
| Pooled | 12 TRAIN speakers | 61,362 / 560 | **.162125** | **.012828** | **.011473** |

Temporal P@560 pooled=.01964 (11 true positives), P@5%=.01564; static P@560=.02143 (12 positives), P@5%=.01499; MB03 P@560=.24286 (136 positives). Fold-specific values and all supports are in receipt and `scores.npz`. Supervisor generated full per-fold and pooled PR coordinates for MB03 neural, MB07 temporal and static controls in `reports/METHOD_BENCH_07_INSERTION_PR_CURVES.csv.gz` after exact manifest-derived gap-ID/label alignment.

## Phương pháp, thực thi, tái lập

ID `RQ1-MB07-B-SEQACOUSTIC-S101`. Input: frozen `facebook/wav2vec2-base-960h` snapshot `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, hidden layer 6, 768D, 20-ms frame sequence from MB03 feature cache. Its extraction receipt records `gold_timing_used=false`, `realized_phones_used_for_features=false`, and `gold_error_labels_used_for_features=false`; TRAIN WAV manifest SHA-256 is `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff`. Với gap j của N+1, vị trí trung tâm được ước lượng bằng rank warping `round((j+0.5)T/(N+1)-0.5)`; lấy cửa sổ liên tục 9 frame với edge replication. Temporal model Conv1d hai lớp + MLP, 314,017 parameters. Đối chứng static-local dùng mean/std/last-minus-first của cùng cửa sổ, logistic regression. Từng outer fold train chỉ trên 9 speaker còn lại: seed 101, 10 epochs × 150 steps, batch 256, mỗi batch 64 positives + 192 negatives, AdamW lr 4e-4, weight decay 1e-3. CUDA GTX 1650, runtime 80.86 s. Held-out labels chỉ dùng evaluation; không dùng realized-phone/gold-timing inference; không dùng DEV_CAL/DEV_SEARCH.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/rq1_method_bench_07/agent_b/train.py
.venv/bin/python experiments/rq1_method_bench_07/agent_b/verify.py
```

Nguồn/receipt/artifacts: `experiments/rq1_method_bench_07/agent_b/{train.py,verify.py,receipt.json,scores.npz,train.log,mb03_artifact_verification.json}`. Supervisor kiểm tra độc lập `scores.npz`: 61,362 `ids` duy nhất theo dạng `utterance_id::gap_idx`, 560 positives, đủ 12 speakers. Manifest SHA-256 `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff`; feature extraction receipt MB03: `experiments/rq1_method_bench_03/agent_a/w2v2_h6_features/EXTRACTION_RECEIPT.json`.

## Diễn giải và giới hạn

Mô hình dùng chuỗi frame thật, nhưng vị trí gap là tỷ lệ canonical rank/time, không phải dự đoán ranh giới bằng monotonic/CTC alignment. Vì vậy kết quả chủ yếu cho thấy proportional rank-warp + local CNN không định vị insertion tốt; không kiểm tra được liệu alignment có điều kiện trên reference sẽ tạo ra thông tin mới hay không. Static-local, temporal và neural có cùng grouped held-speaker folds, nhưng neural là historical matched artifact, không retrain trong chiến dịch này. Không lặp seed. Temporal tăng nhẹ pooled AP so với prevalence nhưng foldwise chỉ nhỉnh hơn static ở folds 1–3; P@560 còn thấp hơn static.

Một follow-up có cơ sở nhưng chưa chạy là alignment có điều kiện reference (fold-valid CTC posterior/Viterbi hoặc forward-backward occupancy), rồi đưa boundary uncertainty/đoạn âm học chưa giải thích vào explicit insertion-state gap score. Không được dùng gold interval ở held-out inference. Không có cơ sở tích hợp mô hình insertion mới hiện tại.

**Resource receipt deviation:** run dùng CUDA trên GTX 1650 nhưng không có cooperative lease entry trong `coordination/PHASE3_GPU_LEDGER.jsonl`; GPU chỉ được supervisor thấy idle ở lần kiểm tra trước đó. Giữ kết quả, không rerun negative experiment chỉ để lặp lại compute.
