# EVAL-V3 — Đối soát khoa học và giới hạn nghiệm thu

Nguồn đóng băng: `03e4968ffd8908782d0bf463b6d101e6b6872372`; HEAD trước publication: `03e4968ffd8908782d0bf463b6d101e6b6872372`. Chỉ thêm artifact EVAL-V3; giữ nguyên **1499** nguồn đã hash và archive chỉ đọc **930** output lịch sử.

## Kết quả thực thi

- Inventory: **290** dòng; kết quả: **500** dòng, trong đó **268 FULL_SCORE_REPLAY** và **47 COUNTS_ONLY_PARTIAL**. Đây không phải tuyên bố mọi experiment có đủ score.
- Subset đối sánh: **98** dòng. Không gộp chỉ vì cùng tên DEV_SEARCH; cohort/event IDs, ontology, hash và policy phải được kiểm tra.
- Nguồn DEV_SEARCH: **599 utterances / 4 speakers / 19.755 targets / 20.354 gaps**, gold C/S/O/I=17.778/1.768/209/125; TRAIN/DEV_CAL/DEV_SEARCH không giao speaker.
- Evaluator đánh giá discrimination bằng stepwise average precision, không áp FWR acceptance veto. FWR≤0,20 chỉ là reference operating point; không chứng nhận an toàn sư phạm.

| replay_status | Dòng inventory |
|---|---:|
| COUNTS_ONLY_PARTIAL | 48 |
| FULL_SCORE_REPLAY | 57 |
| INVALID_SUPERSEDED | 86 |
| MISSING_ARTIFACT | 23 |
| PROTOCOL_NONCOMPARABLE | 76 |

## Các sửa sai được xác nhận

1. Narrative 23.044 targets/23.643 gaps không khớp manifest hiện tại; số đúng 19.755/20.354. Giá trị cũ giữ nguyên trong legacy evidence, không viết lại báo cáo cũ.
2. CPL01 omission union tính trực tiếp raw flags: **48+68−31=85/209**, không phải94. Không cộng score diagnosis conditional thành end-to-end F1.
3. Frozen Gold seed101 shared thresholds: TP S/O/I=**1/3/29**, FP=**0/15/48**, tổng63. Separated S/O: TP=**27/3/29**, FP=**12/15/48**, tổng75. Không đổi checkpoint/continuous scores; realized risk tăng, không phải cùng FWR.
4. I-Precision lịch sử tại threshold báo cáo là **11/51=21,5686%**, không phải66,7%; không suy ra ưu thế so B102/B103 khi cohort/protocol khác. Macro3-target(C/S/O) không phải macro3-error(S/O/I). Macro4 là mixed-domain average, không phải4-way softmax.
5. `UPR_BASELINE` trong MASTER báo macro3=0,3441 và macro4=0,2581 nhưng từ chính reported F1 cells chỉ ra macro3_target=0,3076667 và macro4=0,251975; đây là arithmetic-only finding, không backfill thành observed metrics. `EXP-B101` no-target-warning policy trên canonical gold có Correct TP/FP/FN=17.778/1.977/0, Correct F1=0,94732635, khác0,923 ghi trong MASTER. `REPORTED_METRIC_ARITHMETIC.csv` giữ nguyên row cũ và phép tính mới. UPR03 frozen FWR113/599=0,1886477462.
6. `45cd3d2` chứa Phase3 execution/evaluation; source diagnostic hiện tại thuộc HEAD03e4968. Không gán ngược source mới cho commit cũ. Superseded KD/GRU/threshold artifacts không được dùng làm champion.

## Kiểm toán số độc lập

Luna lane C dùng arithmetic score-tie groups riêng, không import canonical evaluator. Raw Gold seed101 DEV_SEARCH: AP S/O/I=**0,3646020948/0,1594330191/0,1867152405**, macro3_error_ap=**0,2369167848**. Argmax TP S/O/I=863/144/43, FP=1557/900/200, false_warning_count2657, joint_fwr2657/599=4,4357262104. DEV_CAL denominator đúng300:1271/300=4,2366666667.
Chi tiết ledger/hash, shared-vs-separated regression, union85 và matrices: `experiments/evaluation_v3/INDEPENDENT_AUDIT.json`, `reports/EVAL_V3_INDEPENDENT_AUDIT.md`. Bảy control cũ thực tế có19.664 targets/20.263 gaps, với49 extra và140 missing target/gap slots so canonical; không coi cùng cohort chỉ vì599utterances. Hai synthesis controls khớp đầy đủ.

## WAV, non-oracle và provenance

Genuine sampled WAV+orthographic prompt→local CMUDict→local frozen CTC→student CUDA: **PASS_SAMPLED_WAVEFORM_ONLY**, 39 targets/40 gaps; peak413.535MiB, end-to-end0.8902s gồm model load. Sáu gold-field sentinels bị từ chối trước inference. Model student78661 parameters. Đây không phải warm-streaming/full-population latency.
**UNVERIFIED cho cached full-population reference provenance:** parser tạo target identities/order từ TextGrid. Việc waveform code không nạp gold timing không chứng minh sequence intended độc lập annotation; sampled orthographic phonemization khác manifest. Không đổi event ontology, không dùng sequence mới để dựng thay holdout hoặc thay canonical gold.
Raw softmax/binary scores chưa mặc nhiên calibrated. Isotonic fit/assess cùng DEV_CAL phải mang `IN_SAMPLE_CALIBRATION`; checkpoint/threshold selection reuse DEV_CAL được khai báo. DEV_SEARCH speaker-disjoint nhưng development-exposed, không phải final independent test.

## Robustness và thống kê

Đã xuất **1.567.419 PR points** (84 source/class curves, ties và no-warning endpoint), **275.951 risk–coverage points** với full-population FN,1.056 strata,144 fixed-K rows,360 reliability bins,36 CAL→SEARCH shift rows và36 raw-vs-isotonic calibration comparisons. `MATCHED_CONFIDENCE_POLICY_STATISTICS.csv` có70 metric/policy comparisons qua seeds101/202/303; bootstrap200 lần trên4 speaker chỉ exploratory. `PR_CURVES.csv.gz`/`RISK_COVERAGE.csv.gz` là bản versioned lossless; CSV lớn vẫn giữ local và có hash/tái lập đầy đủ. Fixed K25/50/100/200, tie-break score rồi event ID, không gold-aware sorting.
- `ap_substitution`: GOLD=0.364352±0.006570; KD_ALL=0.366889±0.006057; paired KD−GOLD=+0.002537±0.001360.
- `ap_omission`: GOLD=0.154674±0.007391; KD_ALL=0.153960±0.006983; paired KD−GOLD=-0.000714±0.000409.
- `ap_insertion_exact_gap`: GOLD=0.184074±0.004472; KD_ALL=0.188680±0.006504; paired KD−GOLD=+0.004606±0.002309.
- `macro3_error_ap`: GOLD=0.234367±0.002628; KD_ALL=0.236510±0.003610; paired KD−GOLD=+0.002143±0.001034.

## Trạng thái nghiệm thu

Đã có code executable, score replay, independent arithmetic và sampled waveform evidence; **không tuyên bố mọi scientific validity gate đã PASS**, không tuyên bố zero leakage hoặc KD superiority. Missing continuous scores không được suy AP/ECE từ confusion. Checkpoint-only replay chỉ được công nhận khi đã thực chạy; file tồn tại không phải inference evidence.
Không retrain, restart Phase3B, mở protected TEST_SEALED/GPG/replication, tải dataset/mô hình mạng, push remote hoặc triển khai production. Không tìm thấy materialized Phase3B rõ ràng trong scope discovery; không dựng artifact thay thế.
Legacy leakage tests đọc sealed/replication bị loại khỏi command; suite mới kiểm speaker disjointness chỉ trên ba authorized splits. Exact test counts và final review nằm ở `coordination/EVAL_V3_ACCEPTANCE_TESTS.log` và `reports/EVAL_V3_FINAL_REVIEW.md`.
