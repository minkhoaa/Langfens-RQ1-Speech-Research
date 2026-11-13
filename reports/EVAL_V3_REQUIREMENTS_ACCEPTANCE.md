# EVAL-V3 — Ma trận acceptance khi đóng task

Trạng thái cuối theo scope được phép; `CONDITIONAL`/`BLOCKED` không được đổi thành PASS để đủ danh sách. Báo cáo chính: `reports/EVAL_V3_CLOSURE_REPORT.md`.

| Yêu cầu | Kết luận | Verification surface |
|---|---|---|
| Giữ nguồn/lịch sử và định danh exact-ID | PASS trong scope | Source freeze; `closure/CLOSURE_RECEIPT_AUDIT_FINAL.json`: 1490 verified, 9 protected paths không mở lại |
| Một canonical evaluator, target/gap ontology, correct FP/FN/TN/FWR | PASS | `research/evaluation_v3/evaluator.py`; 159 tests; publication verification |
| Kiểm kê và disposition mọi inventory row | PASS | `closure/inventory_final/`: 290 original rows, không source candidate pending; publication thêm 1 variant có identity riêng |
| Frozen inference cho các float checkpoints recoverable | PASS, không mặc nhận historical parity | 50 legacy NN receipts, CUDA; source/head mode ghi riêng |
| Frozen teacher inference đúng native precision | PASS | 9 AMP CUDA receipts; 9/9 SEARCH ledger delta0 |
| Waveform, feature và student-score parity | PASS có lineage caveat | `closure/waveform_full/`: 2699 WAV; feature delta0; 28/28 student comparisons delta0 |
| Intended reference từ independent orthography | CONDITIONAL | 2660 passes/39 OOV abstentions; annotation-derived canonical references chưa xác minh độc lập |
| Gold-free inference routing | PASS cho accepted new mode; lịch sử có findings | Source validity audit: 38 gold-conditioned methods; None-head variants tách khỏi lịch sử |
| Deterministic controls | PASS | 3 full CAL/SEARCH receipts; 3/3 legacy500 sample parity delta0, không gọi full historical parity |
| Composition/cascade/rule policies | PASS như canonical variants mới | 24 composition/router + 6 policy receipts; CAL-first; no source-param/historical-parity fabrication |
| Calibration fit chỉ CAL, identity/speaker bindings | PASS | Four confidence bundles, calibration provenance tests; score/label/ID/mapping hashes |
| Raw AP không đổi chỉ vì decision-policy đổi | PASS | `release_final/PUBLICATION_VERIFICATION.json`: 94 score-vector groups; CPL I-bank changes có vector identity khác |
| Full PR/recall/FP-count data | PASS, semantic scope rõ | Four lossless chunked PR tables; per-class FP density không tự là joint policy FWR |
| Top-K invariant dưới shuffle/ties | PASS | `classification_selective_v2/CANONICAL_EVENT_ID_TOP_K.csv`: 682 rows; canonical numeric-slot tie-break test |
| Full-population classification-selective risk và abstention FN | PASS | 92 models/175 tasks/15447 curve rows; explicit abstentions không biến mất khỏi denominator |
| Reference family, CTC alignment, speaker và exact-gap context | PASS | `stratified_full/`: 9664 rows, đủ 92 models/175 tasks; frame-seconds test |
| Fine-grained error subtype | CONDITIONAL đúng dữ liệu hiện có | 3354 derived evaluation-only recall/FN rows; không có dataset-native subtype labels; không fake AP trên all-positive groups |
| Reliability trước/sau calibration | PASS | `reliability_figures_final/`: 92 SVGs, 341 supported-head panels; unsupported heads NOT_ESTIMABLE |
| Calibration/SEARCH drift, speaker/group diagnostics | PASS, không blind certification | Confidence bundles: strata, reliability bins, CAL→SEARCH shift; SEARCH development-exposed |
| Multi-seed statistical comparison | PASS nơi có matched seeds; không bịa seed | Baseline matched-seed statistics giữ nguyên; legacy single-checkpoint rows không có fake SD/CI |
| Independent source/adapter audit | PASS trong reviewed scope | 342/342 helper hashes; 50 receipts/100 outputs; provenance/source-chain reports |
| Independent numerical/PR audit | PASS trong reviewed scope | Five NN/composition models + teacher S_EXP01, exact counts; 119696 PR rows; direct composition delta0 |
| Runtime/params/VRAM | PASS như measured receipts, không uniform architecture benchmark | WAV invocation265.887s/486.606MiB, có cache reuse; cached-head timings tách end-to-end; không cascade speedup claim |
| Unified publication và leaderboards | PASS | `release_final/`: 1340 results, 840 new operating rows, 865 primary/matched rows, 291 experiment/seed inventory rows |
| Dynamic quantized native inference | BLOCKED | 5 files, dynamic CPU dispatch có/CUDA không; không CPU forward/dequantize |
| Original fitted F4 và historical missing seed runs | BLOCKED | 4 missing LR gates; EXP18/A/B/C102/103 thiếu persisted fitted artifacts, không retrain |
| M107 native cascade | BLOCKED | M96 INT8 dependency + native SEARCH threshold selection; không float substitution |
| Replication/external protocols | OUT OF SCOPE, đóng disposition | M32/46/M14 original cohort và external09/35/41/51/78; không truy cập/thay cohort |
| Independent blind validation/zero-leakage/deployment safety | CHƯA ĐƯỢC CHỨNG NHẬN | Reference independence chưa xác minh; SEARCH đã exposed; không mở sealed data hoặc deploy |
| Incident transparency và scope safety | PASS về disclosure/remediation, không xóa vi phạm đã xảy ra | `coordination/EVAL_V3_EXECUTION_INCIDENTS.md`; rejected CPU/draft evidence không dùng cho kết luận |

Không còn lane chạy nền cần tiếp tục. Các blocker trên cần artifacts/authorization/runtime contract mới; không được tự ý nới scope để đổi trạng thái. Không mở thêm iteration sau khi bàn giao.
