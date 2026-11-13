# RC2 H-005 — Hiệu chuẩn độ tin cậy và độ bền theo speaker

## Kết luận

**H-005: chưa được kiểm nghiệm đầy đủ; bằng chứng hiện có chỉ hỗ trợ một phần cho isotonic trên checkpoint EXP-24.** Chưa chạy thí nghiệm mới. Kiểm tra CPU đọc các artifact EVAL-V3 hiện hữu. DEV_SEARCH đã development-exposed nên các số bên dưới không phải xác nhận độc lập.

## Bằng chứng EVAL-V3 hiện có

Mapping isotonic canonical được fit trên DEV_CAL gồm 10.043 target và 10.343 gap thuộc hai speaker ERMS/THV. Receipt ghi rõ checkpoint selection cũng dùng DEV_CAL và đánh dấu calibration trên chính DEV_CAL là in-sample. Đánh giá chuyển sang DEV_SEARCH gồm 19.755 target và 20.354 exact-gap thuộc bốn speaker; đây là đánh giá held-out theo speaker nhưng vẫn development-exposed.

| EXP-24 head | Brier thô | Brier isotonic | Baseline theo prevalence | ECE thô | ECE isotonic |
|---|---:|---:|---:|---:|---:|
| Substitution | 0,101525 | 0,077436 | 0,081487 | 0,165060 | 0,076090 |
| Omission | 0,016898 | 0,011893 | 0,010468 | 0,020753 | 0,015668 |
| Insertion | 0,009312 | 0,005327 | 0,006104 | 0,010242 | 0,002080 |

Isotonic giảm Brier và ECE so với score thô ở cả ba head trong artifact này. Tuy nhiên, Brier omission sau hiệu chuẩn vẫn kém baseline prevalence-only. Đây là kết quả reliability trên một mapping/checkpoint và cohort phát triển bốn speaker, không chứng minh hiệu chuẩn tốt trên speaker hoặc corpus mới. Artifact hiện có ghi nhận isotonic; không có đối chiếu canonical giữa temperature, beta và isotonic. Không suy ra AP tăng: raw AP vẫn là chỉ số phân biệt chính, và isotonic có thể tạo ties.

## Hạng mục chưa đo

- Chưa xác nhận có dự đoán TRAIN out-of-fold với provenance speaker-fold để so sánh phương pháp hiệu chuẩn theo speaker. Điểm in-sample từ checkpoint đã học trên cùng speaker không hợp lệ làm đánh giá held-out.
- Chưa chạy temperature scaling, beta calibration, class-conditional alternatives, hay risk–coverage mới trong Cycle 2.
- Chỉ hai speaker ở DEV_CAL không đủ để khẳng định một ánh xạ isotonic tổng quát cho unseen speakers.
- Không fit mapping theo gold speaker identity tại inference.

## Tái lập và giới hạn

Các số hiện hữu được trích từ `experiments/evaluation_v3/closure/confidence_recovered_v1/CAL_SEARCH_SHIFT.csv`, chọn EXP-24, split `DEV_SEARCH`, ba head S/O/I và hai nguồn `RAW_NATIVE` / `CALIBRATED_HELDOUT_BY_SPEAKER`. Fit provenance nằm trong `calibrator_recipes.json`; evaluator có phiên bản 3.1 tại `research/evaluation_v3/evaluator.py`. SHA-256 lần lượt: CSV `8787404db4659cc4b1541cce5c838a9d7c0d8d87273c89d328ab8b38951103ef`; recipes `37914fc5cab1351fa7311f4421f29f8b6c997fb00c79007125dae151f043ce31`; evaluator `76e41f906e5d953ee55cee8156379644bdd96bf96efab69e991b87d7f857d4a2`. Đây là đọc kết quả closure, không phải lần chạy mới. Không có checkpoint, prediction hay metric artifact Cycle 2 mới; không có claim về TEST_SEALED.

Kế hoạch tiếp theo là tạo/kiểm tra frozen TRAIN OOF scores theo speaker-fold, fit từng calibrator bên trong outer folds, báo Brier/ECE/log loss và risk–coverage theo speaker, rồi khóa mapping trước mọi đánh giá DEV_CAL/SEARCH. Nếu không có OOF scores hợp lệ thì cần inference/checkpoint folds có provenance trước khi so sánh calibrators.
