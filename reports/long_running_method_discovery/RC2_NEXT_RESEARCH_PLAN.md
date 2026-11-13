# LANGFENS RC2 — kế hoạch nghiên cứu tiếp theo

## Ưu tiên 1: kiểm tra độ bền của H-001

1. Giữ immutable frozen-emission run và tái dựng A/B ở speaker-grouped TRAIN folds để ước lượng biến thiên theo speaker, không dùng in-sample calibrator cho claim reliability.
2. Thêm một ablation để collapse/điều tiết hai SIL state kề quanh phone bị xóa; giữ target/gap keys không đổi.
3. Báo cáo score ranking, operating points CAL-frozen, short-correct false deletion, repeated-phone strata và mọi speaker.
4. Chỉ dùng tên CTC likelihood khi tìm được checkpoint phoneme-CTC có objective/token/blank provenance được xác minh; hiện cached CharSiU frame classifier không đạt hợp đồng này.

## Ưu tiên 2: neural arms sau khi CUDA phục hồi

Kiểm tra live driver, GTX 1650 capacity, CUDA lease và active Phase 4 trước mỗi job. Giữ tối đa ba CUDA-heavy jobs và giảm concurrency theo VRAM/throughput thực tế.

- H-002: xác nhận checkpoint pretrained-only local; nếu chỉ dùng ASR-finetuned Wav2Vec2 thì đăng ký thành sensitivity arm riêng. Cache một lần theo immutable spec; matched small head và acoustic/reference controls.
- H-006: chốt place/manner/voicing ontology trước training; auxiliary branch không được dùng reference one-hot trong acoustic-only control.
- H-004: bắt đầu Gold-Only/raw-OOF scalar-KD matched seeds; dùng fold-local preprocessing/calibration. Chạy feature transfer/disagreement riêng từng intervention sau provenance gate.
- H-003: chỉ tiếp tục nếu có lý do mới sau local-pooling âm tính; attention, energy/voicing và transition trajectory là giả thuyết riêng, không nhập chung.

## Ưu tiên 3: calibration và xác nhận độc lập

- Tạo frozen speaker-grouped TRAIN OOF probability scores; so sánh raw/temperature/isotonic/beta theo outer held-out folds.
- Đóng băng mapping trước DEV_CAL/SEARCH; giữ raw AP làm ranking chính, và ghi riêng calibration ties.
- Tìm một cohort speaker độc lập được phép trước claim generalization; không dùng TEST_SEALED.
- Giải quyết provenance của intended reference/G2P/OOV và variant selection trước claim reference independence.

## Quy tắc quyết định

Không tích hợp thay đổi Cycle 2 vào Phase 4 khi chưa có phối hợp/ủy quyền riêng. Không coi SEARCH cao hơn là winner; không đặt FWR 0,20 làm cổng loại. Duy trì event population N/N+1 và evaluator EVAL-V3. Giữ kết quả âm tính H-003 và mọi lần chạy lỗi trong versioned namespaces.

## Cập nhật sau các screen CUDA130 RC2

Các ưu tiên phía trên là kế hoạch ban đầu, được ghi trước khi GPU được khôi phục. Các screen một seed đã chạy và chỉ được dùng như kết quả SEARCH thăm dò:

- **H-002 là ứng viên cần xác nhận tiếp:** frozen Wav2Vec2 ASR-finetuned layer-6 attention target head đạt AP S/O/I `0.377810 / 0.167294 / 0.186715`, Macro3 AP `0.243940` so với Gold-Only `0.236917`. Gap score giữ từ checkpoint Gold-Only. Ở CAL budget 0.10, Macro3 F1 `0.054638` so với `0.034250`. Dùng paired grouped TRAIN folds, reference-only model được huấn luyện lại matched, và ba seed trước khi tuyên bố cải thiện ổn định. Checkpoint/prediction hashes, runtime provenance và limitations ở [H-002 report](RC2_H002_SSL.md), [H-002 EVAL-V3.1 receipt](../../research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/EVAL_V3_RECEIPT.json).
- Một phép ghép score độc lập đã nối target H-002 với gap Gold-Only theo `(utterance_id, slot_idx/gap_idx)`, xác nhận key duy nhất/đầy đủ, không tạo TRAIN predictions. Reproduction: `../../research/long_running_method_discovery/cycle_02/unified_candidate/assemble_screened_bundle.py`, sau đó `evaluate_screened_bundle.py`. [Screened bundle manifest](../../research/long_running_method_discovery/cycle_02/unified_candidate/RC2_SCREENED_UNIFIED_BUNDLE.json) phân biệt rõ score bundle với checkpoint thống nhất. EVAL-V3.1 tái lập đúng H-002 ở budget 0.10 và 0.20.
- **H-001 optional scalar + acoustic features:** Macro3 AP `0.229014`, thấp hơn Gold-Only; không tích hợp. H-001 frozen max-sum A/B vẫn là kết quả ranking exploratory tách biệt, chưa tạo O F1.
- **H-004 raw OOF scalar KD:** Macro3 AP `0.239828` nhưng F1 ở budget 0.20 `0.104513` so với `0.104897`. **Feature KD:** Macro3 AP `0.231038`, cả S/O/I AP đều giảm. Chưa tích hợp; disagreement objective chưa thử.
- **H-006 combined articulatory auxiliary:** Macro3 AP `0.239060`, mixed per-class deltas; F1 ở budget 0.20 `0.113712` so với `0.104897`. Một seed và bốn speaker SEARCH chưa đủ để chọn tích hợp.
- **H-003 và H-005** giữ kết luận/artefact hiện tại trong [RC2 hypothesis results](RC2_HYPOTHESIS_RESULTS.csv); không suy rộng từ các nhánh mới.

### Thứ tự công việc kế tiếp

1. Xác nhận intended-reference provenance và dùng speaker-grouped TRAIN folds cho H-002; chỉ dùng local checkpoint có quyền sử dụng. Tách encoder pretraining-only và ASR-finetuned thành hai arm nếu checkpoint hợp lệ xuất hiện.
2. Chạy ba seed matched cho H-002 target head và Gold-Only; báo cáo từng seed, mọi speaker-fold, AP class-wise/Macro3, confidence, operating curves và compute. Chọn kiến trúc/hyperparameter trên TRAIN grouped folds, khóa trước DEV_CAL; SEARCH chỉ mô tả.
3. Nếu H-002 vượt qua xác nhận, huấn luyện một mô hình unified mới để dự đoán target và exact-gap cùng lúc; không dùng score assembly làm bằng chứng rằng đã có một checkpoint chung. Giữ đúng N+1 gaps, và so sánh với Gold-Only trong cùng seeds/budget.
4. Chỉ thử tích hợp H-006 hoặc KD nếu có một giả thuyết mới từ error analysis; mỗi thành phần thêm vào sau ablation đơn lẻ, không fusion hàng loạt. H-001 neural scalar hiện âm tính.
5. Đánh giá calibration bằng TRAIN speaker-grouped held-out mapping rồi khóa mapping; DEV_CAL chỉ cho policy/calibration cuối. Báo FWR ở nhiều điểm như metric bổ sung, không dùng ngưỡng 0.20 làm cổng thành công.
6. Không sửa Phase 4 hay khẳng định TEST_SEALED/generalization. Intended-reference independence và external speaker validation vẫn là blockers khoa học cần giải quyết trước khuyến nghị triển khai.
