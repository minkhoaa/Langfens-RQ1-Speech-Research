# Hợp đồng tích hợp EVAL-V3 v1

Nguồn chỉ đọc: `/home/khoa/Projects/New-Method-Research`, HEAD ban đầu `03e4968ffd8908782d0bf463b6d101e6b6872372`. Không sửa bất kỳ artifact cũ nào; không training, mạng hoặc protected TEST_SEALED/GPG. DEV_SEARCH là development-exposed.

Ba lane ghi độc lập `/tmp/langfens-evalv3-03e4968/{A,B,C}`. Parent duy nhất tích hợp vào repository. Không spawn thêm agents. Tất cả reports tiếng Việt.

## API v1

Package đích `research/evaluation_v3`. Lane A cung cấp `evaluator.py:UnifiedEvaluatorV3` với `evaluate(target_predictions, gap_predictions, gold_targets, gold_gaps, expected_utterances=None)` tương thích cấu trúc input `evaluator/unified_evaluator.py`, nhưng giữ dataset_id/speaker_id trong event key. Các alias input position/target_index, gap_position/gap_index được hỗ trợ. Không tạo score từ decision khi thiếu score. Có thể cung cấp thêm `evaluate_events` nhưng không thay API v1.

Output chứa metric IDs từ directive, per-class confusion integer, statuses và full population counts; lane B có thể flatten nested output. Lane A cung cấp policies hoạt động trên numpy `p_target (N,3)`, `p_gap (M,)`, chọn chỉ DEV_CAL, deterministic S/O arbitration. A ghi rõ API trong specification.md; B có thể tự viết adapter policies với contract này. Để không block nhau, B xây migration CLI hỗ trợ `--root`, `--output` và import evaluator sau khi tích hợp. B không gọi evaluator cũ để bịa AP. C tính arithmetic/AP độc lập, không import evaluator under test.

## Phân quyền

A: schemas/evaluator/policies/calibration/legacy_adapter, tests, specification và scientific protocol.
B: migration.py + inventory/replays/CSV/report generation; giữ toàn bộ old rows dưới legacy JSON/columns. Tạo code trước, chạy sau integration nếu cần. Inventory không được chỉ gồm minimum names.
C: independent_audit.py + independent audit JSON/report, provenance/known defects/waveform-readiness investigation. Không neural forward trong lane C (parent quản lý GPU).
Parent: source/hash freeze, genuine bounded waveform CUDA smoke nếu đủ authorized artifacts, tích hợp, chạy suite/replay, báo cáo tổng hợp, scoped commit.
