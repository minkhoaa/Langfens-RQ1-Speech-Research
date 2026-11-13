# EVAL-V3 closure — Hợp đồng v2

Nguồn read-only cho workers: `/home/khoa/Projects/New-Method-Research`, HEAD `cc4e33117a73974859c0dc2cdea66a823be40751`. Mốc này có69tests,290 inventory entries/500 result rows/268 FULL_SCORE rows. Operator yêu cầu tiếp tục đóng toàn bộ yêu cầu, không xem mốc commit là kết thúc.

Ba lane độc lập chỉ ghi `/tmp/langfens-evalv3-closure-cc4e331/{A,B,C}`. Parent duy nhất ghi repository và điều phối CUDA. Không child agents, training, network dataset/model download, protected TEST_SEALED/GPG/replication hoặc Phase3B restart. Không mở external corpus dù tên experiment nhắc corpus đó. Với protected historical administrative experiment, chỉ dùng registry metadata hiện có, không probe nội dung holdout.

A: evaluator/test/acceptance gap audit; đề xuất và implement isolated fixes/tests khi có defect thực; không viết migration/inventory/provenance lane khác.
B: kiểm23MISSING +48PARTIAL và76non-comparable, tìm artifact/linked metadata thực trong authorized root; phát hiện administrative entries không phải experiment model; tạo recoverability evidence và adapter code nếu có đường frozen replay không retrain. Không CUDA trong child. Không kết luận unavailable chỉ vì lookup đầu tiên sai đường/schema.
C: provenance audit toàn2699authorized manifests; text→local CMUDict independently, quantify OOV/phone-sequence mismatches và distinction annotation-derived intended reference vs prohibited timing/realized/labels at inference. Viết script/output/report isolated. Không neural forward. Không đổi canonical gold/IDs.

Parent: bounded WAV pilot, sau đó mở rộng waveform/feature/checkpoint inference nếu cần và có evidence; chốt acceptance matrix từng yêu cầu, cập nhật reports/data với legacy receipts bảo tồn và thêm scoped commit. Baseline hash của85file V3 trước closure: `experiments/evaluation_v3/closure/CLOSURE_BASELINE_SHA256.json`. Không ghi đè initial1499source freeze.
