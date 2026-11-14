# Kiểm toán validity inference native — AST hiệu chỉnh

Đã rà soát source tĩnh; không import/chạy model, không load checkpoint, không dùng dữ liệu holdout/protected. M12 replication branch bỏ qua. SHA256 từng file source thực sự quét nằm trong JSON. Các checkpoint chỉ được ghi nhận bằng đường dẫn metadata trong root, không đọc nội dung.

## Số lượng và phân loại
Tổng **108** method: `{"VERIFIED_GOLD_CONDITIONED_INFERENCE": 38, "VERIFIED_NO_FORBIDDEN_INPUT": 2, "STATIC_CANDIDATE_GOLD_CONDITIONING": 1, "UNVERIFIED": 67}`. Verified không đồng nghĩa toàn bộ phương pháp sạch: chỉ M31/M34 có kết luận no-forbidden-input bị giới hạn theo call path đã rà; upstream X lineage đầy đủ chỉ kiểm chứng qua helper ba utterance ở M31/M34/M38, không thể suy rộng ra 108 method.

`VERIFIED_GOLD_CONDITIONED_INFERENCE` gán cho M38/M48/M94/M104 và các checkpoint-bearing candidate chỉ sau trace riêng từng actual AST Call, prediction/evaluation context, no_grad/inference_mode, item/batch target_m binding, builder dependency/hash, và CAL/SEARCH source binding. File/line/hash cụ thể cùng chain nằm trong JSON. Các method còn lại có pattern AST đủ điều kiện nhưng chưa được rà interprocedurally là `STATIC_CANDIDATE_GOLD_CONDITIONING`, không kết luận leakage. Huấn luyện-only call không được verified.

M38: gate `run.py:429-430`; gold builder `:251-262`; inference `predict_sequences :558-567`. M48: gate `:85-92`; evaluation call `:412-425`. M94: gate `:206-210`; eval call `:415-427`. M104 imports M94 class `:68-78`, eval call `:260-271`, saves own EXP-104 checkpoint `:192`; source does not prove transferred checkpoint. AST call/provenance rows include exact source hashes.

M31/M34 clean calls are bounded claims only; M34 TRAIN-fitted priors are not per-example inference labels. Parent evidence of full-population feature replay 0 delta and 28/28 scores 0 delta is acknowledged but was not reproduced here and cannot certify native score callsites. Native `target_m=None` fallback, if used, is a distinct frozen-weight inference mode—not historical parity.

Đính chính feature: chỉ competitor margin dim 46 lệch (-10 student, -15 legacy); silence margin dim 47 khớp (-15).
