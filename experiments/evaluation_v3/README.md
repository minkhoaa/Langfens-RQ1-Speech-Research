# Artifact EVAL-V3

**Bàn giao cuối:** `closure/release_final/`; báo cáo `reports/EVAL_V3_CLOSURE_REPORT.md`. Có1340result rows/840new operating rows/92closure models và291inventory experiment-seed rows. Các file ở root được giữ nguyên như baseline500rows; không nhầm chúng với publication cuối. Task đã đóng; các blocker có bằng chứng được liệt kê trong acceptance report.

- `EXPERIMENT_INVENTORY.csv`:290 experiment/variant entries, mỗi entry một replay status.
- `RESULTS_FULL.csv`:500 rows, trong đó268 FULL_SCORE_REPLAY; primary subset98 rows/14 current student arm–seed pairs.
- `EVIDENCE_MANIFEST_SHA256.json`:initial1499 source hashes;930 output cũ giữ trong archive read-only. Không ghi đè freeze này khi tái lập.
- `PR_CURVES.csv`:1.567.419 complete tie-group PR points; `RISK_COVERAGE.csv`:275.951 points với full-population FN. Hai CSV lớn tồn tại local; bản versioned `.csv.gz` là lossless, hash trong `OUTPUT_MANIFEST_SHA256.json`. Dùng `gzip -dk FILE.csv.gz` nếu cần khôi phục CSV chưa tồn tại.
- `SCORE_CURVE_INDEX.csv`:liên kết268 full-score operating rows với84 source/class curves; policy-only changes không tạo score mới.
- `OPERATING_POINTS.csv`, `RECALL_FWR_CURVES.csv`, `FP_COUNT_RECALL_CURVES.csv`:frozen CAL/current và recorded historical policy decisions; không chọn policy bằng SEARCH.
- `RECONCILIATION_RECEIPT.json`, `REPRODUCTION_VERIFICATION.json`, `INDEPENDENT_AUDIT.json`:coverage, reproducibility và independent arithmetic evidence.
- `CORE_*`:output trung gian trước historical reconciliation; CLI tái tạo chúng, không phải leaderboard khác.
- `waveform/`:sampled genuine WAV + orthographic prompt/CMUDict + CUDA receipt/predictions. Không chứng minh full-population reference độc lập TextGrid.

Tái lập bằng `.venv` local, không training/network/holdout; xem `reports/EVAL_V3_REPRODUCTION.md`. DEV_SEARCH development-exposed; missing artifacts, superseded và non-comparable rows không bị ẩn. Không tuyên bố zero leakage hoặc production safety.
