# EVAL-V3 — CLOSED / bàn giao cuối

- Publication: `experiments/evaluation_v3/closure/release_final/`.
- Báo cáo đầy đủ: `reports/EVAL_V3_CLOSURE_REPORT.md`.
- Acceptance: `reports/EVAL_V3_REQUIREMENTS_ACCEPTANCE.md`.
- 92 closure models/variants: 50 NN, 9 teachers, 3 deterministic, 24 composition/router, 6 policies; 184 CAL/SEARCH score hashes verified. Baseline14 students giữ riêng.
- 2699 WAV parity, features delta0; 28/28 student split comparisons delta0; 9/9 teacher SEARCH native-AMP delta0.
- 1340 result rows (500 preserved +840 new operating evidence), 865 primary/matched rows; 291 inventory experiment/seed rows gồm290original+1newvariant.
- 92-model robustness:175domain tasks,9664strata rows,3354derived subtype recall/FN rows. Classification selective:15447points; canonical Top-K682rows. Reliability:92SVGs/341supported heads.
- 159 tests passed,6native NumPy warnings; all-family receipt audit PASS. Source freeze1490verified/9protected paths intentionally not reopened.
- Independent adapter342/342hashes; numerical five NN/compositions+teacherS;119696PR rows verified.
- Constraints/blockers giữ thật: F4missing fits;5dynamic-quantized checkpoint files CUDAunsupported;M107dependency/nativeSEARCHselection; protected replication/external protocols; missing persisted seed weights; independent reference/blind-test certification unverified.
- Accepted learned replays dùng CUDA; rejected synthetic CPU forwards và draft root writes được công bố trong incident log, không dùng làm evidence.
- Không training, push/deployment hoặc thêm research iteration để lấp blocker. Local commit kết thúc task; Git log là nguồn commit ID.
