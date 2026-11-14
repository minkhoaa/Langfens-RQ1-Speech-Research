# EVAL-V3 — Leaderboard mô tả trên subset tương thích

Chỉ cached student scores cùng canonical manifest/ontology/feature lineage. DEV_SEARCH development-exposed; reference provenance full population UNVERIFIED. Không chọn policy bằng SEARCH; mọi CAL policy đã khóa trên DEV_CAL. Đây không phải independent test hoặc champion deployment.

## Năng lực ranking (ARGMAX row, policy không đổi AP)

| Model | Seed | AP S | AP O | AP I exact-gap | macro3_error_ap |
|---|---:|---:|---:|---:|---:|
| STU-05-KD-ALL | 101 | 0.368503 | 0.158438 | 0.193748 | 0.240230 |
| STU-09-KD-ALL-CALIBRATED | 101 | 0.367733 | 0.158278 | 0.192596 | 0.239536 |
| STU-04-KD-I-F1 | 101 | 0.366044 | 0.159076 | 0.193467 | 0.239529 |
| STU-06-CONF-GATED | 101 | 0.366773 | 0.158952 | 0.192510 | 0.239412 |
| STU-08-KD-I-DUAL | 101 | 0.366051 | 0.158872 | 0.192442 | 0.239122 |
| STU-07-KD-I-ALTERNATE | 101 | 0.366093 | 0.158776 | 0.191585 | 0.238818 |
| STU-02-KD-S-ONLY | 101 | 0.366889 | 0.159125 | 0.188288 | 0.238101 |
| STU-03-KD-O-ONLY | 101 | 0.364919 | 0.160098 | 0.185897 | 0.236971 |
| STU-01-GOLD-ONLY | 101 | 0.364602 | 0.159433 | 0.186715 | 0.236917 |
| STU-05-KD-ALL | 202 | 0.371976 | 0.145915 | 0.190946 | 0.236279 |
| STU-01-GOLD-ONLY | 202 | 0.370793 | 0.146159 | 0.186597 | 0.234517 |
| STU-05-KD-ALL | 303 | 0.360189 | 0.157529 | 0.181347 | 0.233021 |
| STU-01-GOLD-ONLY | 303 | 0.357661 | 0.158430 | 0.178910 | 0.231667 |
| STU-00-LIGHT-GOLD | 101 | 0.359436 | 0.138263 | 0.168230 | 0.221976 |

## Matched confidence operating policies

| Policy CAL | Budget | Model | Seed | macro3_error_f1 | Realized joint_fwr |
|---|---|---|---:|---:|---:|
| ARGMAX_DIAGNOSTIC | NOT_APPLICABLE | STU-05-KD-ALL | 303 | 0.303628 | 3.896494 |
| ARGMAX_DIAGNOSTIC | NOT_APPLICABLE | STU-05-KD-ALL | 101 | 0.303608 | 4.383973 |
| ARGMAX_DIAGNOSTIC | NOT_APPLICABLE | STU-05-KD-ALL | 202 | 0.302701 | 4.176962 |
| CAL_MAX_ERROR_MACRO_F1 | UNBOUNDED | STU-06-CONF-GATED | 101 | 0.306554 | 5.692821 |
| CAL_MAX_ERROR_MACRO_F1 | UNBOUNDED | STU-01-GOLD-ONLY | 202 | 0.306291 | 4.617696 |
| CAL_MAX_ERROR_MACRO_F1 | UNBOUNDED | STU-05-KD-ALL | 202 | 0.305625 | 4.602671 |
| CAL_RISK_BUDGETED | 0.1 | STU-02-KD-S-ONLY | 101 | 0.095438 | 0.086811 |
| CAL_RISK_BUDGETED | 0.1 | STU-05-KD-ALL | 202 | 0.094239 | 0.088481 |
| CAL_RISK_BUDGETED | 0.1 | STU-01-GOLD-ONLY | 202 | 0.093945 | 0.086811 |
| CAL_RISK_BUDGETED | 0.2 | STU-06-CONF-GATED | 101 | 0.125641 | 0.136895 |
| CAL_RISK_BUDGETED | 0.2 | STU-09-KD-ALL-CALIBRATED | 101 | 0.122456 | 0.143573 |
| CAL_RISK_BUDGETED | 0.2 | STU-07-KD-I-ALTERNATE | 101 | 0.119037 | 0.136895 |
| CAL_RISK_BUDGETED | 0.5 | STU-06-CONF-GATED | 101 | 0.212483 | 0.439065 |
| CAL_RISK_BUDGETED | 0.5 | STU-09-KD-ALL-CALIBRATED | 101 | 0.211554 | 0.432387 |
| CAL_RISK_BUDGETED | 0.5 | STU-07-KD-I-ALTERNATE | 101 | 0.209727 | 0.452421 |
| CAL_RISK_BUDGETED | 1.0 | STU-02-KD-S-ONLY | 101 | 0.264184 | 1.106845 |
| CAL_RISK_BUDGETED | 1.0 | STU-03-KD-O-ONLY | 101 | 0.259704 | 1.116861 |
| CAL_RISK_BUDGETED | 1.0 | STU-05-KD-ALL | 303 | 0.238920 | 1.110184 |
| CAL_RISK_BUDGETED | 2.0 | STU-06-CONF-GATED | 101 | 0.295410 | 1.854758 |
| CAL_RISK_BUDGETED | 2.0 | STU-09-KD-ALL-CALIBRATED | 101 | 0.293658 | 1.861436 |
| CAL_RISK_BUDGETED | 2.0 | STU-05-KD-ALL | 101 | 0.293332 | 1.873122 |

Các vị trí cao chỉ mô tả model trên development population; không chọn operating policy mới từ bảng. Full bảng giữ mọi seed/policy, không chỉ top3. Mean/sample SD và paired deltas: `MATCHED_SEED_STATISTICS.json`. Histories, teacher domains, oracle và superseded nằm ngoài subset chính.
