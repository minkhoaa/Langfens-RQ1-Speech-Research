# Final executed protocol stratum V2

Binding definitions: [`../BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md) and frozen [`EVALUATION_CONTRACT_V1.md`](../../RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md). Neither historical source nor contract was modified.

- All4checkpoints: same original head/frozen backbone, same batch1CPUFP32,4threads, cached exact original encoder features; both matched and same-speaker manifest-order cyclic-shift1 audio. No model fitting, backward/optimizer or TEST.
- VAL300: NJS/RRBI150each, fixed manifest order. CAL300: NCC/THV, correct-score-only threshold order statistics at5/10/15%FPR. No VAL calibration. Every supported score has separately labeled retrospective VAL diagnostic points.
- O: annotation-canonical9956positions. Z: strictly waveform+frozen written prompt/CMUDict first pronunciation; runtime hashes frozen before annotation access; original `sub_first` offline mapping. Z9919aligned,37unmapped gold abstentions,37extra prompt rows, every9956gold position accounted for. No fullgold AP imputation or O/Z ranking.
- T-OP operation_ce: binary s_error plus identified uncalibrated subtype scores. B-Control binary_collapsed: only s_error identified; subtype scores/metrics null with unavailable semantics.
- O and Z share saved query-independent CTC hypotheses against9881realized reference phones. Edit S/D/I does not represent human annotated omission/substitution event counts.
- Offline word-final consonants follow frozen midpoint-to-word rule;1871gold/95omissions,1866aligned/93omissions. Both speakers separately reported;G=2 is not population significance.
- Canonical-only frozen FIT1799 prior is unavailable; no replacement fit. Historical E0/P-SCORE are archival-only, separate from this new numerical stratum. E0 trained on TRAIN2399 and is not the matched FIT1799 control.

Exact hashes/source/identities/commands: `RUN_MANIFEST.json`, `RECOVERY_MANIFEST.json`, `CHECKPOINT_VERIFICATION.json`, `RAW_ARTIFACT_INDEX.json` and [`../REPRODUCTION_REPORT.md`](../REPRODUCTION_REPORT.md). Results: [`BENCHMARK_COMPARISON.md`](BENCHMARK_COMPARISON.md). Private persistence receipt also covers recomputed outputs generated after the inference manifest. Full-contract completion remains PARTIAL solely for the unexecuted required frozen prior control, not an artifact/data-inference access blocker.
