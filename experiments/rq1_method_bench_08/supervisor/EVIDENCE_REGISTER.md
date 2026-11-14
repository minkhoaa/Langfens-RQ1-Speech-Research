# MB08 supervisor evidence register

Decision: may RQ1-TRAIN-PHASE-01 begin, and which methodology/architecture should it train? Owner: MB08 supervisor. Scope TRAIN speaker-grouped 4-fold, not independent deployment validation. Sources repeatedly use the same 12 speakers and are not independent cohorts.

| ID | Source | Scope / quality | Finding and implication |
|---|---|---|---|
| E1 | MB03 agent_c producer, raw NPZ, receipt | 59562 targets/61362 gaps, seed101, 4 held-speaker folds; source-order legacy IDs recoverable; no saved head checkpoints | Direct neural baseline is strong insertion comparator; retain if new aligner fails. Reproducibility is source/config/prediction, not checkpoint replay. |
| E2 | MB04 final/report/CTC source | Small two-speaker holdout + exposed dev history; no rerun | E4 improves matched weak CTC controls, not proof of frontier or full-system quality. |
| E3 | MB05 grouped E4 heads | 2 seeds x4 outer folds; outer checkpoints separately indexed in source receipts | E4 raw LLR ranks and softmax-simplex ranks differ; do not compare their AP as identical scorers. |
| E4 | MB06 fusion diagnostics | Same TRAIN events; pooled percentile normalization exploratory | Rank-mean apparent gain is not valid nested evidence; excluded from training-primary claims. |
| E5 | MB07 nested raw direct/fusion/inner predictions and 24 inner CTC checkpoints | Both seeds, exact IDs/labels/folds/checkpoint SHA independently verified by supervisor audit_inputs.py | Fixed/learned E4 improve direct H002 target AP; learned S ties MB03 PLAIN, O improves. No independent external confirmation. |
| E6 | MB07 warning source | Source bug verified: realized_target_FWR = total warnings/Nutterances | Historical FWR-labeled tables are not true false-FWR; preserve and supersede, never silently edit. |
| E7 | MB07 phone-prior source + actual train_reference | 39-d reference vs hardcoded last42; supervisor independently finds 14/59562 cases choose acoustic channel | Historical prior is not mathematically pure phone. Small observed incidence does not excuse the control label; correct future reference slice. No outer-speaker leakage introduced. |
| E8 | EVAL-V3.1 research/evaluation_v3 | 33 synthetic contract tests executed; target/gap topology and false wrong-type warnings tested | Use macro3_error, not older target-macro APIs. |
| E9 | MB08 supervisor assembly tests | 4 tests executed, exact IDs/topology/support | Wrong IDs even at equal row count fail closed. Full-system decisions remain missing without gap policy provenance. |
| E10 | MB08 Agent A initial return | No training/code; NOT_TESTED | Not a negative scientific result; resumed only unfinished implementation with pre-outcome recovery amendment. |

## Completed evidence additions

- E11: Agent A retained-child recovery completed both seeds/all four folds. Supervisor independently replayed all insertion metrics, checked eight checkpoint hashes and 13 tiny forward/backward fixtures (`new_insertion_audit.json`). Both seeds lose to MB03; no universal alignment-negative claim.
- E12: Agent B completed true inner false-budget selection and corrected duplicate-budget pooled denominators without refitting. Supervisor checked 960 distinct fold policies, 240 pooled policies and independently reproduced 192 selected/fallback inner thresholds (`policy_verification_receipt.json`).
- E13: Agent C completed independent keyed baseline, metric/hash audit and architecture alternatives. Supplied intended-phone conditioning is supported; rawtext-only upstream lineage remains a distinct bounded concern, not proven prohibited leakage.
- E14: Supervisor assembled selected and ranking-alternative complete score streams, with valid AP but null full-system decision metrics because insertion policy provenance is missing.

Final decision and authority: `reports/METHOD_BENCH_08_READINESS_GATE.md` = CONDITIONAL_GO. Exact chosen training protocol: `reports/METHOD_BENCH_08_TRAIN_PHASE_01_SPEC.md`; independent draft alternatives are not the authoritative selected protocol. No formal training launched.
