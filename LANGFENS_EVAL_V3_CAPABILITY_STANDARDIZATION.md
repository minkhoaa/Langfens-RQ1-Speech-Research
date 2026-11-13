# LANGFENS EVAL-V3 — Scientific Capability Evaluation & Full Experiment Reconciliation

## Autonomous implementation directive for Pi/OMP

Act as the LANGFENS Scientific Evaluation Director. Implement, test, and apply **EVAL-V3** across all recoverable relevant experiments in the existing repository:

`/home/khoa/Projects/New-Method-Research`

**Execute actual code changes and evaluation replays now. This is not a research proposal.** Preserve all historical evidence. Do not retrain models or restart the stopped Phase 3B campaign. Reuse saved full-population scores and frozen checkpoints, performing inference replay only when essential and permitted by available artifacts.

### 0. Source safety and scope

1. Inspect current `pwd`, Git HEAD, branch, status, active sessions, uncommitted user changes, and repository instructions. Do not assume the historical Git refs remain HEAD. Do not reset, force-checkout, discard, or overwrite anything. Do not push to remote.
2. Read Phase 1/2/3 acceptance reports, Phase 3 diagnostic report from commit `45cd3d2`, evaluator and experiment registry, prediction schema, source caches, and test suites. Source paths mentioned below are **discovery hints**, not an instruction to invent missing files.
3. Freeze SHA-256 hashes of all input manifests, predictions, checkpoints, features, evaluator code, and selected configs. Keep a read-only copy of old metric outputs and a source revision receipt.
4. Respect protected TEST_SEALED and encrypted GPG archives. Do not decrypt, restore, probe, or construct substitute versions of protected holdouts. Only authorized L2-ARCTIC research splits may be used. No new network datasets, cloud/Colab, external publication, or production deployment.
5. Run three independent workstreams concurrently when using native Pi/OMP agents: (A) evaluator + tests, (B) registry migration + replay, (C) scientific audit + independent metric verification. Use isolated working files; share interfaces through versioned contracts. The Phase 3B training agents remain stopped.
6. Minimize repeated CUDA forward passes. GPU-first for any necessary neural inference replay, CPU for metrics/statistics. Do not run neural training as part of EVAL-V3.

### 1. Scientific objective and non-negotiable priorities

The primary research objective is **reliable discrimination of real SUBSTITUTION, OMISSION, and exact-gap INSERTION errors**, independent of a single strict FWR threshold.

**Scientific validity gates (must be met or explicitly marked unverified):**
- Genuine waveform + intended reference inference; confirm real audio-to-score path rather than relying solely on precomputed feature tensors.
- Non-oracle input contract: no gold TextGrid timing, realized-phone transcript, or gold error labels at inference; distinguish gold labels legitimately used for fitting and evaluation.
- TRAIN/DEV_CAL/DEV_SEARCH speaker disjointness, provenance, unchanged full-population event IDs.
- Full N target phones and N+1 gaps for each utterance; no selective dropping. Report how any abstention affects FN/coverage.
- Exact target-position classification and exact utterance-plus-gap insertion matching.
- Reproducibility including hashes, seed, model/config version, evaluator revision, score semantics, and input type.

Any failure of a validity gate is a **validity/status finding**, not a reason to silently exclude the experiment or freeze all independent workstreams.

### 2. Canonical data and event ontology

Verify authoritative manifests, do not blindly trust a report:

| Split | Utterances | Speakers | Targets | Gaps | Gold S/O/I |
|---|---:|---:|---:|---:|---|
| TRAIN | 1,800 | 12 | 59,562 | 61,362 | 7,381 / 1,825 / 560 |
| DEV_CAL | 300 | 2 | 10,043 | 10,343 | 1,781 / 462 / 102 |
| DEV_SEARCH | 599 | 4 | 19,755 | 20,354 | 1,768 / 209 / 125 |

DEV_SEARCH phone labels reported: CORRECT=17,778, SUBSTITUTION=1,768, OMISSION=209. Gap positive INSERTION=125. Correct earlier mistaken narrative totals of 23,044 targets / 23,643 gaps without rewriting old reports. Preserve original values in legacy columns and annotate their origin.

Canonical event key: `dataset_id:speaker_id:utterance_id:target|gap:slot_index`. Target domain = N mutually exclusive labels `CORRECT/SUBSTITUTION/OMISSION`. Gap domain = N+1 binary labels `NO_INSERTION/INSERTION`, including leading and trailing gaps. Never collapse a gap into a target slot or use count-only insertion matches.

**Scores are not always probabilities.** Store source-specific raw logits/scores and calibrated probabilities in distinct fields. Binary teacher scores for S and O are not automatically a valid three-class simplex.

### 3. EVAL-V3 metrics — precise definitions

Implement and document a single versioned evaluator entrypoint and a canonical scoring schema. Reuse the existing `UnifiedEvaluator` or create a compatible wrapper; avoid a second, inconsistent scoring implementation.

**A. Threshold-free discrimination (PRIMARY):**
- `ap_substitution`, `ap_omission`: average precision, one-vs-rest over all **target** events.
- `ap_insertion_exact_gap`: average precision over all **gap** events with exact gap IDs.
- `macro3_error_ap = mean(AP_S, AP_O, AP_I)` with clear note that target and gap domains differ.
- `binary_md_ap`: target-level error probability ranking for `SUBSTITUTION OR OMISSION` vs `CORRECT`; define score (e.g. `p_sub+p_omis` for valid simplex or a documented fused score). This is not an insertion metric.
- `roc_auc_*` supplemental. Explicitly separate sklearn-style average precision (stepwise AP) from trapezoidal PR-AUC if both are reported; never silently label one as the other. Preserve score ties. Record prevalences, positive supports, score directions, and normalized lift relative to prevalence (descriptive only).
- Missing continuous scores -> `NOT_ESTIMABLE`, never infer AP from a single confusion matrix.

**B. Decision metrics (PRIMARY at named operating points):**
- TP/FP/FN/TN, precision, recall, F1, support for S, O on target positions and I on exact gap positions; print raw integer confusion cells.
- `correct_precision`, `correct_recall`, `correct_f1` on target positions, clearly marked secondary due to prevalence.
- `macro3_error_f1 = (F1_S+F1_O+F1_I)/3`.
- `macro3_target_f1 = (F1_C+F1_S+F1_O)/3`, retained only where useful for compatibility and **never** mislabeled `macro3_error_f1`.
- `macro4_mixed_domain_f1 = (F1_C+F1_S+F1_O+F1_I)/4`; provide legacy alias `macro_4_f1` with explicit warning this averages a target-domain task and gap-domain task; not a single 4-way softmax F1.
- `binary_md_precision/recall/f1` on target domain, positive iff gold is S or O and predicted error is S or O, irrespective of diagnosis class; do not include insertion.
- `diagnosis_accuracy_on_gold_error` and `correct_type_given_detected` optional with precise denominators; never substitute conditional diagnosis scores for end-to-end F1.
- Define handling of undefined precision/recall/AP for zero-support classes explicitly (`null` + status or formal documented convention), and do not hide unavailable values in macro statistics.

**C. Confidence quality:**
- Per-class reliability plots, Brier score, ECE with fixed bins and clearly specified binning/evaluation population; compare to constant-prevalence Brier baseline.
- Calibration source, method, fit split, held-out split, model-selection overlap and provenance. Calibration fitted and assessed on same DEV_CAL must be labeled `IN_SAMPLE_CALIBRATION`, not unbiased out-of-sample validation.
- Predeclared `precision_at_k`, `recall_at_k`, using fixed K and clearly specified event pool; report fixed K per class and/or normalized alert density (e.g. alerts per 100 utterances) to ensure comparability. Stable tie handling, no gold-aware ranking.
- Risk–coverage and selective prediction metrics with clear definitions: confidence-driven abstentions count towards full-population coverage/FN accounting, never silently remove examples from the evaluation denominator.

**D. Operational risk (SECONDARY, not an acceptance veto):**
- `joint_fwr = (FP_S+FP_O+FP_I)/num_utterances` **using the canonical predicted-error warning accounting**. Wrong-type error prediction is a false warning of the predicted class. Distinguish this from `false_alarms_on_gold_correct_only` and from FP_CORRECT; avoid double-counting one target event.
- `false_warning_count`, fraction of utterances with at least one FP, warnings/utterance distribution, false alerts per 1,000 eligible event positions, per-class FP.
- Complete PR curves, recall–FWR curves and FP-count–recall curves. Report realized FWR, not only nominal budget.
- FWR <= 0.20 remains one *reference point*, never a gating criterion for research model quality. Additional exploratory budgets 0.10, 0.20, 0.50, 1.00, 2.00 and unbounded where meaningful. No arbitrary claim a high-FWR mode is pedagogically safe.

**E. Robustness and comparisons:**
- Per speaker, per reference-phone/family (if stable ontology), support counts, error subtype, alignment characteristics.
- Matched seed mean and sample SD, paired deltas, limited speaker-cluster bootstrap clearly tagged exploratory with small number of clusters.
- CAL->SEARCH prevalence, score-distribution, and calibration shift diagnostics. DEV_SEARCH is speaker-disjoint but development-exposed; it is NOT an independent untouched final test.
- Latency/VRAM/model parameter count and end-to-end waveform latency, separated from cached-feature-only inference timing.

### 4. Confidence-based decision protocol

Separate **model ranking capability** from **operating-point selection**. For each frozen model:

1. Evaluate continuous raw scores and threshold-free AP on all eligible events without imposing FWR cap.
2. Define several *deployable* policies: `ARGMAX_DIAGNOSTIC` (no risk constraint), `CAL_MAX_ERROR_MACRO_F1` (thresholds chosen on DEV_CAL), `CAL_CLASS_SPECIFIC_CONFIDENCE` (distinct S/O/I thresholds plus deterministic S-vs-O conflict rule), `CAL_RISK_BUDGETED` (multiple budgets), and optional `CAL_PRECISION_PRIORITY`. No test-label tuning.
3. For target positions, S and O predictions are mutually exclusive after arbitration. Do not apply one shared S/O threshold by default. For gaps, retain separate insertion score/threshold.
4. Calibrate scores only using authorized development data; record any checkpoint/calibration/threshold reuse on DEV_CAL. Use no DEV_SEARCH gold to fit thresholds, calibrators, or weights.
5. A confidence >= 0.9 is **not** equivalent to 90% empirical correctness unless properly calibrated. Record calibrated and uncalibrated confidence separately.
6. Carry abstentions as explicit states, preserve denominators; define whether abstention produces no warning and how it counts in FN for each relevant class.
7. Keep `SEARCH_ORACLE_UPPER_BOUND` as *diagnostic only*, clearly labeled non-deployable and excluded from champion selection.
8. Preserve the original Phase 3 shared-threshold policy in `LEGACY_REPLAY` for audit, but do not let it define the new research optimum.
9. Do not select the best policy by inspecting DEV_SEARCH metrics; the frozen DEV_CAL selection must be performed first and then applied unchanged.

### 5. Experiment inventory and comprehensive migration

Build a **complete discovery inventory** from actual registries, reports, checkpoint metadata, and files; do not rely only on this list. At minimum include:

- UPR_BASELINE; EXP-A101/B101/C101; EXP-A102/B102/C102; UPR02_SYNTHESIS_CHAMPION; EXP-A103/B103/C103; UPR03_SYNTHESIS_CHAMPION.
- Teacher S Champion, Teacher O Champion, Teacher I EXP01 (I-F1), Teacher I Champion (I-Precision), and all available teacher validation/checkpoints and Phase 1 report variants.
- Phase 2 CPL-01 evidence (including reported omission union correction 85 vs 94), CPL-02 full policy family F0–F5, CPL-03 KD feasibility and linked predictions.
- Phase 3 STU-00 through STU-09 as available, Gold-Only and KD-All seeds 101/202/303, 7 historic FWR budgets, OOF teacher variants, confidence-gated/dual-I/calibrated variants.
- Phase 3 diagnostic counterfactual policies from commit `45cd3d2`, including separate S/O thresholds, class-only oracles, and score-policy diagnostics.
- Already materialized Phase 3B artifacts, if any, without restarting the intentionally stopped training agents.

Assign every registered experiment one unambiguous replay status:

`FULL_SCORE_REPLAY` / `CHECKPOINT_INFERENCE_REPLAY` / `COUNTS_ONLY_PARTIAL` / `MISSING_ARTIFACT` / `PROTOCOL_NONCOMPARABLE` / `INVALID_SUPERSEDED`.

- Prefer complete, frozen prediction ledgers. For available checkpoints without compatible scores, run bounded actual CUDA inference with authorized input features; perform waveform-to-output smoke/full tests where available and report whether full or sampled.
- Historical confusion-only reports may allow F1 arithmetic but **not** AP, ECE, score curves, or confidence analyses. Mark missing values explicitly.
- Preserve full old rows, original reported metrics and old evaluation version; compute `v3_*` columns side-by-side. Never silently change original reports or backfill inferred values as observed.
- Reconcile known past problems: old 23,044/23,643 typo vs manifest 19,755/20,354; baseline macro F1 inconsistencies; mistaken correct-F1 counts; omission union 85 vs 94; I-Precision vs historical B102/B103 precision; UPR03 FWR and target/gap class semantics; old macro3-target vs macro3-error mixing; Phase 2 oracle vs deployable outputs; superseded Phase 3 KD/GRU/threshold artifacts.
- Do not report `same dataset` simply because both rows say DEV_SEARCH: verify manifest hash, event IDs, label counts, evaluator type, feature/prediction provenance, and score semantics.
- Create a comparable subset leaderboard only for rows with matched cohort + ontology + event evaluator. Place other rows in a clearly marked non-comparable inventory.

### 6. Test suite and independent numerical audit

Implement automated tests for:

- Exact N target/N+1 gap accounting and leading/trailing gaps; utterance/gap cross-matching false positives; duplicate/missing event IDs.
- All-CORRECT and all-NO_INSERTION baselines; one perfectly correct event set; shuffled/permuted labels; wrong-type S-vs-O confusion; no-warning/abstain.
- Correct precision/recall/F1 from integer confusion counts; macro3-error, macro3-target, macro4 formulas; reproducible FWR with incorrect-error-type warnings counted once.
- AP correctness on fixed known tiny vectors, ties and reverse ranks; class imbalance/prevalence controls; AP vs trapezoidal AUC distinction; zero-positive or zero-negative edge cases.
- Threshold separate-vs-shared counterfactual from seed101: old Gold S/O/I TP=1/3/29, new separated 27/3/29 with FP counts as previously reported, if exact historical prediction ledgers exist; otherwise mark regression NOT_RUN, not PASS.
- Data leakage sentinels injecting gold-only timing/labels into inference input and ensuring pipeline refuses or ignores them; feature lineage and normalizer train-only tests.
- Speaker partition disjointness and calibration data-provenance checks.
- Stable results under row shuffling, reproducible sorting/tie-breaking, and source hashes.
- Legacy evaluator parity on a frozen fixture, and explanation of every intentional V3 definition change.

Use an **independent reviewer/verification agent** to recompute selected cases from raw event ledgers using separate short arithmetic code, not by simply calling the implementation under test.

### 7. Deliverables and paths

Create versioned, reproducible artifacts inside the existing repo. Suggested names, adapt to current layout:

```
research/evaluation_v3/
  specification.md
  schemas.py
  evaluator.py
  policies.py
  calibration.py
  legacy_adapter.py
  migration.py
  test_evaluator_v3.py
reports/EVAL_V3_SCIENTIFIC_PROTOCOL.md
reports/EVAL_V3_RECONCILIATION_REPORT.md
reports/EVAL_V3_COMPARABILITY.md
reports/EVAL_V3_LEADERBOARD.md
reports/EVAL_V3_REPRODUCTION.md
experiments/evaluation_v3/EXPERIMENT_INVENTORY.csv
experiments/evaluation_v3/RESULTS_FULL.csv
experiments/evaluation_v3/RESULTS_PRIMARY_COMPARABLE.csv
experiments/evaluation_v3/OPERATING_POINTS.csv
experiments/evaluation_v3/PR_CURVES.csv
experiments/evaluation_v3/RISK_COVERAGE.csv
experiments/evaluation_v3/HISTORICAL_VS_V3_DELTAS.csv
experiments/evaluation_v3/MIGRATION_COVERAGE.csv
experiments/evaluation_v3/EVIDENCE_MANIFEST_SHA256.json
coordination/EVAL_V3_PROGRESS_LATEST.md
```

Each `RESULTS_FULL.csv` record must include experiment_id, arm/seed, population, manifest hash, split/speaker count, label support, event domain, checkpoint/prediction hashes, evaluator version, data exposure flag, replay_status, scientific validity status, threshold policy and fit source, raw/calibrated score status, P/R/F1/AP for S/O/I, Correct F1, binary MD F1, macro3-error F1, macro3-error AP, macro4-mixed-domain F1, FWR, ECE/Brier when computable, latency provenance, reproducibility command, and missing-value reason.

All reports must be in **Vietnamese** with exact English metric IDs retained, and list concrete output rather than only generic process steps.

### 8. Acceptance criteria

EVAL-V3 is ready only when:

1. Canonical schema and formulas are documented and executable.
2. Deterministic unit, regression, leakage, and cross-population tests pass; attach exact counts.
3. All recoverable registered experiments have been inventoried and assigned a replay status. Do not claim 100% fully re-scored if missing scores/checkpoints remain.
4. Every `FULL_SCORE_REPLAY` row has AP, class-specific P/R/F1, correctly named Macro-3/4, full event counts, and provenance.
5. `COUNTS_ONLY_PARTIAL` rows do not contain fabricated threshold-free/calibration results.
6. DEV_CAL-selected deployable confidence policies are separated from DEV_SEARCH oracle diagnostics.
7. A single reproducible comparable leaderboard identifies the best models by **class AP, error Macro-3 F1, and matched confidence operating policies**; no universal FWR<=0.20 gate.
8. The same fixed checkpoints yield unchanged scores when only the decision policy changes; explain any intentional metric differences from historical reports.
9. Confirm a genuine WAV+reference inference path with traceable provenance, and label any cached-only row accurately.
10. Retain old artifacts, commit only scoped code/reports with clean provenance, and leave Phase 3B stopped.

### 9. Immediate execution sequence

- Inspect Git/source state, freeze hashes, and initialize three parallel evaluation workstreams.
- Implement evaluator/schema/policies and test fixtures before updating any leaderboard.
- Build inventory in parallel; replay existing full-population ledgers while tests are developed.
- Independently reconcile known arithmetic/event-definition defects.
- Run actual required bounded CUDA forward replays, without retraining.
- Apply EVAL-V3 to the complete recoverable experiment set; preserve all original results alongside new ones.
- Produce comparison matrices, PR/risk/coverage curves, per-speaker reports, missingness/provenance flags, and a Vietnamese scientific assessment.
- Deliver exact paths, output hashes, tests and replay commands; report which rows are fully re-evaluated versus partially recoverable.

**FINAL GOAL:** A trustworthy, versioned, consistent capability-first benchmark across LANGFENS, where scientific model discrimination is separated from confidence-policy behavior, all old experiments remain auditable, and no false claims of independent blind validation, zero leakage, or superior model performance are made.

**START IMPLEMENTATION NOW. DO NOT RESTART MODEL TRAINING.**
