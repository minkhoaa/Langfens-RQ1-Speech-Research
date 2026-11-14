# Agent B continuation — joint S/O/I policy and MB03 gap refits

## Status and classification

**MEASURED bounded mixed-source audit; NOT a formal Phase-01 training run or independent cohort.** The previous handoff and freeze receipt were preserved unchanged. The source/provenance blockers stated in that earlier handoff were corrected: `experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py` supplies the exact producer (`get_arrays`, `Detector`, `fit_predict`), and the parent supplied independently verified eligible target-inner predictions. Sources, exact fit/held event IDs, seeds and inputs were amended/frozen in `results/continuation_freeze.json` before gap fitting; target-source and inner-gap inputs/policy code were frozen in `results/policy_freeze.json` before cutoff selection; cutoff JSON was frozen before any outer gap fitting; evaluation code/inputs were frozen in `results/evaluation_freeze.json` before outer policy application.

No target or CTC model was trained. Reused target scores are archived MB07 seed101 predictions; new models are MB03 PLAIN-style exact-gap fits. No GPU, protected/TEST_SEALED data, DEV_SEARCH tuning, or out-of-lane writes were used.

## Recovered MB03 procedure and execution

Imported the legacy producer with `importlib` and called `get_arrays()` only (no legacy `main`). Training uses the producer's exact `Detector` class and the PLAIN `fit_predict` training semantics: fit-only mean/std normalizer with `sd=max(std,1e-4)`; `sqrt(n/(2*max(class_count,1)))` weights normalized to mean one; weighted cross-entropy; AdamW `lr=8e-4, wd=1e-4`; batch 1024; fixed 5 epochs; `PLAIN` task loss only. The otherwise-unused speaker branch is instantiated and the forward call returns speaker logits, preserving its parameter initialization and RNG/dropout sequence. The only protocol change is the required phase-01 seed schedule.

- **12/12 inner gap fits MEASURED**, 6 fit speakers / 3 held validation speakers per fit, seed `101+10*outer+inner`; inner predictions cover each outer-training gap once per outer fold (184,086 keyed rows across four eligible pools, 61,362 unique TRAIN gaps, three outer-pool predictions per gap).
- **4/4 outer gap refits MEASURED**, 9 fit speakers / 3 untouched outer speakers, seed `201+outer`; all 61,362 held-out gap IDs exactly once.
- Combined refit scope: **33.57 CPU seconds** (inner receipt 21.74s plus outer receipt 11.83s), versus 1,200-second hard cap. CPU, two threads. No cap truncation.
- Each checkpoint retains exact fit/held event IDs, speaker IDs, state dict, fitted mean/std, class counts/weights, seed and recipe. Checkpoint SHA-256 and normalizer SHA-256 are in per-fit receipts. Raw keyed score arrays and command logs/exit receipts are retained in `results/`.

## Frozen policy and outer results

Joint warning policy uses inner OOF only: B∈{.2,.5,1,2} false warnings/utterance, allocated .8 target/.2 gap; target false includes any S/O wrong type; gap false is insertion on gold-negative; SHA tie order is descending first-eight SHA-256 bytes for `LANGFENS-MB07-warning-tie-v1:` + event ID. For every outer fold, all target and gap inner rows were checked for exact-once coverage and **per-event** fit exclusion (each prediction's fit speakers exactly equal the outer-training speakers minus that inner validation group). 48 cutoffs were frozen with fit IDs, thresholds, SHA cutoffs, inner selected counts and FP counts before outer application. Direct, E4-only and fixed .5 neural/E4 target comparisons share the same new gap score stream.

| Target stream | AP-S | AP-O | AP-I (shared) | Macro-3 Error AP | Macro-3 Error F1 at B=.2 | Joint FWR at B=.2 | Outer warnings / 1800 | Wrong-type FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct H-002 | .384080 | .265655 | .174642 | .274792 | .110711 | .143333 | 624 | 9 |
| E4-only | .272225 | .229009 | .174642 | .225292 | .140457 | .280000 | 887 | 99 |
| Fixed .5 neural + .5 E4 | .402742 | .330783 | .174642 | .302723 | .161780 | .202222 | 920 | 29 |

B=.2 classwise P/R/F1 (same shared gap stream):

| Stream | S P/R/F1 | O P/R/F1 | I P/R/F1 |
|---|---|---|---|
| Direct | .7116 / .0308 / .0590 | .6087 / .0307 / .0584 | .3897 / .1482 / .2147 |
| E4-only | .5029 / .0117 / .0228 | .4254 / .1173 / .1838 | .3897 / .1482 / .2147 |
| Fixed | .7578 / .0331 / .0634 | .5948 / .1255 / .2072 | .3897 / .1482 / .2147 |

The shared gap stream at B=.2 has 83 TP / 130 FP / 477 FN (AP-I .174642). Target plus gap false-warning counts are wrong-type-inclusive. FWR is not interchangeable with total warnings: pooled achieved FWR at B=.2 is .1433 / .2800 / .2022 for direct/E4/fixed, while total warning counts are 624 / 887 / 920. At all budgets the results and exact counts are in `results/joint_policy_pooled.csv`; AP is invariant across operating budgets, while F1/FWR/warnings vary. These are inner-nominal budgets, **not matched achieved outer FWR**; do not claim exact deployable-rate superiority.

Descriptive speaker uncertainty: at nominal B=.2, fixed-stream achieved speaker FWR ranges .0867–.3600 (mean .2022, sample SD .0840); direct .0267–.2733; E4 .1067–.4733. Fold and speaker breakdowns are in `joint_policy_fold.csv` and `joint_policy_speaker.csv`. This is repeated TRAIN grouped-OOF variability, not independent-cohort confidence. No bootstrap was run. Full primary metrics, all four budgets, supports, class TP/FP/FN, warnings, source hashes and output hashes are in `joint_policy_results.json`.

## Raw artifacts and verification

- `results/inner_predictions.npz`, `outer_predictions.npz`: keyed raw gap probabilities, labels, speaker/fold metadata; inner has per-event outer+inner IDs.
- `results/gap_inner_*.pt`, `gap_outer_*.pt`: 16 checkpoints plus scaler and exact fit/held IDs.
- `results/inner_receipt.json`, `outer_receipt.json`, `inner_run.log`, `outer_run.log`, `*_exit.txt`: fit counts, seeds, timings, hashes, exact status/exit.
- `results/thresholds_frozen.json`: 48 frozen thresholds and exact fit IDs / SHA cutoffs / budget receipts.
- `results/joint_policy_raw_keyed.csv` (1,451,088 keyed application rows), `joint_policy_pooled.csv`, `joint_policy_fold.csv`, `joint_policy_speaker.csv`, `joint_policy_results.json`.
- `results/completed_run_audit.json`: post-run audit verified source hashes, fit schedule, checkpoint hashes, per-event speaker exclusion, fit-ID counts, threshold/prediction coverage, raw keyed rows, and 33.57s total scope.
- The parent-provided `preflight/supervisor/inner_target_verification.json` states `VERIFIED_ELIGIBLE_KEYED_INNER_TARGETS`; its NPZ SHA-256 is recorded in policy/evaluation receipts. Existing MB07 target/CTC sources remain read-only.

Tests: six `unittest` cases passed, including wrong-type FP counting, partial SHA tie cutoff behavior, NaN rejection, duplicate IDs, full nested union coverage and per-event leakage rejection. Keyed legacy ordering was independently verified earlier; the completed-run audit rechecked event identity and coverage.

## Interpretation and remaining caveats

This is an **audit of archived target scores combined with newly refit seed101 MB03-style gap scores**, not formal controlled Phase-01 training, not a newly trained target+CTC+gap system, and not a new cohort. Thresholds were fit from eligible inner-only streams; outer outcomes did not select cutoffs. The target and gap modules have different historical/source provenance, so report as mixed-source. Historical MB03 checkpoint lineage for the old stream is not substituted; new refit checkpoint/scaler provenance is complete. No evidence here supports deployment or safety claims. Target-only class AP values reproduce archived MB07 direct/fixed/E4 streams; the complete-system values reported here are this audit's measured result, not a formal run verdict.

Supervisor owns the readiness verdict. No further fitting or tuning is recommended in this lane; retain the mixed-source qualification and CONDITIONAL/INCONCLUSIVE status for formal-run readiness until supervisor review.
