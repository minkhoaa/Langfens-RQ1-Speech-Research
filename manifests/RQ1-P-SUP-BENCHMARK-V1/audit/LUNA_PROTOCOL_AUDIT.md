# Luna read-only P-SUP benchmark audit

**Disposition: PRIOR FOUR-CHECKPOINT INFERENCE OUTPUTS EXIST LOCALLY, BUT THEY DO NOT ESTABLISH A VERIFIED O-versus-Z P-SUP BENCHMARK.** The key defect is a protocol-label/input mismatch: the P-SUP evaluator’s saved prediction rows say `Z-PROMPT-V1`, while its evaluation dataset obtains canonical phones and labels from the validation TextGrids. That is Protocol O input, not a demonstrated gold-free written-prompt/G2P evaluation. Do not present those rows as evidence of Z-PROMPT performance.

**Audit boundary:** read-only inspection; no files changed, no tests or inference run, no optimizer/backward pass, no sealed TEST content accessed, no credentials printed. Repository HEAD was `40bc3770776ee1c99462d44101fdd145602b76f5` on `review/psup-recovery-benchmark-v1`; pre-existing dirty/untracked state was preserved.

## Evidence and artifact availability

The frozen V1 contract is `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md` (SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`). It defines O versus Z, strict metric populations, thresholds/calibration, output provenance, and controls. The relevant prior P-SCORE specification and records are in `/home/khoa/Projects/KLTN/langfens-rq1-p-score/manifests/RQ1-P-SCORE/EXPERIMENT_SPEC.md` and `.../evaluation/`.

All four P-SUP checkpoints are present under `checkpoints/RQ1-P-SUP/{t_op,b_control}/`; their SHA-256 values match the hashes recorded in the evaluation summaries:

| Configuration | Checkpoint SHA-256 | Local metrics and raw rows |
|---|---|---|
| T-OP, fixed 3,600 | `a8559b382ff6ec09f9c90ed4eccba59fa45ff41e20e10aae1cea9eff7697fbd9` | `manifests/RQ1-P-SUP/evaluation/t_op_fixed_budget_3600/` |
| T-OP, best | `3b492049399608a1f0319accee0404de50fe4972340ff5cc3c930f58867a14e9` | `.../t_op_best/` |
| B-CONTROL, fixed 3,600 | `e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201` | `.../b_control_fixed_budget_3600/` |
| B-CONTROL, best | `2dbad5a0160224f9d936e564a6c5d52dffe6a8efdbde8e409c50a8cc393be93f` | `.../b_control_best/` |

Each directory contains `metrics.json`, `runtime_predictions.jsonl`, and `offline_token_evaluation.jsonl`. The shared summary is `manifests/RQ1-P-SUP/evaluation/FINAL_EVALUATION_SUMMARY.json`. **These files were inspected but their raw-row checksums and metrics were not independently recomputed.** The checkpoint’s historical remote-receipt claim is not a substitute for that verification.

The authorized Fold-1 TRAIN/VAL archive exists at `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip`; its SHA-256 matched the frozen digest `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`. The P-SCORE FP32 frozen-E0 cache at `cache/rq1_p_score_fold1_val_emissions.json` also matched its recorded digest `a699fd9378c86cd348a0628a144ecacfef491ea8f90e0edd23b4e46932e3e829`. That CTC-emission cache is **not** a P-SUP feature cache and cannot stand in for P-SUP model inference.

### P-SUP reported four-cell metrics

These are values read from the local `metrics.json` files, not independently recalculated:

| Configuration | Binary AP | Legacy omission AP | Dedicated omission AP | Substitution AP | PER |
|---|---:|---:|---:|---:|---:|
| T-OP, fixed 3,600 | 0.3240808445 | 0.0376102296 | 0.1077242032 | 0.3127669188 | 0.2419795567 |
| T-OP, best | 0.3217066868 | 0.0353791892 | 0.0990175950 | 0.3118446595 | 0.2446108693 |
| B-CONTROL, fixed 3,600 | 0.3226327131 | 0.0380328924 | null | 0.3112319148 | 0.2419795567 |
| B-CONTROL, best | 0.3272717163 | 0.0346919838 | null | 0.3197348820 | 0.2529096245 |

All report 300 VAL utterances, 9,956 prompt positions, and zero abstentions/failures. `null` dedicated omission AP for B-CONTROL means no separate omission score was reported; it must not be treated as zero or compared as if a measured subtype score existed. The legacy omission AP uses `s_error`, so it remains comparable across arms.

## Protocol root cause: O is mislabeled as Z

In the P-SUP worktree at `9ad1e5ebcd9a5c8203d6ab45fda1f4c8095dae15`:

- `src/langfens_prep/p_sup/evaluator.py` constructs `PSupL2ArcticDataset(..., split="val")`, then evaluates `batch["prompt_phone_ids"]` position-by-position against labels from that dataset.
- `src/langfens_prep/p_sup/dataset.py` reads each audio file **and its TextGrid**, parses the gold annotation, and builds operation labels from it.
- The saved P-SUP runtime rows identify their `protocol` as `Z-PROMPT-V1`, but the evaluator/dataset path above does not demonstrate a frozen CMUDict/G2P sequence generated only from written prompt text. It supplies annotation-derived canonical identities to scoring.

That is **Protocol O** under the frozen contract: TextGrid-derived canonical phone identities are oracle information even when timestamps are not used. A genuine Z run must take waveform plus written prompt through the pinned prompt table/dictionary/G2P and OOV policy, and defer TextGrid use to an offline join *after predictions and hashes are frozen*. Correct the provenance claim; do not relabel existing output as Z based only on a row’s `protocol` field.

By contrast, the P-SCORE run manifest declares `protocol: Z-PROMPT-V1`, `runtime_inputs: waveform_and_written_prompt_only`, frozen-cache mode, FP32 log-probability emissions, and no calibration. Its experiment spec identifies cache and scorer hashes. This is useful prior Z-method evidence, but it is **not** a contemporaneous P-SUP Z control. The P-SCORE records use an aligned-only cohort (`N=9,919`), while P-SUP metrics use all 9,956 positions; direct ranking across those reports is invalid.

## Correct metrics, denominators, and operating points

The frozen contract’s gold population is 9,956 canonical positions: 8,799 correct, 1,012 substitutions, and 145 deletions. For the legacy-comparable endpoints:

- Binary: substitutions + deletions versus correct, `N=9,956`, score `s_error`.
- Omission: deletions versus correct, **exclude substitutions**, `N=8,944` (145 positive, 8,799 negative), score `s_error` for legacy-comparable AP.
- Substitution: substitutions versus correct, **exclude deletions**, `N=9,811` (1,012 positive, 8,799 negative), score `s_error`.
- Dedicated T-OP omission AP uses `s_omit` on the same omission-versus-correct population; it is a separate endpoint and is unavailable for the binary control.
- PER uses greedy CTC decoding against the realized-phone reference (`N=9,881` historically), not omission labels. A CTC recognition deletion is not a learner omission.

The P-SCORE aligned-only population has `N=9,919` and excludes 37 unaligned positions (25 correct, 4 substitutions, 8 deletions). It has 137 aligned deletions and 8,774 aligned correct positions. Label results with this cohort and its coverage; never pool its AP with full-population AP.

The P-SCORE operating curves’ `FPR15` thresholds were computed retrospectively on VAL. They are diagnostic, **not CAL-derived guarantees**. V1 requires threshold selection on CAL correct-token scores, choosing the lowest threshold that satisfies the empirical ceiling, and separately reports achieved FPR. Existing P-SUP runtime rows have `threshold: null` and `calibration_reference: null`; the metrics do not establish calibrated probabilities or thresholded deployment performance. P-SUP’s reduced-train CAL speakers are NCC/THV, SELECT speakers MBMPS/TLV, and VAL speakers NJS/RRBI. The contract additionally notes that frozen E0 was trained on original TRAIN including SELECT/CAL, so its calibration speakers are not independent of historical E0 acoustic training.

## Missing controls and word-final qualification

V1 registers a deterministic **within-speaker cyclic shift-1 audio mismatch** (no fixed points, saved mapping/hash) and a canonical phone/position-only empirical prior. These are essential checks against speaker identity and text-position priors masquerading as acoustic evidence. The inspected P-SUP output directories contain matched-condition rows only; they do not establish either control. The P-SCORE report records an acoustic-gap control but does not fulfill this P-SUP matched-versus-mismatched/control comparison.

For prior reference, the frozen registry cites canonical empirical prior AP `0.303063` and a neural margin of `+0.013958`; this is registry-reported rather than independently recomputed here. It must be scored on the same population and mapping as the candidate, with construction restricted to permitted training data. A prior-only model or learned calibration/fusion requires its own approved, frozen procedure.

P-SCORE’s word-final-consonant stratum is derived offline in `audit_scripts/evaluate_p_score_benchmarks.py`: canonical-phone interval midpoint is assigned to a containing word-tier interval; the final canonical phone midpoint in that word is marked word-final, with single-phone words separately classified; only consonants enter that breakdown. The report’s 93/137 deletion count therefore depends on gold alignment/word-tier information. Treat it as an **offline descriptive stratum**, disclose the definition and coverage, and never supply TextGrid times/word boundaries to runtime. The existing P-SUP rows do not provide this stratification.

## Execution feasibility, privacy, and authorization

A CPU evaluator exists: `/home/khoa/Projects/KLTN/langfens-rq1-p-sup/audit_scripts/evaluate_p_sup.py`, backed by `src/langfens_prep/p_sup/evaluator.py`; its CLI accepts `--checkpoint`, `--split-manifest`, `--zip-path`, and `--output-dir`. Its implementation auto-selects CUDA when available and otherwise CPU, so any explicitly CPU-only replay must first confirm that the evaluator is forced to CPU rather than silently selecting a device. CPU inference may be technically feasible, but I did **not** benchmark its runtime or execute it; the P-SCORE emission cache does not eliminate P-SUP’s audio/model forward pass. The runner’s separate frozen 80-utterance command is not a substitute for this 300-utterance P-SUP VAL benchmark.

Read-only remote status checks found:

- Colab: `0.00` compute units, `0` active assignments. No active-session listing was obtained: `colab list` is not a supported command for the installed CLI.
- Kaggle: `27.03h` GPU quota remaining; both T-OP and B-CONTROL kernels report `COMPLETE`.
- Existing reconstruction records say baseline HF is public and the two P-SUP notebook API records reported `is_private=false`; no privacy settings were changed. Treat public/model or notebook metadata as an exposure/privacy blocker until reviewed; do not upload or stage new sensitive outputs there.
- Kaggle deployment policy requires `kaggle kernels push ... --no-run` for staging and a user attaching `HF_TOKEN` then starting execution in the Web UI (“Save & Run All”). That Web UI start requirement cannot be bypassed by CLI or this audit. No push, launch, or remote mutation was performed.

No sealed TEST audio, labels, predictions, or derived content were read or accessed. Existing TEST speaker names are used only as denylist metadata in the contract.

## Recommended validation before claiming a benchmark

1. Freeze the intended four cells, evaluator source revision, protocol, input-ID digest, precision, and output destinations. Keep fixed-budget and best-selection checkpoints as distinct reporting strata; do not imply they are independent model replications.
2. Verify checkpoint, dataset, prompt/G2P, source, dependency, and contract hashes from the actual evaluator inputs. Verify private durable storage and authorization before any remote inference.
3. Run P-SUP under **two explicitly separate protocols**: O for archive-compatible diagnostic comparison, and genuine Z-PROMPT-V1 with the frozen written-prompt frontend. Keep results in separate tables and preserve failures/unaligned positions; do not merge populations.
4. For every cell, recompute metrics from full-precision raw records. Verify exactly one row per expected prompt position and gold position, score orientation, finite values, duplicate keys, count consistency, no gold runtime fields, and complete failure/abstention accounting. Reconcile runtime prompt IDs against the pinned written-prompt/G2P output for Z; O rows must be explicitly marked oracle-canonical.
5. Report AP with grouped-tie convention and explicit prevalence/denominator; report binary, legacy omission, dedicated omission (when available), substitution, PER counts, per-speaker metrics, and coverage. Keep VAL FPR curves retrospective; report CAL thresholds/achieved FPR only after permitted calibration is independently validated.
6. Add the matched within-speaker cyclic audio shift-1 and train-only canonical empirical prior controls. Add speaker-level paired comparisons; with only two VAL speakers, do not claim population-level significance.
7. Define word-final strata explicitly and compute them only in the offline evaluation join. Validate midpoint/word-tier edge cases, deletions, single-phone words, and missing/multiple word assignments.
8. Recommended tests: source-level test that Z runtime has no TextGrid/label imports or calls; O/Z input-provenance integration test; output-schema and key-coverage checks; strict full-versus-aligned denominator reconciliation; score-calculation and `null` handling tests; FPR tie/threshold tests; mismatch permutation tests proving same-speaker/no-fixed-point mapping; prior baseline reproducibility test; word-final boundary/deletion fixtures; and protected TEST-ID rejection tests that inspect metadata only.

**Tests run in this audit: none.** No PASS is claimed.

---

**Managed output destination:** `/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/f07cb35c-ee66-413d-9df0-e7bf68b7fac9/audit/LUNA_PROTOCOL_AUDIT.md`