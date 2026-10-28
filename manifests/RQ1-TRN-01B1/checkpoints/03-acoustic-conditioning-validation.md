# LANGFENS — RQ1-TRN-01B1 / 03-Acoustic Conditioning Validation

**Status:** COMPLETED

**Recorded at:** 2026-10-25T19:25:00+07:00

**Evidence source:** Commit `73eae7e`, `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Empirically test whether multi-task Wav2Vec 2.0 with cross-attention relies on acoustic speech features or merely memorizes canonical prompt statistics, and freeze the CTC decoder protocol.

## 2. Inputs & Dependencies
- Checkpoint: Trained pilot model from Cell 1.
- Holdout test items: Train-only holdout partitions.
- Runner: `audit_scripts/run_rq1_trn_01b1_r2_final_acoustic_gate.py`.

## 3. Work Performed
1. Constructed canonical-only baseline predicting error solely from canonical phone prior probabilities.
2. Evaluated matched-audio versus mismatched-audio inputs to measure acoustic sensitivity.
3. Froze greedy CTC decoding protocol without external language model rescoring.
4. Conducted end-to-end MDD evaluation under calibrated thresholds.

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-01B1-R2: PASS`.
- Full acoustic model achieved significant AP and F1 gains over canonical-only baseline.
- Mismatched audio produced catastrophic drop in detection F1, proving genuine acoustic feature dependence.
- Documented in [CANONICAL_ONLY_BASELINE_COMPARISON.md](../reconciliation/final_acoustic_gate/CANONICAL_ONLY_BASELINE_COMPARISON.md) and [DIAGNOSIS_ACOUSTIC_CONTRIBUTION.md](../reconciliation/final_acoustic_gate/DIAGNOSIS_ACOUSTIC_CONTRIBUTION.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/CANONICAL_ONLY_BASELINE_COMPARISON.md`, `DIAGNOSIS_ACOUSTIC_CONTRIBUTION.md`, `CTC_PROTOCOL_AND_DECODER_FREEZE.md`.
- Metrics: `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/canonical_only_baselines.json`, `task_b_acoustic_contribution.json`.

## 6. Restart / Recovery Instructions
- Inspect baseline comparison metrics:
  ```bash
  python -c "import json; print(json.load(open('manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/canonical_only_baselines.json')))"
  ```
- No user authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Baseline scaling to full Fold 1.
- Prohibited actions: Never assume text-only priors are sufficient for MDD.
- Next phase: [RQ1-TRN-02A](../../RQ1-TRN-02A/PROCESS_CHECKPOINT.md).
