# LANGFENS SM13-ASYMMETRIC-FUSION-01: REGRESSION AND INTEGRITY RESULTS

**Execution Date:** 2026-11-11  
**Test Suite:** `tests/test_sm13_asymmetric_fusion_regression.py`  
**Environment:** Python 3.11.17 (`.venv/bin/python`), Pytest 9.2.0  
**Status:** **ALL 8 TESTS PASSED (100% SUCCESS)**

---

## 1. Test Summary

| Test Function | Verification Target | Status | Duration |
|---|---|---|---|
| `test_frozen_original_policy_integrity` | Verifies SHA-256 hash `dc7da3bcbc...` of `FROZEN_CALIBRATED_POLICY.json` | **PASSED** | 0.01s |
| `test_checkpoints_and_sealed_firewall_integrity` | Cryptographic check of target, gap, and CTC weights + test_sealed manifest | **PASSED** | 0.05s |
| `test_candidate_asymmetric_policy_schema_and_integrity` | Schema conformance, required provenance keys, and routing semantics | **PASSED** | 0.01s |
| `test_deterministic_conflict_resolution_rules` | Boundary conditions and determinism for all 5 S/O arbitration rules | **PASSED** | 0.01s |
| `test_independent_event_counts_and_fwr_calculation` | Full mathematical identity with `UnifiedEvaluatorV3` down to integer event counts | **PASSED** | 2.58s |
| `test_asymmetric_budget_threshold_monotonicity` | Strict monotonicity of $(\tau_S, \tau_O)$ under budget scaling | **PASSED** | 0.32s |
| `test_modulo42_counterfactual_accounting` | Complete event-level accounting of modulo-42 hits, misses, and hit rates | **PASSED** | 0.02s |
| `test_deliverables_exist_and_schema` | Verification of `POLICY_COMPARISON.csv`, `PER_SPEAKER_METRICS.csv`, and JSON files | **PASSED** | 0.04s |

---

## 2. Event-Level Accounting Identity Verification

In `test_independent_event_counts_and_fwr_calculation`, all metric outputs from fast vector evaluation were head-to-head verified against the authoritative `UnifiedEvaluatorV3` engine across all 59,562 target slots and 61,362 gap slots of Full-TRAIN Seed 101:

1. **Target Error Accounting Identity:**
   $$\text{False Warnings}_{\text{target}} = \text{FP}_S + \text{FP}_O = 360$$
   $$\text{False Alarms on Correct Only} = 309$$
   $$\text{Wrong-Type Warnings} = 51$$
   $$\text{Total Target FP} = 309 + 51 = 360 \quad (\Delta = 0)$$

2. **Total Warning Budget Identity:**
   $$\text{Total False Warnings} = \text{FP}_S + \text{FP}_O + \text{FP}_I = 90 + 198 + 72 = 360$$
   $$\text{Joint FWR} = \frac{360}{1,800} = 0.200000 \quad (\Delta < 10^{-12})$$

3. **Per-Class Support Identity:**
   $$\text{Support}_S = \text{TP}_S + \text{FN}_S = 278 + 7,103 = 7,381 \quad (\Delta = 0)$$
   $$\text{Support}_O = \text{TP}_O + \text{FN}_O = 296 + 1,554 = 1,850 \quad (\Delta = 0)$$
   $$\text{Support}_I = \text{TP}_I + \text{FN}_I = 65 + 495 = 560 \quad (\Delta = 0)$$

---

## 3. Cryptographic Hash Ledger

| File Artifact | SHA-256 Checksum | Integrity Status |
|---|---|---|
| `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json` | `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1` | Preserved Untouched |
| `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt` | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` | Preserved Untouched |
| `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt` | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` | Preserved Untouched |
| `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt` | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` | Preserved Untouched |
| `data/derived/l2arctic/manifests/test_sealed.jsonl` | `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | Firewall Sealed |
| `data/derived/l2arctic/manifests/dev_cal.jsonl` | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` | Diagnostic Only |
| `data/derived/l2arctic/manifests/dev_search.jsonl` | `6fa40f951e60dcbe8f8db5a691e8eaecfca48c12fa4b22c836bfa72e5eeef736` | Diagnostic Only |
