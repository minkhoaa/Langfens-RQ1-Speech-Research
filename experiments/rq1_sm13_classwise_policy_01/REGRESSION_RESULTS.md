# LANGFENS SM13-CLASSWISE-POLICY-01: Regression Results & Verification Report

**Date:** 2026-11-11  
**Auditor:** Principal Speech ML Systems Engineer & Scientific Auditor  
**Suite:** `tests/test_sm13_classwise_policy_regression.py`  
**Status:** **ALL TESTS PASSED (6/6)**

---

## 1. Executive Summary

This document certifies the systematic regression testing and cryptographic integrity verification for the **SM13-CLASSWISE-POLICY-01** investigation. The test suite evaluates independent classwise threshold application ($\tau_S, \tau_O, \tau_I$), deterministic S/O conflict resolution, budget allocation monotonicity, and strict mathematical equivalence with `UnifiedEvaluatorV3`.

All tests executed with zero warnings, zero regressions, and complete reproducibility across seeds 101, 202, and 303.

---

## 2. Integrity & Firewall Audit

### 2.1 Preservation of Existing Frozen Policy Artifacts
The prompt requires:
> *"Preserve FROZEN_CALIBRATED_POLICY.json; save any candidate separately. Do not silently replace the existing policy."*

- **File:** `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json`
- **Expected SHA-256:** `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1`
- **Observed SHA-256:** `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1`
- **Result:** **PASSED (UNMODIFIED)**. The existing frozen DEV_CAL policy is completely intact.

### 2.2 Firewall Compliance
- `test_sealed.jsonl` was **never accessed, opened, or parsed**. Its SHA-256 checksum remains locked at `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`.
- `dev_cal.jsonl` and `dev_search.jsonl` were **never accessed for tuning, searching, or fitting**.
- DEV_CAL was evaluated **strictly post-hoc** as a fixed diagnostic comparison using thresholds frozen exclusively on TRAIN OOF.
- No neural acoustic model was retrained; no wav audio inference was executed.

---

## 3. Test Suite Specification & Results

| Test Function | Target Property | Verification Mechanism | Status |
|---|---|---|---|
| `test_frozen_original_policy_integrity` | Immutability of baseline policy | Exact SHA-256 hash match against pre-registered digest | **PASS** |
| `test_candidate_policy_schema_and_integrity` | Candidate policy schema compliance | Schema keys, valid float ranges $[0, 1]$, and provenance firewall | **PASS** |
| `test_deterministic_conflict_resolution_rules` | Deterministic S/O arbitration | Synthetic boundary cases: isolated alerts, conflicts, and ties | **PASS** |
| `test_independent_event_counts_and_fwr_calculation` | Mathematical parity with EVAL-V3.1 | Bit-for-bit equivalence of TP/FP/FN, FWR, and wrong-type warnings | **PASS** |
| `test_budget_threshold_monotonicity` | Threshold selection monotonicity | Monotonic non-increasing $\tau_c$ as budget $B_c$ increases | **PASS** |
| `test_deliverables_exist_and_non_empty` | Deliverable completeness | File presence, non-zero byte size, and format validity | **PASS** |

---

## 4. Deterministic S/O Conflict Resolution Audit

Target events possess continuous scores $(p_S, p_O)$ on the 3-simplex ($p_C + p_S + p_O = 1.0$). When independent thresholds $\tau_S$ and $\tau_O$ are applied, an event triggers:
- Substitution warning if $p_S \ge \tau_S$
- Omission warning if $p_O \ge \tau_O$

### 4.1 Boundary Analysis: When Do Conflicts Occur?
Because $p_S + p_O \le 1.0$:
$$\text{If } \tau_S + \tau_O > 1.0, \quad \text{Conflicts are mathematically impossible}.$$

In the nominal budget regime ($B = 0.20$ false warnings/utterance across 1,800 utterances):
- $\tau_S$ values selected on TRAIN OOF range from **0.7570 to 0.8669**.
- $\tau_O$ values selected on TRAIN OOF range from **0.7916 to 0.8826**.
- Sum of thresholds: $\tau_S + \tau_O \ge 1.5486 > 1.0000$.
- **Observed empirical conflict count across all 59,562 TRAIN target events:** **0 conflicts**.
- **Observed empirical conflict count across all 10,043 DEV_CAL target events:** **0 conflicts**.

### 4.2 Low-Threshold Arbitration Verification
To ensure full specification robustness in high-budget or exploratory scenarios where $\tau_S < 0.50$ (e.g. $\tau_S = 0.35, \tau_O = 0.45$), `test_deterministic_conflict_resolution_rules` explicitly verifies 5 arbitration rules:
1. `score_margin`: $\text{argmax}(p_S - \tau_S, p_O - \tau_O)$
2. `norm_margin`: $\text{argmax}\left(\frac{p_S - \tau_S}{1 - \tau_S}, \frac{p_O - \tau_O}{1 - \tau_O}\right)$
3. `prob_comparison`: $\text{argmax}(p_S, p_O)$
4. `priority_substitution`: Selects Substitution
5. `priority_omission`: Selects Omission

Ties in margins or probabilities are resolved by deterministic SHA-256 event hash:
$$\text{tie\_hash}(e) = \text{int.from\_bytes}(\text{sha256}("LANGFENS-MB07-warning-tie-v1:" + e)[:8], "big")$$
If tie_hash is odd, class 1 (Substitution) is assigned; if even, class 2 (Omission). Zero random numbers, zero nondeterminism.

---

## 5. Event Count & FWR Parity Proof

`UnifiedEvaluatorV3` defines false warnings as:
$$\text{false\_warnings} = \text{FP}_S + \text{FP}_O + \text{FP}_I$$
where:
- $\text{FP}_S = \sum \mathbf{1}(\hat{y}_t = 1 \land y_t \ne 1) = \text{C}\to\text{S} + \text{O}\to\text{S}$
- $\text{FP}_O = \sum \mathbf{1}(\hat{y}_t = 2 \land y_t \ne 2) = \text{C}\to\text{O} + \text{S}\to\text{O}$
- $\text{FP}_I = \sum \mathbf{1}(\hat{y}_g = \text{True} \land y_g = 0) = \text{NoIns}\to\text{Ins}$

Decomposing target false alarms:
$$\text{false\_alarms\_gold\_correct} = \sum \mathbf{1}(y_t = 0 \land \hat{y}_t \in \{1, 2\}) = \text{C}\to\text{S} + \text{C}\to\text{O}$$
$$\text{wrong\_type\_warnings} = \sum \mathbf{1}(y_t \in \{1, 2\} \land \hat{y}_t \in \{1, 2\} \land y_t \ne \hat{y}_t) = \text{O}\to\text{S} + \text{S}\to\text{O}$$

Thus:
$$\text{false\_warnings} = \text{false\_alarms\_gold\_correct} + \text{wrong\_type\_warnings} + \text{FP}_I$$
$$\text{joint\_fwr} = \frac{\text{false\_warnings}}{N_{\text{utterances}}}$$

In `test_independent_event_counts_and_fwr_calculation`:
- On Seed 101 TRAIN OOF (1,800 utterances):
  - $\text{FP}_S = 67$
  - $\text{FP}_O = 192$
  - $\text{FP}_I = 163$
  - $\text{false\_alarms\_gold\_correct} = 207$
  - $\text{wrong\_type\_warnings} = 52$
  - $\text{false\_warnings} = 67 + 192 + 163 = 422$
  - Decomposition check: $207 + 52 + 163 = 422$ (**Exact identity verified**).
  - Joint FWR: $422 / 1800 = 0.234444$ (**Exact match with UnifiedEvaluatorV3**).

---

## 6. Pytest Execution Transcript

```
============================= test session starts ==============================
platform linux -- Python 3.11.17, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/khoa/Projects/New-Method-Research
configfile: pytest.ini
plugins: platformdirs-4.14.4, anyio-4.16.0
collected 6 items

tests/test_sm13_classwise_policy_regression.py ......                    [100%]

============================== 6 passed in 3.22s ===============================
```

### Full Repository Regression Suite
```
tests/test_hard_negative_mining.py ..                                    [  8%]
tests/test_phase2_artifacts.py ....                                      [ 24%]
tests/test_policy_lookup_regression.py .....                             [ 44%]
tests/test_sm13_calibration_regression.py .....                          [ 64%]
tests/test_sm13_classwise_policy_regression.py ......                    [ 88%]
tests/test_upr02_engine_and_evaluator.py ...                             [100%]

============================== 25 passed in 5.86s ==============================
```

All 25 repository tests pass with 100% success.
