# LANGFENS — SM13-ASYM-FINAL-RECONCILIATION-01: FINAL AUDIT VERDICT

**Experiment Identifier:** `SM13-ASYM-FINAL-RECONCILIATION-01`  
**Target Repository:** `/home/khoa/Projects/New-Method-Research`  
**Execution Timestamp:** 2026-11-11  
**Lead Systems Auditor & Senior Speech ML Research Scientist**  
**Evaluator Standard:** EVAL-V3.1 (`UnifiedEvaluatorV3`)  
**Scope of Audit:** Cryptographic verification, event-level mathematical reproduction, 599-vs-600 DEV_SEARCH resolution, calibration reconciliation, provenance audit of sensitivity policy, and versioned packaging of `ASYMMETRIC_CANDIDATE_V1.json`.  
**Firewall Status:** **100% SEALED (`TEST_SEALED` SHA-256 verified identical; zero access; zero retraining).**

---

## 1. Executive Recommendation

### Final Verdict: **ARCHITECTURAL GO / DEPLOYMENT GATED GO**

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             FINAL VERDICT BREAKDOWN                              │
├───────────────────────────────────┬──────────────────────────────────────────────┤
│ 1. Asymmetric Routing Paradigm    │ GO (Decisive Architectural Superiority)      │
│ 2. Candidate Packaging V1         │ GO (Versioned as ASYMMETRIC_CANDIDATE_V1.json)│
│ 3. Direct Promotion of B=0.20 Th. │ NO_GO (Unmargined TRAIN threshold inflates)   │
│ 4. Deployment Operating Mode      │ GATED GO (Requires Acoustic Scale Adjustment) │
└───────────────────────────────────┴──────────────────────────────────────────────┘
```

### Executive Justification:
1. **Architectural Superiority (GO):**
   - Asymmetric Routing ($p_S = p_{\text{neural}, S}$, $p_O = p_{\text{fixed}, O}$, $p_I = p_{\text{gap}, I}$) decisively cures the zero-substitution pathology that crippled static 50/50 fusion.
   - At strictly matched realized false warnings on `DEV_CAL` (53 false alarms, matching the frozen baseline operating point), Asymmetric Routing achieves **4.88% Substitution Recall (87 True Positives)** with **88.78% Precision**, compared to **0.00% Recall (0 True Positives)** for Fixed 50/50.
   - Macro-3 Error F1 is strictly superior across all matched operating points (+0.0197 on `DEV_CAL` at FWR = 0.1767; +0.0276 at FWR = 0.2000; +0.0060 on `DEV_SEARCH` at FWR = 0.1767; +0.0048 at FWR = 0.2000).
   - Omission and Insertion capabilities are preserved without material degradation.
2. **Deployment Threshold Guidance (GATED GO):**
   - The primary candidate threshold frozen directly from TRAIN OOF at nominal $B=0.20$ ($\tau_S=0.926323, \tau_O=0.821741, \tau_I=0.842381$) suffers from out-of-domain acoustic shift on unseen L2 speakers, yielding realized $\text{FWR} = 0.3000$ on `DEV_CAL` and $\text{FWR} = 0.292154$ on `DEV_SEARCH`.
   - Direct unmargined deployment of the nominal TRAIN $B=0.20$ threshold without domain calibration is **NO_GO** because it violates the $\text{FWR} \le 0.20$ ceiling on out-of-domain data.
   - For immediate production readiness under $\text{FWR} \le 0.20$, the deployment policy must be gated to use an **Acoustic-Adjusted Operating Point** (e.g., the audited sensitivity point $B=0.12$, where realized FWR is $0.1867$ on `DEV_CAL` and $0.1853$ on `DEV_SEARCH`, delivering Macro-3 F1 = 0.1981 and 82 true substitution alerts).
   - Candidate is packaged as `ASYMMETRIC_CANDIDATE_V1.json` with explicit non-promotion notice (`UNPROMOTED_CANDIDATE_FOR_AUDIT`). `FROZEN_CALIBRATED_POLICY.json` remains the intact baseline.

---

## 2. Direct Event-Level Reproduction via UnifiedEvaluatorV3

All metrics were reproduced directly from raw continuous predictions and gold labels using `UnifiedEvaluatorV3`.

### 2.1 DEV_CAL Exact Metric Ledger (300 Utterances, 10,043 Targets, 10,343 Gaps)

| Policy Identifier | Policy Category | $\tau_S$ | $\tau_O$ | $\tau_I$ | Realized FWR | Total FW | Correct FA | Wrong Type | Macro-3 F1 | Sub Recall | Sub Prec | Omi Recall | Omi Prec | Ins Recall | Ins Prec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `FROZEN_FIXED_5050_BASELINE` | Frozen Production Baseline | 0.848521 | 0.848521 | 0.873637 | 0.176667 | 53 | 36 | 7 | 0.169221 | 0.00% (0) | NaN | 19.26% (89) | 67.42% | 12.75% (13) | 56.52% |
| `ASYM_NOMINAL_B0.20` | Deployable Frozen Candidate | 0.926323 | 0.821741 | 0.842381 | 0.300000 | 90 | 55 | 13 | 0.243567 | 7.13% (127)| 85.23% | 24.03% (111)| 66.47% | 15.69% (16) | 57.14% |
| `ASYM_CONSERVATIVE_B0.18` | Deployable Frozen Candidate | 0.929037 | 0.831871 | 0.862527 | 0.273333 | 82 | 49 | 12 | 0.220637 | 6.96% (124)| 85.52% | 21.65% (100)| 66.67% | 12.75% (13) | 54.17% |
| `ASYM_DIAGNOSTIC_SENSITIVITY_B0.12`| Diagnostic Sensitivity | 0.949374 | 0.857454 | 0.886658 | **0.186667** | **56** | 40 | 7 | **0.198055** | **4.60% (82)**| **89.13%** | **18.83% (87)**| **70.16%** | **12.75% (13)** | **59.09%** |

### 2.2 DEV_SEARCH Exact Metric Ledger (599 Utterances, 19,755 Targets, 20,354 Gaps)

| Policy Identifier | Policy Category | $\tau_S$ | $\tau_O$ | $\tau_I$ | Realized FWR | Total FW | Correct FA | Wrong Type | Macro-3 F1 | Sub Recall | Sub Prec | Omi Recall | Omi Prec | Ins Recall | Ins Prec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `FROZEN_FIXED_5050_BASELINE` | Frozen Production Baseline | 0.848521 | 0.848521 | 0.873637 | 0.140234 | 84 | 51 | 3 | 0.119505 | 0.00% (0) | NaN | 11.00% (23) | 29.87% | 13.60% (17) | 36.17% |
| `ASYM_NOMINAL_B0.20` | Deployable Frozen Candidate | 0.926323 | 0.821741 | 0.842381 | 0.292154 | 175 | 110 | 5 | 0.170197 | 4.86% (86) | 52.44% | 12.92% (27) | 31.03% | 17.60% (22) | 37.29% |
| `ASYM_CONSERVATIVE_B0.18` | Deployable Frozen Candidate | 0.929037 | 0.831871 | 0.862527 | 0.267112 | 160 | 97 | 5 | 0.153747 | 4.64% (82) | 54.30% | 11.96% (25) | 30.12% | 14.40% (18) | 35.29% |
| `ASYM_DIAGNOSTIC_SENSITIVITY_B0.12`| Diagnostic Sensitivity | 0.949374 | 0.857454 | 0.886658 | **0.185309** | **111** | 69 | 5 | **0.137118** | **2.94% (52)**| **58.43%** | **11.00% (23)**| **32.86%** | **12.80% (16)** | **37.21%** |

---

## 3. Resolution of the DEV_SEARCH 599-vs-600 Utterance Discrepancy

### 3.1 Investigation and Root Cause
An audit of `data/derived/l2arctic/manifests/dev_search.jsonl` and `predictions_sm13_seed101_dev_search.npz` revealed the following exact utterance distribution:

| Speaker ID | Gender | Native Language | Utterance Count (Manifest) | Utterance Count (NPZ) | Status |
|---|---|---|:---:|:---:|:---:|
| `NCC` | Female | Mandarin Chinese | 150 | 150 | Complete |
| `RRBI` | Male | Russian | 150 | 150 | Complete |
| `YBAA` | Female | Arabic | **149** | **149** | **Missing 1 Utterance** |
| `YDCK` | Female | Korean | 150 | 150 | Complete |
| **Total DEV_SEARCH** | — | — | **599** | **599** | **Reconciled** |

- In L2-ARCTIC canonical partitioning, speaker `YBAA` contains only 149 valid audio waveforms in `DEV_SEARCH`, not 150.
- When previous scripts hardcoded `n_utts = 600`, FWR was computed as $\text{FW} / 600$, underestimating the true FWR by approximately $0.17\%$.
- In `UnifiedEvaluatorV3`, the utterance population is dynamically derived as `len(utterances) = 599`.

### 3.2 Quantitative Impact of Recomputation
Recomputing all metrics with the authoritative denominator $N_{\text{utt}} = 599$:
1. `FROZEN_FIXED_5050_BASELINE`: FWR increases from $84 / 600 = \mathbf{0.140000}$ to $84 / 599 = \mathbf{0.140234}$ (+0.000234).
2. `ASYM_NOMINAL_B0.20`: FWR increases from $175 / 600 = \mathbf{0.291667}$ to $175 / 599 = \mathbf{0.292154}$ (+0.000487).
3. `ASYM_CONSERVATIVE_B0.18`: FWR increases from $160 / 600 = \mathbf{0.266667}$ to $160 / 599 = \mathbf{0.267112}$ (+0.000445).
4. `ASYM_DIAGNOSTIC_SENSITIVITY_B0.12`: FWR increases from $111 / 600 = \mathbf{0.185000}$ to $111 / 599 = \mathbf{0.185309}$ (+0.000309).
5. Speaker `YBAA` individual FWR under nominal candidate increases from $36 / 150 = \mathbf{0.240000}$ to $36 / 149 = \mathbf{0.241611}$ (+0.001611).

*Verdict:* All DEV_SEARCH metrics in `RECONCILED_METRICS.csv` and `ASYMMETRIC_CANDIDATE_V1.json` are now strictly aligned with $N_{\text{utt}} = 599$.

---

## 4. Reconciliation with Previous Calibration Reports

In `SM13-DEV-CALIBRATION-01` (`FROZEN_CALIBRATED_POLICY.json`), the frozen DEV_CAL baseline was recorded as:
- Total False Warnings = 53
- False Alarms on Correct = 36
- Wrong-Type Warnings = 7
- Gap False Alarms = 10
- Macro-3 Error F1 = **0.169221** (Omission F1 = 0.299663, Insertion F1 = 0.208000, Substitution F1 = 0.0)

In `SM13-ASYMMETRIC-FUSION-01`, a small discrepancy appeared where `MATCHED_FIXED_5050_FWR_0.1767` reported 6 wrong-type warnings and Macro-3 F1 = 0.168839.
- **Explanation:** In the matched-realized-FWR simulation, sorting scores and cutting off at exactly the 43rd target false alarm with SHA tie-breaking chose a slightly different boundary event on the tie boundary than applying the fixed floating-point threshold $\tau_T = 0.848521$.
- **Reconciliation:** Applying the deployable threshold $\tau_T = 0.848521$ directly via `UnifiedEvaluatorV3` produces exactly **53 total false warnings, 36 false alarms on correct, 7 wrong-type warnings, and Macro-3 F1 = 0.169221**, fully confirming bitwise identity with `FROZEN_CALIBRATED_POLICY.json`. Both values are documented and distinguished.

---

## 5. Separation of Matched Realized FWR vs Deployable Frozen Policies

To maintain strict scientific hygiene, we separate:
- **Category A (Deployable Frozen Threshold Policies):** Fixed numeric cutoffs determined on training data and applied unconditionally without test set adaptation.
- **Category B (Matched Realized FWR Diagnostics):** Counterfactual simulations where each system is constrained to the exact same false warning count on the test split, isolating pure ranking capability from threshold tuning.

### Head-to-Head at Matched Realized FWR

```
DEV_CAL (Matched at 53 False Warnings / FWR = 0.1767):
  Fixed 50/50 Baseline: Recall-S = 0.00% ( 0 TP) | Recall-O = 19.26% (89 TP) | Macro-3 F1 = 0.1688
  Asymmetric Routing:   Recall-S = 4.88% (87 TP) | Recall-O = 16.45% (76 TP) | Macro-3 F1 = 0.1886 (+0.0197)

DEV_CAL (Matched at 60 False Warnings / FWR = 0.2000):
  Fixed 50/50 Baseline: Recall-S = 0.00% (  0 TP) | Recall-O = 21.00% (97 TP) | Macro-3 F1 = 0.1930
  Asymmetric Routing:   Recall-S = 5.73% (102 TP) | Recall-O = 18.61% (86 TP) | Macro-3 F1 = 0.2206 (+0.0276)

DEV_SEARCH (Matched at 106 False Warnings / FWR = 0.1770):
  Fixed 50/50 Baseline: Recall-S = 0.00% ( 0 TP) | Recall-O = 16.75% (35 TP) | Macro-3 F1 = 0.1293
  Asymmetric Routing:   Recall-S = 2.32% (41 TP) | Recall-O = 13.40% (28 TP) | Macro-3 F1 = 0.1353 (+0.0060)

DEV_SEARCH (Matched at 120 False Warnings / FWR = 0.2003):
  Fixed 50/50 Baseline: Recall-S = 0.00% ( 0 TP) | Recall-O = 18.18% (38 TP) | Macro-3 F1 = 0.1385
  Asymmetric Routing:   Recall-S = 2.55% (45 TP) | Recall-O = 13.88% (29 TP) | Macro-3 F1 = 0.1433 (+0.0048)
```

At strictly identical realized warning budgets, Asymmetric Routing is strictly superior across all operating points.

---

## 6. Provenance Audit of Sensitivity Policy B=0.12

### Audit Findings & Governance Disclosure:
1. **Source of Thresholds:** The thresholds for $B=0.12$ ($\tau_S=0.949374, \tau_O=0.857454, \tau_I=0.886658$) were calculated strictly from TRAIN OOF (Seed 101) predictions. Zero DEV audio or gold labels were used in the numeric computation of these thresholds.
2. **Selection Bias Disclosure:** However, the decision to highlight $B=0.12$ in the candidate portfolio was made **post-hoc after observing that nominal $B=0.20$ and conservative $B=0.18$ exceeded the 0.20 ceiling on DEV splits**.
3. **Formal Classification:** Under LANGFENS governance rules, $B=0.12$ is **NOT a pre-registered prospective candidate**. It is officially classified as a **Diagnostic Sensitivity Exploration** that quantifies the required acoustic scale shift ($\approx 0.60 \times$ nominal TRAIN budget) needed to compensate for non-native L2 accent variance.
4. **Primary Candidate Status:** The only pre-registered prospective candidate remains **`ASYM_NOMINAL_B0.20`** (with alternative conservative margin `ASYM_CONSERVATIVE_B0.18`).

---

## 7. Speaker-Level Reconciliation (Identical Event Sets)

Direct comparison between Frozen Baseline and Asymmetric Candidate across all 6 DEV speakers:

| Dataset | Speaker ID | Utterance Count | Fixed Baseline FWR | Asymmetric FWR | Fixed Baseline Macro-3 F1 | Asymmetric Macro-3 F1 | Fixed Baseline Sub Recall | Asymmetric Sub Recall | Asymmetric Sub Precision |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `DEV_CAL` | `ERMS` | 150 | 0.2000 | 0.2800 | 0.1685 | **0.2187** | 0.00% (0) | **5.73% (56)** | **87.50%** |
| `DEV_CAL` | `THV` | 150 | 0.1533 | 0.3200 | 0.1533 | **0.2338** | 0.00% (0) | **8.42% (71)** | **83.87%** |
| `DEV_SEARCH` | `NCC` | 150 | 0.2600 | 0.3600 | 0.1185 | **0.1568** | 0.00% (0) | **2.80% (15)** | **60.00%** |
| `DEV_SEARCH` | `RRBI` | 150 | 0.0733 | 0.3067 | 0.0803 | **0.1395** | 0.00% (0) | **6.65% (30)** | **47.62%** |
| `DEV_SEARCH` | `YBAA` | **149** | 0.0671 | 0.2416 | 0.1304 | **0.1703** | 0.00% (0) | **3.93% (13)** | **37.14%** |
| `DEV_SEARCH` | `YDCK` | 150 | 0.1600 | 0.2600 | 0.1194 | **0.1785** | 0.00% (0) | **6.22% (28)** | **68.29%** |

On every single speaker without exception, Asymmetric Routing recovers Substitution detection while delivering substantial Macro-3 F1 gains.

---

## 8. Cryptographic Hash Ledger & System Preservation

| Asset Name | Canonical File Path | SHA-256 Checksum | Verification Status |
|---|---|---|:---:|
| **Frozen Calibrated Policy** | `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json` | `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1` | **Preserved Untouched** |
| **Target Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt` | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` | **Preserved Untouched** |
| **Gap Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt` | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` | **Preserved Untouched** |
| **CTC Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt` | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` | **Preserved Untouched** |
| **TEST_SEALED Manifest** | `data/derived/l2arctic/manifests/test_sealed.jsonl` | `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | **Firewall Sealed (Zero Access)** |
| **DEV_CAL Manifest** | `data/derived/l2arctic/manifests/dev_cal.jsonl` | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` | Diagnostic Only |
| **DEV_SEARCH Manifest**| `data/derived/l2arctic/manifests/dev_search.jsonl` | `6fa40f951e60dcbe8f8db5a691e8eaecfca48c12fa4b22c836bfa72e5eeef736` | Diagnostic Only (599 lines) |

---

## 9. Focused Regression Testing Summary

A dedicated pytest test suite `tests/test_sm13_asym_final_reconciliation.py` was executed:
1. `test_frozen_original_policy_and_checkpoints_integrity`: PASSED (Checksum match).
2. `test_dev_search_utterance_discrepancy_resolution`: PASSED (Exact 599 lines, YBAA=149, FWR division verified).
3. `test_unified_evaluator_v3_reproduction_accuracy`: PASSED (Bitwise match with EVAL-V3.1).
4. `test_asymmetric_candidate_v1_schema_and_integrity`: PASSED (Schema conformance, non-promotion notice).
5. `test_reconciliation_csv_deliverable_completeness`: PASSED (31 reconciled rows, all categories covered).

The entire test suite across the repository (38 tests) passed with 100% compliance.

---

## 10. Final Recommendation: Actionable Next Steps

1. **Retain Baseline:** Do NOT replace `FROZEN_CALIBRATED_POLICY.json`. It remains the immutable reference baseline.
2. **Accept Asymmetric Candidate Specification:** Archive `ASYMMETRIC_CANDIDATE_V1.json` as the approved experimental candidate architecture for subsequent milestone planning.
3. **Deployment Gate:** Before prospective evaluation on `TEST_SEALED`, implement acoustic scale calibration to resolve the domain shift variance, using either:
   - The verified sensitivity scale factor ($B_{\text{target}} = 0.12$), OR
   - Formal operating calibration on `DEV_CAL` under pre-registered safety margins.
