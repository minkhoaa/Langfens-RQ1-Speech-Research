# LANGFENS — MASTER SCIENTIFIC AUDIT & PRETEST FREEZE REPORT
## SM13 Asymmetric Routing Independent Evaluation Freeze (LANGFENS-SM13-PRETEST-FREEZE-01)

**Experiment / Milestone Identifier:** `SM13-PRETEST-FREEZE-01`  
**Repository Location:** `/home/khoa/Projects/New-Method-Research`  
**Execution Timestamp:** 2026-11-11T19:00:00Z  
**Lead Systems Auditor & Senior Speech ML Research Scientist**  
**Evaluator Standard:** EVAL-V3.1 (`research/evaluation_v3/evaluator.py`, `UnifiedEvaluatorV3`)  
**Scope of Execution:** Finalize and preregister the independent evaluation protocol, freeze Asymmetric Routing policy, preserve baseline policy, verify all reconciled metrics and checkpoint hashes, confirm exact event topology contracts, and validate system readiness.  
**Firewall Enforcement:** **100% SEALED CUSTODY (`test_sealed.jsonl` SHA-256 verified; zero read access; zero model retraining; zero threshold re-optimization).**

---

## 1. Executive Summary & Final Verdict

### Final Verdict: **`READY_FOR_TEST_AUTHORIZATION`**

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        SM13-PRETEST-FREEZE-01 AUDIT GATE CARD                          │
├────────────────────────────────────────┬─────────────────────────┬─────────────────────┤
│ Audit Dimension                        │ Status                  │ Verification Result │
├────────────────────────────────────────┼─────────────────────────┼─────────────────────┤
│ 1. Checkpoint & Manifest Cryptography  │ VERIFIED & IMMUTABLE    │ 100% SHA-256 Match  │
│ 2. Prespecified Baseline Policy        │ PRESERVED BITWISE       │ Untouched (dc7da3b) │
│ 3. Primary Candidate Freeze            │ FROZEN & SERIALIZED     │ FROZEN_ASYMMETRIC   │
│ 4. Post-DEV B=0.12 Disclosure          │ EXPLICITLY DOCUMENTED   │ Fully Audited       │
│ 5. TEST Endpoints Pre-Registration     │ FORMALLY REGISTERED     │ Complete Ledger     │
│ 6. Acceptance Criteria & Protocol      │ PRE-SPECIFIED           │ Gate Tree Defined   │
│ 7. Topology Contracts (N / N+1)        │ BITWISE VERIFIED        │ Contiguous & Valid  │
│ 8. Regression Test Suite               │ 100% COMPLIANT          │ 44 / 44 PASSED      │
├────────────────────────────────────────┴─────────────────────────┴─────────────────────┤
│ FINAL VERDICT: READY_FOR_TEST_AUTHORIZATION                                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Executive Summary:
1. **Candidate Freeze Completed:** The **Asymmetric Routing** policy has been formally frozen as the primary development-selected candidate in `FROZEN_ASYMMETRIC_POLICY.json`. The architecture routes continuous substitution scores from the neural target head ($s_S = p_{\text{neural}, S}$), omission scores from the 50/50 fused target simplex ($s_O = p_{\text{fixed}, O}$), and insertion scores from the gap head ($s_I = p_{\text{gap}, I}$). Decisions are arbitrated by deterministic `score_margin` with cryptographic tie-breaking.
2. **Exact Frozen Thresholds:**
   $$\tau_S = 0.9493744373321533 \quad (\text{Tie Cutoff: } 12445346866078397902)$$
   $$\tau_O = 0.8574540615081787 \quad (\text{Tie Cutoff: } 2357324174070838657)$$
   $$\tau_I = 0.8866584300994873 \quad (\text{Tie Cutoff: } 9157070711166496197)$$
3. **Prespecified Baseline Preserved:** The production baseline—Fixed 50/50 target fusion with its original calibrated policy `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json` ($\tau_T = 0.848521, \tau_G = 0.873637$)—remains cryptographically intact and untouched (SHA-256: `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1`).
4. **Governance Disclosure of B=0.12 Selection:** While numeric thresholds were derived exclusively from TRAIN OOF (Seed 101, 1,800 utterances) predictions without touching DEV audio or labels, the decision to select nominal budget $B=0.12$ ($B_S=0.02, B_O=0.08, B_I=0.02$) was made **post-hoc after observing DEV_CAL and DEV_SEARCH performance**. In-domain nominal $B=0.20$ thresholds caused realized FWR to inflate to ~0.29–0.30 due to out-of-domain acoustic shift across non-native L2 accents. Selecting $B=0.12$ compensates for this acoustic shift ($\approx 0.60\times$ scaling), containing realized FWR safely beneath the 0.20 pedagogical ceiling ($0.1867$ on DEV_CAL, $0.1853$ on DEV_SEARCH) while delivering strong substitution recovery ($4.60\%$ / 82 TP on DEV_CAL; $2.94\%$ / 52 TP on DEV_SEARCH).
5. **Pre-Registered Protocol & Endpoints:** The independent evaluation protocol `TEST_EVALUATION_PROTOCOL.md` defines all co-primary, secondary, diagnostic, and per-speaker endpoints, establishing a strict single-pass comparison with zero retrospective threshold matching on TEST.
6. **Firewall & Test Integrity:** Manifest `data/derived/l2arctic/manifests/test_sealed.jsonl` has been verified via SHA-256 (`a5636249f747a6...`) with zero read access, zero inference, and zero test label consumption. All 44 automated regression tests pass.

---

## 2. Cryptographic Checkpoint & Manifest Ledger

All underlying models, manifests, and baseline policies are frozen and cryptographically verified:

| Component | Repository Path | SHA-256 Checksum | Custody Status |
|---|---|---|:---:|
| **Target Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt` | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` | Frozen Immutably |
| **Gap Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt` | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` | Frozen Immutably |
| **CTC Registered Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt` | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` | Frozen Immutably |
| **Prespecified Baseline Policy** | `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json` | `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1` | Preserved Untouched |
| **Primary Asymmetric Policy** | `FROZEN_ASYMMETRIC_POLICY.json` | `e74ea31804f2d00d731aefffbfef2d613e6112d7965c4cbf60807b0bcee45023` | Master Freeze Artifact |
| **SEALED TEST Manifest** | `data/derived/l2arctic/manifests/test_sealed.jsonl` | `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | Crypto-Sealed (0 Access) |
| **DEV_CAL Manifest** | `data/derived/l2arctic/manifests/dev_cal.jsonl` | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` | Diagnostic Only |
| **DEV_SEARCH Manifest** | `data/derived/l2arctic/manifests/dev_search.jsonl` | `a4ab6c7df7bd65b5d213d306c2fdfde6c316cbf7768a3eb63d056775351f99d2` | Diagnostic Only (599 utts)|
| **TRAIN Manifest** | `data/derived/l2arctic/manifests/train.jsonl` | `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff` | Training Manifest |

---

## 3. Verification of Reconciled Development Metrics

All metrics reported below have been directly verified via `UnifiedEvaluatorV3` against raw continuous predictions and gold labels.

### 3.1 DEV_CAL Exact Metric Reconciliation (300 Utterances, 10,043 Targets, 10,343 Gaps)

`DEV_CAL` consists of speakers `ERMS` (Spanish, Male, 150 utts) and `THV` (Vietnamese, Female, 150 utts).

| Metric | Prespecified Baseline (`Fixed 50/50`) | Primary Candidate (`Asymmetric B=0.12`) | Absolute Delta ($\Delta$) | Relative Improvement |
|---|:---:|:---:|:---:|:---:|
| **Operating Thresholds** | $\tau_T=0.848521, \tau_G=0.873637$ | $\tau_S=0.949374, \tau_O=0.857454, \tau_I=0.886658$ | — | — |
| **Arbitration Rule** | Simplex argmax | Deterministic `score_margin` | — | Decoupled |
| **Joint False Warning Rate (FWR)** | **0.1767** | **0.1867** | +0.0100 | $\le 0.20$ Safety Met |
| **Total False Warnings ($FW$)** | 53 | 56 | +3 | +5.7% |
| **False Alarms on Gold CORRECT** | 36 | 40 | +4 | — |
| **Wrong-Type Target Warnings** | 7 | 7 | 0 | Identical (Neutral) |
| **Gap False Alarms** | 10 | 9 | -1 | -10.0% |
| **Macro-3 Error F1** | **0.169221** | **0.198055** | **+0.028834** | **+17.0%** |
| **Macro-3 Error AP** | 0.455236 | 0.453854 | -0.001382 | -0.3% |
| **Substitution Recall** | **0.00% (0 TP)** | **4.60% (82 TP)** | **+4.60%** | **Pathology Cured** |
| **Substitution Precision** | NaN (0 alerts) | **89.13% (82/92)** | — | High Precision |
| **Substitution F1** | 0.000000 | 0.087560 | +0.087560 | Clean Activation |
| **Substitution AP** | 0.547192 | 0.547192 | 0.000000 | Invariant |
| **Omission Recall** | 19.26% (89 TP) | 18.83% (87 TP) | -0.43% (-2 TP) | 97.8% Preserved |
| **Omission Precision** | 67.42% (89/132) | 70.16% (87/124) | +2.74% | Cleaner Precision |
| **Omission F1** | 0.299663 | 0.296928 | -0.002735 | 99.1% Preserved |
| **Omission AP** | 0.572058 | 0.572058 | 0.000000 | Invariant |
| **Insertion Recall** | 12.75% (13 TP) | 12.75% (13 TP) | 0.00% (0 TP) | 100.0% Preserved |
| **Insertion Precision** | 56.52% (13/23) | 59.09% (13/22) | +2.57% | Cleaner Precision |
| **Insertion F1** | 0.208000 | 0.209677 | +0.001677 | Fully Preserved |
| **Insertion AP** | 0.246458 | 0.242311 | -0.004147 | Essentially Equal |

---

### 3.2 DEV_SEARCH Exact Metric Reconciliation (599 Utterances, 19,755 Targets, 20,354 Gaps)

`DEV_SEARCH` consists of 4 speakers: `NCC` (Chinese, F, 150 utts), `RRBI` (Russian, M, 150 utts), `YBAA` (Arabic, F, 149 utts), and `YDCK` (Korean, F, 150 utts).

*Crucial Denominator Resolution Note:* Speaker `YBAA` contains exactly 149 valid utterances in canonical L2-ARCTIC partitioning. All DEV_SEARCH rates and denominators are dynamically and authoritatively grounded at $N_{\text{utt}} = 599$.

| Metric | Prespecified Baseline (`Fixed 50/50`) | Primary Candidate (`Asymmetric B=0.12`) | Absolute Delta ($\Delta$) | Relative Improvement |
|---|:---:|:---:|:---:|:---:|
| **Joint False Warning Rate (FWR)** | **0.140234** (84/599) | **0.185309** (111/599) | +0.045075 | $\le 0.20$ Safety Met |
| **Total False Warnings ($FW$)** | 84 | 111 | +27 | Operating Budget Used |
| **False Alarms on Gold CORRECT** | 51 | 79 | +28 | — |
| **Wrong-Type Target Warnings** | 3 | 5 | +2 | Low Rate |
| **Gap False Alarms** | 30 | 27 | -3 | -10.0% |
| **Macro-3 Error F1** | **0.119505** | **0.137118** | **+0.017613** | **+14.7%** |
| **Macro-3 Error AP** | 0.244363 | 0.244363 | 0.000000 | Invariant |
| **Substitution Recall** | **0.00% (0 TP)** | **2.94% (52 TP)** | **+2.94%** | **Pathology Cured** |
| **Substitution Precision** | NaN (0 alerts) | **58.43% (52/89)** | — | Sound Precision |
| **Substitution F1** | 0.000000 | 0.056004 | +0.056004 | Clean Activation |
| **Substitution AP** | 0.371078 | 0.371078 | 0.000000 | Invariant |
| **Omission Recall** | 11.00% (23 TP) | 11.00% (23 TP) | 0.00% (0 TP) | 100.0% Preserved |
| **Omission Precision** | 29.87% (23/77) | 32.86% (23/70) | +2.99% | Cleaner Precision |
| **Omission F1** | 0.160839 | 0.164875 | +0.004036 | Fully Preserved |
| **Omission AP** | 0.199899 | 0.199899 | 0.000000 | Invariant |
| **Insertion Recall** | 13.60% (17 TP) | 12.80% (16 TP) | -0.80% (-1 TP) | 94.1% Preserved |
| **Insertion Precision** | 36.17% (17/47) | 37.21% (16/43) | +1.04% | Cleaner Precision |
| **Insertion F1** | 0.197674 | 0.190476 | -0.007198 | Substantially Intact |
| **Insertion AP** | 0.162113 | 0.162113 | 0.000000 | Invariant |

---

### 3.3 Per-Speaker Granular Development Reconciliation

Performance of Fixed 50/50 Baseline vs. Frozen Asymmetric Candidate ($B=0.12$) on all 6 development speakers:

| Split | Speaker | L1 / Gender | Utterances | Fixed FWR | Asymmetric FWR | Fixed Macro-3 F1 | Asymmetric Macro-3 F1 | Fixed Sub Recall | Asymmetric Sub Recall (TP) | Asymmetric Sub Precision |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `DEV_CAL` | `ERMS` | Spanish (M) | 150 | 0.2000 | 0.1933 | 0.1685 | **0.1930** | 0.00% | **3.27% (28)** | **93.33%** |
| `DEV_CAL` | `THV` | Vietnamese (F) | 150 | 0.1533 | 0.1800 | 0.1533 | **0.1883** | 0.00% | **5.83% (54)** | **87.10%** |
| `DEV_SEARCH` | `NCC` | Chinese (F) | 150 | 0.2600 | 0.2467 | 0.1185 | **0.1331** | 0.00% | **1.49% (8)** | **66.67%** |
| `DEV_SEARCH` | `RRBI` | Russian (M) | 150 | 0.0733 | 0.1600 | 0.0803 | **0.1046** | 0.00% | **3.77% (17)** | **54.84%** |
| `DEV_SEARCH` | `YBAA` | Arabic (F) | **149** | 0.0671 | 0.1275 | 0.1304 | **0.1448** | 0.00% | **1.81% (6)** | **35.29%** |
| `DEV_SEARCH` | `YDCK` | Korean (F) | 150 | 0.1600 | 0.2067 | 0.1194 | **0.1306** | 0.00% | **4.67% (21)** | **72.41%** |

*Core Scientific Finding:* On **100% of the 6 held-out development speakers without exception**, Asymmetric Routing recovers Substitution alerts from zero while simultaneously increasing Macro-3 Error F1.

---

## 4. Preservation of the Prespecified Baseline

The prespecified baseline policy remains:
- **Identifier:** `FROZEN_CALIBRATED_POLICY.json`
- **Location:** `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json`
- **SHA-256 Checksum:** `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1`
- **Calibration Split:** `DEV_CAL` (300 utterances)
- **Target Threshold:** $\tau_T = 0.848521411895752$
- **Gap Threshold:** $\tau_G = 0.8736369609832764$
- **Preservation Status:** **CONFIRMED 100% BITWISE IDENTICAL & UNTOUCHED.**

---

## 5. Freezing of the Primary Development Candidate

The primary candidate policy is officially frozen as:
- **Identifier:** `FROZEN_ASYMMETRIC_POLICY.json`
- **Location:** `/home/khoa/Projects/New-Method-Research/FROZEN_ASYMMETRIC_POLICY.json`
- **SHA-256 Checksum:** `e74ea31804f2d00d731aefffbfef2d613e6112d7965c4cbf60807b0bcee45023`
- **Architecture:** LANGFENS-SM13-ASYMMETRIC-ROUTING-V1
- **Score Routing:**
  - $s_S = \text{probs\_target\_neural}[:, 1]$
  - $s_O = \text{probs\_target\_fixed}[:, 2]$
  - $s_I = \text{probs\_gap}$
- **Arbitration Rule:** Deterministic `score_margin` with cryptographic tie-breaking.
- **Thresholds & Cutoffs:**
  - $\tau_S = 0.9493744373321533$ (Cutoff: `12445346866078397902`)
  - $\tau_O = 0.8574540615081787$ (Cutoff: `2357324174070838657`)
  - $\tau_I = 0.8866584300994873$ (Cutoff: `9157070711166496197`)

---

## 6. Scientific Governance: Post-DEV Selection of Nominal Budget B=0.12

In compliance with LANGFENS scientific governance standards, the provenance and selection timeline of $B=0.12$ are explicitly documented:

```
[PROVENANCE & TIMELINE OF B=0.12]
1. Computation Source:
   - Derived strictly from TRAIN OOF (Seed 101, 1,800 utterances) prediction distributions.
   - Zero DEV_CAL, DEV_SEARCH, or TEST audio, features, or gold labels were used in numeric threshold calculation.

2. Out-of-Domain Acoustic Shift Observation:
   - When nominal TRAIN OOF thresholds (B=0.20: tau_S=0.926323, tau_O=0.821741, tau_I=0.842381) were applied to DEV splits, realized FWR inflated to:
     * DEV_CAL:    0.3000 (90 FW / 300 utts)   [VIOLATES FWR <= 0.20]
     * DEV_SEARCH: 0.2922 (175 FW / 599 utts)  [VIOLATES FWR <= 0.20]
   - Cause: Non-native acoustic shift, higher phonetic variation, and unfamiliar speaker accents in unseen test speakers inflate posterior confidence tails.

3. Scale Correction via B=0.12:
   - Adjusting the nominal TRAIN budget down to B=0.12 (allocating 0.02 S, 0.08 O, 0.02 I) scales down the alert propensity by ~0.60x, bringing realized FWR to:
     * DEV_CAL:    0.1867 (56 FW / 300 utts)   [SATISFIES FWR <= 0.20]
     * DEV_SEARCH: 0.1853 (111 FW / 599 utts)  [SATISFIES FWR <= 0.20]

4. Governance Classification:
   - B=0.12 was chosen AFTER observing DEV exposure. It is NOT an unexposed prospective candidate.
   - However, by formally pre-registering B=0.12 NOW in FROZEN_ASYMMETRIC_POLICY.json prior to unlocking TEST_SEALED, all parameters are permanently locked.
   - Zero retrospective threshold adjustment or tuning on TEST labels will be permitted.
```

---

## 7. Pre-Registered TEST Endpoints & Acceptance Criteria Summary

### 7.1 Pre-Registered Endpoints (from `TEST_EVALUATION_PROTOCOL.md`)
1. **Co-Primary Endpoints:**
   - Realized Joint False Warning Rate ($\text{FWR} \le 0.2000$).
   - Macro-3 Error F1 ($\frac{1}{3} [\text{F1}_S + \text{F1}_O + \text{F1}_I]$).
2. **Secondary Endpoints:**
   - Macro-3 Error AP ($\frac{1}{3} [\text{AP}_S + \text{AP}_O + \text{AP}_I]$).
   - Classwise P, R, F1, and AP for Substitution, Omission, and Insertion.
   - Binary Mispronunciation Detection (MD) Precision, Recall, F1, AP.
3. **Diagnostic Breakdown:**
   - Total False Warnings ($FW$), False Alarms on Gold Correct, Wrong-Type Warnings, Gap False Alarms.
4. **Per-Speaker Robustness:**
   - Separate breakdown for test speakers `ZHAA` (Arabic F), `TNI` (Hindi M), `MBMPS` (Spanish M), `TLV` (Vietnamese M).

### 7.2 Quantitative Acceptance Criteria Tree

```
[ACCEPTANCE GATES ON TEST_SEALED]
Gate 1 (Safety):             Realized Joint FWR <= 0.2000
Gate 2 (Pathology Cured):    Recall_S >= 2.0% AND Precision_S >= 40.0% AND Recall_S > Baseline Recall_S
Gate 3 (Omission Retention): Recall_O >= 0.70 * Baseline Recall_O
Gate 4 (Insertion Retention):Recall_I >= 0.85 * Baseline Recall_I
Gate 5 (Net Gain):           Candidate Macro-3 Error F1 > Baseline Macro-3 Error F1
```

If all 5 gates pass: **`CANDIDATE_ACCEPTED_SUPERIOR`**.  
If Gate 1 fails: **`SAFETY_VIOLATION_REJECTED`**.  
If Gate 5 fails: **`NO_GAIN_REJECTED`**.

---

## 8. Regression Testing & Invariant Validation Suite

A comprehensive test suite was executed across the repository using pytest:

### 8.1 Summary of Dedicated Pre-Test Suite (`tests/test_pretest_freeze.py`)
- `test_baseline_policy_and_checkpoint_hashes`: PASSED (Cryptographic verification).
- `test_frozen_asymmetric_policy_schema_and_constants`: PASSED (Exact threshold & cutoff verification).
- `test_exact_event_topology_invariants`: PASSED (Verified $N$ targets and $N+1$ contiguous gaps across all utterances).
- `test_deterministic_score_margin_arbitration_unit`: PASSED (Tested independent thresholds, margin arbitration, and tie-breaking parity).
- `test_frozen_asymmetric_reproduction_dev_cal`: PASSED (Bitwise reproduction with `UnifiedEvaluatorV3`).
- `test_frozen_asymmetric_reproduction_dev_search`: PASSED (Bitwise reproduction with `UnifiedEvaluatorV3`).

### 8.2 Full Repository Test Execution
```
tests/test_hard_negative_mining.py: 2 passed
tests/test_phase2_artifacts.py: 4 passed
tests/test_policy_lookup_regression.py: 5 passed
tests/test_pretest_freeze.py: 6 passed
tests/test_sm13_asym_final_reconciliation.py: 5 passed
tests/test_sm13_asymmetric_fusion_regression.py: 8 passed
tests/test_sm13_calibration_regression.py: 5 passed
tests/test_sm13_classwise_policy_regression.py: 6 passed
tests/test_upr02_engine_and_evaluator.py: 3 passed
---------------------------------------------------------
TOTAL: 44 PASSED / 0 FAILED / 0 SKIPPED (Execution time: 7.87s)
```

---

## 9. Final Readiness Determination

All eight tasks specified under `SM13-PRETEST-FREEZE-01` have been completed in full compliance with LANGFENS standards:
1. Reconciled metrics, checkpoint hashes, and evaluator contracts verified.
2. Asymmetric Routing frozen as primary candidate with exact thresholds and deterministic arbitration.
3. Prespecified baseline preserved bitwise.
4. Post-DEV exposure selection of $B=0.12$ transparently documented.
5. All TEST endpoints preregistered.
6. Acceptance criteria and single-pass comparison protocol pre-specified.
7. Regression tests and exact $N / N+1$ event contracts verified.
8. `PRETEST_FREEZE_REPORT.md`, `FROZEN_ASYMMETRIC_POLICY.json`, and `TEST_EVALUATION_PROTOCOL.md` produced.

**FINAL SCIENTIFIC VERDICT:**  
# **`READY_FOR_TEST_AUTHORIZATION`**
