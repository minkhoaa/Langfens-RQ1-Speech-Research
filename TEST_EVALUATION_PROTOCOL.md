# LANGFENS — INDEPENDENT EVALUATION PROTOCOL FOR SM13 ASYMMETRIC ROUTING
## (LANGFENS-SM13-TEST-EVALUATION-PROTOCOL-V1.0)

**Protocol Identifier:** `LANGFENS-SM13-TEST-EVALUATION-PROTOCOL-V1.0`  
**Target Repository:** `/home/khoa/Projects/New-Method-Research`  
**Execution Mode:** Prospective Pre-Registration & Independent Evaluation Standard  
**Governing Standard:** EVAL-V3.1 (`UnifiedEvaluatorV3`, exact $N$ targets and $N+1$ gaps)  
**Lead Systems Auditor & Senior Speech ML Research Scientist**  
**Creation & Pre-Registration Timestamp:** 2026-11-11T19:00:00Z  
**Firewall Status:** **SEALED CUSTODY (`test_sealed.jsonl` SHA-256 verified; zero read access; zero inference before protocol execution).**

---

## 1. Executive Protocol Summary & Scientific Purpose

This protocol establishes the prospective, preregistered, and immutable evaluation framework for comparing the development-selected candidate system—**SM13 Asymmetric Routing**—against the prespecified production baseline—**Fixed 50/50 Target Fusion**—on the held-out, crypto-sealed test partition **`TEST_SEALED`**.

The objective of SM13 Asymmetric Routing is to eliminate the severe "zero-substitution pathology" inherent in static 50/50 Target Fusion (where substitution recall was 0.00% under calibrated operating budgets) while maintaining strict adherence to the pedagogical safety ceiling of **$\text{FWR} \le 0.20$ false warnings per utterance**, preserving omission and insertion detection capability, and improving overall unweighted Macro-3 Error F1.

To guarantee complete scientific integrity and prevent all forms of post-hoc p-hacking, retrospective threshold fishing, or test data leakage, **this protocol completely locks and freezes all model checkpoints, routing logic, decision thresholds, tie-breaking mechanisms, evaluation endpoints, and quantitative acceptance criteria prior to unlocking `TEST_SEALED`**.

---

## 2. Prespecified System Architectures & Checkpoint Ledger

Both evaluation systems operate on the identical underlying neural checkpoints formally trained during `RQ1-SM13-FORMAL-TRAIN-01` (Seed 101 refit across 12 speakers / 1,800 utterances). No parameters are updated, retrained, or fine-tuned.

### 2.1 Cryptographic Checkpoint Ledger

| Asset Name | Canonical Repository Path | SHA-256 Checksum | Preserved Status |
|---|---|---|:---:|
| **Target Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt` | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` | **VERIFIED IMMUTABLE** |
| **Gap Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt` | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` | **VERIFIED IMMUTABLE** |
| **CTC Registered Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt` | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` | **VERIFIED IMMUTABLE** |
| **Prespecified Baseline Policy** | `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json` | `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1` | **VERIFIED IMMUTABLE** |
| **Primary Asymmetric Policy** | `FROZEN_ASYMMETRIC_POLICY.json` | `e74ea31804f2d00d731aefffbfef2d613e6112d7965c4cbf60807b0bcee45023` | **FROZEN CANDIDATE** |
| **TEST_SEALED Manifest** | `data/derived/l2arctic/manifests/test_sealed.jsonl` | `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | **CRYPTO-SEALED** |

---

## 3. Mathematical Foundations & Evaluator Contract (EVAL-V3.1)

All metric calculations must be executed via `UnifiedEvaluatorV3` (`research/evaluation_v3/evaluator.py`).

### 3.1 Event Topology Invariants
For every utterance $u \in \mathcal{U}$:
- **Target Event Domain:** Contains exactly $N_u$ target phone events with contiguous slot indices $i \in \{0, 1, \dots, N_u - 1\}$. Gold labels $\in \{\text{CORRECT}, \text{SUBSTITUTION}, \text{OMISSION}\}$.
- **Gap Event Domain:** Contains exactly $N_u + 1$ inter-phoneme gap events with contiguous slot indices $j \in \{0, 1, \dots, N_u\}$. Gold labels $\in \{\text{NO\_INSERTION}, \text{INSERTION}\}$.
- **Total Utterance Population ($N_{\text{utt}}$):** Authoritative denominator dynamically computed from the manifest. On `TEST_SEALED`, $N_{\text{utt}} = 600$ utterances across 4 held-out speakers (`ZHAA`, `TNI`, `MBMPS`, `TLV`).

### 3.2 Authoritative Endpoint Formulations
1. **False Warning Count ($FW$):**
   $$FW = FP_{\text{target}} + FP_{\text{gap}} = (\text{False Alarms on Gold CORRECT}) + (\text{Wrong-Type Target Warnings}) + (\text{Gap False Alarms})$$
   where a *Wrong-Type Target Warning* occurs when gold is an error (e.g. SUBSTITUTION) but the model issues an alert of the opposite error type (e.g. OMISSION), or vice versa.
2. **Joint False Warning Rate (FWR):**
   $$\text{Joint FWR} = \frac{FW}{N_{\text{utt}}}$$
3. **Macro-3 Error F1:**
   $$\text{Macro-3 Error F1} = \frac{1}{3} \left( \text{F1}_{\text{SUB}} + \text{F1}_{\text{OMI}} + \text{F1}_{\text{INS}} \right)$$
   where $\text{F1}_c = \frac{2 \cdot TP_c}{2 \cdot TP_c + FP_c + FN_c}$. If $TP_c + FP_c + FN_c = 0$, $\text{F1}_c = 0.0$.
4. **Macro-3 Error Average Precision (AP):**
   $$\text{Macro-3 Error AP} = \frac{1}{3} \left( \text{AP}_{\text{SUB}} + \text{AP}_{\text{OMI}} + \text{AP}_{\text{INS}} \right)$$
   measured across continuous decision scores before discrete thresholding.
5. **Classwise Recall and Precision:**
   $$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad \text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

---

## 4. Prespecified Reference Baseline System: Fixed 50/50 Fusion

The reference baseline is the formally calibrated production policy defined in `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json`.

### 4.1 Baseline Scoring and Decision Rules
- **Target Simplex:** 50/50 static linear combination of neural and acoustic target probabilities:
  $$\mathbf{p}_{\text{fixed}} = 0.50 \cdot \mathbf{p}_{\text{neural}} + 0.50 \cdot \mathbf{p}_{\text{acoustic}}$$
- **Target Confidence Score:** $s_{\text{target}} = \max(p_{\text{fixed}, \text{SUB}}, p_{\text{fixed}, \text{OMI}})$
- **Target Prediction Rule:**
  $$\hat{y}_t = \begin{cases} 
  \text{SUBSTITUTION} & \text{if } s_{\text{target}} \ge \tau_T \text{ and } p_{\text{fixed}, \text{SUB}} \ge p_{\text{fixed}, \text{OMI}} \\
  \text{OMISSION} & \text{if } s_{\text{target}} \ge \tau_T \text{ and } p_{\text{fixed}, \text{OMI}} > p_{\text{fixed}, \text{SUB}} \\
  \text{CORRECT} & \text{if } s_{\text{target}} < \tau_T 
  \end{cases}$$
- **Gap Prediction Rule:**
  $$\hat{y}_g = \begin{cases} 
  \text{INSERTION} & \text{if } p_{\text{gap}} \ge \tau_G \\
  \text{NO\_INSERTION} & \text{if } p_{\text{gap}} < \tau_G
  \end{cases}$$

### 4.2 Baseline Frozen Thresholds
- **Target Threshold ($\tau_T$):** `0.848521411895752`
- **Gap Threshold ($\tau_G$):** `0.8736369609832764`
- **Baseline Diagnostic Reference Performance:**
  - `DEV_CAL` (300 utts): $\text{FWR} = 0.1767$ ($53$ false warnings), $\text{Macro-3 F1} = 0.169221$, $\text{Recall}_S = 0.00\%$ ($0$ TP), $\text{Recall}_O = 19.26\%$ ($89$ TP), $\text{Recall}_I = 12.75\%$ ($13$ TP).
  - `DEV_SEARCH` (599 utts): $\text{FWR} = 0.140234$ ($84$ false warnings), $\text{Macro-3 F1} = 0.119505$, $\text{Recall}_S = 0.00\%$ ($0$ TP), $\text{Recall}_O = 11.00\%$ ($23$ TP), $\text{Recall}_I = 13.60\%$ ($17$ TP).

---

## 5. Primary Development-Selected Candidate System: Asymmetric Routing

The primary candidate system is frozen in `FROZEN_ASYMMETRIC_POLICY.json`.

### 5.1 Asymmetric Routing Scoring Logic
Instead of forcing target probabilities into a shared 3-class simplex where acoustic omission scores overwhelm neural substitution alerts, Asymmetric Routing decouples the error detection channels:
- **Substitution Score ($s_S$):** Evaluated strictly from the neural target head:
  $$s_S = p_{\text{neural}, \text{SUB}} = \text{probs\_target\_neural}[:, 1]$$
- **Omission Score ($s_O$):** Evaluated strictly from the robust 50/50 fused target simplex:
  $$s_O = p_{\text{fixed}, \text{OMI}} = \text{probs\_target\_fixed}[:, 2]$$
- **Insertion Score ($s_I$):** Evaluated strictly from the gap head:
  $$s_I = p_{\text{gap}} = \text{probs\_gap}$$

### 5.2 Deterministic `score_margin` Arbitration Contract
Because $s_S$ and $s_O$ originate from different scoring channels, both channels may independently cross their respective operating thresholds on the same phoneme. All such conflicts are arbitrated by the deterministic `score_margin` rule:

```
[TARGET DECISION ALGORITHM]
1. Independent Threshold Evaluation:
   flag_S = (s_S > tau_S) or (s_S == tau_S and tie_hash(event_id) >= cut_S)
   flag_O = (s_O > tau_O) or (s_O == tau_O and tie_hash(event_id) >= cut_O)

2. Deterministic Arbitration:
   if flag_S and not flag_O:
       pred_label = "SUBSTITUTION"
   elif flag_O and not flag_S:
       pred_label = "OMISSION"
   elif flag_S and flag_O:
       margin_S = s_S - tau_S
       margin_O = s_O - tau_O
       if margin_S > margin_O:
           pred_label = "SUBSTITUTION"
       elif margin_O > margin_S:
           pred_label = "OMISSION"
       else:
           # Exact margin tie broken by deterministic cryptographic event hash parity
           pred_label = "SUBSTITUTION" if (tie_hash(event_id) % 2 == 1) else "OMISSION"
   else:
       pred_label = "CORRECT"

3. Gap Decision:
   pred_has_insertion = (s_I > tau_I) or (s_I == tau_I and tie_hash(event_id) >= cut_I)
```

where `tie_hash(event_id)` is defined as the first 8 bytes of `sha256("LANGFENS-MB07-warning-tie-v1:" + str(event_id))` parsed as big-endian `uint64`.

### 5.3 Primary Candidate Frozen Thresholds
The candidate thresholds frozen in `FROZEN_ASYMMETRIC_POLICY.json` are:

| Parameter | Floating-Point Threshold | Exact Tie-Breaking Cutoff | Nominal TRAIN Budget |
|---|:---:|:---:|:---:|
| **Substitution ($\tau_S$)** | `0.9493744373321533` | `12445346866078397902` | $B_S = 0.020$ FP/utt |
| **Omission ($\tau_O$)** | `0.8574540615081787` | `2357324174070838657` | $B_O = 0.080$ FP/utt |
| **Insertion ($\tau_I$)** | `0.8866584300994873` | `9157070711166496197` | $B_I = 0.020$ FP/utt |
| **Total Nominal Budget ($B$)** | — | — | **$B = 0.120$ FP/utt** |

---

## 6. Governance Disclosure: Post-DEV Selection of Nominal Budget B=0.12

To preserve scientific rigor, the history and rationale of selecting $B=0.12$ are formally disclosed:

1. **Mathematical Provenance of Cutoffs:** The numeric values of $\tau_S, \tau_O, \tau_I$ were derived solely from TRAIN OOF (Seed 101, 1,800 utterances) predictions. Zero audio frames or gold labels from `DEV_CAL`, `DEV_SEARCH`, or `TEST_SEALED` entered into the computation of these thresholds.
2. **Post-DEV Selection Trigger:** The decision to select nominal budget $B=0.12$ as the primary candidate (rather than nominal $B=0.20$ or conservative $B=0.18$) was made **after observing DEV_CAL and DEV_SEARCH performance**.
3. **Root Cause of Acoustic Shift:** When the nominal in-domain TRAIN $B=0.20$ threshold ($\tau_S=0.926323, \tau_O=0.821741, \tau_I=0.842381$) was transferred to unseen L2 speakers, acoustic variance and non-native phone variations inflated realized FWR to **$0.3000$ on DEV_CAL** and **$0.2922$ on DEV_SEARCH**, violating the pedagogical safety limit of $\text{FWR} \le 0.20$.
4. **Correction via Scale Factor:** Exploration on TRAIN OOF demonstrated that scaling the nominal budget by $\approx 0.60\times$ (nominal $B=0.12$) yields realized $\text{FWR} = 0.1867$ on DEV_CAL and $\text{FWR} = 0.1853$ on DEV_SEARCH, successfully centering operating performance beneath the 0.20 safety ceiling while recovering substantial substitution recall ($4.60\%$ / 82 TP on DEV_CAL; $2.94\%$ / 52 TP on DEV_SEARCH).
5. **Pre-Test Lock:** By pre-registering $B=0.12$ in this protocol and freezing it in `FROZEN_ASYMMETRIC_POLICY.json` *prior to test set unlocking*, this parameter choice is treated as a fixed, development-selected model specification, completely prohibiting retrospective threshold adjustments on TEST.

---

## 7. Preregistered TEST Endpoints

Evaluation on `TEST_SEALED` must compute and report the following standardized endpoints:

### 7.1 Co-Primary Endpoints
1. **Realized Joint FWR:** Must satisfy $\text{FWR} \le 0.2000$.
2. **Macro-3 Error F1:** Primary multi-class efficacy benchmark.

### 7.2 Secondary Efficacy & Discrimination Endpoints
1. **Macro-3 Error Average Precision (AP):** Invariant ranking metric across error types.
2. **Classwise Substitution Performance:**
   - Substitution Recall ($\text{Recall}_S$), Precision ($\text{Precision}_S$), F1 ($\text{F1}_S$), and Average Precision ($\text{AP}_S$).
3. **Classwise Omission Performance:**
   - Omission Recall ($\text{Recall}_O$), Precision ($\text{Precision}_O$), F1 ($\text{F1}_O$), and Average Precision ($\text{AP}_O$).
4. **Classwise Insertion Performance:**
   - Insertion Recall ($\text{Recall}_I$), Precision ($\text{Precision}_I$), F1 ($\text{F1}_I$), and Average Precision ($\text{AP}_I$).
5. **Secondary Target Correct Metrics:** Precision, Recall, F1 for the gold-correct majority class.
6. **Binary Mispronunciation Detection (MD):** Precision, Recall, F1, AP treating all error types as a single positive class.

### 7.3 Diagnostic & Quality Breakdown Endpoints
1. **Total False Warnings ($FW$):** Exact integer count.
2. **False Alarms on Gold Correct:** Spurious alerts on clean phonemes.
3. **Wrong-Type Error False Warnings:** Alerts where an error was present but misdiagnosed (S $\leftrightarrow$ O).
4. **Gap False Alarms:** Spurious insertion alerts on clean inter-phoneme boundaries.
5. **False Alert Rate per 1,000 Eligible Positions:** Normalized diagnostic alert density across target and gap positions.

### 7.4 Per-Speaker Robustness Endpoints
Separate computation of the above metrics for each of the 4 held-out sealed test speakers:
- `ZHAA` (Arabic, Female)
- `TNI` (Hindi, Male)
- `MBMPS` (Spanish, Male)
- `TLV` (Vietnamese, Male)

---

## 8. Quantitative Acceptance Criteria & Decision Rules

Evaluation on `TEST_SEALED` shall follow an objective, gate-based decision tree:

```
[TEST EVALUATION ACCEPTANCE GATES]
Gate 1 (Safety Ceiling):
  Candidate Joint FWR <= 0.2000
  [CRITICAL: If FWR > 0.2000, Candidate is IMMEDIATELY REJECTED for pedagogical violation]

Gate 2 (Pathology Recovery):
  Candidate Recall_S >= 2.0% AND Candidate Precision_S >= 40.0%
  AND Candidate Recall_S > Baseline Recall_S (which is 0.00%)
  [CRITICAL: Must prove genuine recovery of substitution detection]

Gate 3 (Omission Non-Degradation):
  Candidate Recall_O >= 0.70 * Baseline Recall_O
  [CRITICAL: No more than 30% relative loss in omission recall]

Gate 4 (Insertion Non-Degradation):
  Candidate Recall_I >= 0.85 * Baseline Recall_I
  [CRITICAL: Insertion performance must remain substantially intact]

Gate 5 (Macro-3 Superiority):
  Candidate Macro-3 Error F1 > Baseline Macro-3 Error F1
  [CRITICAL: Net multi-class improvement across all error types]
```

### 8.1 Formal Verdict Definitions

| Verdict Code | Condition | Operational Consequence |
|---|---|---|
| **`CANDIDATE_ACCEPTED_SUPERIOR`** | Passes all Gates 1, 2, 3, 4, 5 | Candidate promoted to Production Master / Core Landmark Benchmark |
| **`SAFETY_VIOLATION_REJECTED`** | Fails Gate 1 ($\text{FWR} > 0.2000$) | Candidate rejected for deployment; requires further domain adaptation |
| **`PATHOLOGY_PERSISTS_REJECTED`** | Fails Gate 2 ($\text{Recall}_S < 2.0\%$) | Asymmetric routing failed to generalize substitution recovery |
| **`TRADE_OFF_COLLAPSE_REJECTED`** | Fails Gate 3 or Gate 4 | Severe collateral damage to omission or insertion channels |
| **`NO_GAIN_REJECTED`** | Fails Gate 5 | Substitution gain offset by other channels; net F1 is negative or zero |

---

## 9. Single-Pass Execution Protocol (Strict Rules Before TEST Access)

To enforce zero-leakage custody, prospective evaluation must adhere strictly to this sequence:

1. **Step 1 — Pre-Test Readiness Verification:** Verify all regression tests pass (`pytest tests/`), checkpoint hashes match, and `FROZEN_ASYMMETRIC_POLICY.json` is validated.
2. **Step 2 — Single-Pass Audio Inference:** Execute waveform forward pass through frozen checkpoints on `TEST_SEALED` waveforms to generate continuous posterior arrays (`probs_target_neural`, `probs_target_fixed`, `probs_gap`).
3. **Step 3 — Blind Policy Application:**
   - Apply prespecified baseline policy (`FROZEN_CALIBRATED_POLICY.json`) unconditionally.
   - Apply primary asymmetric policy (`FROZEN_ASYMMETRIC_POLICY.json`) unconditionally using deterministic `score_margin`.
4. **Step 4 — Single-Pass Scoring:** Feed discrete predictions and continuous scores into `UnifiedEvaluatorV3` alongside gold labels.
5. **Absolute Prohibitions:**
   - **NO** iterative threshold sweeping on TEST.
   - **NO** post-hoc margin adjustments or tie-breaking modifications.
   - **NO** re-weighting of components or model retraining.
   - **NO** counterfactual threshold matching using TEST labels. Both systems are judged strictly on their frozen prospective operating points.

---

## 10. Pre-Test Audit Signatures

```
Lead Systems Auditor: Senior Speech ML Research Scientist
Evaluation Authority: EVAL-V3.1 Unified Validation Suite
Repository: /home/khoa/Projects/New-Method-Research
Pre-Registration Status: FULLY PREREGISTERED & CRYPTOGRAPHICALLY FROZEN
Firewall Status: SEALED (Zero TEST Access Prior to Formal Unlock Authorization)
```
