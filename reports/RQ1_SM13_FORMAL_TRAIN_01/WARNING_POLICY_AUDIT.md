# LANGFENS RQ1-SM13-FORMAL-TRAIN-01 — Warning Policy & Operational Calibration Audit

**Auditor:** Principal Speech ML Engineer & Scientific Reproducibility Auditor  
**Method:** `LANGFENS-SM13-FUSION-V1`  
**Evaluation Standard:** EVAL-V3.1  
**Operating Standard:** Nominal Budget $B \in \{0.2, 0.5, 1.0, 2.0\}$ False Warnings per Utterance  
**Audit Status:** **POLICY READY — EXTERNAL FWR CONTROL STRICTLY UNCONFIRMED**  

---

## 1. Executive Summary & Calibration Governance

The educational utility of an automated pronunciation diagnostic system hinges on predictable warning rates. Excessive false alarms induce student fatigue, while under-triggering misses critical errors. 

In the LANGFENS evaluation protocol:
- Operating thresholds $(\tau_S, \tau_O, \tau_I)$ are derived strictly on **inner out-of-fold (OOF)** validation splits.
- **Zero Outer-Label Tuning:** Outer evaluation fold labels are never inspected or used to tune operating points.
- Operating points are evaluated across nominal budgets $B \in \{0.2, 0.5, 1.0, 2.0\}$ false alarms per utterance.

This audit evaluates the calibration provenance of the reference policy, quantifies the empirical false warning rate (FWR) overshoot observed on development cross-validation, analyzes the posthoc matched-realized-FWR diagnostic curves, and formally defines the boundary of operational validity.

---

## 2. Policy Provenance & Budget Partitioning

### 2.1 The 80/20 Pedagogical Warning Budget Allocation
For a nominal total false warning budget $B$ (in units of false alarms per utterance):
- **Target Error Budget:** $0.80 \times B$ per utterance.
  - Controls false alarms on target substitutions ($S$) and omissions ($O$).
  - A warning on a target phone is considered a false alarm if the gold label is `CORRECT` or if a wrong error type is triggered (e.g. diagnosing an omission when the speaker made a substitution).
- **Gap Insertion Budget:** $0.20 \times B$ per utterance.
  - Controls false alarms on inter-phoneme insertion ($I$).
  - A warning on a gap position is a false alarm if the gold label is `NO_INSERTION`.

### 2.2 Inner OOF Derivation Protocol
In each outer fold (comprising 9 fit speakers and 3 held-out speakers), the 9 fit speakers are partitioned into 3 inner cross-validation folds (6 fit, 3 held). Models are fit on the 6 inner speakers and predict the 3 inner held speakers. All inner predictions are concatenated into an inner OOF pool ($N_{\text{inner}} = 1,350$ utterances).

Operating thresholds are derived by sorting inner predictions by confidence and selecting the threshold that caps cumulative false alarms to $\lfloor B_{\text{domain}} \times N_{\text{inner}} \rfloor$:
$$\tau_T(B) = \arg\max_\tau \left\{ \sum_{i=1}^{N_{\text{inner}}} \mathbf{1}(\text{conf}_i \ge \tau \land \text{is\_false}_i) \le 0.80 \cdot B \cdot N_{\text{inner}} \right\}$$
$$\tau_G(B) = \arg\max_\tau \left\{ \sum_{j=1}^{N_{\text{inner}}} \mathbf{1}(p_{j, I} \ge \tau \land y_{j, I} = 0) \le 0.20 \cdot B \cdot N_{\text{inner}} \right\}$$

---

## 3. Audit of Development Calibration vs Nominal Target

### 3.1 Nominal Budget vs Achieved Realized FWR
When the inner-derived thresholds are applied to outer held-out speakers across all 4 folds (1,800 utterances total), the achieved empirical False Warning Rate (FWR) exhibits systematic overshoot:

| Nominal Budget ($B$) | Target Budget ($0.8 B$) | Gap Budget ($0.2 B$) | Achieved Dev FWR | Absolute Delta | Relative Overshoot | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$B = 0.20$** *(Primary)* | 0.160 | 0.040 | **$0.235370$** | $+0.035370$ | **$+17.69\%$** | **OVERSHOOT** |
| **$B = 0.50$** | 0.400 | 0.100 | **$0.542407$** | $+0.042407$ | **$+8.48\%$** | **OVERSHOOT** |
| **$B = 1.00$** | 0.800 | 0.200 | **$1.050741$** | $+0.050741$ | **$+5.07\%$** | **SLIGHT OVERSHOOT** |
| **$B = 2.00$** | 1.600 | 0.400 | **$2.005185$** | $+0.005185$ | **$+0.26\%$** | **NEAR-EXACT** |

### 3.2 Root-Cause Forensic Analysis of the $+17.7\%$ FWR Overshoot
1. **Speaker Acoustic Disparity:** L2-ARCTIC exhibits substantial cross-speaker accent diversity (Vietnamese, Chinese, Arabic, Hindi, Korean, Spanish). Thresholds calibrated on 9 speakers do not generalize perfectly to 3 held-out speakers with differing acoustic baseline energies and phone durations.
2. **Scalar-Mixed Head Confidence Calibration:** Because `ScalarMixingTargetHead` improves discriminative separation between correct phones and substitutions, the distribution of neural confidence scores is sharper than the single-layer Phase-01 baseline. At the strict $B=0.20$ tail, a slight distribution shift on held-out speakers produces a modest increase in triggered warnings.
3. **Wrong-Type False Warnings:** In EVAL-V3.1, misclassifying a realized substitution as an omission counts as a false warning. In phonetically blurred non-native speech (e.g. reduced unstressed vowels), substitution vs omission boundaries are inherently ambiguous.

---

## 4. Matched-Realized-FWR Diagnostic Evidence

Because nominal operating points produce differing realized false warning rates across candidate models, scientific comparisons must be verified on **matched realized false warning rates**. 

In `MATCHED_FWR_ANALYSIS.csv`, both systems were evaluated along their continuous threshold curves to identify their exact performance when forced to emit identical empirical false alarms:

| Operating Level | Phase-01 Official Baseline Macro-3 F1 | LANGFENS SM13 Fusion Macro-3 F1 | Absolute Gain ($\Delta \text{F1}$) | Relative F1 Gain |
|---|:---:|:---:|:---:|:---:|
| **Strictly Matched $\text{FWR} = 0.1000$** | 0.107083 | **0.119345** | $+0.012262$ | **$+11.45\%$** |
| **Strictly Matched $\text{FWR} = 0.1500$** | 0.132440 | **0.148685** | $+0.016245$ | **$+12.27\%$** |
| **Strictly Matched $\text{FWR} = 0.2000$** | 0.155884 | **0.172369** | $+0.016485$ | **$+10.58\%$** |
| **Strictly Matched $\text{FWR} = 0.2044$** *(Baseline Nominal Level)* | 0.158420 | **0.173022** | $+0.014602$ | **$+9.22\%$** |
| **Strictly Matched $\text{FWR} = 0.2356$** *(SM13 Nominal Level)* | 0.168308 | **0.184047** | $+0.015739$ | **$+9.35\%$** |
| **Strictly Matched $\text{FWR} = 0.3000$** | 0.188586 | **0.203590** | $+0.015004$ | **$+7.96\%$** |
| **Strictly Matched $\text{FWR} = 0.4000$** | 0.212569 | **0.227499** | $+0.014930$ | **$+7.02\%$** |
| **Strictly Matched $\text{FWR} = 0.5000$** | 0.230745 | **0.241702** | $+0.010957$ | **$+4.75\%$** |

### Critical Scientific Principle:
Matched-realized-FWR analysis is **posthoc diagnostic evidence** that proves SM13's performance gain is genuine and not an artifact of running at a higher false alarm rate. However, **matched-FWR thresholds cannot be deployed directly** because computing realized FWR requires ground-truth labels that do not exist at deployment time.

---

## 5. Formal Operational Limitation & Handoff Verdict

### 5.1 Scientific Limitations Disclosed
1. **Outer Label Tuning Prohibited:** We refuse to tune deployable thresholds on outer-fold test data. Such practice constitutes label leakage and invalidates scientific generalization.
2. **Unconfirmed External FWR Control:** Because the only eligible calibration source was the TRAIN partition, the degree of FWR control on external, unseen speakers and acoustic environments remains **UNCONFIRMED**.

### 5.2 Official Operational Status
The warning policy status for `LANGFENS-SM13-FUSION-V1` is certified as:
$$\textbf{POLICY\_READY (External FWR Control: UNCONFIRMED)}$$

**Operational Directive:** Downstream deployment or production release is **PROHIBITED** until the model and its frozen policy thresholds undergo the independent validation gate on held-out calibration partitions (`DEV_CAL` and `DEV_SEARCH`).
