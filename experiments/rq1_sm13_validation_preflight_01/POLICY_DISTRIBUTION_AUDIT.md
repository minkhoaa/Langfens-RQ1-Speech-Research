# LANGFENS RQ1-SM13-VALIDATION-PREFLIGHT-01 — Warning Policy Distribution Audit

**Auditor:** Independent Speech ML Validation Lead & Scientific Auditor  
**Campaign:** `RQ1-SM13-VALIDATION-PREFLIGHT-01` (Lane A)  
**Evaluated Artifact:** `LANGFENS-SM13-FUSION-V1` (Full-TRAIN Seed 101, replication Seeds 202/303)  
**Standard:** EVAL-V3.1 (Exact $N$ targets, $N+1$ insertion gaps)  
**Operating Standard:** Nominal False Warning Budget $B = 0.20$ false warnings per utterance ($0.16$ targets, $0.04$ gaps)  
**Audit Verdict:** **POLICY TRANSFER BLOCKED / EXTERNAL FWR CONTROL UNCONFIRMED**  

---

## 1. Executive Summary & Audit Scope

This audit investigates the severe numerical discrepancy observed between:
1. **Cross-Validation Realized FWR ($B=0.20$):** **$0.235370$** false warnings per utterance across 4 outer folds (1,800 utterances total).
2. **Full-TRAIN In-Sample FWR:** **$0.776111$** (Seed 101), **$0.496667$** (Seed 202), and **$0.550000$** (Seed 303).
3. **Target Nominal Policy Budget:** **$0.200000$** false warnings per utterance.

Through independent script execution, score distribution profiling, threshold provenance tracing, and AST inspection of the formal training engine, this audit establishes the exact mathematical, forensic, and behavioral causes of this discrepancy without using training labels for posthoc tuning.

---

## 2. Independent Recomputation on Retained Predictions

All calculations were independently executed on disk-persisted prediction arrays using `UnifiedEvaluatorV3`:
- CV OOF Predictions: `experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed{101,202,303}.npz`
- Full-TRAIN In-Sample Predictions: `experiments/rq1_sm13_formal_train_01/predictions/predictions_sm13_seed{101,202,303}_fulltrain.npz`

### 2.1 Recomputed Evaluation Metrics Table

| Evaluation Regime | Seed | Macro-3 AP | Macro-3 F1 ($B=0.2$) | Joint FWR ($B=0.2$) | Total False Alarms | Gold Correct FP | Wrong-Type Warnings | Gap False Insertions |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CV Outer OOF (METHOD-SELECT-02R)** | 101 | 0.324357 | 0.184115 | 0.234444 | 422 | 207 | 52 | 163 |
| **CV Outer OOF (METHOD-SELECT-02R)** | 202 | 0.320912 | 0.196812 | 0.281667 | 507 | 348 | 81 | 78 |
| **CV Outer OOF (METHOD-SELECT-02R)** | 303 | 0.336630 | 0.182595 | 0.190000 | 342 | 222 | 45 | 75 |
| **CV OOF Multi-Seed Mean** | **Pooled** | **0.327300** | **0.187841** | **0.235370** | **423.7** | **259.0** | **59.3** | **105.3** |
| | | | | | | | | |
| **Full-TRAIN In-Sample (SM13-FORMAL-01)** | 101 | 0.580630 | 0.417067 | **0.776111** | 1,397 | 437 | 15 | 945 |
| **Full-TRAIN In-Sample (SM13-FORMAL-01)** | 202 | 0.593743 | 0.459214 | **0.496667** | 894 | 533 | 15 | 346 |
| **Full-TRAIN In-Sample (SM13-FORMAL-01)** | 303 | 0.589308 | 0.449221 | **0.550000** | 990 | 465 | 25 | 500 |
| **Full-TRAIN In-Sample Mean** | **Pooled** | **0.587894** | **0.441834** | **0.607593** | **1,093.7** | **478.3** | **18.3** | **597.0** |

*Verification Finding:* The numbers $0.776111$, $0.496667$, and $0.550000$ exactly match the `joint_fwr` entries stored in `FORMAL_TRAINING_RECEIPT.json`. Event counting, wrong-type error assignment, and insertion gap indexing follow official EVAL-V3.1 specifications.

---

## 3. Forensic Root-Cause Diagnosis of the Discrepancy

Forensic inspection reveals that the discrepancy is driven by **three compounding factors**:

### 3.1 Primary Bug: Silent Path Failure and Uncalibrated Dummy Fallback
In `experiments/rq1_sm13_formal_train_01/train_sm13_formal.py` (lines 315–329):
```python
# Load reference warning policy thresholds (from inner OOF derivation of seed 101)
ref_policy_path = ROOT / "experiments/rq1_train_phase_01/formal/seed_101/outer_0/inner_0/policy.json"
t_thresh_02 = 0.70
g_thresh_02 = 0.50
if ref_policy_path.exists():
    pol_data = json.loads(ref_policy_path.read_text())
    t_thresh_02 = pol_data["policies"]["fixed/0.2"]["target"]["threshold"]
    g_thresh_02 = pol_data["policies"]["fixed/0.2"]["gap"]["threshold"]
```

1. **Missing File Path:** The directory `.../outer_0/` contains `policies.json` (at the outer fold level), not `inner_0/policy.json`.
2. **Silent Failure:** `ref_policy_path.exists()` evaluated to `False`. The code did not raise an exception or assertion error.
3. **Arbitrary Dummy Values Applied:** The training script executed inference using the fallback constants:
   $$\tau_T = 0.70, \quad \tau_G = 0.50$$
4. **Catastrophic Gap Over-Triggering:** In the gap insertion distribution, setting $\tau_G = 0.50$ triggers **945 false alarms** on negative gap positions (gold $y_G = 0$) for Seed 101, accounting for $\frac{945}{1800} = 0.525$ FWR on its own!

### 3.2 Actual Calibrated Thresholds in Cross-Validation
In `METHOD-SELECT-02R`, the actual inner-OOF-derived thresholds for nominal $B=0.20$ were:
- Fold 0 (`ABA, ASI, BWC` held): $\tau_T = 0.8348, \; \tau_G = 0.6784$
- Fold 1 (`EBVS, HJK, HKK` held): $\tau_T = 0.8408, \; \tau_G = 0.7642$
- Fold 2 (`HQTV, LXC, NJS` held): $\tau_T = 0.7977, \; \tau_G = 0.8281$
- Fold 3 (`PNV, SKA, SVBI` held): $\tau_T = 0.8379, \; \tau_G = 0.6873$
- **Mean Calibrated Thresholds:** $\bar{\tau}_T \approx 0.8278, \; \bar{\tau}_G \approx 0.7395$

When the mean calibrated thresholds ($\tau_T = 0.82, \tau_G = 0.74$) are applied to the full-TRAIN predictions:
- Target False Alarms: 121 ($0.0672$ / utt; budget is $0.160$)
- Gap False Alarms: 332 ($0.1844$ / utt; budget is $0.040$)
- Total FWR: **$0.2517$** false alarms per utterance (comparable to the CV realized FWR of $0.235$, and completely resolving the $0.776$ anomaly).

### 3.3 Score Distribution Sharpening Under Full-TRAIN Refitting
Comparing the score distributions between CV OOF predictions and Full-TRAIN predictions demonstrates that refitting on all 1,800 utterances changes confidence sharpness:

| Model / Split | Feature | Mean | Median (P50) | P90 | P95 | P99 | Max |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CV OOF Seed 101** | Target Neural Conf | 0.2011 | 0.0810 | 0.6267 | 0.7769 | 0.9203 | 0.9993 |
| **Full-TRAIN Seed 101** | Target Neural Conf | 0.2162 | 0.0692 | 0.7179 | 0.8353 | 0.9471 | 0.9989 |
| **CV OOF Seed 101** | Target Fixed Conf | 0.2318 | 0.1865 | 0.5015 | 0.6231 | 0.8545 | 0.9920 |
| **Full-TRAIN Seed 101** | Target Fixed Conf | 0.2355 | 0.1848 | 0.5142 | 0.6592 | 0.8826 | 0.9960 |
| **CV OOF Seed 101** | Gap Insertion Prob | 0.0485 | 0.0176 | 0.1138 | 0.1996 | 0.5182 | 0.9968 |
| **Full-TRAIN Seed 101** | Gap Insertion Prob | 0.0736 | 0.0316 | 0.1744 | 0.2900 | 0.7272 | 0.9995 |

*Key Distribution Insight:* Full-data fitting on 12 speakers produces higher confidence at the upper tail (e.g., the 95th percentile target confidence shifts from $0.777$ to $0.835$, and gap P99 shifts from $0.518$ to $0.727$). Consequently, **thresholds calibrated on 9-speaker OOF folds cannot be directly assumed to maintain exact false alarm budgets when transferred to full-data models**.

---

## 4. Matched-Realized-FWR Diagnostic Evidence vs Deployable Policy

To ensure scientific integrity, a strict distinction must be maintained between **diagnostic matched-FWR curves** and **deployable policy performance**:

```
+-----------------------------------------------------------------------------+
| 1. DIAGNOSTIC MATCHED-REALIZED-FWR CURVES (POSTHOC SCIENTIFIC EVIDENCE)     |
| - Computed by tracing continuous operating curves using ground truth labels.|
| - Forced equal realized false warning rate (e.g. exactly FWR = 0.2000).     |
| - Proves SM13's F1 gain (+10.58% at FWR=0.20) is genuine, not an artifact    |
|   of running at a higher false alarm rate.                                  |
| - CANNOT BE DEPLOYED: Requires gold labels which do not exist at inference. |
+-----------------------------------------------------------------------------+
                                     VS
+-----------------------------------------------------------------------------+
| 2. DEPLOYABLE WARNING POLICY (OPERATIONAL MECHANISM)                        |
| - Uses fixed, frozen numerical thresholds (tau_T, tau_G) derived a priori.  |
| - Operates purely on predicted confidence scores at test time.             |
| - Realized test FWR may deviate from nominal target due to acoustic drift. |
| - MUST BE FROZEN BEFORE TEST ACCESS: External FWR control must be certified |
|   on a held-out calibration partition (DEV_CAL).                            |
+-----------------------------------------------------------------------------+
```

---

## 5. Preregistered DEV_CAL Policy Calibration Protocol

Because the full-TRAIN models must never be calibrated using their own training labels, we preregister a calibration protocol on the isolated `DEV_CAL` partition.

### 5.1 Protocol Specifications
1. **Calibration Partition:** `DEV_CAL` (`data/derived/l2arctic/manifests/dev_cal.jsonl`).
   - Population: 300 utterances from 2 unseen speakers (`ERMS` [Spanish M], `THV` [Vietnamese F]).
   - Total Target Events: $N_T = 9,927$.
   - Total Gap Positions: $N_G = 10,227$.
2. **Pedagogical Budget Allocation (EVAL-V3.1):**
   For nominal false warning budget $B \in \{0.20, 0.50, 1.00, 2.00\}$:
   - Target False Alarm Quota:
     $$K_T(B) = \lfloor 0.80 \cdot B \cdot 300 \rfloor \quad \implies \quad K_T(0.20) = 48 \text{ false alarms}$$
   - Gap False Insertion Quota:
     $$K_G(B) = \lfloor 0.20 \cdot B \cdot 300 \rfloor \quad \implies \quad K_G(0.20) = 12 \text{ false alarms}$$
3. **Threshold Selection Rule:**
   - On `DEV_CAL`, run frozen Full-TRAIN Seed 101 forward inference to obtain fixed target probabilities $\mathbf{p}_i \in \Delta^2$ and gap probabilities $q_j \in [0, 1]$.
   - Target confidence: $c_i = \max(p_{i, \text{SUB}}, p_{i, \text{OMI}})$; predicted class $\hat{y}_i = \arg\max_{c \in \{\text{SUB}, \text{OMI}\}} p_{i, c}$.
   - Binary target false alarm indicator: $e_i = \mathbf{1}(\hat{y}_i \neq y_i^*)$, where $y_i^*$ is the gold label.
   - Sort target events by confidence descending, with deterministic SHA-256 tie-breaking:
     $$\tau_T^*(B) = \arg\max_\tau \left\{ \sum_{i=1}^{N_T} \mathbf{1}(c_i \ge \tau \land e_i = 1) \le K_T(B) \right\}$$
   - Gap false alarm indicator: $g_j = \mathbf{1}(y_{j, \text{gap}}^* = 0)$.
   - Sort gap events by $q_j$ descending with deterministic SHA-256 tie-breaking:
     $$\tau_G^*(B) = \arg\max_\tau \left\{ \sum_{j=1}^{N_G} \mathbf{1}(q_j \ge \tau \land g_j = 1) \le K_G(B) \right\}$$
4. **Serialization and Freezing:**
   Save all thresholds to `FROZEN_CALIBRATED_POLICY.json` with SHA-256 hash. These values become immutable prior to any evaluation on `TEST_SEALED`.

---

## 6. Audit Verdict and Mandatory Gate Actions

- **LANE A VERDICT:** **POLICY TRANSFER BLOCKED / UNCONFIRMED**
- **Blocking Condition:** The full-TRAIN artifact does not possess a validly calibrated deployable policy. The training receipt values ($0.776 / 0.497 / 0.550$) represent an uncalibrated dummy fallback ($0.70 / 0.50$).
- **Mandatory Next Action:** The operator must authorize execution of the preregistered DEV_CAL calibration protocol before independent testing on `TEST_SEALED` can proceed.
