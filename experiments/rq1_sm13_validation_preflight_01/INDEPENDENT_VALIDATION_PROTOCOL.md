# LANGFENS RQ1-SM13-VALIDATION-PREFLIGHT-01 — Independent Prospective Validation Protocol

**Auditor:** Independent Speech ML Validation Lead & Scientific Auditor  
**Campaign:** `RQ1-SM13-VALIDATION-PREFLIGHT-01`  
**Governing Standard:** EVAL-V3.1 (Exact $N$ targets, $N+1$ insertion gaps)  
**Target Candidate Artifact:** `LANGFENS-SM13-FUSION-V1` (Full-TRAIN Seed 101, replication Seeds 202/303)  
**Protocol Version:** `VALIDATION-PROTOCOL-V1.0` (Pre-Registered & Frozen)  

---

## 1. Objectives and Scientific Mission

The objective of this protocol is to define an airtight, contamination-proof, and reproducible procedure for evaluating the prospective generalization and operational policy adherence of formally trained `LANGFENS-SM13-FUSION-V1` artifacts before any production deployment or formal scientific publication.

This protocol enforces **strict separation of custody** between:
1. **Calibration Phase:** Conducted solely on the isolated `DEV_CAL` partition to freeze operating thresholds.
2. **Prospective Evaluation Phase:** Conducted on the sealed `TEST_SEALED` holdout partition using frozen models and frozen thresholds.

---

## 2. Partition Architecture & Custody Firewall

### 2.1 Partition Registry & Demographics

| Partition | Role | Speakers | Utterances | Target Events ($N$) | Gap Events ($N+1$) | L1 Demographics | Firewall Status |
|---|---|---|:---:|:---:|:---:|---|:---:|
| `TRAIN` | Model Fitting Only | `ABA, SKA, BWC, LXC, ASI, SVBI, HKK, HJK, EBVS, NJS, HQTV, PNV` | 1,800 | 59,562 | 61,362 | Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese (6M / 6F) | FIT COMPLETED |
| `DEV_SEARCH` | Prior Method Ablation | `YBAA, NCC, RRBI, YDCK` | 599 | 19,845 | 20,444 | Arabic, Chinese, Hindi, Korean (2M / 2F) | EXPOSED (Ineligible for holdout) |
| **`DEV_CAL`** | **Operating Policy Calibration** | **`ERMS, THV`** | **300** | **9,927** | **10,227** | **Spanish (M), Vietnamese (F)** | **AUTHORIZED CALIBRATION ONLY** |
| **`TEST_SEALED`** | **Prospective Generalization Benchmark** | **`ZHAA, TNI, MBMPS, TLV`** | **600** | **19,845** | **20,445** | **Arabic (F), Hindi (M), Spanish (M), Vietnamese (M)** | **CRYPTO-SEALED UNTOUCHED** |

### 2.2 Mandatory Custody Firewall Invariants
1. **Zero Access During Preflight:** Neither audio nor labels of `DEV_CAL` or `TEST_SEALED` may be accessed during preflight auditing.
2. **Sequential Unlocking:** `TEST_SEALED` shall **NOT** be unlocked or accessed until `DEV_CAL` calibration is executed, audited, and cryptographically sealed.
3. **One-Way Data Flow:** Labels from `TEST_SEALED` shall only be accessed by the frozen `UnifiedEvaluatorV3` during final score tabulation. Predictions and model parameters must be locked prior to metric computation.

---

## 3. Explicit Contamination & Integrity Checks

Prior to executing prospective evaluation, the auditor must programmatically certify the following four contamination checks:

```
[CONTAMINATION CHECK MATRIX]
+-----------------------------------------------------------------------------------------+
| CHECK 1: BIOLOGICAL SPEAKER DISJOINTNESS                                                |
| Assert: set(TRAIN_SPEAKERS) & set(TEST_SEALED_SPEAKERS) == {}                           |
| Assert: set(DEV_CAL_SPEAKERS) & set(TEST_SEALED_SPEAKERS) == {}                         |
| Assert: set(DEV_SEARCH_SPEAKERS) & set(TEST_SEALED_SPEAKERS) == {}                      |
+-----------------------------------------------------------------------------------------+
| CHECK 2: PROHIBITION OF ALIAS CONTAMINATION (LXC / LXC_test)                            |
| Assert: 'LXC' not in set(TEST_SEALED_SPEAKERS)                                           |
| Assert: all(not rec['utterance_id'].startswith('LXC') for rec in TEST_SEALED)           |
+-----------------------------------------------------------------------------------------+
| CHECK 3: CRYPTOGRAPHIC INTEGRITY OF SEALED MANIFEST                                     |
| Compute: sha256(data/derived/l2arctic/manifests/test_sealed.jsonl)                      |
| Assert: actual_hash == "a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6" |
+-----------------------------------------------------------------------------------------+
| CHECK 4: NO PARAMETER RETRAINING OR GRADIENT UPDATES                                    |
| Assert: model_target.training == False and model_gap.training == False                  |
| Assert: torch.is_grad_enabled() == False                                                |
+-----------------------------------------------------------------------------------------+
```

---

## 4. Phase 1 — Preregistered Calibration Protocol on `DEV_CAL`

### 4.1 Calibration Execution Steps
1. **Operator Authorization:** Obtain explicit operator signature authorizing access to `data/derived/l2arctic/manifests/dev_cal.jsonl`.
2. **Inference Forward Pass:** Run frozen Full-TRAIN Seed 101 forward inference on all 300 `DEV_CAL` utterances:
   - Extract 13-layer acoustic representations from Wav2Vec2.
   - Run CharSiu forced alignment DP with canonical intended phones to determine centers.
   - Run SM13 Target and Gap mixing heads.
   - Run LinearCTC on Layer 6 and compute E4 sequence likelihood ratios.
   - Compute fixed 50/50 target fusion: $\mathbf{P}_{\text{fused}} = 0.5 \cdot \mathbf{P}_{\text{neural}} + 0.5 \cdot \mathbf{P}_{\text{E4}}$.
3. **Threshold Optimization under EVAL-V3.1 Budgets:**
   For each nominal budget $B \in \{0.20, 0.50, 1.00, 2.00\}$ false alarms per utterance:
   - Target False Warning Quota: $K_T(B) = \lfloor 0.80 \cdot B \cdot 300 \rfloor$.
     For $B = 0.20$: $K_T(0.20) = 48$ false alarms.
   - Gap False Insertion Quota: $K_G(B) = \lfloor 0.20 \cdot B \cdot 300 \rfloor$.
     For $B = 0.20$: $K_G(0.20) = 12$ false alarms.
   - Sort target predictions by confidence $\max(P_S, P_O)$ descending; select highest threshold $\tau_T^*(B)$ that limits false alarms (gold is `CORRECT` or wrong error type) to $\le K_T(B)$.
   - Sort gap predictions by $P_{\text{insert}}$ descending; select highest threshold $\tau_G^*(B)$ that limits false insertions (gold $y_G = 0$) to $\le K_G(B)$.
4. **Policy Serialization and Lock:**
   Serialize derived operating points into `experiments/rq1_sm13_validation_preflight_01/FROZEN_CALIBRATED_POLICY.json`:
   ```json
   {
     "calibration_partition": "DEV_CAL",
     "calibration_utterances": 300,
     "primary_budget": 0.20,
     "policies": {
       "0.20": {
         "target_threshold": "FLOAT",
         "target_tie_cutoff": "INT",
         "gap_threshold": "FLOAT",
         "gap_tie_cutoff": "INT"
       }
     }
   }
   ```
   Compute and log its SHA-256 checksum.

---

## 5. Phase 2 — Prospective Validation Protocol on `TEST_SEALED`

### 5.1 Prospective Evaluation Steps
1. **Input Contract:**
   - Input to inference: Native 16 kHz WAV audio files and canonical intended phone sequences (from prompt text via audited CMUDict reference lookup).
   - Prohibited Inputs: No gold segment timing, no realized phone transcriptions, no error labels.
2. **Inference Execution:**
   - Execute the frozen Full-TRAIN Seed 101 pipeline across all 600 `TEST_SEALED` utterances.
   - Save full continuous prediction arrays (`test_sealed_predictions_seed101.npz`).
3. **Operating Policy Application:**
   - Apply the frozen thresholds $(\tau_T^*(B), \tau_G^*(B))$ from `FROZEN_CALIBRATED_POLICY.json`.
   - Generate categorical target decisions ($0 = \text{CORRECT}, 1 = \text{SUBSTITUTION}, 2 = \text{OMISSION}$) and binary gap decisions ($0 = \text{NO\_INSERTION}, 1 = \text{INSERTION}$).
4. **Metric Tabulation via Frozen `UnifiedEvaluatorV3`:**
   Execute standard EVAL-V3.1 scoring against `TEST_SEALED` gold labels.

### 5.2 Metrics Reporting Hierarchy

```
[METRIC EVALUATION HIERARCHY]

1. PRIMARY CONFIRMATORY METRICS (DEPLOYABLE POLICY):
   - Macro-3 Error AP (Threshold-Free Area Under Precision-Recall Curve)
   - Macro-3 Error F1 at Nominal B = 0.20
   - Realized Empirical FWR at Nominal B = 0.20 (Target: <= 0.200000)

2. PEDAGOGICAL SAFETY & BIAS METRICS:
   - Target False Warning Rate (Target: <= 0.160000)
   - Gap False Warning Rate (Target: <= 0.040000)
   - Wrong-Type False Warnings Count
   - Fraction of Utterances with False Positive Warnings

3. SECONDARY DIAGNOSTIC METRICS:
   - Strictly Matched-Realized-FWR Macro-3 F1 (Matched at FWR = 0.2000)
   - Exact-Gap Insertion Precision @ Top-560
   - Per-Class Average Precision: AP-Substitution, AP-Omission, AP-Insertion
   - Macro-3 Target F1 (Target Domain Only) and Macro-4 Mixed-Domain F1
```

---

## 6. Generalization Scope Disclosures & Governance

The prospective validation report must explicitly state:
1. **L1 Demographic Generalization:**
   Results on `TEST_SEALED` represent prospective generalization to **Arabic, Hindi, Spanish, and Vietnamese** adult non-native speakers. Generalization to **Chinese and Korean** speakers is unassessed due to the historical composition of `TEST_SEALED`.
2. **Corpus Style Generalization:**
   The evaluation tests read speech from the ARCTIC phonetically balanced prompt set. Spontaneous conversational speech is not evaluated.
3. **Hardware Runtime Parity:**
   Reported inference speed must document whether execution occurred on CPU or GPU using the audited runtime equations established in `END_TO_END_RUNTIME_AUDIT.md`.
