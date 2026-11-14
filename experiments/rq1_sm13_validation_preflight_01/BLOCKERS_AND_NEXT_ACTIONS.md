# LANGFENS RQ1-SM13-VALIDATION-PREFLIGHT-01 — Blockers & Next Actions

**Auditor:** Independent Speech ML Validation Lead & Scientific Auditor  
**Campaign:** `RQ1-SM13-VALIDATION-PREFLIGHT-01`  
**Target Candidate Artifact:** `LANGFENS-SM13-FUSION-V1` (Full-TRAIN Seed 101)  
**Readiness Status:** **NO-GO (PREFLIGHT GATED — OPERATOR APPROVAL REQUIRED)**  

---

## 1. Executive Summary & Blocking Decision

Following an exhaustive forensic audit across Artifact Integrity (Phase 0), Warning Policy Transfer (Lane A), Split & Generalization Integrity (Lane B), and Complete End-to-End Inference (Lane C), the Independent Speech ML Validation Lead issues a formal **NO-GO** on unsealing `TEST_SEALED`.

Independent validation on the sealed benchmark is **BLOCKED** pending the resolution of two operational calibration discrepancies. Proceeding directly to `TEST_SEALED` with the current uncalibrated artifact would result in severe false warning rate failure ($> 0.77$ FWR vs $0.20$ target) and permanently spoil the unexposed status of the benchmark.

---

## 2. Inventory of Blocking & Non-Blocking Discrepancies

### 2.1 Critical Scientific Blockers (Must Be Resolved Prior to TEST_SEALED Access)

| # | Discrepancy / Blocker | Severity | Forensic Root Cause | Resolution Required |
|---|---|:---:|---|---|
| **BLK-01** | **Uncalibrated Warning Policy on Full-TRAIN Artifacts** | **CRITICAL / BLOCKING** | A silent file path error in `train_sm13_formal.py` caused thresholds to default to uncalibrated dummy constants ($\tau_T = 0.70, \tau_G = 0.50$), producing in-sample FWR values of $0.776111 / 0.496667 / 0.550000$. No deployable policy is currently frozen for the full-TRAIN model. | Execute the preregistered calibration protocol on `DEV_CAL` to derive and freeze $(\tau_T^*, \tau_G^*)$ for nominal $B \in \{0.2, 0.5, 1.0, 2.0\}$. |
| **BLK-02** | **Preregistered DEV_CAL Calibration Authorization Pending** | **PROCEDURAL / BLOCKING** | Per zero-leakage research protocol, the auditor cannot access `data/derived/l2arctic/manifests/dev_cal.jsonl` without explicit written authorization from the Lead Operator. | Lead Operator must authorize access to `DEV_CAL` for threshold calibration only. |
| **BLK-03** | **Chinese / Korean Generalization Blind Spot in TEST_SEALED** | **SCIENTIFIC LIMITATION** | `TEST_SEALED` contains only Arabic, Hindi, Spanish, and Vietnamese speakers. Chinese and Korean speakers are absent. Historical attempts to use `LXC` (from TRAIN) as an evaluation speaker represent direct speaker contamination. | Disclose in all prospective benchmark reports that generalization to Chinese and Korean remains unconfirmed until external corpora are authorized. Bar any use of `LXC` or `LXC_test`. |
| **BLK-04** | **Latency Underreporting Correction** | **ENGINEERING DOCUMENTATION** | Published runtime benchmark ($74.03\text{ ms}$) omitted CharSiu forward pass, Viterbi alignment DP, and CTC/E4 sequence likelihood scoring ($67.8\%$ of true runtime). | Update all engineering specifications and deployment documents to reflect the true audited latency: **$223.09\text{ ms}$ on GPU** and **$840.66\text{ ms}$ on CPU**. |

### 2.2 Verified Non-Blocking Invariants (Certified Ready)

| Dimension | Audit Status | Forensic Evidence |
|---|:---:|---|
| **Artifact & Code Lineage** | **PASS** | Bit-exact verification of 79/79 files in `rq1_sm13_formal_train_01`, 56/56 in `rq1_method_select_02r`, and 24/24 in `rq1_method_select_02`. Zero corrupted checkpoints. |
| **Checkpoint Loadability** | **PASS** | Target Head, Gap Head, and LinearCTC state dicts load cleanly across all three registered seeds (101, 202, 303). |
| **Event Cardinality** | **PASS** | Exact $N$ targets and $N+1$ gaps strictly guaranteed across arbitrary phoneme sequence lengths. |
| **Real-Time Deployability** | **PASS** | True RTF of $0.0549$ (GPU, $18.2\times$ faster than real time) and $0.2130$ (CPU, $4.7\times$ faster than real time) satisfies interactive tutoring constraints ($\text{RTF} < 0.25$). |
| **Memory Footprint** | **PASS** | Peak GPU VRAM of $855.9\text{ MB}$ leaves $> 79\%$ headroom on a 4GB GTX 1650. |
| **Speaker Disjointness** | **PASS** | `L2_ARCTIC_SPLIT.yaml` maintains zero speaker overlap across fit and evaluation partitions. No `LXC_test` exists in `TEST_SEALED`. |

---

## 3. Four-Step Unblocking Roadmap

```
+-----------------------------------------------------------------------------+
| STEP 1: OPERATOR FORMAL REVIEW & AUTHORIZATION                              |
| - Lead Operator reviews preflight findings (VALIDATION_PREFLIGHT_FINAL.md). |
| - Operator signs off on DEV_CAL calibration authorization.                  |
| Status: AWAITING OPERATOR INPUT                                             |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| STEP 2: DEV_CAL THRESHOLD CALIBRATION EXECUTION                             |
| - Load full-TRAIN Seed 101 model.                                           |
| - Run complete inference on 300 DEV_CAL utterances (ERMS, THV).             |
| - Derive tau_T*(B) and tau_G*(B) under EVAL-V3.1 80/20 budget allocations.  |
| - Generate FROZEN_CALIBRATED_POLICY.json with immutable SHA-256 lock.       |
| Status: READY FOR IMMEDIATE EXECUTION UPON AUTHORIZATION                    |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| STEP 3: PRE-TEST AUDIT GATE CONFIRMATION                                    |
| - Auditor verifies FROZEN_CALIBRATED_POLICY.json integrity.                 |
| - Auditor verifies TEST_SEALED pristine hash (a5636249f7...).                |
| - Operator authorizes one-way unsealing of TEST_SEALED.                     |
| Status: GATED BEHIND STEP 2                                                 |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| STEP 4: PROSPECTIVE INDEPENDENT TEST EXECUTION                              |
| - Execute frozen Seed 101 on 600 TEST_SEALED utterances.                    |
| - Apply frozen calibrated policy (B=0.20 primary, B in {0.5, 1.0, 2.0}).    |
| - Score with frozen UnifiedEvaluatorV3; publish final prospective report.   |
| Status: GATED BEHIND STEP 3                                                 |
+-----------------------------------------------------------------------------+
```

---

## 4. Operator Sign-Off Request Template

The Lead Operator is requested to provide formal authorization for the next validation phase:

```markdown
### OPERATOR AUTHORIZATION DECISION:

[ ] OPTION 1: AUTHORIZE DEV_CAL CALIBRATION ONLY
    Permit execution of Phase 1 calibration on data/derived/l2arctic/manifests/dev_cal.jsonl
    to derive and freeze FROZEN_CALIBRATED_POLICY.json. Keep TEST_SEALED strictly sealed.

[ ] OPTION 2: ABORT / REQUEST FURTHER INVESTIGATION
    Require additional architecture or training revisions before proceeding.

Signed: ___________________________ (Lead Operator)
Date:   ___________________________
```
