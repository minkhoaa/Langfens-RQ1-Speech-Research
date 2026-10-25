# LANGFENS — RQ1-TRN-01B0-R: Corrected Training Subset Evidence & Data Leakage Audit

**Gate Reference:** RQ1-TRN-01B0-R — Real-Data Learning Diagnostics & Evidence Reconciliation  
**Deliverable Path:** `manifests/RQ1-TRN-01B0/reconciliation/CORRECTED_SUBSET_LEAKAGE_AUDIT.md`  
**Authoritative Split Reference:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`  
**Authoritative Training Subset:** `manifests/RQ1-TRN-01B0/TRAIN_ONLY_SUBSET_MANIFEST.json`  
**Superceded Preliminary Audit:** `manifests/RQ1-TRN-01B0/DATA_LEAKAGE_AUDIT.md` (Retained for provenance)  
**Lead Research Engineers & Reviewers:** Senior Speech ML Research Engineer, PyTorch Training Engineer, Scientific Reproducibility Reviewer  
**Audit Status:** `DATA_LEAKAGE_AUDIT: PASS (Zero Leakage Formally Certified)`  

---

## 1. Executive Summary & Provenance Addendum

During independent scientific reproducibility review of gate **RQ1-TRN-01B0**, a critical evidence discrepancy was identified between the documentation in `DATA_LEAKAGE_AUDIT.md` and the actual training input manifest `TRAIN_ONLY_SUBSET_MANIFEST.json`:
- **Discrepancy:** The markdown table in Section 3 of `DATA_LEAKAGE_AUDIT.md` listed 32 utterance IDs (e.g., `BWC/arctic_a0003`, `BWC/arctic_a0006`, `ERMS/arctic_a0003`, `ERMS/arctic_a0007`) which had **zero overlap** with the 32 utterance IDs in `TRAIN_ONLY_SUBSET_MANIFEST.json` (e.g., `BWC/arctic_a0571`, `BWC/arctic_b0415`, `ERMS/arctic_a0269`, `ERMS/arctic_b0150`).
- **Root Cause Analysis:** During the drafting of RQ1-TRN-01B0, an early prototype listing of candidate utterances (the first two alphanumeric recordings per Fold 1 train speaker, e.g., `a0001`–`a0007`) was tabulated into the report template. Concurrently, `audit_scripts/select_train_subset.py` was executed with random seed 42 and strict error-coverage heuristics (prioritizing utterances containing both phoneme substitutions and deletions to maximize representation). The training script `audit_scripts/run_training_sanity.py` ingested `TRAIN_ONLY_SUBSET_MANIFEST.json`. However, the documentation table in `DATA_LEAKAGE_AUDIT.md` was never regenerated from the serialized JSON manifest.
- **Scientific Impact:** Both the prototype list and the selected manifest list belong strictly to Fold 1 train speakers ($S_{\text{train}}$) and Fold 1 train utterances ($U_{\text{train}}$). Neither list contained any validation ($U_{\text{val}}$) or test ($U_{\text{test}}$) samples. Thus, **no parameter updates or feature representations were exposed to validation or test data**. However, because the audit documentation presented a completely different set of IDs, scientific provenance was broken.
- **Resolution:** This document replaces the obsolete documentation table with the authoritative, programmatically verified table derived directly from `TRAIN_ONLY_SUBSET_MANIFEST.json`. The original report `DATA_LEAKAGE_AUDIT.md` is retained intact for audit provenance.

---

## 2. Partition Disjointness Proof (Fold 1)

From the frozen split manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`):

### 2.1 Speaker Partition Disjointness
- **Train Speakers ($S_{\text{train}}$, 16 speakers):**  
  `['BWC', 'ERMS', 'HKK', 'HQTV', 'MBMPS', 'NCC', 'SKA', 'SVBI', 'THV', 'TLV', 'TNI', 'TXHC', 'YBAA', 'YDCK', 'YKWK', 'ZHAA']`
- **Validation Speakers ($S_{\text{val}}$, 2 speakers):**  
  `['NJS', 'RRBI']`
- **Test Speakers ($S_{\text{test}}$, 6 speakers):**  
  `['ABA', 'ASI', 'EBVS', 'HJK', 'LXC', 'PNV']`

**Formal Set Relations:**
$$S_{\text{train}} \cap S_{\text{val}} = \emptyset \quad (\text{Disjointness Certified})$$
$$S_{\text{train}} \cap S_{\text{test}} = \emptyset \quad (\text{Disjointness Certified})$$
$$S_{\text{val}} \cap S_{\text{test}} = \emptyset \quad (\text{Disjointness Certified})$$
$$|S_{\text{train}}| + |S_{\text{val}}| + |S_{\text{test}}| = 16 + 2 + 6 = 24 = |S_{\text{corpus}}|$$

### 2.2 Utterance Partition Disjointness
- **Train Utterances ($U_{\text{train}}$):** Exactly $2,399$ utterances.
- **Validation Utterances ($U_{\text{val}}$):** Exactly $300$ utterances.
- **Test Utterances ($U_{\text{test}}$):** Exactly $900$ utterances.
- **Total Corpus Accounted ($U_{\text{corpus}}$):** Exactly $3,599$ utterances.

**Formal Set Relations:**
$$U_{\text{train}} \cap U_{\text{val}} = \emptyset \quad (\text{Zero Utterance Overlap})$$
$$U_{\text{train}} \cap U_{\text{test}} = \emptyset \quad (\text{Zero Utterance Overlap})$$
$$U_{\text{val}} \cap U_{\text{test}} = \emptyset \quad (\text{Zero Utterance Overlap})$$
$$|U_{\text{train}}| + |U_{\text{val}}| + |U_{\text{test}}| = 2,399 + 300 + 900 = 3,599$$

---

## 3. Programmatic Training Subset Audit ($N=32$)

The table below was generated programmatically by inspecting every entry in `TRAIN_ONLY_SUBSET_MANIFEST.json` and asserting its membership against the sets $U_{\text{train}}$, $U_{\text{val}}$, and $U_{\text{test}}$:

| # | Utterance Full ID | Speaker ID | Duration | Canonical | Realized | Sub | Del | Cor | In $U_{\text{train}}$? | In $U_{\text{val}}$? | In $U_{\text{test}}$? | Status |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `BWC/arctic_a0571` | `BWC` | 3.41 s | 29 | 26 | 7 | 2 | 20 | **YES** | **NO** | **NO** | **CLEAN** |
| 2 | `BWC/arctic_b0415` | `BWC` | 3.81 s | 30 | 29 | 3 | 1 | 26 | **YES** | **NO** | **NO** | **CLEAN** |
| 3 | `ERMS/arctic_a0269` | `ERMS` | 6.60 s | 39 | 37 | 10 | 2 | 27 | **YES** | **NO** | **NO** | **CLEAN** |
| 4 | `ERMS/arctic_b0150` | `ERMS` | 4.88 s | 36 | 35 | 5 | 1 | 30 | **YES** | **NO** | **NO** | **CLEAN** |
| 5 | `HKK/arctic_a0369` | `HKK` | 4.39 s | 46 | 45 | 4 | 1 | 41 | **YES** | **NO** | **NO** | **CLEAN** |
| 6 | `HKK/arctic_b0355` | `HKK` | 5.37 s | 51 | 50 | 4 | 1 | 46 | **YES** | **NO** | **NO** | **CLEAN** |
| 7 | `HQTV/arctic_a0103` | `HQTV` | 4.57 s | 33 | 32 | 7 | 1 | 25 | **YES** | **NO** | **NO** | **CLEAN** |
| 8 | `HQTV/arctic_a0112` | `HQTV` | 2.50 s | 18 | 16 | 3 | 2 | 13 | **YES** | **NO** | **NO** | **CLEAN** |
| 9 | `MBMPS/arctic_a0046` | `MBMPS` | 4.14 s | 32 | 31 | 5 | 1 | 26 | **YES** | **NO** | **NO** | **CLEAN** |
| 10 | `MBMPS/arctic_a0295` | `MBMPS` | 3.83 s | 36 | 35 | 3 | 1 | 32 | **YES** | **NO** | **NO** | **CLEAN** |
| 11 | `NCC/arctic_a0053` | `NCC` | 4.41 s | 39 | 38 | 5 | 1 | 33 | **YES** | **NO** | **NO** | **CLEAN** |
| 12 | `NCC/arctic_a0459` | `NCC` | 2.65 s | 23 | 22 | 4 | 1 | 18 | **YES** | **NO** | **NO** | **CLEAN** |
| 13 | `SKA/arctic_a0036` | `SKA` | 2.76 s | 17 | 16 | 1 | 1 | 15 | **YES** | **NO** | **NO** | **CLEAN** |
| 14 | `SKA/arctic_b0490` | `SKA` | 4.29 s | 37 | 35 | 3 | 2 | 32 | **YES** | **NO** | **NO** | **CLEAN** |
| 15 | `SVBI/arctic_a0032` | `SVBI` | 5.37 s | 46 | 45 | 6 | 1 | 39 | **YES** | **NO** | **NO** | **CLEAN** |
| 16 | `SVBI/arctic_b0411` | `SVBI` | 4.09 s | 36 | 35 | 6 | 1 | 29 | **YES** | **NO** | **NO** | **CLEAN** |
| 17 | `THV/arctic_a0048` | `THV` | 3.19 s | 27 | 25 | 4 | 2 | 21 | **YES** | **NO** | **NO** | **CLEAN** |
| 18 | `THV/arctic_b0380` | `THV` | 4.54 s | 36 | 32 | 6 | 4 | 26 | **YES** | **NO** | **NO** | **CLEAN** |
| 19 | `TLV/arctic_a0029` | `TLV` | 3.65 s | 37 | 34 | 4 | 3 | 30 | **YES** | **NO** | **NO** | **CLEAN** |
| 20 | `TLV/arctic_a0111` | `TLV` | 3.48 s | 39 | 36 | 4 | 3 | 32 | **YES** | **NO** | **NO** | **CLEAN** |
| 21 | `TNI/arctic_a0031` | `TNI` | 2.89 s | 25 | 23 | 6 | 2 | 17 | **YES** | **NO** | **NO** | **CLEAN** |
| 22 | `TNI/arctic_a0079` | `TNI` | 2.30 s | 14 | 13 | 1 | 1 | 12 | **YES** | **NO** | **NO** | **CLEAN** |
| 23 | `TXHC/arctic_a0027` | `TXHC` | 5.28 s | 44 | 43 | 5 | 1 | 38 | **YES** | **NO** | **NO** | **CLEAN** |
| 24 | `TXHC/arctic_b0422` | `TXHC` | 3.32 s | 24 | 23 | 2 | 1 | 21 | **YES** | **NO** | **NO** | **CLEAN** |
| 25 | `YBAA/arctic_a0100` | `YBAA` | 2.50 s | 15 | 13 | 1 | 2 | 12 | **YES** | **NO** | **NO** | **CLEAN** |
| 26 | `YBAA/arctic_a0105` | `YBAA` | 4.39 s | 42 | 41 | 4 | 1 | 37 | **YES** | **NO** | **NO** | **CLEAN** |
| 27 | `YDCK/arctic_a0039` | `YDCK` | 3.48 s | 29 | 27 | 1 | 2 | 26 | **YES** | **NO** | **NO** | **CLEAN** |
| 28 | `YDCK/arctic_a0042` | `YDCK` | 3.99 s | 32 | 31 | 2 | 1 | 29 | **YES** | **NO** | **NO** | **CLEAN** |
| 29 | `YKWK/arctic_a0022` | `YKWK` | 4.54 s | 52 | 49 | 5 | 3 | 44 | **YES** | **NO** | **NO** | **CLEAN** |
| 30 | `YKWK/arctic_a0075` | `YKWK` | 2.94 s | 28 | 27 | 2 | 1 | 25 | **YES** | **NO** | **NO** | **CLEAN** |
| 31 | `ZHAA/arctic_a0107` | `ZHAA` | 2.65 s | 22 | 21 | 1 | 1 | 20 | **YES** | **NO** | **NO** | **CLEAN** |
| 32 | `ZHAA/arctic_b0495` | `ZHAA` | 4.79 s | 40 | 36 | 2 | 4 | 34 | **YES** | **NO** | **NO** | **CLEAN** |

### Summary Verification:
- **Selected Utterances in Fold 1 Train ($U_{\text{train}}$):** **32 / 32 (100.0%)**
- **Selected Utterances in Fold 1 Val ($U_{\text{val}}$):** **0 / 32 (0.0%)**
- **Selected Utterances in Fold 1 Test ($U_{\text{test}}$):** **0 / 32 (0.0%)**
- **Distinct Fold 1 Train Speakers:** **16 / 16 (100.0%)**
- **Utterances per Train Speaker:** **Exactly 2**
- **Validation or Test Speakers Encountered:** **0**

---

## 4. Cryptographic Manifest Checksums

The integrity of split and data artifacts was established and verified using SHA-256 digests:
- `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`:  
  SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`
- `manifests/RQ1-TRN-01B0/TRAIN_ONLY_SUBSET_MANIFEST.json`:  
  SHA-256: `18f8ef56b9c9f4cbe5e003058eb7b8ee83b4ba43867ff7c37463f8489a74bfad`
- `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip`:  
  File size: `7,515,454,814 bytes` (Immutable raw archive)

---

## 5. Audit Conclusion

$$\mathbf{DATA\_LEAKAGE\_AUDIT: PASS}$$

**Formal Certification:** The 32 utterances used in the sanity training experiment of RQ1-TRN-01B0 and the repeat experiment in RQ1-TRN-01B0-R originate strictly and exclusively from Fold 1 training speakers and utterances. Zero validation or test utterances, speakers, boundaries, or labels were accessed or ingested during optimization.
