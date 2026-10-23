# LANGFENS — RQ1-TRN-01A: Repository Corrections & Audit Ledger

**Phase:** `RQ1-TRN-01A` — Training Implementation & GPU Smoke Test  
**Focus Area:** Phase 1 — Repository Corrections, Discrepancy Reconciliation & Git History Audit  
**Author / Engineering Lead:** Senior Speech ML Engineer & Research Reproducibility Reviewer  
**Deliverable Path:** `manifests/RQ1-TRN-01A/REPOSITORY_CORRECTIONS.md`  
**Git Branch:** `feat/rq1-trn-01a-training-smoke-test`  
**Base Commit:** `0b85ab9`  
**Status:** `CORRECTIONS_APPLIED: PASS`  

---

## 1. Executive Summary & Audit Scope

This report documents the formal correction, reconciliation, and audit of the project repository prior to initiating neural model training in Phase `RQ1-TRN-01A`.

All ten mandatory Phase 1 items have been rigorously audited, rectified, and validated against empirical code and frozen split manifests.

---

## 2. Item-by-Item Reconciliation Ledger

### Item 1: Root README.md Added
- **Status:** **COMPLETED**
- **Action:** Created comprehensive, accurate root `README.md` documenting project objectives, repository architecture, dual execution environments (`.venv` CPU vs `.venv-cuda` GPU), model registry specifications, and usage instructions.

---

### Item 2: Correction of $S_{\text{eval}}$ from 13,829 to 13,840
- **Status:** **CORRECTED**
- **Root Cause Analysis:**
  - Total substitutions in L2-ARCTIC v5.0 gold annotations: $S_{\text{raw}} = \mathbf{14,098}$.
  - Total intervals tagged with the uncertainty code `'err'` across the entire corpus: $N_{\text{err, total}} = \mathbf{269}$.
  - Earlier draft specifications (e.g. initial `MDD_LABEL_SCHEMA.md` and `EVALUATION_PROTOCOL.md`) mistakenly subtracted all 269 `'err'` intervals from substitutions ($14,098 - 269 = 13,829$).
  - **Empirical Census Truth:** In `manifests/RQ1-STD-01/reconciliation/CORRECTED_LABEL_ACCOUNTING.md:215-225`, exact interval classification confirmed that:
    - Exactly **258** `'err'` tokens occur within substitution intervals (`CPL,err,s`).
    - The remaining **11** `'err'` tokens occur within epenthetic additions (`sil,err,a`).
  - Therefore, the exact evaluable 39-class substitution diagnosis denominator is:
    $$S_{\text{eval}} = S_{\text{raw}} - N_{\text{err, sub}} = 14,098 - 258 = \mathbf{13,840 \text{ tokens}}$$
- **Actions Taken:** Updated `configs/huggingface_registry.yaml`, `configs/huggingface_model_card.md`, and project documentation to reflect $S_{\text{eval}} = 13,840$.

---

### Item 3: Vietnamese LOSO Partition Specification (21 Train / 2 Val / 1 Test)
- **Status:** **CORRECTED**
- **Root Cause Analysis:** Informal notes referenced "training on the other 23 speakers", omitting the 2 validation speakers.
- **Empirical Split Verification:** Inspection of `manifests/RQ1-SPLIT-01/l2arctic_loso_vietnamese_splits.json` (`regime_a_cross_vietnamese_loso`) confirms that each of the 4 rounds (`HQTV`, `PNV`, `THV`, `TLV`) is partitioned into:
  - **Train:** **21 speakers** (including 3 Vietnamese speakers),
  - **Validation:** **2 speakers** (used strictly for early stopping and threshold calibration),
  - **Test:** **1 speaker** (held-out Vietnamese speaker).
  - Total: $21 + 2 + 1 = \mathbf{24 \text{ speakers}}$.
- **Actions Taken:** Reconciled in documentation and `README.md`.

---

### Item 4: Vietnamese Zero-Shot Partition Specification (18 Train / 2 Val / 4 Test)
- **Status:** **CORRECTED**
- **Root Cause Analysis:** Informal notes cited "training strictly on the 20 non-Vietnamese speakers", omitting the internal train/validation split.
- **Empirical Split Verification:** Inspection of `manifests/RQ1-SPLIT-01/l2arctic_loso_vietnamese_splits.json` (`regime_b_zero_shot_vietnamese_transfer`) confirms:
  - **Train:** **18 non-Vietnamese speakers** across 5 L1s (Arabic, Chinese, Hindi, Korean, Spanish),
  - **Validation:** **2 non-Vietnamese speakers**,
  - **Test:** **4 Vietnamese speakers** (`HQTV`, `PNV`, `THV`, `TLV`) for zero-shot transfer evaluation.
  - Total: $18 + 2 + 4 = \mathbf{24 \text{ speakers}}$.
- **Actions Taken:** Reconciled in documentation and `README.md`.

---

### Item 5: Source-Code vs Dataset Licensing Distinction
- **Status:** **FORMALIZED**
- **Contract Boundary:**
  - **Source Code & Tools:** Permissive **MIT License** (declared in `LICENSE` and `pyproject.toml`).
  - **L2-ARCTIC v5.0 Corpus & Annotations:** **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. All trained checkpoints, phoneme representations, and derived weights are strictly restricted to non-commercial academic research.
  - **SpeechOcean762 Corpus:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

---

### Item 6: Hugging Face Model Card Marked as PLANNED / NOT TRAINED
- **Status:** **COMPLETED**
- **Action:** Updated `configs/huggingface_model_card.md` with explicit status banner:
  `**Status:** PLANNED / NOT TRAINED (Architecture & Contract Smoke Testing Phase RQ1-TRN-01A)`

---

### Item 7: Proposed Hugging Face Repository Visibility Set to Private
- **Status:** **COMPLETED**
- **Action:** Updated `configs/huggingface_registry.yaml` setting `private: true`. Checkpoints will remain private until post-training review and licensing sign-off.

---

### Item 8: Reconciliation of 40-Class vs 43-Token CTC Vocabulary
- **Status:** **RECONCILED**
- **Analysis & Clarification:**
  - **40-Class Formulation (Direct Acoustic CTC Head):**
    In `src/langfens_prep/phoneme_normalizer.py`, the phoneme inventory consists of:
    - Index `0`: `<BLANK>` (reserved non-emitting CTC token),
    - Indices `1..39`: The 39 standard CMUDict ARPAbet phonemes.
    - Total: **40 classes**.
    PyTorch's `nn.CTCLoss(blank=0)` and `LangfensWav2Vec2MDD.ctc_head` operate directly over this 40-class output.
  - **43-Token Formulation (Hugging Face Wav2Vec2 Tokenizer Convention):**
    When exporting to Hugging Face Hub, `Wav2Vec2CTCTokenizer` incorporates additional special tokens:
    - `<pad>` (index 40),
    - `<s>` (index 41),
    - `</s>` (index 42),
    - `<unk>` (index 43, or space delimiter `|`).
  - **Target Sequence Invariance:** All supervision targets emitted by `L2ArcticMDDDataset` and `LangfensMDDCollate` are drawn **strictly from the 39 phone tokens (indices 1..39)**. Special tokens are never emitted as supervision targets.

---

### Item 9: Verification Against Frozen Split Manifests
- **Status:** **VERIFIED (0 Contradictions)**
- **Verification Evidence:** All dataset loaders (`L2ArcticMDDDataset`, `SpeechOcean762ScoreDataset`) consume the immutable JSON split files in `manifests/RQ1-SPLIT-01/`. Payloads were re-verified against their canonical SHA-256 digests with zero discrepancies.

---

### Item 10: Git History Audit for Large Files and Potential Secrets
- **Status:** **AUDITED & REPORTED (No History Rewrite)**
- **Audit Findings:**
  1. **Large Binary Files in History:**
     In commit `3b3d5e9`, two pretrained MFA acoustic model archives were committed:
     - `resources/MFA/acoustic-models/english_mfa.zip` ($92,175,348\text{ bytes} \approx 87.9\text{ MB}$)
     - `resources/MFA/acoustic-models/english_us_arpa.zip` ($91,928,208\text{ bytes} \approx 87.6\text{ MB}$)
     *Resolution:* In commit `0b85ab9`, both files were untracked from Git (`git rm --cached`) and `.gitignore` was updated with `*.zip`. The physical files remain on disk for local experimentation. In accordance with instructions, **Git history was not rewritten**, preserving commit cryptographic hashes. Both files are $< 100\text{ MB}$ and do not block GitHub operations.
  2. **Potential Secrets & Credentials:**
     A regex search across all commits (`git log -p`) for API tokens (`ghp_`, `gho_`, `hf_`, private keys, passwords) confirmed **zero sensitive credentials** in version control. All occurrences of `secret` or `password` in history correspond to CMUDict phonetic dictionary entries (e.g. `secret S IY1 K R AH0 T`).
