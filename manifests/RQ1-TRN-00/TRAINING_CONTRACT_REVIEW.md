# LANGFENS — RQ1-TRN-00: Training Contract Static Review & Loss Verification Report

**Execution Gate:** RQ1-TRN-00 — Modeling Readiness & Reproducibility Preflight  
**Focus Area:** Task C — Training Contract Interfaces, Loss Masking Proofs, Label Semantics & Test Blindness  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Reproducibility Lead  
**Deliverable Associated Path:** `manifests/RQ1-TRN-00/TRAINING_CONTRACT_REVIEW.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-00/`  
**Gate Evaluation Status:** `TRAINING_CONTRACT_VALIDATED: PASS`  

---

## 1. Executive Summary & Training Contract Philosophy

In speech machine learning, silent data contract bugs represent one of the most pernicious sources of scientific invalidation. When training multi-task acoustic architectures for Mispronunciation Detection & Diagnosis (MDD), models must simultaneously optimize:
1. A **binary detection head** (detecting whether a canonical phoneme was mispronounced),
2. A **phonetic diagnosis head** (identifying which substitute phone was articulated), and/or
3. A **temporal CTC alignment head** (transcribing realized speech across continuous time).

Because variable-length batch sequences necessitate padding tensors, naive implementations frequently pass pad tokens (such as `-100`) directly into loss functions, contaminating training gradients. Furthermore, improper handling of deletions, epenthetic additions, and acoustic distortions (`err` tags) distorts empirical error rates.

This report provides static, mathematical, and synthetic empirical proof verifying all **ten mandatory training contract items**.

---

## 2. In-Depth Verification of the 10 Training Contract Items

### Contract Item 1: Binary MDD Targets Distinguish Valid Labels from Padding
- **Requirement:** Ground-truth binary labels for MDD must reside exclusively in $\{0, 1\}$ ($0 = \text{Correct}$, $1 = \text{Error}$). Padded sequence positions must use an unambiguous sentinel value ($-100$) that is never interpreted as a legitimate binary label.
- **Verification Evidence:**
  - In `src/langfens_prep/dataset.py:99-103`, `binary_labels` are constructed strictly from `binary_label_strict` or `binary_label_lenient`, which yield exclusively $0$ or $1$.
  - In `src/langfens_prep/collate.py:34-36`, `LangfensMDDCollate` pads shorter sequences with `padding_value = self.pad_token_id` ($-100$).
  - In `tests/test_training_contracts.py::test_binary_mdd_targets_distinguish_valid_from_padding`, we empirically prove that $\text{Labels}_{\text{active}} \subseteq \{0, 1\}$, $\text{Padding} = \{-100\}$, and $\{0, 1\} \cap \{-100\} = \emptyset$.

---

### Contract Item 2: BCEWithLogitsLoss Explicit Masking & The -100 Catastrophe
- **Requirement:** `nn.BCEWithLogitsLoss` must apply an explicit boolean mask for padded positions. Padded value $-100$ must NEVER enter binary cross-entropy computations.
- **Mathematical Analysis of the Trap:**
  Standard PyTorch `nn.BCEWithLogitsLoss` does **NOT** support an `ignore_index` parameter (unlike `nn.CrossEntropyLoss`).
  The binary cross-entropy loss function is mathematically defined as:
  $$\mathcal{L}(x, y) = - \left[ y \log \sigma(x) + (1 - y) \log (1 - \sigma(x)) \right]$$
  If a developer naively passes a padded tensor containing $y = -100$ into `nn.BCEWithLogitsLoss`, the loss at padded position $i$ evaluates to:
  $$\mathcal{L}_{\text{pad}} = - \left[ -100 \log \sigma(x_i) + 101 \log (1 - \sigma(x_i)) \right] = 100 \log \sigma(x_i) - 101 \log (1 - \sigma(x_i))$$
  The resulting gradient with respect to logit $x_i$ is:
  $$\frac{\partial \mathcal{L}}{\partial x_i} = \sigma(x_i) - y_i = \sigma(x_i) - (-100) = \sigma(x_i) + 100 \approx 100.5$$
  This injects an artificial gradient $100\times$ larger than normal signals on uninformative padding frames, completely destabilizing neural weights and producing catastrophic divergence.
- **Remediation & Verification:**
  - We implemented `MaskedBCEWithLogitsLoss` in `src/langfens_prep/losses.py`. It constructs an explicit boolean mask $M = (y \ne -100)$, asserts that all remaining active targets are in $\{0, 1\}$, and computes loss strictly over $M$.
  - In `src/langfens_prep/collate.py:38`, `LangfensMDDCollate` now explicitly emits `target_binary_mask = (padded_binary_labels != -100)`.
  - In `tests/test_training_contracts.py::test_bce_with_logits_loss_masked_vs_unmasked_catastrophe` and `test_masked_bce_gradient_flow`, we empirically proved:
    1. Unmasked BCE produces distorted loss ($>60$).
    2. `MaskedBCEWithLogitsLoss` produces exact loss equal to computing only on active tokens.
    3. The gradient $\frac{\partial \mathcal{L}}{\partial x_i}$ at padded positions is **identically 0.0**.

---

### Contract Item 3: Multi-Class CrossEntropyLoss Masking for 39-Class Diagnosis
- **Requirement:** When training a phone diagnosis head over the 39 ARPAbet classes, the loss function must use `ignore_index = -100` to mask:
  1. Padded sequence positions,
  2. Deletion tokens (which have no realized sound), and
  3. Indeterminate / distorted productions labeled with `'err'`.
- **Implementation & Verification:**
  - We implemented `MaskedCrossEntropyLoss` in `src/langfens_prep/losses.py`, wrapping `nn.CrossEntropyLoss(ignore_index=-100)`.
  - In `src/langfens_prep/dataset.py:109-114`, `diagnosis_target_indices` maps valid 39-ARPAbet phone realizations to $[0, 38]$ via `PHONE_TO_INDEX`, while deletions and `'err'` tokens are mapped to $-100$.
  - In `tests/test_training_contracts.py::test_masked_crossentropy_loss_masks_deletions_and_err`, we verified that gradients on deletions, `'err'` tags, and padding are exactly $0.0$, and total loss is computed exclusively on valid substitute phones.

---

### Contract Item 4: CTC Blank ID 0 Never Enters Valid Target Sequences
- **Requirement:** For CTC loss (`nn.CTCLoss`), index $0$ is strictly reserved for the non-emitting `<BLANK>` token. Valid target phoneme sequences must NEVER contain index $0$.
- **Verification Evidence:**
  - In `src/langfens_prep/phoneme_normalizer.py:14-16`:
    $$\text{PHONE\_TO\_CTC\_INDEX}(p) = \text{idx} + 1 \in [1, 39], \quad \forall p \in \text{STANDARD\_39\_ARPABET}$$
    $$\text{CTC\_INDEX\_TO\_PHONE}[0] = \text{"<BLANK>"}$$
  - Index $0$ is physically absent from `PHONE_TO_CTC_INDEX.values()`.
  - In `tests/test_training_contracts.py::test_ctc_blank_id_never_in_target_sequences`, all 39 phones were verified to map to indices in $[1, 39]$.

---

### Contract Item 5: CTC Temporal Invariance & Frame Length Validation
- **Requirement:** The fundamental mathematical requirement of CTC alignment dictates that the acoustic frame sequence length $T_{\text{input}}$ must be greater than or equal to the target phoneme sequence length $L_{\text{target}}$:
  $$T_{\text{input}} \ge L_{\text{target}} > 0$$
  If $T_{\text{input}} < L_{\text{target}}$, CTC cannot compute a valid forward-backward lattice, causing PyTorch to raise `RuntimeError` or emit `NaN`.
- **Verification Evidence:**
  - We implemented `validate_ctc_contract` in `src/langfens_prep/losses.py`. It checks:
    1. Batch shape alignment,
    2. Finite values (no NaNs or Infs),
    3. $T_{\text{input}} \ge L_{\text{target}}$ for every utterance in the batch,
    4. Target positivity ($L_{\text{target}} > 0$),
    5. Absence of `<BLANK>` index $0$ within active target lengths $[0, L_{\text{target}}-1]$.
  - Verified in `tests/test_training_contracts.py::test_ctc_contract_validation`.

---

### Contract Item 6: Realized Phone Sequence Semantics (Additions, Deletions, Substitutions)
- **Requirement:** Realized phone sequences must correctly preserve error taxonomy semantics:
  - **Deletions ($D = 3,420$):** Canonical phone omitted by speaker. In TextGrid intervals, realized phone is recorded as `"sil"`. In CTC target sequences, deletions produce no acoustic emission; in diagnosis heads, deletions are masked with $-100$.
  - **Epenthetic Additions ($I = 1,092$):** Unprompted phone inserted by speaker. They are flagged with `is_canonical = False` and excluded from canonical prompt evaluation ($N = 119,151$).
  - **Substitutions ($S = 14,098$):** Canonical phone replaced by another sound.
- **Verification Evidence:** Verified across all 3,599 gold TextGrids in `test_full_census.py` and synthetically in `test_realized_sequence_semantics`.

---

### Contract Item 7: Uncertainty Masking (`err` Tags)
- **Requirement:** Non-native speech frequently contains heavily distorted, incomprehensible, or non-English phonetic fragments labeled `'err'` by linguists ($N_{\text{err}} = 258$ intervals in L2-ARCTIC gold annotations). These tokens must:
  1. Be marked as errors in binary detection ($b_i = 1$),
  2. Be strictly masked ($-100$) from 39-class diagnosis heads (since they do not correspond to any valid English phoneme).
- **Verification Evidence:**
  - In `src/langfens_prep/textgrid_parser.py:168-175`, `has_err` sets `diagnosis_target = None`, which maps to $-100$.
  - Verified in `tests/test_parsers.py` and `tests/test_training_contracts.py::test_realized_sequence_semantics`.

---

### Contract Item 8: Strict vs. Lenient Policy Separation
- **Requirement:** Strict and lenient MDD policies must operate as completely independent targets while maintaining the identical canonical denominator $N = 119,151$:
  - **Strict Policy:** $C_{\text{strict}} = 101,633$, $E_{\text{strict}} = 17,518$ ($14.70\%$ error rate). All base-preserving asterisk allophones (`CPL,CPL*,s`) are treated as errors ($1$).
  - **Lenient Policy:** $C_{\text{lenient}} = 102,745$, $E_{\text{lenient}} = 16,406$ ($13.77\%$ error rate). Base-preserving asterisk allophones ($1,112$ tokens) are reclassified as correct ($0$).
- **Verification Evidence:**
  - `L2ArcticMDDDataset(policy="strict")` vs `(policy="lenient")` verified in `tests/test_dataset_and_loader.py:11-37`.
  - Exact census counts confirmed in `tests/test_full_census.py:27-38`.

---

### Contract Item 9: Test Partition Blinding & Zero Label Leakage
- **Requirement:** Test evaluation samples must be completely blinded: ground-truth labels (`binary_labels`, `ctc_target_indices`, `diagnosis_target_indices`) must be `None`. Test data must never enter model training, hyperparameter tuning, or decision threshold calibration.
- **Verification Evidence:**
  - `L2ArcticMDDDataset(blind_labels=True)` sets all target tensors to `None`.
  - `LangfensMDDCollate` emits `None` for all target keys when blinded samples are passed.
  - Verified in `tests/test_dataset_and_loader.py::test_l2arctic_blinded_test_isolation`.

---

### Contract Item 10: All Training Folds Consume Frozen RQ1-SPLIT-01 Manifests
- **Requirement:** All training, validation, and test partitions must consume the immutable manifests generated in `RQ1-SPLIT-01` without dynamic re-splitting.
- **Verification Evidence:**
  - Verified in `tests/test_provenance.py` via SHA-256 digests against `l2arctic_4fold_splits.json`, `l2arctic_loso_vietnamese_splits.json`, and `speechocean762_official_splits.json`.

---

## 3. Minimal Isolated Code Enhancements Summary

To satisfy all contract requirements, the following minimal, isolated code additions were introduced into the repository:

1. **`src/langfens_prep/losses.py`:** Created dedicated `MaskedBCEWithLogitsLoss`, `MaskedCrossEntropyLoss`, and `validate_ctc_contract` functions.
2. **`src/langfens_prep/phoneme_normalizer.py`:** Added `PHONE_TO_INDEX` and `INDEX_TO_PHONE` (0-indexed $[0, 38]$) for multi-class classification.
3. **`src/langfens_prep/dataset.py`:** Added `diagnosis_target_indices` extraction to `L2ArcticMDDDataset`.
4. **`src/langfens_prep/collate.py`:** Added explicit boolean tensor `target_binary_mask` and padded `target_diagnosis`.
5. **`tests/test_training_contracts.py`:** Implemented 8 comprehensive contract tests covering every edge case.

All 28 tests pass unconditionally.

**Phase Gate Evaluation:** **`TRAINING_CONTRACT_VALIDATED: PASS`**
