# LANGFENS — RQ1-TRN-01A: Training Contract Hardening & Test Report

**Phase:** `RQ1-TRN-01A` — Training Implementation & GPU Smoke Test  
**Focus Area:** Phase 2 — Training Contract Hardening, Edge-Case Remediation & Regression Testing  
**Author / Engineering Lead:** Senior Speech ML Engineer & Research Reproducibility Reviewer  
**Deliverable Path:** `manifests/RQ1-TRN-01A/TRAINING_CONTRACT_TEST_REPORT.md`  
**Total Tests Executed:** 38 tests (38 passed in `.venv-cuda`, 37 passed + 1 skipped in `.venv`)  
**Status:** `TRAINING_CONTRACTS_HARDENED: PASS`  

---

## 1. Executive Summary & Test Suite Execution Results

Phase `RQ1-TRN-01A` conducted a comprehensive static and dynamic review of all training contract interfaces. During edge-case stress testing, four critical vulnerabilities were identified in naive loss and alignment implementations. Minimal, isolated fixes were engineered, and regression tests were added to guarantee stability across multi-gpu training, gradient accumulation, and variable-length batches.

### Empirical Test Execution Summary:
- **CUDA GPU Environment (`.venv-cuda`):** **38 passed, 0 failures** in $48.72\text{ s}$.
- **CPU Reference Environment (`.venv`):** **37 passed, 1 skipped** (CUDA smoke test skipped on CPU) in $39.58\text{ s}$.

---

## 2. Hardened Training Contract Edge Cases

### Edge Case 1: Masked BCE Gradient Connectivity on All-Masked Batches
- **Vulnerability:** If an entire micro-batch contains only padded or blinded targets (`target_binary == -100`), naive implementations return a constant tensor `torch.tensor(0.0)`. This detached leaf tensor severs the autograd computation graph, leaving `logits.grad = None`. In distributed training (DDP) or when calling `clip_grad_norm_()`, `None` gradients cause fatal runtime exceptions.
- **Remediation (`src/langfens_prep/losses.py:27-32`):**
  When no active targets exist (`not mask.any()`), `MaskedBCEWithLogitsLoss` returns `(logits * 0.0).sum()`. This produces a scalar `0.0` while retaining a valid autograd path, guaranteeing that `logits.grad` evaluates to `torch.zeros_like(logits)`.
- **Regression Test:** `tests/test_training_contracts.py::test_masked_bce_all_masked_gradient_connectivity` (**PASSED**).

---

### Edge Case 2: Masked Cross-Entropy NaN Prevention on All-Masked Batches
- **Vulnerability:** Standard PyTorch `nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')` divides accumulated loss by the sum of active sample weights. When all targets in a batch are `-100` (e.g. an utterance consisting exclusively of deletions or `'err'` tags), the denominator is zero ($0 / 0$), causing PyTorch to output `NaN`. Backward propagation through `NaN` injects invalid gradients that corrupt model weights.
- **Remediation (`src/langfens_prep/losses.py:75-81`):**
  `MaskedCrossEntropyLoss` pre-checks target validity. If `not (targets != self.ignore_index).any()`, it bypasses the division by zero and returns `(logits * 0.0).sum()`.
- **Regression Test:** `tests/test_training_contracts.py::test_masked_ce_all_masked_avoids_nan` (**PASSED**).

---

### Edge Case 3: CTC Feasibility with Adjacent Repeated Target Tokens
- **Vulnerability:** Standard CTC loss implementations check the condition $T_{\text{input}} \ge L_{\text{target}}$. However, in Connectionist Temporal Classification, two consecutive identical target phonemes (e.g. `[AA, AA]`) cannot be emitted in adjacent frames without collapsing into a single phoneme. CTC mathematically requires at least one intervening non-emitting `<BLANK>` frame. Consequently, if a target has length $L=2$ with adjacent duplicates and input length is $T=2$, PyTorch `nn.CTCLoss` evaluates to **`inf`** (Infinity).
- **Remediation (`src/langfens_prep/losses.py:126-140`):**
  Updated `validate_ctc_contract` to enforce the exact minimum frame requirement:
  $$T_{\text{min}} = L_{\text{target}} + \sum_{k=1}^{L_{\text{target}}-1} \mathbb{I}(y_k == y_{k+1})$$
  The validator verifies that $T_{\text{input}} \ge T_{\text{min}}$ before CTCLoss execution, preventing infinite loss propagation.
- **Regression Test:** `tests/test_training_contracts.py::test_ctc_contract_adjacent_repeated_tokens` (**PASSED**).

---

### Edge Case 4: Frame-to-Phone Alignment Boundary Clamping
- **Vulnerability:** In non-native speech, phonemes can exhibit extremely brief durations ($< 20\text{ ms}$). When downsampling $16\text{ kHz}$ audio by $320\times$ ($20\text{ ms/frame}$), naive integer truncation can produce equal start and end frame indices ($\lfloor t_{\text{start}} / 0.02 \rfloor = \lfloor t_{\text{end}} / 0.02 \rfloor$), creating an empty slice `[s:s, :]`. Computing `.mean(dim=0)` on an empty tensor returns `NaN`.
- **Remediation (`src/langfens_prep/alignment.py:46-52`):**
  `pool_frames_to_canonical_phones` enforces:
  $$\text{end\_frame} = \max(\text{start\_frame} + 1, \, \min(\text{max\_frames}, \text{end\_frame}))$$
  Ensuring every phoneme receives at least one acoustic frame.
- **Regression Test:** `tests/test_training_contracts.py::test_frame_to_phone_pooling_guarantees` (**PASSED**).

---

## 3. Complete Test Suite Inventory (38 Tests)

| Test Module | Test Name | Target Functionality | Result |
| :--- | :--- | :--- | :---: |
| `test_audio.py` | `test_load_audio_from_l2arctic_nested_archive` | In-memory ZIP streaming & 16kHz resampling | **PASSED** |
| `test_audio.py` | `test_load_audio_from_speechocean_archive` | Tarball extraction & WAV verification | **PASSED** |
| `test_audio.py` | `test_configurable_normalization_and_dc` | DC removal and peak normalization | **PASSED** |
| `test_dataset_and_loader.py` | `test_l2arctic_dataset_loading_and_policy` | Dataset iteration under strict/lenient policy | **PASSED** |
| `test_dataset_and_loader.py` | `test_l2arctic_blinded_test_isolation` | Blinded test mode (None targets) | **PASSED** |
| `test_dataset_and_loader.py` | `test_mdd_batch_collation` | Batch collation & explicit binary masks | **PASSED** |
| `test_dataset_and_loader.py` | `test_speechocean_score_dataset` | SpeechOcean continuous scoring dataset | **PASSED** |
| `test_full_census.py` | `test_full_l2arctic_gold_census` | Full 3,599 gold TextGrid interval census | **PASSED** |
| `test_full_census.py` | `test_speechocean762_validation` | SpeechOcean 5,000 manifested audio census | **PASSED** |
| `test_normalizer.py` | `test_standard_phoneme_normalization` | 39-ARPAbet stress stripping & mapping | **PASSED** |
| `test_normalizer.py` | `test_ax_to_ah_folding` | AX to AH folding and reversibility | **PASSED** |
| `test_normalizer.py` | `test_asterisk_and_err_flags` | Asterisk allophone & 'err' tag extraction | **PASSED** |
| `test_normalizer.py` | `test_typo_correction_v8` | Annotation keyboard typo correction | **PASSED** |
| `test_normalizer.py` | `test_reversibility` | Exact token reconstruction from metadata | **PASSED** |
| `test_normalizer.py` | `test_ipa_mapping` | ARPAbet to IPA character mapping | **PASSED** |
| `test_parsers.py` | `test_parse_gold_textgrid_intervals` | Praat TextGrid interval parsing | **PASSED** |
| `test_parsers.py` | `test_speechocean_parser_manifested` | SpeechOcean scores.json parsing | **PASSED** |
| `test_provenance.py` | `test_verify_split_manifest_hashes` | Split manifest SHA-256 digests | **PASSED** |
| `test_provenance.py` | `test_canonical_payload_sha256` | Canonical JSON serialization digests | **PASSED** |
| `test_provenance.py` | `test_provenance_timestamp_investigation` | Timezone metadata erratum analysis | **PASSED** |
| `test_training_contracts.py` | `test_binary_mdd_targets_distinguish_valid_from_padding` | Binary targets in {0, 1}, padding = -100 | **PASSED** |
| `test_training_contracts.py` | `test_bce_with_logits_loss_masked_vs_unmasked_catastrophe` | Proof of -100 BCE loss explosion & fix | **PASSED** |
| `test_training_contracts.py` | `test_masked_bce_gradient_flow` | 0.0 gradients on padded frames | **PASSED** |
| `test_training_contracts.py` | `test_masked_crossentropy_loss_masks_deletions_and_err` | 0.0 gradients on deletions and 'err' | **PASSED** |
| `test_training_contracts.py` | `test_ctc_blank_id_never_in_target_sequences` | Blank 0 strictly excluded from targets | **PASSED** |
| `test_training_contracts.py` | `test_ctc_contract_validation` | Mathematical CTC preconditions | **PASSED** |
| `test_training_contracts.py` | `test_realized_sequence_semantics` | Deletion, addition & substitution semantics | **PASSED** |
| `test_training_contracts.py` | `test_collate_with_masks_and_diagnosis` | Collation with explicit mask tensors | **PASSED** |
| `test_training_contracts.py` | `test_masked_bce_all_masked_gradient_connectivity` | BCE autograd connectivity on all-masked | **PASSED** |
| `test_training_contracts.py` | `test_masked_ce_all_masked_avoids_nan` | CE 0.0 loss (no NaN) on all-masked | **PASSED** |
| `test_training_contracts.py` | `test_ctc_contract_adjacent_repeated_tokens` | CTC length validation for duplicate tokens | **PASSED** |
| `test_training_contracts.py` | `test_frame_to_phone_pooling_guarantees` | Frame pooling non-empty slice guarantees | **PASSED** |
| `test_model.py` | `test_model_forward_ctc_only` | Forward pass with CTC head | **PASSED** |
| `test_model.py` | `test_model_multitask_forward_and_loss` | Multi-task forward, losses & backward | **PASSED** |
| `test_model.py` | `test_freeze_feature_encoder` | Feature extractor freezing API | **PASSED** |
| `test_model.py` | `test_checkpoint_serialization` | Save / load state_dict roundtrip | **PASSED** |
| `test_model.py` | `test_optimizer_step_adamw` | AdamW parameter update stability | **PASSED** |
| `test_model.py` | `test_gpu_forward_backward_if_available` | GPU forward/backward under AMP fp16 | **PASSED** |

**Phase Gate Evaluation:** **`TRAINING_CONTRACTS_HARDENED: PASS`**
