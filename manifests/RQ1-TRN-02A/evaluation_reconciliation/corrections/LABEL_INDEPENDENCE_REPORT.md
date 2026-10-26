# Label Independence & Inference Interface Audit — RQ1-TRN-02A-EVAL-R2-CORR

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Verification that reference labels do not contaminate prediction logits during model inference  

---

## 1. Architectural Code Audit of `LangfensWav2Vec2MDD`

In speech evaluation, passing ground-truth labels (`target_ctc`, `target_binary`) into a forward pass during evaluation creates the potential appearance of label leakage. We conducted an architectural code audit of `src/langfens_prep/model.py` to determine whether targets exert any mathematical or computational influence on prediction logits.

### Forward Computational Graph Analysis:
1. **Feature Extraction & Transformer Encoding:**
   - Raw waveform is processed by `wav2vec2.feature_extractor` and `wav2vec2.encoder` using only `audio_lengths` and padding masks.
   - Outputs `hidden_states`.
2. **Head 1: Realized CTC Logits:**
   ```python
   ctc_logits = self.ctc_head(hidden_states)
   ```
   Computed strictly as a linear projection of acoustic encoder states. **Zero target dependency.**
3. **Head 2 & 3: Binary MDD & Diagnosis Logits:**
   ```python
   aligned_acoustic, phone_embeds, alignment_weights = self.cross_attention_module(
       frame_features=hidden_states,
       canonical_phone_indices=canonical_phone_indices,
       acoustic_padding_mask=acoustic_padding_mask
   )
   binary_input = torch.cat([aligned_acoustic, phone_embeds], dim=-1)
   binary_logits = self.binary_head(binary_input).squeeze(-1)
   ```
   Computed strictly from cross-attention between `hidden_states` and `canonical_phone_indices`. **Zero target dependency.**
4. **Loss Computation (Conditional):**
   ```python
   if target_ctc is not None and target_lengths is not None:
       ctc_loss = self.ctc_loss_fn(log_probs, target_ctc, ...)
       loss_dict["ctc_loss"] = ctc_loss

   if target_binary is not None and binary_logits is not None:
       binary_loss = self.binary_loss_fn(binary_logits, target_binary, ...)
       loss_dict["binary_loss"] = binary_loss
   ```
   Targets are evaluated **only after `binary_logits` and `ctc_logits` are completely constructed**.

### Mathematical Invariance:
$$\frac{\partial \hat{\mathbf{y}}_{\text{binary}}}{\partial \mathbf{Y}_{\text{target}}} = \mathbf{0}, \quad \frac{\partial \hat{\mathbf{y}}_{\text{ctc}}}{\partial \mathbf{Y}_{\text{target}}} = \mathbf{0}$$
$$\text{Logits}(\mathbf{X}_{\text{audio}}, \mathbf{C}_{\text{prompt}}, \mathbf{Y}_{\text{targets}}) \equiv \text{Logits}(\mathbf{X}_{\text{audio}}, \mathbf{C}_{\text{prompt}}, \emptyset)$$

---

## 2. Regression Test Verification

Dedicated regression tests were implemented in `tests/test_label_independence.py`:

```bash
$ PYTHONPATH=. .venv/bin/pytest tests/test_label_independence.py
tests/test_label_independence.py .. [100%]
2 passed in 3.10s
```

### Verified Assertions:
1. `test_label_independence_on_dummy_inputs`:
   - `out_no_targets.loss is None`
   - `out_with_targets.loss is not None`
   - `torch.equal(out_no_targets.ctc_logits, out_with_targets.ctc_logits)` (**Exact bit-for-bit equality, $\Delta = 0.0$**)
   - `torch.equal(out_no_targets.binary_logits, out_with_targets.binary_logits)` (**Exact bit-for-bit equality, $\Delta = 0.0$**)
2. `test_label_independence_on_real_checkpoint_weights`:
   - Executed on production `best.pt` weights: confirmed `torch.equal` on real validation parameters.

---

## 3. Evaluator Implementation Refactoring

To eliminate any ambiguity and enforce prediction-only execution, `audit_scripts/evaluate_fold1_ctc_binary.py` has been refactored:

```python
# audit_scripts/evaluate_fold1_ctc_binary.py (lines 545-555)
# STRICT PREDICTION-ONLY INFERENCE: Targets are NEVER passed to model.
args_fwd = dict(
    source=ni["source"].to(device),
    audio_lengths=ni["audio_lengths"].to(device),
    canonical_phone_indices=ni["canonical_phone_indices"].to(device),
    canonical_phone_lengths=ni["canonical_phone_lengths"].to(device)
)
out = model(**args_fwd)
```

Target tensors (`batch["target_binary"]`, `batch["target_ctc"]`, `batch["target_lengths"]`) remain strictly outside the model execution boundary and are consumed only by the offline Python scoring metrics layer.
