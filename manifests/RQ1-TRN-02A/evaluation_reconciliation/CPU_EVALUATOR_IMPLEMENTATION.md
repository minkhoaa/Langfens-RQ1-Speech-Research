# Dedicated CPU Evaluator Implementation — RQ1-TRN-02A-EVAL-R2

**Module:** `audit_scripts/evaluate_fold1_ctc_binary.py`  
**Test Suite:** `tests/test_cpu_evaluator.py` (9 unit tests passing)  
**Execution Context:** CPU-only inference environment without CUDA  
**Author:** Senior PyTorch & Speech ML Evaluation Engineer  

---

## 1. Architectural Architecture and Design Principles

The CPU evaluation pipeline is a clean-room, inference-only entrypoint engineered to overcome the training driver's CUDA-only guard (`audit_scripts/train_fold1_ctc_binary.py`) while preserving 100% of the training safety contracts.

### Key Guarantees
1. **Isolated Inference Mode:**
   - Instantiates `LangfensWav2Vec2MDD` with frozen weights (`requires_grad=False` on all parameters).
   - Operates strictly under `model.eval()` and `torch.inference_mode()`.
   - Zero backward passes, optimizer steps, GradScaler adjustments, or learning rate updates.
2. **Model Parameter Immutability:**
   - Computes a full SHA-256 weight fingerprint across all parameter tensors before and after evaluation:
     $$\text{Fingerprint} = \text{SHA256}\left(\bigoplus_{k} \left[\text{name}_k \,\|\, \text{bytes}(\mathbf{W}_k)\right]\right)$$
   - Pre-evaluation fingerprint: `32d12bdf3524508bd891605a728c9207be588d0ce8a1a16bca2f65cab12ae60d`
   - Post-evaluation fingerprint: `32d12bdf3524508bd891605a728c9207be588d0ce8a1a16bca2f65cab12ae60d` (**IDENTICAL**)
3. **Partition & Safety Isolation:**
   - Evaluates strictly on the 300 authorized Fold 1 validation utterances (`NJS` and `RRBI`).
   - The official Fold 1 test partition (900 utterances across 6 speakers) is sealed and structurally rejected by runtime assertions if requested.
   - The training driver's CUDA guard remains intact; no silent CPU training fallback is enabled.
4. **Exact Model & Configuration Pinned Provenance:**
   - Pretrained model: `facebook/wav2vec2-base-960h`
   - Pretrained revision: `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` (verified in local HF cache)
   - Vocabulary: 40 CTC tokens (blank ID = 0, 39 ARPAbet phonemes).
   - Cross-attention module: 4 attention heads, hidden dim 768, feedforward dim 3072.
   - Checkpoint state dict loaded with `strict=True` (zero missing, zero unexpected parameters).

---

## 2. CLI Interface and Configurable Parameters

```bash
PYTHONPATH=. .venv/bin/python audit_scripts/evaluate_fold1_ctc_binary.py \
    --config configs/rq1_trn_02a_fold1.yaml \
    --checkpoint checkpoints/RQ1-TRN-02A/recovered/best.pt \
    --zip-path datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip \
    --device cpu \
    --batch-size 1 \
    --output-dir manifests/RQ1-TRN-02A/evaluation_reconciliation \
    --run-controls
```

### CLI Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | `configs/rq1_trn_02a_fold1.yaml` | Experiment YAML configuration |
| `--checkpoint` | `str` | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | Path to production checkpoint |
| `--zip-path` | `str` | Authorized dataset archive | Path to authorized train/val zip archive |
| `--device` | `str` | `cpu` | PyTorch execution device |
| `--batch-size` | `int` | `1` | Batch size (default 1 ensures unpadded acoustic execution) |
| `--output-dir` | `str` | `manifests/RQ1-TRN-02A/evaluation_reconciliation` | Artifact destination directory |
| `--max-eval-utterances` | `int` | `None` | Optional limit for diagnostic/smoke tests |
| `--run-controls` | `bool` | `True` | Runs paired matched/mismatched and baseline controls |
| `--split` | `str` | `val` | Enforced split (only `'val'` permitted) |

---

## 3. Unit and Regression Verification

Unit tests are implemented in `tests/test_cpu_evaluator.py`:
- `test_compute_sha256`: Validates cryptographic block-wise digest.
- `test_verify_checkpoint_integrity_rejects_missing`: Verifies `FileNotFoundError` on absent path.
- `test_verify_checkpoint_integrity_rejects_invalid_format`: Asserts rejection of non-dict or non-model files.
- `test_generate_derangement_indices`: Asserts valid permutation with zero self-matches ($\pi(i) \neq i, \forall i$).
- `test_generate_derangement_within_speakers`: Asserts speaker-segmented derangement preserving speaker identity.
- `test_get_model_weights_fingerprint_and_immutability`: Confirms weight fingerprint invariance across forward passes.
- `test_calculate_binary_metrics`: Asserts exact AP, prevalence, and threshold sweep logic.
- `test_calculate_ctc_metrics`: Validates greedy PER, Levenshtein edit decomposition (S, D, I), and length ratios.
- `test_canonical_empirical_prior_synthetic`: Verifies train-only Bayesian lookup $P_{\text{train}}(\text{err} \mid c)$ calculation.

**Test execution status:** 9 passed in 2.29s (full repository test suite: 97 passed, 3 skipped due to CUDA).
