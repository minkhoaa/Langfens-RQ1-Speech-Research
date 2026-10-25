# Regression Test Report — Fold 1 Clean Restart

**Status: PASS | Local Tests: 88 Passed | Colab GPU Integration: PASS**

---

## 1. Local Suite Verification (Laptop Environment)

Executed using `.venv` Python 3.14 with `PYTHONPATH=.`:
- **Total Tests Collected**: 91
- **Passed**: 88
- **Skipped**: 3 (strictly CUDA-only pretrained inference tests on non-CUDA laptop)
- **Failed**: 0
- **Warnings**: 1 (expected CTC length contract violation warning from dynamic contract boundary test)

Key test suites verified:
1. `tests/test_training_contracts.py`: Validated two-head parameter filtering, diagnosis head gradient exclusion, binary label masking (-100 vs 0/1), and CTC contract bounds.
2. `tests/test_fold1_training_driver.py`: Validated Levenshtein edit distance decomposition (S, D, I, total edits), greedy CTC decoding, atomic checkpoint persistence, patience restoration, and GradScaler state continuity.
3. `tests/test_reconciliation_regression.py`: Validated 4-fold split manifest schemas, speaker balance, disjoint partitions, and loss functions.

---

## 2. Colab GPU Integration and Verification

Executed on Google Colab Tesla T4 GPU (`rq1-trn02a-restart-01`):
1. **Model Architecture & Two-Head Optimizer**: CTC head, cross-attention module, and binary head trainable; Wav2Vec2 backbone frozen in eval mode; diagnosis head frozen with `requires_grad=False`.
2. **Real-Audio Integration Gate (Epoch 1)**: Processed all 2,399 train and 300 validation utterances with batch size 2, accumulation 2 (600 optimizer updates).
3. **AMP Numerical Stability**: 0 AMP skips observed across all 6,600 optimizer updates with initial scale 4,096.
4. **Checkpoint Resume Continuity**: Bounded restart test verified that resuming from `last.pt` restores exact model weights, optimizer momentum, scheduler step, patience counter, and RNG state.
5. **Full Run Early Stopping**: Completed 11 epochs, culminating in clean early-stopping termination after 5 epochs without validation AP improvement beyond the Epoch 6 peak.
