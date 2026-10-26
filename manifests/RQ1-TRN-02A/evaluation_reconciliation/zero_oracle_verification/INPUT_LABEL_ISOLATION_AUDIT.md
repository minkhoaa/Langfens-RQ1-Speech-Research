# Input and Label Isolation Audit — RQ1-TRN-02A-ZERO-ORACLE-VERIFY

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Rigorous proof of TextGrid isolation during input generation and inference, offline scoring independence, and model parameter immutability  

---

## 1. Executive Summary

This audit verifies that the evaluation pipeline for **Protocol Z (Zero-Oracle)** maintains strict boundaries between model inference and ground-truth human annotations:
1. **Zero TextGrid Reads in Inference Path:** Model inference consumes only raw audio waveforms (`.wav` files) and G2P-derived phoneme sequences from prompt text. Praat TextGrids are never opened, read, or referenced during dataset loading or model forward passes.
2. **Strict Offline Scoring Separation:** Ground-truth labels and gold canonical intervals are loaded exclusively inside the post-inference offline scoring component.
3. **Parameter Immutability:** Pre-evaluation and post-evaluation SHA-256 weight fingerprints are identical ($\Delta = 0.0$), confirming zero gradient updates or runtime mutations.

---

## 2. Architectural Boundary Verification

```
+-----------------------------------------------------------------------------------+
|                            INFERENCE COMPONENT (Protocol Z)                       |
|                                                                                   |
|  Written Prompt Text --------> [ Deterministic G2P ] -----> Canonical Phone IDs  |
|                                (resources/CMUDict)                 |              |
|                                                                    v              |
|  Audio Waveform (.wav) ----------------------------------> [ Wav2Vec2 MDD Model ] |
|                                                                    |              |
|                                                                    v              |
|                                                             Predicted Scores      |
+--------------------------------------------------------------------+--------------+
                                                                     |
                                  AIR GAP                            |
         (No TextGrids, No Labels, No Interval Timestamps)           |
                                                                     |
+--------------------------------------------------------------------+--------------+
|                            OFFLINE SCORING COMPONENT                              |
|                                                                    v              |
|  Gold TextGrids -------------> [ Gold Parser ] -----------> [ Levenshtein ]       |
|  (Validation ONLY)                                          [  Alignment  ]       |
|                                                                    |              |
|                                                                    v              |
|                                                             Evaluated Metrics     |
|                                                             (PR-AUC, F1, Matrix)  |
+-----------------------------------------------------------------------------------+
```

### Verification Checklist:
- [x] **`ZeroOracleL2ArcticDataset.__getitem__`:** Calls only `load_l2arctic_audio_from_zip(..., speaker, utterance_id)` which reads exclusively `f"{speaker}/wav/{utterance_id}.wav"`.
- [x] **No TextGrid Reads:** `read_annotation_text` is never called within `ZeroOracleL2ArcticDataset`.
- [x] **No Target Labels:** `binary_labels` and `ctc_target_indices` are hardcoded to `None`, and `is_blinded = True`.
- [x] **No Padding/Target Masks in Model Execution:** `LangfensWav2Vec2MDD` is invoked with `source`, `audio_lengths`, `canonical_phone_indices`, and `canonical_phone_lengths` only. No `target_ctc` or `target_binary` tensors are instantiated or passed.
- [x] **Zero Test Set Contamination:** The evaluation entrypoint strictly requires `split="val"`, verifying against frozen manifest SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`. Official test speakers (`ABA`, `SKA`, `YDCK`, `ZHAA`) remain sealed.

---

## 3. Cryptographic State Verification

### Checkpoint Provenance:
- **Model Checkpoint:** `checkpoints/RQ1-TRN-02A/recovered/best.pt`
- **Checkpoint SHA-256:** `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`
- **Training Epoch Stored:** 18 (display epoch 19)
- **Training Updates:** 14,300 gradient steps

### Weights Fingerprint Immutability:
A SHA-256 digest of all 210 parameter tensors in memory was computed before and after full inference across all 300 validation utterances:
- **Pre-Inference Fingerprint:**  `32d12bdf3524508bd891605a728c9207be588d0ce8a1a16bca2f65cab12ae60d`
- **Post-Inference Fingerprint:** `32d12bdf3524508bd891605a728c9207be588d0ce8a1a16bca2f65cab12ae60d`
- **Bitwise Delta:** Exactly **$0.0$** (zero parameter drift, zero optimizer state, strict `inference_mode`).

---

## 4. Label Independence & Leakage Tests

A dedicated test suite in `tests/test_zero_oracle_isolation.py` proves that:
1. Instantiating `ZeroOracleL2ArcticDataset` and evaluating model forward outputs yields identical predictions regardless of whether ground truth TextGrids are present or absent on the file system.
2. Passing dummy or corrupted targets to an offline evaluator does not alter the model's forward logits.
3. Attempting to pass `split="test"` to the evaluation entrypoint is caught and rejected by assertion guards.

### Conclusion:
Protocol Z satisfies all requirements for rigorous, zero-oracle evaluation. The model predictions are completely decoupled from human TextGrid annotations.
