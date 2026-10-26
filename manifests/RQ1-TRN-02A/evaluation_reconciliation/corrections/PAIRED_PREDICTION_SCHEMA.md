# Paired Prediction Records: Schema & Independent Reconstruction Guide — RQ1-TRN-02A-EVAL-R2-CORR

**Artifact:** `manifests/RQ1-TRN-02A/evaluation_reconciliation/corrections/PAIRED_PREDICTION_RECORDS.json`  
**Test Suite:** `tests/test_prediction_reconstruction.py` (PASS)  
**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  

---

## 1. Overview and Purpose

`PAIRED_PREDICTION_RECORDS.json` provides an auditable, lightweight, non-binary record of model predictions across all 300 Fold 1 validation utterances (`NJS` and `RRBI`).

It enables complete statistical reconstruction of:
- Binary MDD Average Precision (AP / PR-AUC)
- Greedy CTC PER, Edit counts (S, D, I), reference and hypothesis token lengths
- Confusion matrices and F1 scores across any threshold
- Paired matched vs. mismatched score statistics ($\Delta S$, agreement, sensitivity)
- Utterance-clustered bootstrap distributions

**Zero model weights, audio binaries, or CUDA GPUs are required to verify the audit findings.**

---

## 2. JSON Schema Specification

```json
{
  "metadata": {
    "description": "Per-utterance matched and mismatched prediction records for Fold 1 validation",
    "split": "val",
    "fold": 1,
    "total_utterances": 300,
    "total_phone_tokens": 9956,
    "total_error_tokens": 1157,
    "evaluator_source_sha": "<git-sha>",
    "checkpoint_sha256": "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
    "schema": { ... }
  },
  "utterances": [
    {
      "record_id": "l2arctic_NJS_arctic_a0003",
      "speaker": "NJS",
      "utterance_id": "arctic_a0003",
      "mismatched_audio_record_id": "l2arctic_NJS_arctic_a0005",
      "canonical_phone_indices": [15, 6, 28, 12, ...],
      "binary_reference_labels": [0, 0, 0, 0, 1, 0, ...],
      "matched_binary_scores": [0.021154, 0.067339, ...],
      "mismatched_binary_scores": [0.023037, 0.061570, ...],
      "matched_ctc_reference": [15, 6, 28, 12, ...],
      "matched_ctc_hypothesis": [15, 6, 28, 12, ...]
    }
  ]
}
```

### Field Definitions:
| Field | Type | Description |
|---|---|---|
| `record_id` | `str` | Full internal L2-ARCTIC identifier (`l2arctic_{speaker}_{utt}`) |
| `speaker` | `str` | Speaker code (`NJS` or `RRBI`) |
| `utterance_id` | `str` | ARCTIC utterance name (e.g. `arctic_a0003`) |
| `mismatched_audio_record_id` | `str` | Identifier of donor utterance under within-speaker cyclic derangement ($\text{shift\_1}$) |
| `canonical_phone_indices` | `List[int]` | Integer indices of prompt phonemes passed to cross-attention |
| `binary_reference_labels` | `List[int]` | Ground-truth binary MDD reference labels ($0 = \text{correct}, 1 = \text{error}$) for valid tokens |
| `matched_binary_scores` | `List[float]` | Predicted error probabilities from true matched audio |
| `mismatched_binary_scores` | `List[float]` | Predicted error probabilities from permuted within-speaker audio |
| `matched_ctc_reference` | `List[int]` | Ground-truth CTC reference phone token indices |
| `matched_ctc_hypothesis` | `List[int]` | Greedy decoded CTC hypothesis phone token indices |

---

## 3. Independent Reconstruction Recipe

To verify all validation metrics directly from Python without invoking the model:

```python
import json
import numpy as np
from audit_scripts.evaluate_fold1_ctc_binary import average_precision_binary, calculate_ctc_metrics

with open("manifests/RQ1-TRN-02A/evaluation_reconciliation/corrections/PAIRED_PREDICTION_RECORDS.json") as f:
    data = json.load(f)

labels = []
matched_scores = []
mismatched_scores = []
refs, hyps = [], []

for u in data["utterances"]:
    labels.extend(u["binary_reference_labels"])
    matched_scores.extend(u["matched_binary_scores"])
    mismatched_scores.extend(u["mismatched_binary_scores"])
    refs.append(u["matched_ctc_reference"])
    hyps.append(u["matched_ctc_hypothesis"])

# 1. Binary AP
matched_ap = average_precision_binary(labels, matched_scores)
mismatched_ap = average_precision_binary(labels, mismatched_scores)
print(f"Matched AP: {matched_ap:.6f}")       # 0.317021
print(f"Mismatched AP: {mismatched_ap:.6f}") # 0.294722

# 2. Confusion matrix at 0.50
preds_05 = (np.array(matched_scores) >= 0.5).astype(int)
y = np.array(labels)
tp = int(((preds_05 == 1) & (y == 1)).sum()) # 201
fp = int(((preds_05 == 1) & (y == 0)).sum()) # 208
fn = int(((preds_05 == 0) & (y == 1)).sum()) # 956
tn = int(((preds_05 == 0) & (y == 0)).sum()) # 8591

# 3. CTC PER
ctc = calculate_ctc_metrics(refs, hyps, blank_frames=150654, total_frames=190877)
print(f"PER: {ctc['greedy_per']:.6f}")       # 0.236818
print(f"Edits: {ctc['total_edits']}")        # 2340
```

This reconstruction is formally verified in continuous integration via `tests/test_prediction_reconstruction.py`.
