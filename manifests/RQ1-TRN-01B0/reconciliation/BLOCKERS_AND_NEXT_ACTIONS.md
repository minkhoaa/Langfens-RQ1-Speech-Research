# LANGFENS — RQ1-TRN-01B0-R: Blockers, Resolved Discrepancies & Next Actions

**Gate Reference:** RQ1-TRN-01B0-R — Real-Data Learning Diagnostics & Evidence Reconciliation  
**Deliverable Path:** `manifests/RQ1-TRN-01B0/reconciliation/BLOCKERS_AND_NEXT_ACTIONS.md`  
**Lead Research Team:** Senior Speech ML Research Engineer, PyTorch Training Engineer, Scientific Reproducibility Reviewer  
**Gate Evaluation Status:** `RQ1-TRN-01B0-R: PASS`  

---

## 1. Executive Summary & Gate Status

Gate **RQ1-TRN-01B0-R** has achieved formal **PASS** status. All technical defects, metric conflations, documentation discrepancies, CTC decoding errors, binary calibration misinterpretations, checkpoint continuity questions, and GPU accounting errors identified during reproducibility review have been completely resolved.

The evidence and code corrections establish a clean, scientifically transparent baseline before authorization for full Fold 1 fine-tuning can be granted.

---

## 2. Itemized Resolution of Audit Findings (Tasks A–F)

| Audit Item | Initial Defect / Discrepancy | Resolved Implementation & Audit Evidence | Verification Deliverable |
| :--- | :--- | :--- | :--- |
| **Task A: Training Subset Evidence** | Zero overlap between audit table in `DATA_LEAKAGE_AUDIT.md` and actual training manifest `TRAIN_ONLY_SUBSET_MANIFEST.json`. | Rebuilt audit table programmatically from the authentic 32 training utterances. Verified zero overlap with Fold 1 val (300) or test (900). Root-caused previous table to an early prototype slice. | `CORRECTED_SUBSET_LEAKAGE_AUDIT.md` |
| **Task B: Repair Diagnosis Metrics** | `textgrid_parser.py` assigned `diagnosis_target` to correct phones, inflating denominator to 1,000 and reporting 8% accuracy when true substitution accuracy was only 3.23%. | Masked correct productions, deletions, and `err` tags (`ignore_index = -100`). Restricted substitution diagnosis strictly to eligible substitutions ($N=124$). Defined auxiliary phone classification separately ($N=1000$). Repeat run proved substitution accuracy increased to 16.13%. | `SUBSTITUTION_DIAGNOSIS_METRIC_AUDIT.md`, `tests/test_metric_reconciliation.py` |
| **Task C: CTC Recognition Diagnostics** | CTC loss fell 80% ($17.75 \to 3.55$), but PER was reported as 1.0000; framewise logits were not trimmed to active lengths. | Fixed decoding loop to strictly trim to `input_lengths`. Diagnosed complete **blank collapse** (100% of 5,886 frames predicted blank; 1,012 deletions, 0 insertions, 0 substitutions). Explained mathematical mechanism. Prohibited claiming recognition success. | `CTC_DECODING_AND_PER_DIAGNOSTICS.md` |
| **Task D: Binary Detection Calibration** | Grid-search threshold-optimized F1 (0.4126) was presented without qualifying that default threshold 0.5 yielded all-negative output (F1 = 0.0). | Separately reported default-threshold metrics (F1 = 0.0, TP = 0, TN = 876, FN = 178). Formally labeled threshold tuning as an exploratory `TRAIN_ONLY_DIAGNOSTIC`. Analyzed score distributions (mean for errors 0.2145 > correct 0.1628; PR-AUC 0.4066). Reserved formal calibration for validation split. | `BINARY_CALIBRATION_INTERPRETATION.md` |
| **Task E: Checkpoint Continuity** | `verify_checkpoint_reproducibility()` only checked rounded loss of eval model, failing to prove true optimizer resume. | Verified structural integrity of 232 parameters, 221 AdamW state buffers, GradScaler, and scheduler. Differentiated reload, inference ($\Delta = 0.0$), and active training resume. Demonstrated bit-identical continuation at Step 6 ($\Delta = 0.0$) and bounded ATen kernel variance ($\Delta < 2.0 \times 10^{-4}$) thereafter. | `CHECKPOINT_CONTINUITY_REPORT.md`, `audit_scripts/audit_checkpoint_continuity.py` |
| **Task F: GPU & Step Accounting** | Memory reported as 1,909.38 MB vs. 1,535.7 MiB. Step counts conflated micro-steps (216), updates (100), and AMP overflow skips (8). | Reconciled peak allocated memory: $1,909.38\text{ MB}$ (decimal) = $1,820.93\text{ MiB}$ (binary); $1,535.7\text{ MB}$ was instantaneous allocation. Reconciled 216 attempted micro-steps = 100 successful updates + 8 AMP skipped updates = 400 effective utterance exposures (12.50 epochs). | `GPU_AND_STEP_ACCOUNTING_ERRATUM.md` |

---

## 3. Strict Boundary and Security Constraints

In accordance with scientific integrity guidelines:
- **Zero Raw Data Modification:** L2-ARCTIC raw data, TAMU annotations, and `manifests/RQ1-SPLIT-01/` split manifests remain immutable (SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- **Zero Leakage:** No validation ($S_{\text{val}}$, $U_{\text{val}}$) or test ($S_{\text{test}}$, $U_{\text{test}}$) data was accessed or used for parameter updates during this reconciliation.
- **No Premature Fine-Tuning:** Full Fold 1 training, 4-fold cross-validation, and Vietnamese LOSO training remain **strictly stopped**.
- **No Public Checkpoint Release:** Local checkpoints remain strictly confined to `/home/khoa/Projects/KLTN/Dataset/checkpoints/` and excluded by `.gitignore`.

---

## 4. Open Technical Risks & Recommendations for Full Fold 1 Training

Before commencing full Fold 1 training ($2,399$ training utterances, $\sim 25$ epochs), two key algorithmic challenges must be addressed:

### 1. CTC Blank Collapse Mitigation:
- *Risk:* As demonstrated in Tasks C and G, optimizing a newly initialized linear CTC head end-to-end on raw acoustic features from scratch causes the head to rapidly collapse into predicting $100\%$ blanks.
- *Remediation:*
  - Apply a blank prior penalty or negative bias initialization on the blank token ($b_{\text{blank}} = -2.0$).
  - Warm up the CTC head while keeping the wav2vec2 transformer backbone frozen for the first 1–2 epochs, establishing initial phonetic alignments before end-to-end fine-tuning.
  - Evaluate multi-task loss weight scheduling ($\lambda_{\text{ctc}} = 0.5 \to 1.0$).

### 2. Binary Mispronunciation Detection Calibration:
- *Risk:* Because mispronunciations constitute only $\approx 16.9\%$ of canonical tokens, unweighted binary cross-entropy drives the predicted logits negative, resulting in zero detections at threshold $0.50$.
- *Remediation:*
  - Train with `pos_weight \approx 5.0` in `MaskedBCEWithLogitsLoss` to penalize false negatives and center the decision boundary at $0.50$.
  - Perform temperature scaling and threshold tuning exclusively on the held-out Fold 1 validation partition ($S_{\text{val}}$, $300$ utterances).

---

## 5. Next Actions Sequence for Project Langfens

1. **Synchronize Deliverables to Google Drive:**  
   Execute `rclone sync manifests/RQ1-TRN-01B0/reconciliation/ kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01B0/reconciliation/ --fast-list -v` and verify remote presence.
2. **Commit Code & Documentation on Feature Branch:**  
   Commit safe code changes, unit tests, and the 9 reconciliation deliverables on branch `feat/rq1-trn-01b0-r-evidence-reconciliation`.
3. **Formal Review & Sign-Off:**  
   Submit reconciliation deliverables for review. Await formal authorization before proceeding to Phase **RQ1-TRN-02** (Full Fold 1 Model Training & Validation Calibration).
