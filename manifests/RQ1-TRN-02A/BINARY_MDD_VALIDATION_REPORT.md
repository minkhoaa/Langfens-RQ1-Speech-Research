# Binary MDD Validation Report — Fold 1

**Status: COMPLETED | Evaluation Partition: Fold 1 Validation (300 utterances, 9,956 valid phone tokens, 1,157 errors)**

Full Fold 1 validation evaluation conducted on Google Colab Tesla T4 GPU using `audit_scripts/train_fold1_ctc_binary.py` at commit `caa56783113d40eac438e5f458cdb313b8932218`. All evaluations strictly respect the strict label policy, with padded positions (-100) masked and unpadded targets strictly binary (0=correct, 1=mispronounced). Gold TextGrid boundaries are never used as model inputs.

---

## 1. Summary Metrics (Best Checkpoint: Epoch 6 / `best.pt`)

| Metric | Measured Value | Baseline / Context | Gain / Lift |
|---|---:|---:|---:|
| **Binary Average Precision (PR-AUC)** | **0.316062** | 0.116211 (Validation Prevalence) | **+0.199851 (2.72x)** |
| **Validation Base Rate (Prevalence)** | 0.116211 (11.62%) | 1,157 errors / 9,956 total phones | — |
| **Standard F1 ($\tau = 0.50$)** | **0.2532** | Precision: 0.4865, Recall: 0.1711 | — |
| **Optimal Calibrated F1 ($\tau = 0.20$)** | **0.3775** | Precision: 0.2911, Recall: 0.5367 | **+0.1243 F1 gain** |

---

## 2. Confusion Matrices at Key Operating Points

### A. Standard Operating Point ($\tau = 0.50$)
- **Precision**: 48.65% (198 / 407)
- **Recall**: 17.11% (198 / 1,157)
- **F1 Score**: 0.2532
- **Specificity**: 97.62% (8,590 / 8,799)

| | Predicted Correct (0) | Predicted Error (1) | Total |
|---|---:|---:|---:|
| **Actual Correct (0)** | **8,590** (TN) | **209** (FP) | 8,799 |
| **Actual Error (1)** | **959** (FN) | **198** (TP) | 1,157 |
| **Total** | 9,549 | 407 | 9,956 |

### B. Optimal Calibrated Operating Point ($\tau = 0.20$)
- **Precision**: 29.11% (621 / 2,133)
- **Recall**: 53.67% (621 / 1,157)
- **F1 Score**: 0.3775
- **Specificity**: 82.82% (7,287 / 8,799)

| | Predicted Correct (0) | Predicted Error (1) | Total |
|---|---:|---:|---:|
| **Actual Correct (0)** | **7,287** (TN) | **1,512** (FP) | 8,799 |
| **Actual Error (1)** | **536** (FN) | **621** (TP) | 1,157 |
| **Total** | 7,823 | 2,133 | 9,956 |

---

## 3. Full Threshold Calibration Sweep ($\tau \in [0.05, 0.95]$)

| Threshold $\tau$ | Precision | Recall | F1 Score | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1446 | **0.9196** | 0.2499 | 1,064 | 6,296 | 93 | 2,503 |
| 0.10 | 0.1794 | 0.8358 | 0.2954 | 967 | 4,422 | 190 | 4,377 |
| 0.15 | 0.2354 | 0.6525 | 0.3460 | 755 | 2,452 | 402 | 6,347 |
| **0.20 (Optimal F1)** | **0.2911** | **0.5367** | **0.3775** | **621** | **1,512** | **536** | **7,287** |
| 0.25 | 0.3566 | 0.3846 | 0.3701 | 445 | 803 | 712 | 7,996 |
| 0.30 | 0.4129 | 0.3258 | 0.3643 | 377 | 536 | 780 | 8,263 |
| 0.35 | 0.4512 | 0.3077 | 0.3659 | 356 | 433 | 801 | 8,366 |
| 0.40 | 0.4583 | 0.2947 | 0.3588 | 341 | 403 | 816 | 8,396 |
| 0.45 | 0.4666 | 0.2532 | 0.3283 | 293 | 335 | 864 | 8,464 |
| **0.50 (Standard)** | **0.4865** | **0.1711** | **0.2532** | **198** | **209** | **959** | **8,590** |
| 0.55 | 0.4678 | 0.0691 | 0.1205 | 80 | 91 | 1,077 | 8,708 |
| 0.60 | 0.4400 | 0.0190 | 0.0365 | 22 | 28 | 1,135 | 8,771 |
| 0.65 | 0.2500 | 0.0009 | 0.0017 | 1 | 3 | 1,156 | 8,796 |
| $\ge 0.70$ | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1,157 | 8,799 |

---

## 4. Per-Speaker Validation Breakdown

| Speaker | Phone Tokens | Error Tokens | Prevalence | Binary AP | F1 ($\tau=0.50$) | Precision ($\tau=0.50$) | Recall ($\tau=0.50$) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **NJS** | 5,011 | 641 | 0.1279 | **0.3950** | **0.3104** | **0.6453** | 0.2044 |
| **RRBI** | 4,945 | 516 | 0.1043 | **0.2503** | 0.1861 | 0.3284 | 0.1298 |
| **Combined** | 9,956 | 1,157 | 0.1162 | **0.3161** | 0.2532 | 0.4865 | 0.1711 |

---

## 5. Early Stopping and Convergence Summary

- **Total Epochs**: 11 epochs completed.
- **Predeclared Early Stopping Rule**: Patience = 5 epochs without validation AP improvement.
- **Peak Performance**: Epoch 6 attained the highest validation AP of `0.316062` with healthy CTC blank fraction `0.7894` (< 0.99 threshold).
- **Epochs 7–11**: AP remained below peak (`0.3147`, `0.3151`, `0.3068`, `0.3117`, `0.3044`). Patience counter reached 5 at Epoch 11, properly triggering early termination.
- Checkpoint `best.pt` safely preserves Epoch 6 weights and states.
- Official test partition remained sealed.
