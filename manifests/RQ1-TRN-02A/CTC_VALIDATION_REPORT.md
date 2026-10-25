# CTC Validation Report — Fold 1

**Status: COMPLETED | Evaluation Partition: Fold 1 Validation (300 utterances, 9,881 reference phones)**

Full Fold 1 training and validation were executed on Google Colab Tesla T4 GPU using `audit_scripts/train_fold1_ctc_binary.py` at commit `caa56783113d40eac438e5f458cdb313b8932218`. CTC decoding was evaluated strictly using greedy argmax with consecutive token deduplication and blank token (0) elimination. Zero gold TextGrid timestamps or labels were used during inference.

---

## 1. Overall CTC Recognition Performance (Best Checkpoint: Epoch 6 / `best.pt`)

| Metric | Measured Value | Percentage / Rate |
|---|---:|---:|
| **Phone Error Rate (Greedy PER)** | **0.236413** | **23.64%** |
| Total Reference Tokens | 9,881 | 100.00% |
| Total Decoded Hypothesis Tokens | 9,648 | 97.64% |
| Total Edit Distance | 2,336 | 23.64% |
| **Substitutions (S)** | **1,699** | **17.19%** |
| **Deletions (D)** | **435** | **4.40%** |
| **Insertions (I)** | **202** | **2.04%** |
| **Decoded Length Ratio** | **0.976419** | **97.64%** |
| Framewise Blank Probability Fraction | 0.789390 | 78.94% |

---

## 2. Per-Speaker Validation Breakdown

| Speaker | Utterances | Ref Tokens | Hyp Tokens | S | D | I | Total Edits | PER | Decoded Length Ratio | Blank Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **NJS** | 150 | 4,964 | 4,875 | 848 | 198 | 109 | 1,155 | **0.2327** (23.27%) | 0.9821 (98.21%) | 0.7852 |
| **RRBI** | 150 | 4,917 | 4,773 | 851 | 237 | 93 | 1,181 | **0.2401** (24.01%) | 0.9707 (97.07%) | 0.7936 |

Both holdout validation speakers demonstrate balanced recognition performance with negligible inter-speaker divergence (|23.27% - 24.01%| = 0.74%).

---

## 3. Epoch-by-Epoch CTC Convergence

| Epoch | Train CTC Loss | Val PER | Substitutions | Deletions | Insertions | Decoded Length Ratio | Blank Fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.8420 | 0.3112 | 2,031 | 720 | 324 | 0.9599 | 0.7980 |
| 2 | 1.0047 | 0.2737 | 1,914 | 512 | 278 | 0.9763 | 0.7926 |
| 3 | 0.9191 | 0.2608 | 1,838 | 496 | 243 | 0.9744 | 0.7920 |
| 4 | 0.8694 | 0.2539 | 1,776 | 531 | 202 | 0.9667 | 0.7947 |
| 5 | 0.8374 | 0.2388 | 1,710 | 444 | 206 | 0.9759 | 0.7902 |
| **6 (best.pt)** | **0.8136** | **0.2364** | **1,699** | **435** | **202** | **0.9764** | **0.7894** |
| 7 | 0.7958 | 0.2302 | 1,626 | 467 | 182 | 0.9712 | 0.7929 |
| 8 | 0.7813 | 0.2320 | 1,662 | 454 | 176 | 0.9719 | 0.7917 |
| 9 | 0.7707 | 0.2224 | 1,595 | 402 | 201 | 0.9797 | 0.7894 |
| 10 | 0.7607 | 0.2221 | 1,607 | 413 | 175 | 0.9759 | 0.7911 |
| 11 | 0.7529 | 0.2247 | 1,591 | 485 | 144 | 0.9655 | 0.7935 |

**Findings**:
1. Zero blank collapse observed across all 11 epochs: decoded length ratio remained stable between 96.0% and 98.0%.
2. CTC PER steadily converged from 31.12% down to 22.21% (lowest in Epoch 10; 23.64% at AP-optimal Epoch 6).
3. Official Fold 1 test partition (900 utterances, 6 test speakers) remained sealed throughout.
