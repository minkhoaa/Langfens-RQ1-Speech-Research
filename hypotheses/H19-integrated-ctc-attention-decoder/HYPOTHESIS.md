# Hypothesis H19: Joint Soft-Attention & CTC Phonetic Diagnosis (JSA-CTC)

## 1. Proposition
Forced alignment assigns discrete, non-overlapping frame intervals $[t_s, t_e]$ to each target phone. In heavy L2 accents, boundary jitter ($\pm 20-40$ ms) forces phonetic transitions into adjacent windows, creating artificial acoustic confusion that degrades substitution precision and causes false omission warnings.

By replacing hard segmentation intervals with a soft Multi-Head Cross-Attention distribution:
$$\alpha_{i, t} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_t / \sqrt{d})}{\sum_{t'=1}^T \exp(\mathbf{q}_i^\top \mathbf{k}_{t'} / \sqrt{d})}$$
where query $\mathbf{q}_i$ represents the target phoneme and keys $\mathbf{k}_t$ represent acoustic frame embeddings, and pooling frame logits via soft attention $\bar{\boldsymbol{\ell}}_i = \sum_{t=1}^T \alpha_{i, t} \boldsymbol{\ell}_t$:
1. Boundary jitter is eliminated as each phoneme dynamically attends to its salient acoustic evidence across its true coarticulatory span.
2. `SUBSTITUTION` F1 will break past **0.385** and `OMISSION` F1 will break past **0.410** on `DEV_REPLICATION`.
3. Total Weighted Utility will advance to an all-time campaign high ($> 0.435$).

## 2. Experimental Plan
- Method ID: `M24-ctc-attention-joint-decoder`
- Run ID: `EXP-24-FINAL-ROADMAP-EXPANSION`
- Build soft cross-attention sequence network (`JSAC_Net`).
- Train on `TRAIN` for 30 epochs with joint multi-task loss.
- Calibrate on `DEV_CAL`.
- Primary locked evaluation on `DEV_SEARCH` and `DEV_REPLICATION`.
