# Hypothesis H106: Multi-Stream Acoustic-Articulatory Fusion Network (MSAF-Net)

## 1. Context & Motivation
Across Phase 2 investigations, the laboratory isolated four distinct, validated mechanisms:
1. **Physical Area Function Inversion (EXP-101 VTI-Net):** Disambiguates constriction location and drops calibration error to 0.0196 ECE.
2. **Low-Rank Adaptation Refiner (EXP-102 PEAR-Net):** Adapts non-native acoustic deviations, surging Substitution Recall to 56.18%.
3. **Differentiable Lattice Graph Decoding (EXP-103 DSGA-Net):** Eliminates contradictory slot-gap predictions, lifting Omission AP to 0.4308 and Macro AP to 0.6178.
4. **Articulatory Disentanglement (EXP-94 ADAE-Net):** Decouples vocal timbre from phonetic gestures, achieving standalone Utility of 0.5041 and Macro F1 of 0.6285.

In prior experiments, these mechanisms were tested in isolation. We now construct **MSAF-Net**, uniting all four mechanisms into a single end-to-end standalone architecture.

## 2. Hypothesis Statement
We hypothesize that:
1. Synthesizing these four mechanisms into MSAF-Net:
   - Input Low-Rank Adaptation (LoRA $r=8$) on the 72-dim acoustic stream.
   - Concatenation with 8-dim physical vocal tract log-area functions $A_{\text{tract}}$ (80 dims total).
   - Orthogonal articulatory subspace partitioning (manner: 32, place: 32, voicing: 16, timbre: 48) with Frobenius covariance penalty and reconstruction decoding.
   - Continuous lattice graph anti-concurrence loss $\mathcal{L}_{\text{lattice}}$ coupling slot omission and gap insertion.
   - Positive-weighted BCE (`pos_weight = 7.0`) on triad-embedded gaps.
2. MSAF-Net will establish an all-time standalone record for an individual neural model:
   - Standalone **Macro F1 $\ge 0.6300$ (63.00%)** on held-out `DEV_REPLICATION`.
   - Standalone **Weighted Utility $\ge 0.5080$ (50.80%)** (surpassing ADAE-Net 0.5041 and matching the 21-model super-ensemble 0.5079!).
   - Standalone **Macro AP $\ge 0.6200$ (62.00%)**.
   - Standalone **Substitution F1 $\ge 0.4950$** and **Omission F1 $\ge 0.4700$**.
   - Standalone **Insertion F1 $\ge 0.4250$**.
   - Overall ECE $\le 0.0240$ (2.40%) and False Warning Rate $\le 2.60$ false alarms per utterance.
3. Falsification will occur if multi-stream fusion fails to outperform ADAE-Net (Macro F1 < 0.6285 or Utility < 0.5041).
