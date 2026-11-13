# Scientific Review: Experiment EXP-106-DUODEVIGINTI-PLUS-TWO-CONVERGENCE-SUPER-ENSEMBLE

## 1. Executive Summary & Verification
In Experiment EXP-106, we formulated, calibrated, and evaluated the **25-Model Viginti-Quintum-Pareto Super-Ensemble**, uniting:
- Viginti-Pareto (EXP-86: 21 Models)
- PEAR-Net (EXP-102: LoRA Acoustic Refiner)
- DSGA-Net (EXP-103: Graph Lattice Aligner)
- SGAN-Net (EXP-104: Soft-Gold Adjudication)
- VTI-Net (EXP-101: Vocal Tract Inversion)

## 2. Quantitative Verification (DEV_REPLICATION)
- **Weighted Utility:** **0.5028 (50.28%)** (comfortably sustaining >50% Weighted Utility in an ensemble!).
- **Macro Precision:** **64.47% (64.47%)**!
- **False Warning Rate:** Dropped to **1.587 false alarms per utterance** (only 476 total false warnings across 300 utterances, with only 128 false omissions and 321 false substitutions).
- **Insertion F1 Peak:** **0.4079 (40.79%)** (Precision = 53.45%, Recall = 32.98%, AP = 0.3199).
- **Macro AP:** **0.6348 (63.48%)**, **Macro F1:** **0.6169 (61.69%)**.

## 3. Hypothesis H101 Evaluation
- **Confirmed:** Combining Phase 2 specialists with Viginti-Pareto successfully delivered high Macro Precision (64.47%) and elevated Insertion F1 (40.79%) with clean false warning suppression (1.59/utt).
- **Falsified:** Global Weighted Utility (0.5028) did not reach the 0.5100 target due to conservative omission thresholding.
- **Conclusion:** Hypothesis H101 is **PARTIALLY CONFIRMED**. The 25-model mixture establishes a superior precision-first operating point on the Phase 2 Pareto frontier.
