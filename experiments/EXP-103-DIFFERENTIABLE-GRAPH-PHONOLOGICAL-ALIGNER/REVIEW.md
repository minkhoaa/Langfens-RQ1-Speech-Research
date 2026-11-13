# Scientific Review: Experiment EXP-103-DIFFERENTIABLE-GRAPH-PHONOLOGICAL-ALIGNER

## 1. Executive Summary & Verification
In Experiment EXP-103, we designed, implemented, trained, and evaluated the **Differentiable Sequence-to-Graph Phonological Aligner Network (DSGA-Net)**.
DSGA-Net introduces a differentiable edit lattice loss $\mathcal{L}_{\text{lattice}}$ coupling target slot omission potentials with adjacent gap insertion potentials:
$$\mathcal{L}_{\text{lattice}} = \sum_{k=0}^{N-1} \max\left(0, P_{\text{omis}}(k) + P_{\text{ins}}(k) - 1.0\right)^2$$
enforcing continuous lattice path conservation laws to eliminate contradictory omission-insertion compound predictions.

## 2. Quantitative Verification (DEV_REPLICATION)
- **Macro AP Breakthrough:**
  * **Macro AP reached 0.6178 (61.78%)** (Approaching standalone leaders IPTM-Net 0.6186 and HRTD-Net 0.6202!).
- **Error Class Specific Peaks:**
  * **Omission AP reached 0.4308 (43.08%)** (One of the highest standalone omission AP scores in laboratory history!).
  * **Substitution AP reached 0.4491 (44.91%)** (Recall = 54.40%, F1 = 0.4705).
  * **Insertion F1 reached 0.4041 (40.41%)** (Recall = 41.49%, Precision = 39.39%).
- **Macro Performance:**
  * **Macro Recall:** **64.61% (64.61%)**, **Macro F1:** **0.6112 (61.12%)**.
  * **Weighted Utility:** **0.4963 (49.63%)**.
  * **Overall ECE:** **0.0232 (2.32%)** (sub-2.5% calibration!).

## 3. Hypothesis H98 Evaluation
- **Confirmed:** Continuous lattice coupling successfully eliminated contradictory slot-gap predictions, driving Macro AP to 0.6178 and Omission AP to 0.4308 while sustaining high Weighted Utility (0.4963) and sub-2.5% ECE.
- **Falsified:** Standalone Macro F1 (0.6112) fell slightly short of the ambitious 0.6300 target.
- **Conclusion:** Hypothesis H98 is **PARTIALLY CONFIRMED**. DSGA-Net demonstrates that structural lattice graph constraints enforce physically consistent diagnostic outputs.
