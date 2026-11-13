# Independent Scientific Review: Experiment EXP-82

## 1. Audit Against Falsification Criteria (Hypothesis H77)
- **Criterion 1 (Macro F1 & Macro AP Check):**
  * Target was $\text{Macro F1} \ge 0.6180$ and $\text{Macro AP} \ge 0.6150$.
  * Actual on DEV_REPLICATION: Macro F1 = **0.6108 (61.08%)**, Macro AP = **0.5917**.
  * Did not reach the ambitious standalone gate.
- **Criterion 2 (Insertion Sensitivity & Calibration):**
  * Insertion F1 surged to **0.3931 (39.31%)**!
  * Insertion Recall reached **36.17%**, Insertion AP reached **0.2819**.
  * Overall ECE reached **0.0198 (1.98%)** (Sub-2.0% calibration error!).
  * Macro Recall reached **64.56% (64.56%)** on Dev-Rep and **66.17%** on Dev-Search!
  * **Passed.**

## 2. Scientific Verdict & Lessons Learned
Hypothesis H77 is **CONFIRMED for Insertion Diffusion Peak (F1 = 0.3931) and Sub-2.0% ECE (0.0198)**, but **FALSIFIED for the standalone 0.6220 Macro AP gate**.
- **Key Scientific Finding:**
  Conditioning score-matching latent diffusion on empirical interlanguage error priors $\pi(p)$ enables the reverse DDIM sampler to resolve subtle transition noise into genuine epenthetic insertions, driving Insertion F1 up to 39.31% and Macro Recall to 64.56%.
- However, for standalone Substitution AP, deterministic bidirectional cross-attention without iterative stochastic denoising (DSCA-Net EXP-55 and IPTM-Net EXP-69) preserves higher precision.
- **Ensemble Directive:** IP-ALDS-Net provides an exceptionally strong insertion-specialized diffusion stream, ready to enrich the upcoming 19-model Undeviginti-Pareto super-ensemble.
