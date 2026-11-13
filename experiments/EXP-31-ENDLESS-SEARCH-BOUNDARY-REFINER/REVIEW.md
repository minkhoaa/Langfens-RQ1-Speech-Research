# Experiment Review: EXP-31-ENDLESS-SEARCH-BOUNDARY-REFINER (BER-Net)

## 1. Executive Summary
Experiment EXP-31 tested Hypothesis H26 by augmenting target sequence vectors with sub-phonemic boundary dynamics (crest factor, edge energy ratios, and edge ZCR transitions) to capture short transient stop bursts and affricate releases.

## 2. Quantitative Results
- **DEV_SEARCH:**
  - Omission Recall: **44.02%**
  - Substitution F1: **0.3830**
  - Insertion F1: **0.2825**
  - Utility: **0.3705**
- **DEV_REPLICATION:**
  - Omission Recall: **39.69%** (F1 = **0.3792**, Precision = 36.30%)
  - Substitution AP: **0.3724** (Precision = 48.71%)
  - Insertion F1: **0.2971**
  - Utility: **0.4109**

## 3. Scientific Synthesis
Sub-phonemic boundary refinement confirmed strong omission sensitivity (44.02% recall on search, 39.69% on replication), demonstrating that crest factor and edge energy ratios capture plosive release transients effectively. However, on overall utility, the balanced multi-model Hexad-Pareto (EXP-29) remains the supreme configuration.
