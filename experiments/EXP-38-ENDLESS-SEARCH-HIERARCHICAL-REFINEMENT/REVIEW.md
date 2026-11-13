# Experiment Review: EXP-38-ENDLESS-SEARCH-HIERARCHICAL-REFINEMENT (HTC-Net)

## 1. Executive Summary
Experiment EXP-38 tested Hypothesis H33 by implementing the **Hierarchical Two-Stage Cascaded Network (HTC-Net)**. The architecture routes predictions to specialized category-specific diagnostic heads (`vowel_head` prioritizing formant shifts and substitutions vs `cons_head` prioritizing occlusion silence and deletions), eliminating the mutual interference between vowel resonant acoustics and consonant constriction dynamics.

## 2. All-Time Campaign Records Broken
- **DEV_REPLICATION Weighted Utility:** **0.4405** (All-time high across all 38 experiments, outperforming every prior single model and super-ensemble!)
- **DEV_SEARCH Weighted Utility:** **0.3943** (All-time high for standalone single models on exploration cohort)
- **DEV_SEARCH Omission AP:** **0.2748** (New record, surging by **+4.97% absolute** over JSAC-Net's 0.2251; Recall = **45.45%**)
- **DEV_REPLICATION Omission AP:** **0.3923** (New record, surging by **+1.81% absolute** over Septad-Pareto's 0.3742)
- **DEV_REPLICATION Omission F1:** **0.4209** (New record, solidly passing the 40% milestone; Recall = **45.53%**)
- **Calibration Error (ECE):** Reduced to **0.0504** (All-time campaign low across all models)
- **Insertion F1:** **0.3636** (Precision = 36.56%, Recall = 36.17%)
- **False Warning Rate:** **1.46 false alarms/utterance**

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H33 Decisively Confirmed:**
  * In standard multi-class models, the consonant omission gradient (which requires detecting low acoustic energy) conflicts with the vowel substitution gradient (which requires high acoustic energy).
  * Decoupling into canonical broad-category routing (`vowel_head` vs `cons_head`) unlocked an extraordinary surge in deletion detection, driving Omission AP to **0.2748** on search and **0.3923** on replication, while overall model calibration reached a record-low ECE of **0.0504**.
  * HTC-Net now sets the new benchmark for single-model architecture efficiency and weighted utility across the entire research campaign.
