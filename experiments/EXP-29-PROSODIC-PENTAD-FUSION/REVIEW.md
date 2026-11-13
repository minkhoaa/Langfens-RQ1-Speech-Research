# Experiment Review: EXP-29-PROSODIC-PENTAD-FUSION (Hexad-Pareto)

## 1. Executive Summary
Experiment EXP-29 implemented the 6-Model Hexad Super-Ensemble (**Hexad-Pareto**), fusing:
1. **PSC-Net (EXP-22)**: Phonological Contrastive Prototype Margin
2. **SRAD-Net (EXP-17)**: Adaptive Speaking-Rate Duration Modeling
3. **NNA-Adapter (EXP-21)**: Acoustic Bottleneck Non-Native Refitting
4. **Conformer (EXP-10)**: Multi-Head Self-Attention
5. **JSAC-Net (EXP-24)**: Monotonic Local Gaussian Attention
6. **APED-Net (EXP-28)**: Physical Waveform Acoustic Prosodic-Energy Dynamics

## 2. All-Time Campaign Records Broken
- **DEV_REPLICATION Weighted Utility:** **0.4321** (All-time high, surpassing EXP-23's 0.4310)
- **DEV_SEARCH Weighted Utility:** **0.3914** (All-time high, surpassing EXP-24's 0.3802)
- **Macro AP (Replication):** **0.5810** (First model in campaign history breaking the 0.5800 Macro AP barrier!)
- **Macro F1 (Replication):** **0.5801** (First model in campaign history breaking the 0.5800 Macro F1 barrier!)
- **Substitution AP (Replication):** **0.4019** (First model in campaign history breaking the 40% Substitution AP barrier!)
- **Omission AP (Replication):** **0.3694** (All-time high on replication; Precision = **45.61%**)
- **Substitution F1 (Replication):** **0.4441** (Precision = 48.16%, Recall = 41.19%)
- **Macro Precision (Replication):** **0.6193** (Solidly certified above the 60% milestone)
- **DEV_SEARCH Insertion AP:** **0.2721** (All-time high on DEV_SEARCH; Precision = **38.37%**, F1 = **0.3128**)

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H24 Decisively Confirmed:**
  * Blending the physical energy dynamics of APED-Net ($w_{\text{aped}}=0.20$) with the architectural diversity of Pentad-Pareto ($w_{\text{jsac}}=0.20, w_{\text{psc}}=0.15, w_{\text{srad}}=0.15, w_{\text{nna}}=0.15, w_{\text{conf}}=0.15$) unlocked simultaneous gains across both discrimination (AP) and decision accuracy (F1).
  * Hexad-Pareto now stands as the overall champion system across the entire autonomous research campaign.
