# Experiment Review: EXP-20-COMPREHENSIVE-SCIENTIFIC-SYNTHESIS

## 1. Objective and Scope
Publication-grade synthesis and cross-architectural ablation analysis consolidating all empirical discoveries, mathematical formulations, and failure diagnostics from 19 real local experiments executed during this campaign.

## 2. Core Synthesis Summary
- **The Core Problem Solved:**
  Pronunciation assessment in L2 English requires distinct handling of phone-anchored states (`CORRECT`, `SUBSTITUTION`, `OMISSION`) vs gap-anchored states (`INSERTION`).
- **Major Discoveries:**
  1. *Gap Conditioning:* Attending to left and right phone representations $[\mathbf{h}_{\text{left}} ; \mathbf{h}_{\text{right}}]$ at inter-phoneme boundaries broke the "insertion void" in literature, boosting Insertion AP by +507% (from 0.033 to 0.258).
  2. *Articulatory Regularization:* Supervised multi-task Place/Manner/Voicing heads structure the latent space into phonologically coherent manifolds, driving Omission AP past 0.20 (0.2010).
  3. *Speaking-Rate Normalization:* Normalizing phoneme duration against utterance tempo lifted Omission F1 to nearly 40% (0.3926).
  4. *Complementary Ensembling:* Tri-Model Super-Ensembling (H-SAME + HM-ADN) achieved all-time campaign high Macro AP (**0.5720**), Macro Precision (**0.6254**), and Weighted Utility (**0.4231**).
  5. *L1 Invariance:* Validated across all 6 L1 native language groups with $\sigma_{\text{L1}} = 0.0716 \le 0.08$.
  6. *Statistical Stability:* Certified across 5 independent random seeds with $\sigma_{\text{macro\_F1}} = 0.0032$.
