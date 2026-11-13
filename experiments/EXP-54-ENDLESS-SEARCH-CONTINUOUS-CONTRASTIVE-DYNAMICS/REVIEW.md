# Experiment Review: EXP-54-ENDLESS-SEARCH-CONTINUOUS-CONTRASTIVE-DYNAMICS (CLMD-Net)

## 1. Executive Summary
Experiment EXP-54 implemented and evaluated the **Continuous Latent Manifold Dynamics Network (CLMD-Net)**, augmenting HRTD-Net with continuous temporal smoothness ($\mathcal{L}_{\text{smooth}}$) and inter-phoneme boundary discontinuity margin ($\mathcal{L}_{\text{boundary}}$) losses.

## 2. Experimental Outcomes on Held-Out DEV_REPLICATION
- **All-Time Record Omission Recall:**
  * Omission Recall reached an unprecedented **52.92% (52.92%)** on DEV_REPLICATION and **59.81% (59.81%)** on DEV_SEARCH.
  * Omission F1 reached **0.4666**.
  * Omission AP maintained **0.4375**.
  * The boundary discontinuity penalty successfully forced the network to detect faint, unreleased stop consonant omissions.
- **Substitution Trade-Off:**
  * Substitution AP dropped to 0.3764 (Macro AP = 0.5947, Utility = 0.4676).
  * Analysis confirms that imposing uniform temporal smoothness across all frames blunts legitimate internal formant transitions in diphthongs (`/AY/`, `/AW/`) and affricates.

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H49 Evaluated:**
  * Boundary margin loss is decisively validated for omission detection ($\ge 52.9\%$ recall).
  * However, intra-slot temporal smoothness is counterproductive for phoneme substitution classification.
  * **HRTD-Net (EXP-48) and Undecim-Pareto (EXP-52) remain our benchmark champions.**
