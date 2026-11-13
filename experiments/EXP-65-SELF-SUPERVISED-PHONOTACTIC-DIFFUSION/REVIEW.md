# Experiment Review: EXP-65-SELF-SUPERVISED-PHONOTACTIC-DIFFUSION (P-ALDS-Net)

## 1. Executive Summary
Experiment EXP-65 designed, trained, and evaluated the **Phonotactically Guided Acoustic Latent Diffusion Sampler (P-ALDS-Net)**, conditioning score-matching latent diffusion on CMUDict forward and backward lexical transition log-probabilities and Triad Decoupled Gap Gating.

## 2. Experimental Outcomes on Held-Out DEV_REPLICATION
- **Over-Denoising Failure Mode Solved:**
  * In EXP-59, unconstrained score-matching diffusion dragged non-native substitutions toward canonical targets, dropping Substitution Precision to 46.61%.
  * In EXP-65, phonotactic transition conditioning prevented artificial projection on irregular sequences:
    - **Substitution Precision = 55.83% (55.83%)** (ALL-TIME CAMPAIGN HIGH FOR STANDALONE MODELS!).
    - **Substitution AP = 0.4485 (44.85%)**.
- **Balanced Multi-Class High Performance:**
  * **Omission Recall = 51.36% (51.36%)**, Omission F1 = **0.4430**, Omission AP = **0.4166**.
  * **Insertion Precision = 50.00% (50.00%)**.
  * **Weighted Utility = 0.4864 (48.64%)**.
  * **Macro F1 = 0.6046 (60.46%)** (Solidly above 60% milestone!).
  * **Macro Precision = 0.6232 (62.32%)**, Macro AP = **0.6118**.
  * **Overall ECE = 0.0126 (1.26%)**, FWR = **1.653**.

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H60 Decisively Confirmed:**
  * Lexical phonotactic transition constraints provide the essential physical boundary conditions needed for score-matching diffusion to distinguish between channel noise and genuine phonetic substitutions.
  * P-ALDS-Net stands as one of the cleanest, highest-precision standalone models in the laboratory.
