# Hypothesis H52: Phonetically Grounded Adaptive Slot Thresholding (AMT-Engine)

## 1. Context & Motivation
Across 56 experiments, decision gating has relied on static, uniform thresholds: a single global triplet $(\theta_{\text{subs}}, \theta_{\text{omis}}, \theta_{\text{ins}})$ applied uniformly across every phoneme in an utterance.

However, human phoneticians and speech science demonstrate that error thresholds cannot be uniformly flat across all phonetic classes:
1. **Phonetic Manner Disparity:**
   - **Vowels** have long durations (150-300ms) and high energy; a subtle acoustic dip rarely signifies an omission, but formant shifts reliably indicate substitutions.
   - **Stops (`/p/`, `/t/`, `/k/`)** have brief transient bursts (5-15ms); unreleased plosives easily trigger false omission alarms under static thresholds.
   - **Fricatives & Liquids** have continuous noisy or formant spectra where coarticulation creates high variance.
2. **Speaking Rate / Slot Duration Conditioning:**
   - A phoneme slot compressed to $< 35\%$ of normal duration has an order-of-magnitude higher empirical prior of omission.
   - A prolonged phoneme slot ($> 150\%$ duration) has near-zero chance of omission, meaning any omission probability below 0.80 is almost certainly a false alarm.
3. **Acoustic Entropy / Confidence:**
   - Frames with high posterior entropy indicate recording channel noise or coarticulatory blending, requiring conservative decision thresholds to prevent false warnings.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Designing the **Adaptive Multi-Threshold Inference Engine (AMT-Engine)**, which modulates decision thresholds dynamically for each slot $k$ based on canonical manner category, relative duration ratio, and posterior entropy:
   $$\theta_o(k) = \text{clip}\left(\theta_{o, \text{base}} + \Delta_o(M_k) + \alpha_o (r_k - 1.0) + \beta_o (H_k - \bar{H}), 0.20, 0.85\right)$$
   $$\theta_s(k) = \text{clip}\left(\theta_{s, \text{base}} + \Delta_s(M_k) - \alpha_s (r_k - 1.0) + \beta_s (H_k - \bar{H}), 0.20, 0.85\right)$$
   $$\theta_i(g) = \text{clip}\left(\theta_{i, \text{base}} + \Delta_i(\text{cluster}) + \gamma_i (dur_g - \bar{dur}_g), 0.30, 0.90\right)$$
   will resolve manner-specific false alarms without requiring model retraining.
2. Applying AMT-Engine to **Undecim-Pareto (EXP-52)** and **DSCA-Net (EXP-55)** will:
   - Break the **0.5000 (50.00%) Weighted Utility barrier** for the first time in laboratory history ($\text{Utility} \ge 0.5000$).
   - Push **Macro AP** past $\ge 0.6300$ (63.00%).
   - Push **Macro F1** past $\ge 0.6150$ (61.50%).
   - Suppress False Warning Rate by $\ge 15\%$ (FWR $\le 1.45$ false alarms/utterance).

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (Broke 50% milestone!).
2. **DEV_REPLICATION Macro AP:** $\ge 0.6300$ (All-time campaign record).
3. **DEV_REPLICATION Macro F1:** $\ge 0.6150$.
4. **DEV_REPLICATION FWR:** $\le 1.45$ false alarms/utterance.
