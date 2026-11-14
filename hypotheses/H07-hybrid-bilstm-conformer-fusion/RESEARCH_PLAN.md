# Research Plan: Hypothesis H07 (H-SAME Ensemble)

1. **Model Loading & Inference:**
   - Load pinned checkpoint `bilstm_context_model.pt` from `EXP-08`.
   - Load pinned checkpoint `conformer_phonetic_scorer.pt` from `EXP-10`.
   - Generate probability arrays on `DEV_CAL` and `DEV_SEARCH`.
2. **Mixture Weight & Threshold Optimization on `DEV_CAL`:**
   - Sweep target mixture weight $w_{\text{target}} \in [0.3, 0.4, 0.5, 0.6, 0.7]$.
   - Sweep gap mixture weight $w_{\text{gap}} \in [0.2, 0.3, 0.4, 0.5, 0.6]$.
   - Sweep operating thresholds $\theta_{\text{subs}} \in [0.45, 0.55]$, $\theta_{\text{omis}} \in [0.40, 0.55]$, $\theta_{\text{ins}} \in [0.45, 0.65]$.
   - Find optimal Pareto operating configuration.
3. **Primary Locked Evaluation on `DEV_SEARCH`:**
   - Freeze all mixture parameters and evaluate on `DEV_SEARCH`.
4. **Packaging & Report:**
   - Persist full predictions, scorecard, and compute ledger.
