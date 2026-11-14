# Research Plan: Hypothesis H18 (QC-Pareto Sweeper)

1. **Model Assembly:**
   - Model 1: `contrastive_hierarchical_model.pt` from `EXP-22` (PSC-Net)
   - Model 2: `adaptive_hierarchical_model.pt` from `EXP-17` (SRAD-Net)
   - Model 3: `nna_adapter_hierarchical_model.pt` from `EXP-21` (NNA-Adapter)
   - Model 4: `conformer_phonetic_scorer.pt` from `EXP-10` (Conformer)
2. **Probability Generation:**
   - Extract raw prediction distributions for target phones and gaps across `DEV_CAL`, `DEV_SEARCH`, and `DEV_REPLICATION`.
3. **Exhaustive Grid Sweep on `DEV_CAL`:**
   - Grid over target weights: $(w_1, w_2, w_3, w_4)$ summing to 1.
   - Grid over gap weights: $(v_1, v_2, v_3, v_4)$ summing to 1.
   - Sweep decision thresholds: $\theta_{\text{subs}} \in [0.45, 0.60]$, $\theta_{\text{omis}} \in [0.45, 0.60]$, $\theta_{\text{ins}} \in [0.45, 0.65]$.
   - Maximize Pareto objective: $(F1_c + 2 F1_s + 2 F1_o + 1.5 F1_i) / (1 + 0.1 \cdot \text{FWR})$.
4. **Frozen Evaluation:**
   - Freeze all parameters and evaluate on `DEV_SEARCH` and `DEV_REPLICATION`.
5. **Artifacts & Ledger:**
   - Save artifacts in `experiments/EXP-23-COMPOSITE-UTILITY-SWEEP/`.
