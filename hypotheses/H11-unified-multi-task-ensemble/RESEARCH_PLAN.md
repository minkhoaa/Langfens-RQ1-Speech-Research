# Research Plan: Hypothesis H11 (Unified Super-Ensemble)

1. **Model Loading:**
   - Model 1: `bilstm_context_model.pt` from `EXP-08`
   - Model 2: `conformer_phonetic_scorer.pt` from `EXP-10`
   - Model 3: `hierarchical_articulatory_model.pt` from `EXP-13`
2. **Mixture Optimization on `DEV_CAL`:**
   - Grid search over $(w_1, w_2, w_3)$ on simplex $\sum w_i = 1$.
   - Grid search over gap mixture weights $(v_1, v_2, v_3)$.
   - Pareto operating threshold search: $\theta_{\text{subs}}, \theta_{\text{omis}}, \theta_{\text{ins}}$.
3. **Evaluation:**
   - Evaluate on `DEV_SEARCH` with primary locked evaluator.
   - Evaluate on `DEV_REPLICATION` for confirmation.
4. **Packaging & Report:**
   - Persist predictions, metrics, and artifact hashes.
