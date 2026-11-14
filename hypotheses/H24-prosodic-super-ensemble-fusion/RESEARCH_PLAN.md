# Research Plan: Hypothesis H24 / Experiment EXP-29

## 1. Objective
Implement, optimize, and evaluate the 6-Model Hexad Super-Ensemble (**Hexad-Pareto**) fusing PSC-Net, SRAD-Net, NNA-Adapter, Conformer, JSAC-Net, and APED-Net on `DEV_CAL` and `DEV_REPLICATION` under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Model Checkpoints:**
   - Model 1: PSC-Net (`experiments/EXP-22-PHONOLOGICAL-CONTRASTIVE-MARGIN/CHECKPOINTS/contrastive_hierarchical_model.pt`)
   - Model 2: SRAD-Net (`experiments/EXP-17-ADAPTIVE-DURATION-CTC/CHECKPOINTS/adaptive_hierarchical_model.pt`)
   - Model 3: NNA-Adapter (`experiments/EXP-21-ENDLESS-EXPLORATION-ACOUSTIC-REFIT/CHECKPOINTS/nna_adapter_hierarchical_model.pt`)
   - Model 4: Conformer (`experiments/EXP-10-CONFORMER-ATTENTION/CHECKPOINTS/conformer_phonetic_scorer.pt`)
   - Model 5: JSAC-Net (`experiments/EXP-24-FINAL-ROADMAP-EXPANSION/CHECKPOINTS/jsac_monotonic_model.pt`)
   - Model 6: APED-Net (`experiments/EXP-28-PROSODIC-ENERGY-DYNAMICS/CHECKPOINTS/aped_net.pt`)
2. **Feature Input Generation:**
   - 55-dim acoustic sequences for Models 1, 2, 3, 5.
   - 51-dim sequences for Model 4.
   - 60-dim target / 9-dim gap prosodic sequences for Model 6.
3. **Global Pareto Grid Sweep on DEV_CAL:**
   - Search 6-component convex mixture weights: $(w_1, w_2, w_3, w_4, w_5, w_6)$.
   - Search gap mixture weights: $(v_1, v_2, v_3, v_4, v_5, v_6)$.
   - Search operating thresholds: $(\theta_{\text{subs}}, \theta_{\text{omis}}, \theta_{\text{ins}})$.
4. **Independent Evaluation:**
   - Evaluate on `DEV_SEARCH` (599 utterances) and `DEV_REPLICATION` (300 utterances).
   - Log all metrics, PR curves, ECE, FWR, and Utility score.
5. **Ledger & Artifact Audit:**
   - Save configuration, metrics, predictions, logs, and SHA-256 hashes under `experiments/EXP-29-PROSODIC-PENTAD-FUSION/`.
