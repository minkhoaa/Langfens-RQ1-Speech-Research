# Hypothesis H40: 10-Model Decimal Super-Ensemble (Decimal-Pareto)

## 1. Context and Problem Statement
In EXP-44, Dynamic Masked Context Training (DMCT-Net) officially achieved the historic **60% Macro AP milestone (0.6062)** and all-time record Utility of **0.4535**, while Nonad-Pareto (EXP-43) established record Substitution AP (**0.4161**) and record-low False Warning Rate (**1.180**).

These two landmark achievements represent complementary strengths:
- DMCT-Net excels at deletion and insertion recall under coarticulatory corruption by training on masked latent context.
- The 9-model Nonad ensemble excels at discrimination and false-alarm suppression by averaging out architectural noise across nine distinct inductive priors.

Fusing all ten models into a **Decimal-Pareto Super-Ensemble** combines the coarticulatory resilience of masked token recovery with the high-precision stability of the 9-model ensemble.

## 2. Formal Hypothesis
Forming a 10-component convex Bayesian mixture:
$$P_{\text{decimal}}(y \mid x) = \sum_{k=1}^{10} w_k P_k(y \mid x), \quad \sum_{k=1}^{10} w_k = 1, \quad w_k \ge 0$$
calibrated via global Pareto optimization on `DEV_CAL`, will:
1. **Break the 60% Macro F1 Milestone:** Push Macro F1 to $\ge 0.5900$ (targeting $\ge 0.6000$) on `DEV_REPLICATION`.
2. **Elevate Macro AP beyond 0.6100:** Achieve $\text{Macro AP} \ge 0.6100$ on `DEV_REPLICATION`.
3. **Elevate Weighted Utility beyond 0.4550:** Achieve $\text{Utility} \ge 0.4550$.
4. **Preserve Sub-1.25 False Warning Rate:** Maintain $\text{FWR} \le 1.25$ false alarms/utterance while achieving $\text{Macro Precision} \ge 0.6400$.

## 3. Predicted Evidence & Significance
If validated, Decimal-Pareto represents the pinnacle of multi-paradigm acoustic-linguistic ensemble learning in the campaign, bringing both Macro AP and Macro F1 into the $\ge 60\%$ milestone band.
