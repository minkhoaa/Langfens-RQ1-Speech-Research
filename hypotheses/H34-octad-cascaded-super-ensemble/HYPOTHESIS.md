# Hypothesis H34: 8-Model Octad Super-Ensemble (Octad-Pareto)

## 1. Context and Problem Statement
Throughout the campaign, each breakthrough addressed a distinct scientific barrier:
1. **PSC-Net (EXP-22)**: Structured phonetic confusion via contrastive cosine margin.
2. **SRAD-Net (EXP-17)**: Resolved speaking-rate tempo variations.
3. **NNA-Adapter (EXP-21)**: Adapted non-native acoustic feature space.
4. **Conformer (EXP-10)**: Captured long-range sequence context with low ECE.
5. **JSAC-Net (EXP-24)**: Eliminated frame boundary jitter via monotonic local attention.
6. **APED-Net (EXP-28)**: Injected raw physical waveform energy ratios and ZCR.
7. **PNC-Net (EXP-34)**: Filtered out phonotactically illegal acoustic hypotheses.
8. **HTC-Net (EXP-38)**: Decoupled vowel substitution diagnosis from consonant deletion diagnosis via broad-category canonical routing, setting an all-time campaign high Utility of $0.4405$ and Omission AP of $0.3923$.

Each of these eight models contributes a mathematically orthogonal inductive bias. Combining all eight models into an **Octad-Pareto Super-Ensemble** provides a complete integration of sequence modeling, representation learning, physical acoustics, lexical linguistics, and hierarchical phonetic routing.

## 2. Formal Hypothesis
Forming an 8-component convex Bayesian mixture:
$$P_{\text{octad}}(y \mid x) = \sum_{k=1}^8 w_k P_k(y \mid x), \quad \sum_{k=1}^8 w_k = 1, \quad w_k \ge 0$$
calibrated via global Pareto optimization on `DEV_CAL`, will:
1. Surpass the previous all-time campaign high for Weighted Utility, achieving $\text{Utility} \ge 0.4420$ on `DEV_REPLICATION`.
2. Push Macro AP to $\ge 0.5840$ while maintaining Macro Precision $\ge 0.6200$ (well above the 60% milestone).
3. Simultaneously achieve:
   - Substitution AP $\ge 0.4000$ and Substitution Precision $\ge 52.0\%$.
   - Omission AP $\ge 0.3800$ and Omission F1 $\ge 0.4000$.
   - Insertion F1 $\ge 0.3500$.
   - Calibration ECE $\le 0.0650$.
   - False Warning Rate $\le 1.35$ false alarms/utterance.

## 3. Predicted Evidence & Significance
If validated, Octad-Pareto establishes the supreme multi-paradigm system in the laboratory, proving that hierarchical phonetic routing synergizes constructively with multi-model Bayesian ensembling.
