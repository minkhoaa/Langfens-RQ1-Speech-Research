# Hypothesis H98: Differentiable Sequence-to-Graph Phonological Aligner (DSGA-Net)

## 1. Context & Motivation
As established in `reports/SCIENTIFIC_REVIEW_EXP100.md` (Section 6, Direction C), independent frame-level classification decoupled from inter-phone gap classification produces contradictory diagnostic feedback:
A model may predict an omission on an expected target phone while simultaneously predicting an insertion in the adjacent gap, violating the physical continuity of speech realization.
Phonetic realization is fundamentally an alignment path through a directed acyclic edit lattice (trellis graph).

## 2. Hypothesis Statement
We hypothesize that:
1. Formulating the **Differentiable Sequence-to-Graph Phonological Aligner (DSGA-Net)** where:
   - Acoustic frames and target phonemes are jointly decoded over a lattice with three continuous transition potentials:
     * $\phi_{\text{match}}(k)$: acoustic likelihood of canonical match or substitution.
     * $\phi_{\text{omis}}(k)$: duration-penalized skip transition without acoustic consumption.
     * $\phi_{\text{ins}}(k)$: inter-phoneme epenthesis branch potential.
   - Posterior probabilities are computed via differentiable forward-backward marginalization over the edit graph, enforcing lattice path conservation laws.
   - Coupled with ADAE-Net's orthogonal articulatory latent spaces and 47-dim canonical IPTM prompt stream.
2. DSGA-Net will achieve:
   - Standalone **Macro F1 $\ge 0.6300$ (63.00%)** on held-out `DEV_REPLICATION`.
   - Standalone **Weighted Utility $\ge 0.5050$ (50.50%)**.
   - **False Warning Rate (FWR) $\le 2.50$ false alarms per utterance** (eliminating contradictory slot-gap false alarms).
   - Expected Calibration Error (ECE) $\le 0.0220$ (2.20%).
3. Falsification will occur if lattice graph decoding fails to improve Macro F1 over ADAE-Net (0.6285) or degrades Weighted Utility below 0.4950.
