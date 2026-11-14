# Research Plan: Experiment EXP-103 (Differentiable Sequence-to-Graph Phonological Aligner - DSGA-Net)

## 1. Graph Lattice Formulation
- Let $k \in \{0, \dots, N-1\}$ be canonical target phone slots, and $g \in \{0, \dots, N\}$ be inter-phone gap slots.
- Joint Transition Potentials:
  * For each target phone slot $k$:
    The network outputs unnormalized log-potentials for Match $[\phi_{\text{corr}}(k), \phi_{\text{subs}}(k)]$ and Skip $[\phi_{\text{omis}}(k)]$.
  * For each gap slot $g$:
    The network outputs Insertion potential $\phi_{\text{ins}}(g)$.
- Continuous Lattice Coupling:
  To eliminate contradictory false alarms (predicting both omission at slot $k$ and insertion at gap $k$):
  We introduce an anti-concurrence penalty in the graph loss:
  $$\mathcal{L}_{\text{lattice}} = \sum_{k=0}^{N-1} \max\left(0, P_{\text{omis}}(k) + P_{\text{ins}}(k) - 1.0\right)^2$$
  ensuring that a slot cannot simultaneously vanish and sprout an adjacent epenthetic phone unless supported by extreme acoustic energy.
- Encoder:
  72-dimensional temporal pyramid features + ADAE-Net orthogonal latent spaces (manner, place, voicing, timbre) + 47-dim canonical IPTM prompt stream.

## 2. Experimental Execution
1. Implement `methods/M103-graph-aligner-trainer/run.py`.
2. Train on GPU for 10 epochs using AdamW and cosine annealing.
3. Calibrate thresholds on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H98 falsification criteria.
6. Save all artifacts in `experiments/EXP-103-DIFFERENTIABLE-GRAPH-PHONOLOGICAL-ALIGNER/`.
