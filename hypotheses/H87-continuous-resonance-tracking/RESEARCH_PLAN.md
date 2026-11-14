# Research Plan: Experiment EXP-92 (Continuous Resonance Dynamics Study)

## 1. Methodology
- Compute 6-dim LPC formant resonance trajectories ($F_1, F_2, F_3, F_1/F_2, \Delta F_1, \Delta F_2$) using 16kHz audio waveforms across train, cal, search, and replication.
- Concatenate formant trajectories with 67-dim base HRTD features (total 73-dim acoustic stream).
- Train Formant-Resonance-Aware Network (FRTE-Net) with bidirectional cross-attention.
- Evaluate vowel-specific versus consonant-specific substitution accuracy on `dev_replication`.
- Audit against H87 falsification criteria.
