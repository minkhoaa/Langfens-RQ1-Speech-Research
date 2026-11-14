# Hypothesis H41: Multi-Speaker Stress-Perturbation Transfer Invariance

## 1. Context & Motivation
Across 45 experiments, the laboratory has demonstrated that Dynamic Masked Context Training (DMCT-Net, EXP-44) and the 10-Model Decimal Composite (Decimal-Pareto, EXP-45) break the historic 60% Macro AP milestone on clean evaluations (0.6062 and 0.6056 respectively), with Decimal-Pareto driving False Warning Rate to an all-time low of 1.107 per utterance and Substitution AP to 0.4257.

However, in realistic real-world CAPT (Computer-Assisted Pronunciation Training) deployments, learner audio suffers from diverse acoustic degradations beyond stationary additive white noise:
1. **Pitch Jitter & Glottal Frequency Shifts ($\pm 15\%$ $F_0$ shift)**: Common among adolescent learners and varied vocal tract lengths.
2. **Speaking Rate Dynamic Compression / Expansion ($\pm 20\%$ tempo fluctuation)**: Mimicking hesitation, anxiety, and non-uniform syllable stretching.
3. **Severe Dynamic Range Clipping & Compression**: Simulating cheap smartphone electret microphones and classroom acoustic reverberation.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. **DMCT-Net (EXP-44)**, owing to its self-supervised dynamic sequence token masking (15% random feature dropout + auxiliary phoneme reconstruction), possesses an intrinsic invariant latent phonetic subspace that will retain $\ge 90\%$ of its uncorrupted Macro AP and $\ge 85\%$ of its Omission AP under combined acoustic jitter, tempo dilation ($\times 0.8$ to $\times 1.2$), and dynamic range compression.
2. **Decimal-Pareto (EXP-45)**, by fusing 10 structurally orthogonal feature representations (including VTLN formant warping, CMUDict bigrams, and waveform energy tilts), will provide superior false alarm suppression under severe perturbation, maintaining $FWR \le 1.60$ false warnings per utterance across all stress conditions.

## 3. Predicted Experimental Outcomes
1. **DMCT-Net Under Severe Jitter & Dynamic Stress:**
   - Retained Utility $\ge 0.4000$ (retaining $\ge 88\%$ of baseline 0.4535).
   - Macro AP $\ge 0.5500$.
2. **Decimal-Pareto Under Severe Jitter & Dynamic Stress:**
   - Retained Utility $\ge 0.4000$.
   - FWR $\le 1.50$ false warnings/utterance (demonstrating that multi-paradigm consensus eliminates spurious noise-induced false alarms).
3. **HTC-Net (EXP-38) Control Comparison:**
   - Will suffer greater degradation in vowel diagnosis under pitch shift due to vowel formant reliance without dynamic masking resilience.
