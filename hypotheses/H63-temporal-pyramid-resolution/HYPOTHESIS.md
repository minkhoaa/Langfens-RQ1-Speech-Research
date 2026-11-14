# Hypothesis H63: Multi-Resolution Temporal Acoustic Pyramid (MRTP-Net) for Multi-Scale Phonetic Diagnosis

## 1. Context & Motivation
Our laboratory has discovered that temporal windowing is the single most decisive factor governing omission and substitution sensitivity:
- Standard 10ms frame models (EXP-04 through EXP-27) plateaued around 0.35 Omission AP because transient plosive bursts were smeared across 160-sample windows.
- In EXP-48 (HRTD-Net), extracting 5ms sub-frame dynamics (80 samples) broke the 60% Macro F1 milestone (0.6050) and surged Omission AP to 0.4375.
- In EXP-55 (DSCA-Net), dual-stream cross-attention pushed standalone Weighted Utility to 0.4928.

However, a fixed 5ms window still enforces an artificial acoustic trade-off:
1. **Ultra-Short Scale (2.5ms = 40 samples):** Plosive burst release spikes (e.g. `/p/`, `/t/`, `/k/`) last only 2ms to 4ms. At 5ms, half of the window still contains surrounding closure or aspiration silence, diluting peak burst energy.
2. **Long Steady-State Scale (20ms = 320 samples):** Vowel formant frequencies ($F_1, F_2, F_3$) require longer observation windows to achieve sharp spectral resolution and differentiate subtle vowel substitutions (e.g. `/IH/` vs `/IY/`, `/EH/` vs `/AE/`).

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing a **4-Level Multi-Resolution Temporal Acoustic Pyramid (MRTP-Net)** directly from raw 16kHz audio waveforms:
   - **Level 0 (2.5ms / 40 samples):** Micro-burst crest factor $C_{2.5\text{ms}}$ and initial 40-sample onset gradient $\Delta E_{2.5\text{ms}}$ (capturing true transient plosive release spikes).
   - **Level 1 (5.0ms / 80 samples):** Sub-frame energy ratio and maximum instantaneous $dE/dt$.
   - **Level 2 (10.0ms / 160 samples):** Standard segmental energy, spectral tilt, and zero-crossing rate.
   - **Level 3 (20.0ms / 320 samples):** Macro-segmental vowel formant concentration ($E_{20\text{ms}} / E_{\text{utt}}$) and low-frequency resonance ratio ($< 1$ kHz vs $> 1$ kHz).
2. Augmenting the acoustic stream to **72 dimensions** coupled with bidirectional prompt cross-attention (DSCA-Net) will:
   - Provide simultaneous multi-scale resolution across ultra-fast stop releases and sustained vowel resonances.
   - Be the **FIRST STANDALONE SINGLE MODEL TO BREAK THE 0.5000 (50.00%) WEIGHTED UTILITY MILESTONE**!
   - Elevate standalone **Omission AP past 0.4500 (45.00%)**.
   - Elevate standalone **Substitution AP past 0.4450 (44.50%)**.
   - Elevate standalone **Macro AP past 0.6250 (62.50%)**.
   - Maintain overall ECE $\le 0.0150$ and FWR $\le 1.70$.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Weighted Utility:** $\ge 0.5000$ (First standalone single model to break 50%!).
2. **DEV_REPLICATION Macro AP:** $\ge 0.6250$.
3. **DEV_REPLICATION Omission AP:** $\ge 0.4500$.
4. **DEV_REPLICATION Substitution AP:** $\ge 0.4450$.
