# Research Plan: Experiment EXP-68 (Multi-Resolution Temporal Pyramid Network - MRTP-Net)

## 1. Technical Formulation & Pyramid Feature Extraction
Design, implement, train, and benchmark **MRTP-Net**:
- **Multi-Resolution Temporal Acoustic Pyramid (72 dims):**
  - **Base Acoustic Features (67 dims):**
    - 51 base posteriors/durations + 4 tempo + 8 prosodic/phonotactic + 4 micro-dynamics 5ms.
  - **Pyramid Scale Features (5 new dims):**
    1. $C_{2.5\text{ms}}$: 2.5ms micro-burst crest factor ($\max |x[:40]| / (\text{RMS}_{40} + 1e-6)$).
    2. $\Delta E_{2.5\text{ms}}$: 2.5ms onset energy gradient ($\text{RMS}(x[:40]) - \text{RMS}(x[40:80])$).
    3. $E_{20\text{ms\_ratio}}$: 20ms macro-segmental energy ratio ($\text{RMS}(x[:320]) / (E_{\text{utt}} + 1e-6)$).
    4. $F_{\text{res\_ratio}}$: Low-frequency vowel resonance ratio (energy in low-pass filter relative to high-pass filter over 20ms window).
    5. $S_{\text{pyramid\_tilt}}$: 4-scale temporal energy slope ($\text{slope}(\text{RMS}_{2.5\text{ms}}, \text{RMS}_{5\text{ms}}, \text{RMS}_{10\text{ms}}, \text{RMS}_{20\text{ms}})$).
- **Stream P (Canonical Prompt Stream, 42 dims):**
  - 39 onehot canonical phone representation + 3 canonical articulatory priors (manner, place, voicing).
- **Core Network:**
  - DSCA-Net backbone: Bidirectional Multi-Head Cross-Attention (MHCA, 4 heads) + Monotonic Local Gaussian Attention ($\sigma = 3.0$) + Decoupled Vowel/Consonant Heads.

## 2. Experimental Steps
1. Ingest raw 16kHz audio waveforms across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Extract the 4-level temporal pyramid features and normalize using locked statistics from `train`.
3. Train MRTP-Net on CPU (4 threads) for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), multi-task articulatory loss, and focal cross-entropy.
4. Calibrate decision thresholds on `DEV_CAL`.
5. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
6. Audit against H63 falsification criteria.
