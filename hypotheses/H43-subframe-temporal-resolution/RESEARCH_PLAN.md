# Research Plan: Experiment EXP-48 (High-Resolution Temporal Dynamics Network - HRTD-Net)

## 1. Objective & Architecture
Design, train, and benchmark **HRTD-Net** (High-Resolution Temporal Dynamics Network), augmenting the champion DMCT-Net with 4 micro-temporal acoustic burst features extracted at 5ms windowing:
- **Input Feature Vector:**
  - $D_{\text{target}} = 63 + 4 = 67$ dimensions (incorporating 5ms sub-frame energy ratio, max instantaneous $dE/dt$, micro-temporal crest factor, and high-frequency onset burst ratio).
  - $D_{\text{gap}} = 11 + 2 = 13$ dimensions (incorporating 5ms onset and offset transition gradients).
- **Core Network:**
  - Dynamic token masking with 15% random slot corruption during training.
  - Monotonic Local Gaussian Attention ($\sigma = 3.0$).
  - BiLSTM phonetic context modeling (96 hidden units, 2 layers).
  - Decoupled Hierarchical Vowel and Consonant diagnosis heads.
  - Multi-task articulatory supervision (manner, place, voicing).

## 2. Experimental Steps
1. Compute 5ms sub-frame features directly from 16kHz audio waveforms across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Normalize features using locked statistics from `train`.
3. Train HRTD-Net for 12 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), multi-task articulatory loss, auxiliary token reconstruction loss, and cost-sensitive focal loss ($\alpha_{\text{omis}}=2.0, \alpha_{\text{subs}}=1.5$).
4. Optimize decision thresholds on `DEV_CAL`.
5. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
6. Audit against H43 falsification criteria.
