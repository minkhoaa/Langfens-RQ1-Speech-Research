# Research Plan: Experiment EXP-83 (Cross-Linguistic Consonant-Cluster Sonority Paradigm Network - CCSP-Net)

## 1. Objective & Sonority Hierarchy Formulation
- **Sonority Hierarchy Values:**
  * Vowels: 7.0 (`AA`, `AE`, `AH`, `AO`, `AW`, `AY`, `EH`, `ER`, `EY`, `IH`, `IY`, `OW`, `OY`, `UH`, `UW`)
  * Glides: 6.0 (`W`, `Y`)
  * Liquids: 5.0 (`L`, `R`)
  * Nasals: 4.0 (`M`, `N`, `NG`)
  * Fricatives: 3.0 (`DH`, `F`, `S`, `SH`, `TH`, `V`, `Z`, `ZH`, `HH`)
  * Affricates: 2.0 (`CH`, `JH`)
  * Stops: 1.0 (`B`, `D`, `G`, `P`, `T`, `K`)
  * Silence / Boundaries: 0.0 (`SIL`, `<BOS>`, `<EOS>`)
- **18-Dimensional Augmented Gap Stream ($X_g \in \mathbb{R}^{(T+1) \times 18}$):**
  * 13-dim base HRTD gap representation
  * $\text{Son}(p_k) / 7.0$: left phone normalized sonority
  * $\text{Son}(p_{k+1}) / 7.0$: right phone normalized sonority
  * $\Delta \text{Son} / 7.0$: normalized sonority gradient
  * Sonority reversal flag: $\mathbb{I}[\Delta \text{Son} \le 0 \land \text{both are consonants}]$
  * Coda obstruction flag: $\mathbb{I}[p_k \in \text{Stop} \land p_{k+1} \in \text{Boundary/Pause}]$
- **Acoustic Stream ($X_a \in \mathbb{R}^{T \times 67}$):**
  * 67-dim base HRTD (5ms plosive dynamics, prosody, Charsiu SSL)
- **Prompt Stream ($X_p \in \mathbb{R}^{T \times 47}$):**
  * 42-dim canonical phone + articulatory representation
  * 2-dim CMUDict forward/backward transition priors
  * 3-dim empirical interlanguage error transfer priors $\pi(p)$ from TRAIN

## 2. Experimental Steps
1. Construct 18-dim sonority-augmented gap sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Train CCSP-Net on GPU (GTX 1650) for 10 epochs using AdamW, cosine annealing, and class-weighted multi-task loss.
3. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H78 falsification criteria.
6. Save artifacts, weights, metrics, and checksums under `experiments/EXP-83-CROSS-LINGUISTIC-CONSONANT-CLUSTER-PARADIGM/`.
