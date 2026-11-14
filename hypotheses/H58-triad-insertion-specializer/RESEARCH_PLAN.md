# Research Plan: Experiment EXP-63 (Triad Decoupled Insertion Specializer - TDIS-Net)

## 1. Technical Formulation & Gap Routing Architecture
Design, implement, train, and benchmark **TDIS-Net**:
- **Acoustic & Prompt Backbone:**
  - DSCA-Net (EXP-55): 67-dim acoustic stream (including 5ms sub-frame dynamics) + 42-dim canonical prompt stream with bidirectional multi-head cross-attention.
- **Structural Gap Classification & Routing:**
  For each gap $g \in \{0, \dots, N\}$ between phones $p_{k}$ and $p_{k+1}$:
  - If $p_k$ and $p_{k+1}$ are both consonants (`is_cons(p_k) and is_cons(p_k+1)`): Route to `cluster_epenthesis_head`.
  - Else if $k = N-1$ (pre-coda/word-final boundary): Route to `coda_release_head`.
  - Else (vowel-consonant, vowel-vowel, or pause): Route to `pause_silence_head`.
- **Specialized Head Architectures:**
  - `cluster_epenthesis_head`: 2-layer MLP with focus on spectral zero-crossing rate and 5ms burst energy ratio ($13 + 256 \rightarrow 64 \rightarrow 32 \rightarrow 1$).
  - `coda_release_head`: 2-layer MLP with focus on trailing energy and aspiration duration ($13 + 256 \rightarrow 64 \rightarrow 32 \rightarrow 1$).
  - `pause_silence_head`: 2-layer MLP with conservative silence regularization ($13 + 256 \rightarrow 64 \rightarrow 32 \rightarrow 1$).
- **Combined Gating Output:**
  $$\hat{y}_{\text{gap}}(g) = \begin{cases}
  \text{Head}_{\text{cluster}}(g) & \text{if consonant cluster} \\
  \text{Head}_{\text{coda}}(g) & \text{if word-final coda} \\
  \text{Head}_{\text{pause}}(g) & \text{otherwise}
  \end{cases}$$

## 2. Experimental Steps
1. Ingest structural gap routing flags based on CMUDict consonant definitions across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
2. Warm-start core layers from `DSCA-Net` (EXP-55) and train TDIS-Net for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$) with specialized gap loss weighting ($\text{pos\_weight}=3.5$).
3. Calibrate operating thresholds on `DEV_CAL`.
4. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Audit against H58 falsification criteria.
