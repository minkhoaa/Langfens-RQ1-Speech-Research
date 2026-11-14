# Research Plan: Experiment EXP-82 (Interlanguage Phonotactic Latent Diffusion Network - IP-ALDS-Net)

## 1. Objective & Architecture Design
- **67-Dimensional Acoustic Stream ($X_a \in \mathbb{R}^{T \times 67}$):**
  * 51-dim Charsiu SSL + phone embedding + duration
  * 4-dim duration ratio and tempo dynamics
  * 8-dim prosodic energy dynamics and coda indicators
  * 4-dim 5ms sub-frame temporal micro-dynamics
- **47-Dimensional Conditioning Stream ($X_c \in \mathbb{R}^{T \times 47}$):**
  * 42-dim canonical phone one-hot + phonetic manner/place embedding
  * 2-dim CMUDict forward/backward phonotactic transition priors
  * 3-dim empirical interlanguage error transfer priors $\pi(p) = [P_c, P_s, P_o]$ calculated strictly on `train.jsonl`
- **Bidirectional Audio-Prompt Cross-Attention (MHCA, 4 heads, $d_{\text{model}}=64$):**
  * Local monotonic Gaussian attention ($\sigma=3.0$) preserving natural coarticulatory continuity
- **Score-Matching Latent Diffusion Module:**
  * 50 forward diffusion steps with linear beta schedule.
  * 4-step DDIM deterministic posterior sampling during inference.
  * Score denoiser conditioned on $[z_t, t_{\text{emb}}, p_{\text{emb}}]$.
- **13-Dimensional Triad Decoupled Gap Gating ($X_g \in \mathbb{R}^{(T+1) \times 13}$):**
  * Intra-cluster, coda, and pause transitions.

## 2. Experimental Steps
1. Compute empirical interlanguage error transfer priors $\pi(p)$ strictly on `train.jsonl`.
2. Construct 67-dim acoustic sequences and 47-dim prompt sequences across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
3. Train IP-ALDS-Net on GPU (GTX 1650) for 10 epochs using AdamW, cosine annealing, and multi-task score-matching loss + diagnostic classification loss.
4. Calibrate operating thresholds on `dev_cal` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Evaluate on `dev_search` and `dev_replication`.
6. Audit against H77 falsification criteria.
7. Save artifacts, weights, metrics, and checksums under `experiments/EXP-82-MULTI-ACCENT-INTERLANGUAGE-DIFFUSION-HYBRID/`.
