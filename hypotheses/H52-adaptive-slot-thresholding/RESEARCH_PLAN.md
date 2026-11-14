# Research Plan: Experiment EXP-57 (Adaptive Multi-Threshold Inference Engine - AMT-Engine)

## 1. Technical Formulation & Calibration Grid
Construct and tune the **Adaptive Multi-Threshold Inference Engine (AMT-Engine)**:
- **Conditioning Signals:**
  - $M_k \in \{0, \dots, 7\}$: Canonical Manner of Articulation index (Vowel, Stop, Fricative, Affricate, Nasal, Liquid, Glide).
  - $r_k$: Duration ratio of slot $k$ relative to utterance mean duration ($r_k = dur_k / \bar{dur}_{\text{utt}}$).
  - $H_k$: Posterior entropy of slot $k$ ($H_k = -\sum_c p_c \log p_c$).
  - $dur_g$: Duration frames of inter-phone gap $g$.
- **Offset Grid (Optimized on DEV_CAL):**
  - $\Delta_o(\text{Stop}) \in [-0.05, 0.00, +0.05]$: Stop consonant omission threshold offset.
  - $\Delta_o(\text{Vowel}) \in [0.00, +0.05, +0.10]$: Vowel omission threshold offset (conservative for vowels).
  - $\Delta_s(\text{Vowel}) \in [-0.05, 0.00, +0.05]$: Vowel substitution threshold offset.
  - $\alpha_o \in [0.00, 0.05, 0.10]$: Duration scaling factor for omission.
  - $\beta_s \in [0.00, 0.05, 0.10]$: Entropy scaling factor for substitution.
  - $\gamma_i \in [0.00, 0.02, 0.05]$: Gap duration scaling factor for insertion.

## 2. Experimental Steps
1. Load raw prediction records for:
   - **Undecim-Pareto (EXP-52)**
   - **DSCA-Net (EXP-55)**
   across `DEV_CAL`, `DEV_SEARCH`, and `DEV_REPLICATION`.
2. Extract manner, duration ratio, entropy, and gap duration features for each phone slot.
3. Optimize adaptive parameters on `DEV_CAL` to maximize Weighted Utility.
4. Freeze optimal parameters and evaluate on held-out `DEV_REPLICATION` and `DEV_SEARCH` under locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Audit against H52 falsification criteria.
