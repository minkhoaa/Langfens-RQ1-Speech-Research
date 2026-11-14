# Research Plan: Experiment EXP-69 (Interlanguage Phonotactic Error Prior Mapping - IPTM-Net)

## 1. Technical Formulation & Prior Computation
Design, implement, train, and benchmark **IPTM-Net**:
- **Empirical Interlanguage Prior Computation:**
  - Parse `train.jsonl` (1,800 utterances across 12 speakers and 6 L1s).
  - For each of the 39 canonical phonemes $p \in \text{Arpabet}$:
    $$N_{\text{total}}(p) = N_{\text{correct}}(p) + N_{\text{substitution}}(p) + N_{\text{omission}}(p)$$
    $$\pi(p) = \left[ \frac{N_c(p) + \alpha}{N_{\text{total}}(p) + 3\alpha}, \frac{N_s(p) + \alpha}{N_{\text{total}}(p) + 3\alpha}, \frac{N_o(p) + \alpha}{N_{\text{total}}(p) + 3\alpha} \right]$$
    with Laplace smoothing $\alpha = 1.0$.
  - Strictly zero access to `dev_cal`, `dev_search`, `dev_replication`, or `test_sealed` splits.
- **Stream P (Canonical Prompt Stream, 47 dims):**
  - 39 onehot canonical phone representation + 3 canonical articulatory priors (manner, place, voicing) + 2 CMUDict transition log-probabilities + 3 interlanguage transfer priors $\pi(p) = 47$ dimensions.
- **Stream A (Acoustic Stream, 67 dims):**
  - 51 base features + 4 tempo + 8 prosodic/phonotactic + 4 micro-dynamics 5ms.
- **Core Network:**
  - DSCA-Net backbone: Bidirectional Multi-Head Cross-Attention (MHCA, 4 heads) + Monotonic Local Gaussian Attention ($\sigma = 3.0$) + Decoupled Vowel/Consonant Heads.

## 2. Experimental Steps
1. Compute empirical interlanguage transfer prior matrix strictly on `train.jsonl`.
2. Construct 47-dimensional prompt stream features across `train`, `dev_cal`, `dev_search`, and `dev_replication`.
3. Train IPTM-Net on CPU (4 threads) for 10 epochs using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), multi-task articulatory loss, and focal cross-entropy.
4. Calibrate decision thresholds on `DEV_CAL`.
5. Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
6. Audit against H64 falsification criteria.
