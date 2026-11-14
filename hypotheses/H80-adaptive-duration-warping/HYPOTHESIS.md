# Hypothesis H80: Adaptive-Duration Warped Aligner Network (ADWA-Net)

## 1. Context & Motivation
In non-native English pronunciation, speaking rate varies widely across learners:
- Hesitant learners speak at $< 2.5$ syllables/sec with protracted vowels and pauses.
- Rushed learners speak at $> 5.5$ syllables/sec with compressed plosives and reduced vowels.

In existing architectures (HRTD-Net EXP-48, DSCA-Net EXP-55, IPTM-Net EXP-69), local monotonic attention employs a rigid Gaussian bandwidth $\sigma = 3.0$. This creates a fundamental trade-off:
- For prolonged vowels ($r_{\text{dur}} > 2.0$), $\sigma = 3.0$ is too narrow, failing to aggregate acoustic evidence across the entire prolonged segment.
- For abbreviated flaps or plosive release bursts ($r_{\text{dur}} < 0.35$), $\sigma = 3.0$ is too wide, smearing acoustic energy from adjacent vowels.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **Adaptive-Duration Warped Aligner Network (ADWA-Net)** by:
   - Introducing dynamic temporal attention span gating where the local monotonic attention Gaussian bandwidth $\sigma_k$ is modulated by the non-linear speaking-rate expansion ratio:
     $$\sigma_k = \sigma_0 \cdot \text{Softplus}\left(w_{\sigma} \cdot \log(r_{\text{dur}}(k) + 1e-4) + b_{\sigma}\right)$$
     adapting attention width dynamically ($1.5 \le \sigma_k \le 5.5$) to match phoneme duration.
   - Coupling the duration-warped attention module with 67-dimensional HRTD sub-frame acoustic features and 47-dimensional IPTM prompt features (with empirical interlanguage transfer priors $\pi(p)$).
   - Employing 13-dimensional triad decoupled gap gating.
2. ADWA-Net will achieve:
   - **Standalone Macro AP:** Breaking the 62.2% milestone for the first time in laboratory history ($\text{Macro AP} \ge 0.6220$).
   - **Standalone Macro F1:** Surpassing IPTM-Net to set a new standalone record ($\text{Macro F1} \ge 0.6210$).
   - **Standalone Weighted Utility:** $\ge 0.4950$ (49.50%) (Surpassing DSCA-Net 0.4928).
   - **Substitution AP:** $\ge 0.4500$ with Substitution Precision $\ge 53.50\%$.
   - **Omission Recall:** $\ge 55.00\%$ with Omission F1 $\ge 0.4650$.
   - **Overall ECE:** $\le 0.0120$ (1.20%) with False Warning Rate $\le 1.45$ per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6220$ (All-time standalone peak!).
2. **DEV_REPLICATION Macro F1:** $\ge 0.6210$ (All-time standalone peak!).
3. **DEV_REPLICATION Weighted Utility:** $\ge 0.4950$ (All-time standalone peak!).
4. **Zero-Leakage Compliance:** 100% verified via frozen evaluator and canonical target inputs.
