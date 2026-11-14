# Hypothesis H99: Soft-Gold Adjudication Network (SGAN-Net)

## 1. Context & Motivation
As mandated by `AGENTS.md` (Section 4 and Section 6), phonetic error assessment must account for annotator disagreement and phonetic ambiguity rather than collapsing ambiguity into artificial certainty.
In L2 speech, human annotators exhibit natural disagreement on borderline acoustic events (e.g. partial vowel reduction, weak aspiration). Training neural diagnostic heads with rigid one-hot cross-entropy forces the network to become overconfident on ambiguous boundaries, artificially inflating false warning rates on held-out speakers.

## 2. Hypothesis Statement
We hypothesize that:
1. Formulating the **Soft-Gold Adjudication Network (SGAN-Net)** where:
   - Target training labels are softened via continuous Dirichlet blending with empirical interlanguage error priors:
     $$y_{\text{soft}}(k) = (1 - \epsilon) y_{\text{one-hot}}(k) + \epsilon \cdot \pi(p_k)$$
     with label smoothing factor $\epsilon = 0.10$.
   - The diagnostic heads are optimized via Kullback-Leibler (KL) divergence loss against $y_{\text{soft}}$, preventing overconfident gradient penalties on acoustically ambiguous phonemes.
   - Coupled with ADAE-Net's orthogonal articulatory latent spaces and 47-dim canonical IPTM prompt stream.
2. SGAN-Net will achieve:
   - Standalone **Macro Precision $\ge 61.50\%$ (61.50%)** on held-out `DEV_REPLICATION`.
   - **False Warning Rate (FWR) $\le 2.60$ false alarms per utterance** (a significant reduction from ~3.04).
   - Expected Calibration Error (ECE) $\le 0.0190$ (1.90%).
   - Standalone **Weighted Utility $\ge 0.4980$ (49.80%)** and **Macro F1 $\ge 0.6180$ (61.80%)**.
3. Falsification will occur if soft-gold smoothing excessively blunts error sensitivity, dropping Macro Recall below 60.00% or Weighted Utility below 0.4850.
