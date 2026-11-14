# Hypothesis H21: Unified Contrastive Acoustic Adapter (CAPD-Net)

## 1. Context and Problem Statement
In previous iterations:
- EXP-21 (NNA-Adapter) demonstrated that a residual acoustic bottleneck adapter ($D_{\text{in}} \to 64 \to D_{\text{in}}$) enhances acoustic sensitivity, elevating Omission Recall to 44.98% on `DEV_SEARCH`.
- EXP-22 (PSC-Net) established that supervised cosine margin contrastive loss ($m=0.35$) explicitly penalizes phonetic confusions, yielding record standalone Substitution AP (0.3874) and precision (51.29%).
- EXP-24 (JSAC-Net) proved that Monotonic Local Attention ($\sigma=3.0$) smooths frame-to-phone alignment jitter, raising Omission AP to an all-time high of 0.2251.

However, these three mechanisms operated in separate models. EXP-26 tests whether fusing residual bottleneck adaptation, monotonic local attention, and phonological prototype contrastive regularization into a unified single model—**Contrastive Adapter Phonetic Diagnostic Network (CAPD-Net)**—achieves synergistic gains across both Substitution and Omission detection without ensemble overhead.

## 2. Formal Hypothesis
Jointly training a residual bottleneck adapter ($64$-dim bottleneck with residual connection) under a composite loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \lambda_{\text{contrast}} \mathcal{L}_{\text{margin}} + \lambda_{\text{manner}} \mathcal{L}_{\text{manner}} + \lambda_{\text{voicing}} \mathcal{L}_{\text{voicing}}$$
with $\lambda_{\text{contrast}} = 0.20$ and monotonic local Gaussian attention ($\sigma = 3.0$) will:
1. Surpass standalone baseline Substitution AP ($\ge 0.360$) and standalone Omission AP ($\ge 0.210$) on `DEV_SEARCH`.
2. Maintain high overall Utility ($\ge 0.380$) in a single, compact, streaming-ready model ($< 2.5 \text{ MB}$).

## 3. Predicted Evidence & Significance
If validated, CAPD-Net demonstrates that representation-level contrastive alignment directly amplifies local attention pooling, validating end-to-end representation refitting for low-resource L2 pronunciation assessment.
