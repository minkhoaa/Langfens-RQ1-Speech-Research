# Experiment Review: EXP-26-ENDLESS-SEARCH-CONTRASTIVE-ADAPTER (CAPD-Net)

## 1. Executive Summary
Experiment EXP-26 tested Hypothesis H21 by combining three independently successful mechanisms into a unified end-to-end model:
1. Acoustic bottleneck adapter ($55 \to 64 \to 55$ with residual normalization).
2. Monotonic local Gaussian windowed attention ($\sigma=3.0$).
3. Supervised contrastive prototype margin loss ($m=0.35$).

## 2. Quantitative Results
- **DEV_SEARCH (Exploration Cohort):**
  - Utility: **0.3575**
  - Substitution AP: **0.3675** (Precision = 47.42%, F1 = 0.3648)
  - Omission AP: **0.1555** (Recall = 38.28%, F1 = 0.2548)
  - Insertion AP: **0.2305** (F1 = 0.2437)
  - Correct AP: **0.9771** (F1 = 0.9403)
  - Macro AP: **0.5000**
- **DEV_REPLICATION (Independent Holdout):**
  - Utility: **0.4033**
  - Macro Precision: **0.5967** (near 60% milestone)
  - Macro AP: **0.5503**
  - Macro F1: **0.5494**
  - Omission F1: **0.3722** (Precision = 39.22%, Recall = 35.41%)
  - Substitution AP: **0.3692** (Precision = 48.94%)
  - Insertion F1: **0.2674**
  - False Warning Rate: **1.58** false warnings/utterance

## 3. Scientific Insights & Hypothesis Verdict
- **Hypothesis H21 Partially Supported:**
  * Joint training successfully stabilized without gradient conflict, achieving high Substitution AP (0.3675 search, 0.3692 rep) and high Macro Precision (0.5967).
  * However, Omission AP on `DEV_SEARCH` (0.1555) was lower than pure JSAC-Net (0.2251, EXP-24).
  * **Critical Scientific Insight:** The prototype contrastive loss clusters representations around discrete class centroids, which helps resolve categorical phonetic substitutions, but slightly penalizes the continuous, low-energy acoustic trail indicative of subtle phoneme omissions.
  * Therefore, while the contrastive margin enhances substitution diagnosis, omission detection benefits more from unconstrained continuous attention pooling.
