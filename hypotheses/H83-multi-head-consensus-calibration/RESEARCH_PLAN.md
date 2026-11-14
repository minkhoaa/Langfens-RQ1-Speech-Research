# Research Plan: Experiment EXP-88 (Multi-Head Latent Consensus Calibration Engine - LCC-Engine)

## 1. Objective & Mathematical Architecture
- **Epistemic Disagreement Metrics:**
  * Target Shannon Entropy:
    $$H_k = - \sum_{c=1}^3 \bar{p}_{k, c} \log (\bar{p}_{k, c} + 1e-9)$$
  * Gap Shannon Entropy:
    $$H_j = - \left[\bar{g}_j \log(\bar{g}_j + 1e-9) + (1 - \bar{g}_j) \log(1 - \bar{g}_j + 1e-9)\right]$$
- **Entropy-Gated Dynamic Threshold Modulation:**
  $$\theta_s(k) = \theta_{s, \text{base}} + \lambda_s \cdot \left(\frac{H_k - \bar{H}}{\sigma_H}\right)$$
  $$\theta_o(k) = \theta_{o, \text{base}} + \lambda_o \cdot \left(\frac{H_k - \bar{H}}{\sigma_H}\right)$$
  $$\theta_i(j) = \theta_{i, \text{base}} + \lambda_i \cdot \left(\frac{H_j - \bar{H}_g}{\sigma_{H_g}}\right)$$
  where $\lambda_s, \lambda_o, \lambda_i \in [0.0, 0.08]$ are optimized on `dev_cal.jsonl` using a grid search over Weighted Utility.

## 2. Experimental Steps
1. Load cached predictions from EXP-86 Viginti-Pareto across `DEV_CAL`, `DEV_SEARCH`, and `DEV_REPLICATION`.
2. Compute slot-level epistemic entropy arrays across all three splits.
3. On `DEV_CAL`, optimize entropy scaling coefficients $\lambda_s, \lambda_o, \lambda_i$ to maximize Weighted Utility subject to $\text{FWR} \le 1.50$.
4. Apply the calibrated dynamic thresholds to `DEV_SEARCH` and `DEV_REPLICATION` under locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Audit against H83 falsification criteria.
6. Save predictions, metrics, calibration coefficients, and checksums under `experiments/EXP-88-MULTI-HEAD-LATENT-CONSENSUS-CALIBRATOR/`.
