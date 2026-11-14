# Research Plan: Experiment EXP-89 (Progressive Confidence-Cascaded Viginti Router - PCR-Engine)

## 1. Objective & Architecture Design
- **Tier 1 Fast Acoustic Models ($M_{\text{fast}}$):**
  * BiLSTM Context Model (EXP-08)
  * DMCT-Net Dynamic Masking (EXP-44)
  * HRTD-Net 5ms Sub-frame Dynamics (EXP-48)
  * TDIS-Net Triad Gap Specializer (EXP-63, for gap screening)
  * Combined inference latency: $< 1.5$ ms per utterance on CPU.
- **Confidence Gating Function:**
  $$C_k = \max\left(p_{\text{fast}, \text{CORRECT}}(k), p_{\text{fast}, \text{ERROR}}(k)\right)$$
  If $C_k \ge \tau_{\text{exit}}$, assign fast prediction $\hat{y}_k = y_{\text{fast}, k}$.
- **Tier 2 Heavy Viginti Convergence Ensemble ($M_{\text{heavy}}$):**
  * Invoked only when $C_k < \tau_{\text{exit}}$.
  * Computes full 21-model convex Pareto mixture consensus.

## 2. Experimental Steps
1. Load constituent model prediction arrays from EXP-86 Viginti-Pareto across `DEV_CAL`, `DEV_SEARCH`, and `DEV_REPLICATION`.
2. Optimize confidence threshold $\tau_{\text{exit}} \in [0.75, 0.90]$ on `dev_cal.jsonl` to maximize speedup while constraining $\Delta \text{Utility} \ge -0.003$.
3. Evaluate progressive cascading on `dev_search` and `dev_replication` under locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
4. Measure exit rates, latency, utility retention, and error metrics.
5. Audit against H84 falsification criteria.
6. Save artifacts, metrics, and checksums under `experiments/EXP-89-PROGRESSIVE-CASCADE-VIGINTI-ROUTER/`.
