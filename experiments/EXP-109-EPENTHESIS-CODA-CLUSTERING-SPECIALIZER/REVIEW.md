# Scientific Review: Experiment EXP-109-EPENTHESIS-CODA-CLUSTERING-SPECIALIZER

## 1. Executive Summary & Critical Falsification
In Experiment EXP-109, we implemented the **Coda Release & Epenthesis Clustering Specializer Network (CREC-Net)**, extracting a 16-dimensional gap feature stream (13 base features + 3 post-coda dynamics: $r_{\text{post}}$, $\Delta C_{\text{spec}}$, $\text{ZCR}_{\text{ratio}}$) and training the gap head with binary focal loss ($\gamma=2.0, \alpha=0.75$).

## 2. Experimental Outcomes & Mathematical Failure Mechanism
1. **False Insertion Suppression:**
   - False insertions dropped to an extreme low of only **17 false insertions** on `DEV_REPLICATION` (and only 28 on search).
2. **Catastrophic Recall Collapse & Calibration Distortion:**
   - **Insertion Recall collapsed to 6.38% (0.0638)** (Insertion F1 dropped to **0.1026**).
   - `gap_ece` inflated from ~0.013 to **0.1167 (11.67%)**, driving overall ECE up to **0.0703 (7.03%)**.
   - Weighted Utility fell to **0.4372** (failing the $\ge 0.4900$ guardrail).
3. **Mathematical Cause of Failure:**
   - In insertion detection, positive events represent less than 1.0% of all inter-phone gaps (94 true insertions vs 10,069 true negative gaps).
   - In focal loss $\mathcal{L} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$, the gradient contributions from the 99% majority of negative instances—despite $(1 - p_t)^2$ modulation—overwhelmingly penalize any positive sigmoid logits.
   - This depressed all gap probabilities $\sigma(z_g) \to 0.001$, preventing true positive predictions at standard thresholds while destroying calibrated probability distributions.

## 3. Hypothesis H104 Evaluation
- **Decisively Falsified:** Focal loss on extreme 1:100 gap class imbalance causes gradient starvation on rare positive insertion events, collapsing recall and distorting ECE.
- **Architectural Directive for Phase 2:** Focal loss is strictly prohibited for rare insertion gaps. Standard positive-class weighted cross-entropy (`pos_weight = 7.0`) or class-stratified sampling must be maintained.
