# Falsification Criteria for Hypothesis H19

1. **Attention Superiority Falsification:**
   If JSA-CTC fails to achieve higher Substitution F1 or higher Macro AP than EXP-23 on `DEV_REPLICATION`, H19 is falsified, indicating that soft cross-attention introduces attention leakage across phone boundaries.

2. **Alignment Entropy Gate:**
   If cross-attention entropy $-\sum_t \alpha_{i, t} \log \alpha_{i, t}$ explodes (indicating unaligned, uniform attention), the attention mechanism is rejected as failing to preserve temporal monotonicity.
