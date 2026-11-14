# Falsification Criteria: Hypothesis H76

Hypothesis H76 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating STCA-Net on held-out `DEV_REPLICATION`:

1. **Failure to Beat Standalone Macro F1 Benchmark:**
   - STCA-Net Macro F1 fails to exceed 0.6200 (inferior to IPTM-Net 0.6205).
2. **Failure to Beat Standalone Macro AP Benchmark:**
   - Macro AP falls below 0.6150 (inferior to IPTM-Net 0.6186 and HRTD-Net 0.6202).
3. **Contrastive Oversharpening / Degradation:**
   - Boundary contrastive loss degrades Substitution Precision below 50.00% or Omission Recall below 50.00%.
4. **False Warning Disruption:**
   - False Warning Rate (FWR) exceeds 2.20 false alarms per utterance.
