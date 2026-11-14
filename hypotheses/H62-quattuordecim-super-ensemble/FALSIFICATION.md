# Falsification Criteria: Hypothesis H62

Hypothesis H62 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating Quattuordecim-Pareto on held-out `DEV_REPLICATION`:

1. **Ensemble Stagnation / Regression:**
   - Quattuordecim-Pareto Macro AP fails to exceed 0.6300 (inferior to Tredecim-Pareto 0.6351).
   - Weighted Utility falls below 0.4900.
2. **Diffusion Conflict:**
   - Adding P-ALDS-Net causes Substitution AP to drop below 0.4450 or Omission AP to drop below 0.4600.
3. **Insertion Degradation:**
   - Insertion AP falls below 0.3000.
   - Insertion Precision falls below 48.00%.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 1.90 false alarms per utterance.
