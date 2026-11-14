# Falsification Criteria: Hypothesis H65

Hypothesis H65 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating Quindecim-Pareto on held-out `DEV_REPLICATION`:

1. **Ensemble Stagnation / Regression:**
   - Quindecim-Pareto Macro AP fails to exceed 0.6350 (inferior to Quattuordecim-Pareto 0.6377).
   - Weighted Utility falls below 0.4900.
2. **Prior Interference:**
   - Adding IPTM-Net degrades Omission AP below 0.4650 or Substitution AP below 0.4450.
3. **Insertion Degradation:**
   - Insertion AP falls below 0.3100.
   - Insertion Precision falls below 48.00%.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 1.85 false alarms per utterance.
