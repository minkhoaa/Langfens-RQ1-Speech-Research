# Falsification Criteria: Hypothesis H59

Hypothesis H59 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating Tredecim-Pareto on held-out `DEV_REPLICATION`:

1. **Ensemble Stagnation / Regression:**
   - Tredecim-Pareto Macro AP fails to exceed 0.6250 (inferior to Duodecim-Pareto 0.6341).
   - Weighted Utility fails to exceed 0.4850.
2. **Insertion Plateau:**
   - Insertion F1 falls below 0.3800 (failing to incorporate the gains of TDIS-Net 0.4321).
   - Insertion AP falls below 0.2900.
3. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 1.90 false alarms per utterance.
4. **Precision Loss:**
   - Substitution Precision falls below 51.00%.
   - Insertion Precision falls below 45.00%.
