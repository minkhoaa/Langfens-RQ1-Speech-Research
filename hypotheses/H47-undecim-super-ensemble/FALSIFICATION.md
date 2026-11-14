# Falsification Criteria: Hypothesis H47

Hypothesis H47 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating Undecim-Pareto on held-out `DEV_REPLICATION`:

1. **Ensemble Stagnation / Regression:**
   - Undecim-Pareto Macro AP fails to exceed 0.6150 (inferior to standalone HRTD-Net 0.6202).
   - Weighted Utility falls below 0.4700.
2. **False Alarm Explosion:**
   - False Warning Rate (FWR) exceeds 1.50 false alarms per utterance (failing to maintain the false alarm suppression advantage of Decimal-Pareto).
3. **Precision Regression:**
   - Substitution Precision falls below 52.00%.
   - Macro Precision falls below 60.00%.
4. **Calibration Degradation:**
   - Overall ECE exceeds 0.0800.
