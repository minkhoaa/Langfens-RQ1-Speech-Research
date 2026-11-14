# Falsification Criteria: Hypothesis H43

Hypothesis H43 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating the sub-frame temporal refiner network (HRTD-Net) on `DEV_SEARCH` or `DEV_REPLICATION`:

1. **Failure to Improve Omission Sensitivity:**
   - `DEV_SEARCH` Omission AP fails to exceed 0.2800 (worse than baseline DMCT-Net 0.2965).
   - `DEV_REPLICATION` Omission AP falls below 0.4200.
2. **False Alarm Explosion:**
   - False Warning Rate (FWR) on `DEV_REPLICATION` exceeds 1.60 false alarms per utterance.
3. **Macro Quality Degradation:**
   - Overall Weighted Utility drops below 0.4450.
   - Macro AP drops below 0.6000.
4. **Vowel Diagnosis Interference:**
   - High-frequency burst features interfere with vowel formant discrimination, causing Substitution AP to drop below 0.3900.
