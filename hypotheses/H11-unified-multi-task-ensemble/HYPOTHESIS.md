# Hypothesis H11: Unified Tri-Model Super-Ensemble (HM-ADN + BiLSTM + Conformer)

## 1. Proposition
The three independently developed neural architectures in this campaign specialize in distinct, non-overlapping acoustic-phonological phenomena:
1. **HM-ADN (EXP-13):** Structural articulatory regularisation (Place, Manner, Voicing), producing the highest Omission AP (0.2010) and Insertion AP (0.2447).
2. **BiLSTM-CPN (EXP-08):** Sharp local temporal transition modeling, producing the highest Substitution Precision (50.30%).
3. **ConformerScorer (EXP-10):** Global multi-head prompt attention and depthwise convolution, producing the lowest Calibration Error (ECE: 0.0353).

A convex Bayesian Model Averaging mixture:
$$\mathbf{p}_{\text{target}} = w_1 \mathbf{p}_{\text{HM-ADN}} + w_2 \mathbf{p}_{\text{BiLSTM}} + (1 - w_1 - w_2) \mathbf{p}_{\text{Conformer}}$$
calibrated on `DEV_CAL` will achieve:
- Weighted Utility $> 0.380$ on `DEV_SEARCH` and $> 0.430$ on `DEV_REPLICATION`,
- Macro F1 $> 0.53$ on `DEV_SEARCH` and $> 0.56$ on `DEV_REPLICATION`,
- Further suppression of false warnings (FWR $< 1.50$),
establishing the strongest pronunciation assessment system developed in the campaign.

## 2. Experimental Plan
- Method ID: `M16-unified-super-ensemble`
- Run ID: `EXP-16-SUPER-ENSEMBLE-HMADN-HSAME`
- Calibrate mixture weights $(w_1, w_2, w_3)$ and decision thresholds on `DEV_CAL`.
- Evaluate on `DEV_SEARCH` and `DEV_REPLICATION`.
