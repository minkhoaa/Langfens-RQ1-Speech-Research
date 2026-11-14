# Hypothesis H07: Hybrid Sequential-Attentive Model Ensemble (H-SAME)

## 1. Proposition
Bidirectional LSTMs and Multi-Head Conformers exhibit orthogonal error distributions:
- BiLSTM (EXP-08) possesses a strong inductive bias for local temporal transitions, yielding superior precision and AP on phone-level substitutions and omissions (OMISSION AP: 0.1725, SUBSTITUTION Precision: 0.4462).
- Multi-Head Conformer (EXP-10) leverages global self-attention and depthwise convolution, yielding superior precision and calibration on inter-phonemic gap insertions (INSERTION Precision: 0.2893, F1: 0.2846, ECE: 0.0353).

By forming a calibrated Bayesian mixture of the posterior distributions:
$$\mathbf{p}_{\text{target}} = w \cdot \mathbf{p}_{\text{BiLSTM}} + (1-w) \cdot \mathbf{p}_{\text{Conformer}}$$
$$\mathbf{p}_{\text{gap}} = v \cdot \mathbf{p}_{\text{BiLSTM}} + (1-v) \cdot \mathbf{p}_{\text{Conformer}}$$
with mixture weights $(w, v)$ and decision thresholds fitted on `DEV_CAL`, the ensemble will:
1. Retain the high substitution and omission precision of the BiLSTM,
2. Retain the superior insertion precision and low ECE of the Conformer,
3. Surpass all single-model weighted utilities, breaking past **0.370** on `DEV_SEARCH`.

## 2. Experimental Plan
- Method ID: `M11-hybrid-context-ensemble`
- Run ID: `EXP-11-ENSEMBLE-FUSION`
- Calibrate mixture weights and decision thresholds on `DEV_CAL`.
- Primary locked evaluation on `DEV_SEARCH`.
