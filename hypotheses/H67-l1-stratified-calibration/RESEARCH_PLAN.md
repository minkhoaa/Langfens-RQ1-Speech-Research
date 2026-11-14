# Research Plan: Experiment EXP-72 (L1-Stratified Discriminative Calibration - LDC-Engine)

## 1. Objective & Design
Calibrate the posterior probability distributions and operating thresholds of the 15-model champion ensemble (EXP-70 Quindecim-Pareto) separately for each of the 6 L1 cohorts:
- Arabic (`ABA`, `SKA`, `YBAA`, `ZHAA`)
- Chinese (`BWC`, `LCC`, `NCC`, `TXHC`)
- Hindi (`ASI`, `RRBI`, `SVBI`, `TNI`)
- Korean (`HJK`, `HKK`, `YDCK`, `YKWK`)
- Spanish (`EBVS`, `ERMS`, `MBOH`, `NJS`)
- Vietnamese (`HQTV`, `PNV`, `THV`, `TLV`)

## 2. Experimental Steps
1. Load cached predictions from EXP-70 Quindecim-Pareto across `DEV_CAL`, `DEV_SEARCH`, and `DEV_REPLICATION`.
2. Extract speaker L1 metadata strictly from manifest records (already part of learner registration metadata).
3. On `DEV_CAL`, optimize dialect-specific temperature scaling $T_{\ell} \in [0.8, 1.4]$ and threshold shifts $\Delta \theta_{\ell} = [\Delta \theta_s, \Delta \theta_o, \Delta \theta_i]$ for each L1 cohort to maximize F1 while penalizing false warnings.
4. Apply calibrated policies to held-out `DEV_SEARCH` and `DEV_REPLICATION` using locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Audit against H67 falsification criteria.
6. Save predictions, metrics, calibration parameters, and checksums under `experiments/EXP-72-L1-STRATIFIED-DISCRIMINATIVE-CALIBRATOR/`.
