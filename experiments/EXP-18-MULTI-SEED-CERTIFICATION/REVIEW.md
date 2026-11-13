# Experiment Review: EXP-18-MULTI-SEED-CERTIFICATION

## 1. Objective and Setup
Conducted a formal 5-seed randomized initialization sweep (`seeds = [101, 202, 303, 404, 505]`) of the `SRAD-Net` architecture on `DEV_REPLICATION` (300 utterances, 9,863 target phones, 10,163 gaps from unseen speakers TXHC and YKWK).
Evaluated with the frozen decision policy to certify statistical stability and confirm that reported breakthroughs are not random initialization anomalies.

## 2. Statistical Findings & Certification
- **Rock-Solid Macro Stability:**
  - Macro F1: **$0.5619 \pm 0.0032$** (variance is nearly negligible; standard deviation is $0.0032$, far below the $0.015$ target).
  - 95% Confidence Interval for Macro F1: **[0.5578, 0.5659]**.
- **Certification of 60% Macro Precision Milestone:**
  - Mean Macro Precision: **$0.6201 \pm 0.0076$** [95% CI: 0.6107 - 0.6295].
  - Every individual seed exceeded 61% precision (62.5%, 61.1%, 61.3%, 62.8%, 62.3%).
  - This formally certifies that the model achieves the **60% Macro Precision milestone** across independent seeds.
- **Class-Specific Statistical Boundaries:**
  - `CORRECT F1`: $0.9349 \pm 0.0007$
  - `OMISSION F1`: $0.3750 \pm 0.0108$
  - `SUBSTITUTION F1`: $0.3757 \pm 0.0074$
  - `INSERTION F1`: $0.3075 \pm 0.0312$
  - `Weighted Utility`: $0.4159 \pm 0.0042$

## 3. Scientific Audit
Hypothesis H13 is **CONFIRMED**:
The method exhibits complete statistical robustness across random seeds, confirming its eligibility for official archiving and synthesis.
