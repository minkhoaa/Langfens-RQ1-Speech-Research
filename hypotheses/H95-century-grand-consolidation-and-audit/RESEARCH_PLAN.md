# Research Plan: Experiment EXP-100 (Century Grand Consolidation & Mandatory Scientific Review)

## 1. Audit & Verification Procedures
1. **Cryptographic Sealed Holdout Audit:** Verify `test_sealed.jsonl` SHA-256 against baseline.
2. **Directory & Manifest Integrity Audit:** Check all 99 prior directories (`EXP-01` to `EXP-99`) for completeness and valid SHA-256 hashes.
3. **Raw Prediction Recomputation:**
   Directly recalculate from `PREDICTIONS/dev_replication_predictions.json` using `evaluation/evaluators/evaluator.py`:
   - Standalone: ADAE-Net (EXP-94), CICA-Net (EXP-95), IPTM-Net (EXP-69), DSCA-Net (EXP-55), HRTD-Net (EXP-48).
   - Ensembles: Centum-Pareto (EXP-99), Viginti-Pareto (EXP-86), Undeviginti-Pareto (EXP-84), Septendecim-Pareto (EXP-76), Duodecim-Pareto (EXP-58), Hexad-Pareto (EXP-36).
   - Deployment: PCR-Engine (EXP-89), INT8-ADAE-Net (EXP-96), EDGE-DUO (EXP-66).
4. **10,000-Sample Clustered Bootstrap & Significance Testing:**
   Quantify 95% confidence intervals and permutation $p$-values across models.
5. **Synthesis of `SCIENTIFIC_REVIEW_EXP100.md`:**
   Document verified evidence, denominator accounting, failure mechanisms, literature critique, and Phase 2 roadmap.
6. **State Freezing & Pause Directive:**
   Update `director/PROJECT_STATE.json` and `STATUS.md` with status `PAUSED_FOR_OPERATOR_REVIEW`.
