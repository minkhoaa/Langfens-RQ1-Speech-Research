# Research Plan: Experiment EXP-110 (Phase 2 Master Consolidation Audit)

## 1. Audit Procedures
1. **Sealed Holdout Hash Audit:** Compute SHA-256 of `data/derived/l2arctic/manifests/test_sealed.jsonl` and compare with initial certificate.
2. **Directory & Manifest Audit:** Inspect all 109 experiment directories (`EXP-01` to `EXP-109`) for mandatory files.
3. **Independent Raw Prediction Recomputation:**
   Directly evaluate `PREDICTIONS/dev_replication_predictions.json` using `evaluation/evaluators/evaluator.py` across:
   - EXP-107 (PCR-25 Engine)
   - EXP-106 (Viginti-Quintum-Pareto)
   - EXP-104 (SGAN-Net)
   - EXP-103 (DSGA-Net)
   - EXP-102 (PEAR-Net)
   - EXP-101 (VTI-Net)
   - EXP-94 (ADAE-Net)
   - EXP-86 (Viginti-Pareto)
4. **1,000-Sample Clustered Bootstrap Analysis:**
   Compute 95% confidence intervals on `DEV_REPLICATION` for PCR-25 Engine and Viginti-Quintum-Pareto.
5. **Compile & Seal Receipt:**
   Generate and seal `experiments/EXP-110-PHASE-2-CONSOLIDATION-AND-PROGRESS-SYNTHESIS/PHASE_2_AUDIT_RECEIPT.json`.
