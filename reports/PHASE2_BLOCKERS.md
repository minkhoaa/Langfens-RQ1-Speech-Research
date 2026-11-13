# LANGFENS PHASE 2: BLOCKERS STATUS

| Block ID | Component | Description | Resolution / Status |
|---|---|---|---|
| BLK-P2-01 | Artifact Recovery | Verify Phase 1 teacher checkpoints and prediction ledgers | RESOLVED. Checkpoints and ledgers verified with bit-exact SHA-256 hashes. |
| BLK-P2-02 | DEV_CAL Inference | Ensure teacher predictions exist on calibration split without leakage | RESOLVED. Forward passes executed on CUDA for DEV_CAL, predictions frozen into `experiments/calibration/dev_cal_teachers_predictions.npz`. |
| BLK-P2-03 | Class Mutually Exclusive Simplex | Arbitrate between S (Substitution) and O (Omission) on target positions | RESOLVED. Decision arbitration matrix established. Only 153 target conflicts observed out of 19,755 targets. |
| BLK-P2-04 | Pedagogical FWR Constraint | Maintain false warnings <= 119 across 599 sentences (FWR <= 0.20) | RESOLVED. Operating points `F5-RISK-B020` (53 FPs), `F5-RISK-B030` (110 FPs), and `F3-CALIB-FAMILY` (118 FPs) certified safe. |
| BLK-P2-05 | Distillation Protocol | Establish unified multi-task loss and ablation protocol for student | RESOLVED. Comprehensive 6-component loss designed and verified in `reports/CPL03_KD_EXPERIMENT_DESIGN.md`. |
