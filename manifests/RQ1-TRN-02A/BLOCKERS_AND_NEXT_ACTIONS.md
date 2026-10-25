# Blockers and Next Actions

## Acceptance Gate: `RQ1-TRN-02A-COLAB-RESTART: PASS`

Full Fold 1 two-head (CTC + Binary MDD) training and validation have been cleanly restarted, executed, and completed on a fresh Google Colab Tesla T4 GPU runtime with persistent synchronization to Google Drive.

---

## 1. Resolved Items
1. **Colab-First Execution**: Provisioned a clean Google Colab session (`rq1-trn02a-restart-01`), mounted Drive interactively, copied data to VM disk, and completed all backward passes and optimizer updates on the GPU.
2. **Immutable Fold 1 Dataset Package Reused**: Existing Google Drive archive `fold1_train_val_only.zip` (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`) was verified, copied to VM disk, and verified with zero test speaker contamination (2,399 train, 300 val, 0 test).
3. **Training Driver Upgrades**:
   - Added full Levenshtein CTC PER with substitutions (17.19%), deletions (4.40%), and insertions (2.04%).
   - Added Binary MDD Average Precision (`0.316062` vs prevalence `0.116211`) and full threshold sweep calibration.
   - Fixed early stopping patience preservation and scheduler state persistence across restarts.
   - Guaranteed atomic checkpoint copy and verification on Google Drive after every completed epoch.
4. **Full Production Fold 1 Execution**: Completed 11 epochs (13,200 micro-steps, 6,600 optimizer updates, 0 AMP skips). Predeclared early stopping correctly triggered at Epoch 11 after 5 epochs without validation AP improvement beyond Epoch 6 peak.
5. **Acoustic vs Baseline Comparison**: Validated that the acoustic model outperforms both empirical error prevalence (2.72x gain) and prompt-only prior (+1.30 AP points, +48.65% precision at standard threshold $\tau = 0.50$).
6. **Official Test Sealing**: Official test partition (900 utterances, 6 test speakers) was completely excluded from training, validation, checkpoint selection, and threshold calibration.
7. **Regression Tests**: All 88 tests in `tests/` pass cleanly.

---

## 2. Resource Hygiene and Teardown
- Google Drive artifacts (`best.pt`, `last.pt`, `history.json`, `EVALUATION_REPORT.json`, `BASELINE_COMPARISON.json`) verified.
- The Colab session `rq1-trn02a-restart-01` can now be safely released (`colab stop -s rq1-trn02a-restart-01`) to preserve Google Colab compute units.

---

## 3. Strict Prohibitions Upheld
- No merge into `main`.
- No force-pushing.
- No training or backwards passes on local laptop CPU/GPU.
- No access to or inference on the official Fold 1 test partition.
- No training of the substitution diagnosis head.
- No model weights committed to GitHub.
