# Blockers and next actions

## Current gate: `RQ1-TRN-02A: PARTIAL`

Engineering contract, driver/config, tests, split/pretrained integrity, and bounded Colab T4 GPU feasibility are implemented/passed. Full Fold 1 training and validation have **not** been executed. Validation and acoustic-baseline artifacts are explicitly not generated.

1. **Interactive Drive authorization is pending.** `colab drivemount` requires a human TTY/authorization; this agent must not automate that interaction. The prior session expired after its bounded probe. The feasibility checkpoint was downloaded and synced/verified via local `rclone`.
2. The source branch at the implementation SHA is available remotely. Exact Fold 1 TRAIN+validation audio/TextGrid staging is already complete at `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-02A/authorized_data/fold1_train_val_only.zip` (2,699 listed utterances only; six test speaker archives and every non-manifest utterance excluded; verified Drive size and MD5). After mounting Drive, recreate a Colab T4 session, download/mount this archive, verify its SHA256 and membership, then train.
3. Resolve environment parity or explicitly approve a tested compatible Colab environment: Colab provided torch 2.11.0+cu130 / transformers 5.20.2; repository lock states torch 2.14.1. The feasibility run passed with the former, but this does not establish lock parity.
4. Run full training from freshly initialized heads and pinned backbone, persisting each epoch's checkpoint/history to Drive and verifying each remote object. Stop on OOM, invalid labels, leakage, checkpoint errors, numerical instability, or Colab quota limits.
5. Complete validation-only CTC/Binary metrics, threshold calibration, per-speaker reporting, and matched/mismatched/canonical-prior/prevalence baselines. Keep official test sealed.
6. Historical tau=0.20 erratum is documented. Exact breakdown of the 41 historical tau=0.50 correct cases between `DH→D` and `Z→S` cannot be reconstructed from aggregate artifacts; do not fabricate counts.
7. After all authorized runs, verify Drive presence and remote Git artifacts, then stop the Colab session explicitly.

No merge, test-set use, CV, acoustic-diagnosis training, or checkpoint publication occurred.
