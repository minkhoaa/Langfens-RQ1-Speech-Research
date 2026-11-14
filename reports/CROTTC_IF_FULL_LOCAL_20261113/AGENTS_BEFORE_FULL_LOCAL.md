# AGENTS.md — LANGFENS
Read before repository work. Verify evidence before reporting PASS.

## Active Research Goal
- `GOAL.md` is the sole active research objective, decision `THESIS-METHOD-DECISION-20261113-v1`: develop an improved method using CROTTC-IF as the main research baseline and L2-Arctic-plus/ALMs4Learning to guide feedback evaluation.
- Previous M0–M6 plans, `GOAL_before_*`/`GOAL_after_*` snapshots and completion markers are historical evidence only, not active work queues or proof that the new thesis goal is complete. The supersession archive is `reports/GOAL_DECISION_20261113/`.
- Preserve existing experiment artifacts and authorizations. Follow the current GOAL's declared scope; do not automatically resume old training campaigns or parallel method sweeps.

## Execution Boundaries
- Local laptop: Development, tests, audits, analysis, ALL training, fine-tuning, backward passes, optimizer updates, and GPU ablations. Local training authorized by the user on 2026-11-13; profile memory and record compute budgets before running.
- Google Colab and Kaggle: Not required; do not provision these runtimes for training.
- Local storage: Checkpoints and experiment logs persist locally. Existing Google Drive archives remain unchanged; Drive is optional for backup.
- GitHub: Version-controlled source code, configurations, tests, and lightweight audit artifacts only.
- Local execution is the default; Colab quota/auth must not block local work.

## Reusable Data Staging & Storage
- Keep original dataset archives (`l2arctic_release_v5.0.zip`) unchanged on Google Drive; do not re-upload from laptop.
- Reusable fold packages: Build once per fold from frozen manifests containing ONLY authorized train/validation WAV and TextGrid.
- Official test sealing: Never stage or extract official test audio/labels. Maintain strict speaker-disjoint isolation.
- Verify frozen manifests and SHA-256 of local audio/features before training.
- Read training audio/features from local disk; do not stream through a Google Drive mount.

## Local GPU & Operational Safety
- Check available VRAM, RAM, disk, and existing GPU jobs before training; acquire the shared GPU lock.
- Use explicit, bounded training budgets and retain failed/interrupted run provenance.
- Save resumable checkpoints atomically each epoch. Stop on nonfinite loss, data mismatch, or resource exhaustion; never silently skip failed batches.

## Training Contract & Provenance
- The Fold-1 entrypoints, counts, and two-head objective below apply to the legacy RQ1-TRN-02A experiment. New work follows the current GOAL and its registered protocol; do not mix populations. Local-training authorization from 2026-11-13 remains recorded; the old M2 milestone is historical.
- Split manifest: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- Fold 1 contract: 2,399 train, 300 val (NJS/RRBI), 900 test SEALED. Two-head objective: CTC + Binary MDD only.
- Backbone integrity: Pretrained `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`) kept frozen in eval.
- Start CTC decoding with greedy evaluation; use validation only for checkpoint and threshold selection.
- Never use gold TextGrid timestamps as model inference inputs or claim acoustic diagnosis readiness.
- Training entrypoints: `configs/rq1_trn_02a_fold1.yaml` and `audit_scripts/train_fold1_ctc_binary.py`.
- Atomic checkpointing: Save each epoch (`last.pt`, `best.pt`) with weights, optimizer, scheduler, scaler, RNG, and metadata.
- Persist each epoch checkpoint locally immediately and verify it can be loaded; Google Drive sync is optional.
- Keep datasets, archives, checkpoints, model weights, and credentials out of Git.
- Work on reviewable feature branches. Never force-push, merge into main, or start training without explicit user authorization.
