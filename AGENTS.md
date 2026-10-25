# AGENTS.md — LANGFENS
Read before repository work. Verify evidence before reporting PASS.

## Execution Boundaries
- Local laptop: Code development, unit/regression tests, manifest audits, metric analysis, and lightweight validation.
- Google Colab GPU (official Colab CLI): ALL training, fine-tuning, backward passes, optimizer updates, and GPU ablations.
- Google Drive: Persistent storage for original datasets, reusable fold packages, checkpoints, and experiment logs.
- GitHub: Version-controlled source code, configurations, tests, and lightweight audit artifacts only.
- Strict fallback prohibition: Never fall back to local GPU/CPU training when Colab is unavailable. Stop and report blockers.

## Reusable Data Staging & Storage
- Keep original dataset archives (`l2arctic_release_v5.0.zip`) unchanged on Google Drive; do not re-upload from laptop.
- Reusable fold packages: Build once per fold from frozen manifests containing ONLY authorized train/validation WAV and TextGrid.
- Official test sealing: Never stage or extract official test audio/labels. Maintain strict speaker-disjoint isolation.
- Colab VM storage (`/content`) is ephemeral: Copy reusable fold package from Drive to VM disk and verify SHA-256 before training.
- Read audio directly from local VM disk during training; never stream training audio across a Google Drive mount.

## Colab CLI & Operational Safety
- Provision GPU runtimes via official Colab CLI (`colab new -s <name> --gpu T4` or `colab run`).
- Colab non-interactive rule: Never execute interactive commands (`colab drivemount`, `colab auth`, `colab repl`) in automation.
- Monitor compute units via `colab usage`. If GPU quota is exhausted or auth is missing, STOP and report.
- Resource hygiene: Always release idle runtimes promptly (`colab stop -s <name>`) to preserve compute units.

## Training Contract & Provenance
- Split manifest: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- Fold 1 contract: 2,399 train, 300 val (NJS/RRBI), 900 test SEALED. Two-head objective: CTC + Binary MDD only.
- Backbone integrity: Pretrained `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`) kept frozen in eval.
- Start CTC decoding with greedy evaluation; use validation only for checkpoint and threshold selection.
- Never use gold TextGrid timestamps as model inference inputs or claim acoustic diagnosis readiness.
- Training entrypoints: `configs/rq1_trn_02a_fold1.yaml` and `audit_scripts/train_fold1_ctc_binary.py`.
- Atomic checkpointing: Save each epoch (`last.pt`, `best.pt`) with weights, optimizer, scheduler, scaler, RNG, and metadata.
- Sync each epoch checkpoint to Google Drive immediately; verify remote presence.
- Keep datasets, archives, checkpoints, model weights, and credentials out of Git.
- Work on reviewable feature branches. Never force-push, merge into main, or start training without explicit user authorization.
