# AGENTS.md — LANGFENS
Read before repository work. Verify evidence before reporting PASS.

## Active Research Goal
- `GOAL.md` is the sole active research objective, decision `THESIS-METHOD-DECISION-20261113-v1`: develop an improved method using CROTTC-IF as the main research baseline and L2-Arctic-plus/ALMs4Learning to guide feedback evaluation.
- Previous M0–M6 plans, `GOAL_before_*`/`GOAL_after_*` snapshots and completion markers are historical evidence only, not active work queues or proof that the new thesis goal is complete. The supersession archive is `reports/GOAL_DECISION_20261113/`.
- Preserve existing experiment artifacts and authorizations. Follow the current GOAL's declared scope; do not automatically resume old training campaigns or parallel method sweeps.
- User authorization on2026-11-14 adds `SIDE_GOAL.md`, decision `SM13-BACKBONE-SIDE-20261114-v1`: a bounded SM13 frozen-backbone comparison may proceed alongside current Kaggle training. GOAL remains the primary thesis objective. Local work is CPU development/tests/audits/package preparation only; separate Kaggle resource/admission gates apply to side extraction/fitting, with no interference/reset of the main runtime. No automatic broad sweeps or local production training; retain test sealing and prevent full-TRAIN-adapted backbone leakage into claimed independent TRAIN OOF.
- Latest explicit user instruction2026-11-14 15:55: **SM13 runs local; do not break current full-paper training.** Local S0 frozen-head50epoch run is authorized and active under shared GPU lock; full CROTTC remains on Kaggle unchanged. Reuse hash-verified original TRAIN feature cache, preserve all checkpoints and data sealing. Remote SM13 remains quota-blocked QUEUED (not falsely called cancelled); local bounded window precedes quota reset, with a fail-closed remote-admission guard. Evidence: `reports/SM13_EPOCH50_20261114/LOCAL_LAUNCH_RECEIPT.json`.
- User amendment on2026-11-14 (`SM13-EPOCH50-20261114-v1`): run a bounded SM13 frozen-head epoch-extension experiment, up to50epochs/head, Kaggle first; **local SM13 only** is authorized if Kaggle resource admission fails. Register the failure/resource receipt, shared local GPU lock, budget and resumable state before fallback. This does not authorize local CROTTC restart, backbone fine-tuning, test/DEV_CAL access, or broad sweeps. See SIDE_GOAL.md.
- Latest user decision on 2026-11-14: move full CROTTC-IF training to **Kaggle**, decision `CROTTC-IF-FULL-KAGGLE-20261114-v1` in GOAL.md. This supersedes the previous local-only execution direction. Preserve full architecture, registered stages and resumable state; do not substitute a compact pilot. Existing data-sealing boundaries remain in force.
- Local training ended cleanly for session handoff at update 124, internal epoch 2/cursor320,7/32 pending microbatches. Do not automatically resume local. Read HANDOFF.md and `reports/CROTTC_IF_KAGGLE_20261114/` before Kaggle preparation/migration. Earlier local authorization and artifacts remain valid historical evidence.

## Execution Boundaries
- Local laptop: Development, tests, audits, analysis, package preparation, Kaggle CLI control, and preservation of historical checkpoints/logs. Full training continues on Kaggle under the new decision; do not automatically restart local training.
- Kaggle: User-authorized destination for full training/fine-tuning/backward/optimizer stages. CLI 2.2.5 authentication was verified read-only; profile actual runtime/accelerator/RAM/disk/quota, register bounded windows and a migration receipt before continuing checkpoint 124. Do not assume the local driver is portable without adaptation.
- Google Colab: Not selected; do not provision as an automatic fallback.
- Storage: Preserve local historical checkpoints/logs and original Google Drive archives. Persist and retrieve/verify Kaggle checkpoint outputs across runtime windows; Drive remains optional for backup.
- GitHub: Version-controlled source code, configurations, tests, and lightweight audit artifacts only.
- Kaggle is the default for the next full training campaign. Local preparation can proceed independently of remote runtime availability. Do not ask again for the platform change already explicitly authorized by the user.

## Reusable Data Staging & Storage
- Keep original dataset archives (`l2arctic_release_v5.0.zip`) unchanged on Google Drive; do not re-upload from laptop.
- Reusable fold packages: Build once per fold from frozen manifests containing ONLY authorized train/validation WAV and TextGrid.
- Official test sealing: Never stage or extract official test audio/labels. Maintain strict speaker-disjoint isolation.
- Verify frozen manifests and SHA-256 of local audio/features before training.
- Read training audio/features from the selected runtime's disk; do not stream through a Google Drive mount. Kaggle packages must be private and contain only manifest-allowlisted TRAIN/DEV_SEARCH assets; never upload the whole data tree, raw archives, sealed test or credentials.

## Local GPU & Operational Safety
- Check available VRAM, RAM, disk, and existing GPU jobs before training; acquire the shared GPU lock.
- Use explicit, bounded training budgets and retain failed/interrupted run provenance.
- Save resumable checkpoints atomically each epoch. Stop on nonfinite loss, data mismatch, or resource exhaustion; never silently skip failed batches.
- On Kaggle, preserve weights/optimizer/scheduler/scaler/RNG/cursors and pending gradients, record kernel/dataset versions and runtime identity, and verify exported checkpoints before the next window. Any path/runtime/placement change needs explicit provenance; never rewrite historical receipts or silently discard partial gradients.

## Training Contract & Provenance
- The Fold-1 entrypoints, counts, and two-head objective below apply to the legacy RQ1-TRN-02A experiment. New work follows the current GOAL and its registered protocol; do not mix populations. Local-training authorization from 2026-11-13 is historical; the latest platform decision is Kaggle2026-11-14, and the old M2 milestone is historical.
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
