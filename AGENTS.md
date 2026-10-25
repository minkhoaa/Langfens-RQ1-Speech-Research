# AGENTS.md — LANGFENS
Read this file before repository work. Verify evidence before reporting PASS.

## Where work runs
- Local laptop: develop code, run unit tests, audit manifests, analyze metrics, and do lightweight inference/validation.
- Google Colab GPU via official Colab CLI: ALL training, fine-tuning, backward passes, optimizer updates, and GPU ablations.
- Validate each epoch on the Colab training runtime; local validation is allowed only for light, authorized inference.
- Never silently fall back to local training. If Colab, GPU, quota, or credentials are unavailable, stop and report the blocker.
- Confirm actual GPU, free disk, packages, pinned model revision, and license constraints before execution.

## Reusable data staging
- Google Drive holds persistent datasets, fold packages, checkpoints, and reports; Colab VM storage is temporary.
- Preserve the original L2-ARCTIC v5.0 ZIP unchanged on Drive; do not upload it again from the laptop every run.
- Build each immutable fold-specific package once using frozen manifest IDs; include only needed WAV, TextGrid, and metadata.
- For RQ1-TRN-02A stage train/validation only; never stage official Fold 1 test audio or test labels.
- Preserve expected internal archive paths and record IDs; validate speaker isolation and exact file counts.
- Record original ZIP hash, package SHA-256, manifest hash, package size, and build provenance.
- Each new Colab runtime copies the reusable package from Drive to local VM disk and verifies its SHA-256.
- Read audio from VM disk during training; do not stream thousands of small files from a Drive mount.

## Fold 1 research contract
- Frozen split manifest: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`.
- Fold 1: 2,399 train utterances, 300 validation utterances (NJS/RRBI), 900 test utterances SEALED.
- Two-task training: CTC + Binary MDD only; exclude diagnosis loss and diagnosis optimizer parameters.
- Keep pretrained Wav2Vec2 backbone frozen/eval in the primary protocol; verify no parameter drift.
- Start CTC evaluation with greedy decoding (beta=0); use validation only for model and threshold selection.
- Never use gold TextGrid intervals as model inference inputs or claim acoustic diagnosis readiness.

## Execution and provenance
- Use `configs/rq1_trn_02a_fold1.yaml` and `audit_scripts/train_fold1_ctc_binary.py` as the training entrypoints.
- Run real-audio Colab GPU smoke tests before full epochs (lengths, loss, gradients, AMP, VRAM, resume).
- Save complete checkpoints after each epoch; sync to Drive and verify before relying on them.
- A resume checkpoint needs weights, optimizer, scheduler, scaler, RNG, counters, config, Git SHA, and split/package hashes.
- Keep raw audio, fold archives, credentials, and model weights out of GitHub; commit only source, tests, configs, and safe reports.
- Use reviewable feature branches; do not auto-merge, publish checkpoints, open official test, or launch 4-fold/LOSO without authorization.
- Report measured results, failures, VRAM in MiB, and remaining blockers; never invent completed runs.
