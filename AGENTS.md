# AGENTS.md — LANGFENS
Read before repository work. Verify evidence before reporting PASS.

## Execution Boundaries
- Local laptop: Code development, unit/regression tests, manifest audits, metric analysis, and lightweight validation.
- Remote GPU Platforms: ALL training, fine-tuning, backward passes, optimizer updates, and GPU ablations must run on authorized remote GPU hardware with verified compute allocation:
  - Google Colab GPU (official Colab CLI, Tesla T4, CUDA FP16 AMP)
  - Kaggle GPU (official Kaggle CLI, NvidiaTeslaT4 [T4 × 2], private execution, CUDA FP16 AMP)
- Google Drive & Approved Private Remote Storage: Persistent storage for original datasets, reusable fold packages, checkpoints, and experiment logs. Ephemeral VM/runtime storage (/content, /kaggle/working) must not be treated as durable.
- GitHub: Version-controlled source code, configurations, tests, and lightweight audit artifacts only.
- Strict fallback prohibition: Never fall back to local GPU/CPU training when remote GPU platforms are unavailable. Local training remains strictly prohibited. Stop and report blockers.

## Reusable Data Staging & Storage
- Keep original dataset archives (`l2arctic_release_v5.0.zip`) unchanged on Google Drive; do not re-upload from laptop.
- Reusable fold packages: Build once per fold from frozen manifests containing ONLY authorized train/validation WAV and TextGrid.
- Per-fold test sealing: Never stage or extract active test partition audio/labels for fold $k$ ($S_{\text{test}}^{(k)}$). Maintain strict per-fold speaker-disjoint isolation ($S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$). Across folds, speakers rotate legitimately according to `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`.
- Colab VM storage (`/content`) is ephemeral: Copy reusable fold package from Drive to VM disk and verify SHA-256 before training.
- Read audio directly from local VM disk during training; never stream training audio across a Google Drive mount.

## Colab CLI & Operational Safety
- Provision GPU runtimes via official Colab CLI (`colab new -s <name> --gpu T4` or `colab run`).
- Colab non-interactive rule: Never execute interactive commands (`colab drivemount`, `colab auth`, `colab repl`) in automation.
- Monitor compute units via `colab usage`. If GPU quota is exhausted or auth is missing, STOP and report.
- Resource hygiene: Always release idle runtimes promptly (`colab stop -s <name>`) to preserve compute units.

## Kaggle CLI & Operational Safety
- Provision GPU runtimes via official Kaggle CLI (`kaggle kernels push`) with `machine_shape: NvidiaTeslaT4`, `enable_gpu: true`, `is_private: true`.
- Monitor weekly GPU accelerator quota via `kaggle quota`. Stop if quota is exhausted or insufficient.
- Resource hygiene: Keep notebooks, scripts, and staged datasets strictly private under CC BY-NC 4.0; never publish public assets.
- Ephemeral containment: Read immutable inputs from `/kaggle/input`; write temporary artifacts to `/kaggle/working` (subject to 20 GB ceiling).
- Checkpoint durability: Mirror every epoch checkpoint atomically to an approved private persistent store with cryptographic SHA-256 readback. Ephemeral `/kaggle/working` is non-durable and must never be the sole checkpoint store.
- Numerical integrity: Execute strictly under CUDA FP16 AMP; preserve frozen Wav2Vec2 backbone, two-head CTC+binary architecture, 3,600 max updates per arm, and independent optimizer/scheduler/scaler/RNG states.

## Training Contract & Provenance
- Split manifest: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- Fold 1 contract: 2,399 train, 300 val (NJS/RRBI), 900 test SEALED. Two-head objective: CTC + Binary MDD only.
- Cross-fold contract (Folds 2–4): Independent Baseline R training from original pretrained initialization (`seed: 1337`, `batch_size: 2`, `gradient_accumulation_steps: 2`, `warmup_updates: 100`, cosine schedule over `max_epochs: 25` horizon [~15,000 updates], early stopping patience 5 on validation AP with `blank_fraction < 0.99`). Strictly prohibit warm-starting or fine-tuning from Fold 1 checkpoints.
- Dual checkpoint accounting: Save and report both the fixed-budget 3,600-update (Epoch 6) checkpoint and the validation-selected `best.pt` checkpoint to avoid confounding training exposure with model performance.
- Thesis scope boundaries: SpeechOcean762 cross-corpus evaluation and pedagogical pronunciation feedback belong to RQ1. RQ2 is reserved exclusively for Multi-Agent LLM research.
- Backbone integrity: Pretrained `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`) kept frozen in eval.
- Start CTC decoding with greedy evaluation; use validation only for checkpoint and threshold selection.
- Never use gold TextGrid timestamps as model inference inputs or claim acoustic diagnosis readiness.
- Training entrypoints: `configs/rq1_trn_02a_fold1.yaml` and `audit_scripts/train_fold1_ctc_binary.py`.
- Atomic checkpointing: Save each epoch (`last.pt`, `best.pt`) with weights, optimizer, scheduler, scaler, RNG, and metadata.
- Sync each epoch checkpoint to Google Drive immediately; verify remote presence.
- Keep datasets, archives, checkpoints, model weights, and credentials out of Git.
- Work on reviewable feature branches. Never force-push, merge into main, or start training without explicit user authorization.

## Mandatory Process Checkpoint Policy
For every significant research or engineering process:

1. Create or update a Markdown checkpoint at the completion, interruption or blocking of the process.
2. Record the actual state, inputs, outputs, verification evidence and restart instructions.
3. Update the global checkpoint index (`docs/process-checkpoints/INDEX.md`).
4. Include the checkpoint file in the same reviewable commit as the work whenever possible.
5. Do not report a process as complete without a corresponding checkpoint.
6. Never mark PASS on unexecuted tests or unverified artifacts.
7. On failure, record the exact safe restart point before stopping.
8. On session handoff, read the global index and latest relevant checkpoint before editing or executing.
9. On restart, verify commit SHA, artifacts, checksums and authorization constraints.
10. Preserve historical records; append corrections rather than silently rewriting prior evidence.

Checkpoint documentation is mandatory even when the process ends in BLOCKED or PARTIAL state.

For training and GPU workflows, checkpoints must record execution status, last valid model artifact, successful optimizer update counts, skipped updates, and authorization boundaries.

The Markdown checkpoint is a **research/process checkpoint**, not a substitute for a PyTorch `.pt` training-state checkpoint.
