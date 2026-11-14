# GPU + local RAM continuation — 2026-11-14

User: “tiếp với gpu đi, ko đủ thì dùng ram”. Goal remains active.

Full WavLM Large and all AM heads now compute in FP32 on GTX1650. Native CNN
parameters remain frozen; the same 311,246,528 SSL parameters are trainable.
All 315,453,120 SSL parameters remain in the model. FP32 Adam master parameters,
Adam moments, and accumulated gradients live in local RAM. After each microbatch,
gradients move to RAM and CUDA gradient storage is released. Saved activations
also offload to RAM. This introduces no model reduction or new frozen parameters.

Four prior resource profiles failed with CUDA OOM while accumulated gradients
remained resident on GPU. Their status/logs are preserved. The successful RAM
accumulation profile covered 32 authorized TRAIN sentences around median/p95 and
the longest duration (9.054 seconds), with one real full-model Adam update:
peak allocated CUDA 2,696,370,688 bytes (~2.51 GiB), peak RSS 8,588,914,688 bytes
(~8.0 GiB). These are feasibility measurements, not a repeated speed benchmark.

The native feature-encoder API also disables unnecessary waveform gradients
through the already frozen CNN. A test using the actual native CNN/projection
checks unchanged parameter trainability, outputs and downstream gradients within
FP32 tolerance. Trainable SSL/Conformer/head weights remain learned.

## Checkpoint continuity and evidence

- Existing CPU placement checkpoint epoch0/cursor704/update22 with 3/32 pending
  gradients migrated through `GPU_RAM_PLACEMENT_MIGRATION.json`. Only execution
  placement changes are allowed; data, objective, LR, kernel, batch32 and 300
  epochs cannot change. Cross-device bit identity is not claimed.
- Native GPU continuation reached update23/cursor736 and atomically saved/reloaded.
- A bounded clean stop then saved 12/32 gradients in RAM. Native GPU restart
  continued at microbatch13 and reached update24, then durable update25/cursor800.
- Checkpoint hashing plus scheduler/Adam counters verify optimizer continuity.
  Sixteen unit tests passed; real full-model checks are recorded separately in
  `GPU_RAM_EXECUTION_VERIFICATION.json`.
- Historical profile checkpoint remains unchanged. The live `last.pt` continues
  its registered atomic rolling replacement; no historical checkpoint was deleted.

## Current entrypoint

Inspect `RUNNING_WINDOW.json`, process liveness and GPU lock first. Do not start
a duplicate job. The current four-hour window uses:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv-crottc-official/bin/python -m research.crottc_if_official.gpu_train \
  --stage am_crottc --resume --wall-seconds 14400
```

The migration receipt was used once. Regular continuation now uses `--resume`
without migration. The old CPU driver has a different execution contract and is
not the current resume entrypoint. Keep driver/helper implementation files pinned
while the job runs. Partial gradients are unscaled FP32 CPU master gradients;
weights/optimizer/scheduler/scaler/RNG and cursors continue in every checkpoint.

## Research scope

This is full-size local AM training, not completion of the full paper. Epoch
target remains 300, with checkpoint every five updates/epoch/clean window end.
CTC control and IF stages still follow. IF components, parent-stage gates and full IF trainer are implemented/unit-checked;
native IF resource profiling and stage execution remain pending. See
`IF_TRAINER_PROTOCOL.md` and `IF_TRAINER_IMPLEMENTATION_VERIFICATION.json`. Parent gates reject
unfinished/profile checkpoints. Provisional IF epoch allocation and existing
paper/source augmentation, preprocessing and split differences stay explicit.
Official test remains sealed; no new paper F1 is claimed. Full table coverage
continues in `PAPER_COVERAGE.csv`.
