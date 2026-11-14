# Resume registration — 2026-11-13

User instruction: “Đọc HANDOFF.md và resume GOAL”. This revokes the prior pause;
the existing local-training authorization and test seal remain in force.
The research goal is NOT complete. Branch: `feat/crottc-if-official-20261113`.

## Preserved starting point

`am_crottc_profile/last.pt`: SHA-256
`f4c97a353ce392e166c60965b68aa49868cab598d169ff218bd5626ddde87bd7`,
epoch 0, cursor 32, update 1. This checkpoint is immutable. The preceding trainer
is preserved as `full_train_before_resume.py` in this report directory (SHA-256
`940c97e5b3241abced57fcabb1f90474882b107c69e66fbed5a3b8dcd0be26cf`).
The legacy migration permits only that exact checkpoint, stage AM CROTTC,
paper objective, effective batch 32 and identical data/source/native receipts.
All later checkpoints also check implementation, effective config and runtime.
No fallback to fresh initialization is permitted for a missing resume file.

## Registered execution order and limits

1. Numerical continuation tests: actual serialization and reload of weights,
   BatchNorm buffers, partial gradients, Adam, scheduler and RNG; compare subsequent
   losses and updates with uninterrupted execution on CPU and CUDA.
2. Representative scratch WavLM Large profile: at most 900 seconds, 32 TRAIN
   microbatches, one full Adam update; 16 median neighbors, 8 p95 neighbors,
   8 longest sentences. Load existing Adam state to measure steady-state memory.
   Discard scratch weights; do not advance the production cursor.
3. Real continuation smoke: start `am_crottc` with `--resume-from` the preserved
   checkpoint, at most 60 seconds and one additional update. A clean stop can
   contain partial gradients. Reload that partial state before the sustained run.
4. Initial sustained window: at most 14,400 seconds; epoch target remains 300.
   Check stopping before every microbatch and validation sentence. Grace is one
   in-flight microbatch plus atomic checkpoint finalization. Save every five
   updates, each epoch, and clean stop. Partial groups retain gradients, loss sum,
   group size, consumed cursor and RNG; no partial-size optimizer update.

GPU lock is acquired by each entrypoint. CPU threads: 4. RAM minimum at start:
12 GiB available; stop if below 2 GiB between microbatches. First initialization
requires 12 GiB free disk; resume requires 6 GiB; every atomic save checks estimated
payload size plus 1 GiB reserve. This accounts for partial-gradient checkpoints
of about 4.7 GiB. No historical checkpoint is removed. Clone restored CPU Adam
tensors out of mmap so replacing `last.pt` releases the old file's disk allocation.

## Fidelity freeze for continuation

Continue the **same registered local recipe** that produced the starting checkpoint:
WavLM Large, frozen CNN only, kernel-3 two-layer Conformer, original OTTC operator,
paper Eq.5/6 objective, source speed/DropFreq/DropChunk waveform augmentation,
native SSL SpecAugment enabled, CPU FP32 SSL/Adam and CUDA heads, accumulation 32.
Do not switch to source objective, alter perturbation, or insert a different
SpecAugment implementation while resuming this checkpoint.

The [paper v1, §3.3 and §6.2](https://arxiv.org/html/2604.22133v1) specifies spectral
time warping 80 and ratio-based masks; the pinned source recipe instead defines
waveform perturbations and leaves its warp object unused. The continuation is
therefore explicitly a **local source-augmentation variant**, not a fully faithful
paper recipe. This difference is frozen and machine-readable in the execution
contract, rather than silently claimed resolved. Local split/preprocessing and
microbatch BatchNorm differences also remain. A faithful perturbation variant,
if implemented, must be registered separately and must not overwrite this lineage.

The paper remains ambiguous about the distribution of 200 LM/teacher epochs
between its two IF stages and gives conflicting omega1 values. No IF stage is
started under this registration until its separate contract is settled. Exact
paper evaluation remains NOT_REPRODUCED because its split overlaps local sealed
speakers. Continue permitted training/implementation without opening test data.

## Evidence

- Resume unit tests: `tests/test_crottc_full_resume.py`; initial run 4 passed
  (CPU and CUDA numerical tests plus negative resume/provenance cases). This is
  an exact numerical state-continuation test on a small stochastic model, not
  evidence of bit-identical full WavLM training across all hardware/operators.
- Real representative profile, continuation and sustained-window receipts are
  recorded under `experiments/crottc_if_full_local_20261113/` as they finish.
- No new validation F1 or paper reproduction result is claimed by this document.

- Full-paper table/figure coverage is tracked in `PAPER_COVERAGE.csv`; public-checkpoint
  smoke is never promoted to a training or benchmark reproduction result.
