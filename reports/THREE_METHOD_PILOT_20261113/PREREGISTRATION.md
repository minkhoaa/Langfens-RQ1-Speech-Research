# Three-method local pilot — 2026-11-13

User authorization: implement all three methods in parallel using Luna agents.
Implementation branch: `feat/three-method-local-pilot-20261113`.

The three workers own separate method packages; the parent owns staging and
evaluation. GPU execution is serialized with the existing shared GPU lock.
Initial hardware: GTX 1650 4 GB, 3,729 MiB free reported by nvidia-smi; available
RAM 22,177 MiB; disk approximately 28 GB free. No preexisting GPU compute job was
shown by the initial query.

Frozen scope: M0 fold zero within the historical TRAIN population. Deterministic
selection by hash of seed101+utterance ID, audio duration <=5 seconds; 4 eligible
FIT utterances per FIT speaker and 2 queries per CAL/HELD speaker. Total 24 FIT,
6 CAL, 6 HELD. FIT-only exact-phone filtering; CAL/HELD are not filtered by
realized-phone eligibility. Audio and feature SHA-256 values are checked against
the frozen extraction inventory/preflight. Test data is never accessed.

Budgets are recorded before work in `pilot.json`, `x_runtime_budget.json` and
`per_large_budget.json`. CROTTC compact CPU run: <=2 epochs, <=300 seconds, 2
threads. Retrieval: no optimizer updates, <=600 GPU seconds with lock. X-MDD
adaptation: <=2 FIT utterances, <=2 SFT updates, <=1 GRPO update with group size 2,
<=600 CPU seconds, 2 threads; real execution requires valid Qwen runtime and
canonical TTS. Combined new model weights download <=6 GB and free disk reserve
>=12 GB. Tiny synthetic mechanism tests are correctness checks, not model results.

These are implementation/smoke budgets, not matched scientific comparison budgets.
W2V2 frozen-H6 compact CROTTC and CPU low-rank X-MDD change the original recipes;
the PER paper backbone is attempted separately from the cached HuBERT-base
adaptation. All differences must remain explicit. No published paper score or
product acceptance is claimed from this pilot.

The parent adapts predicted phone sequences to the existing deterministic edit
alignment and EVAL-V3.1 target/exact-gap evaluator. No gold timestamps enter query
inference. Hard sequences provide no continuous event scores, so AP is marked
not estimable instead of inventing confidence. No tuning on HELD. References are
historically annotation-derived supplied phone sequences, not independently
validated prompt G2P; this provenance limitation remains explicit.

Checkpoints must retain per-epoch weights, optimizer, scheduler, scaler, RNG and
metadata, save atomically, and reload. Any resource/nonfinite/data failure remains
an explicit failed/interrupted receipt. Superseded implementation runs remain
separate and are excluded from current method results.
