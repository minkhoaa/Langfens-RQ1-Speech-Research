# RQ1 TRAIN-PHASE-01 controlled runner

Default is **plan only**. No formal training is launched by planning, tests, rehearsal or a GO report.

## Safe commands

```bash
# Full frozen 3-seed / 4-outer / 3-inner schedule, no corpus loading or fits:
.venv/bin/python experiments/rq1_train_phase_01/train.py --mode plan

# Complete orchestration using generated arrays and explicitly tiny fixture models:
# Choose a NEW directory; do not overwrite the retained receipt bundle.
.venv/bin/python experiments/rq1_train_phase_01/train.py --mode rehearse \
  --seed 101 --output /tmp/phase01-new-synthetic-rehearsal

# Resume that same generated fixture with remaining wall allowance:
.venv/bin/python experiments/rq1_train_phase_01/train.py --mode rehearse \
  --seed 101 --output /tmp/phase01-new-synthetic-rehearsal --resume
```

`--mode train` requires BOTH `--authorize-formal` and a separately operator-provided authorization JSON. The file must bind `formal_training_authorized: true`, exact `config_sha256` (`runtime.digest(configuration)`), current `source_hashes`, `environment`, and the selected `seeds`. No command in this README generates authorization. Formal output is restricted to fresh `experiments/rq1_train_phase_01/runs/` directories. This task did not authorize or launch formal training.

Only CPU/two threads are supported by this driver. CUDA and budget/hyperparameter overrides are rejected. Configuration must equal the original frozen MB08 JSON; its historical status fields are not rewritten.

## Modules

- `protocol.py`: official 101/202/303 seeds, local inner0/1/2, nine/six speaker fits, immutable recipes, source hashes.
- `runtime.py`: atomic writes, process output lock, signal-safe stop requests, cumulative persistent per-seed budget.
- `engine.py`: separate module construction/seed reset, finite updates, full optimizer/RNG/sampler state, immutable all-epoch archive/index, fixed-terminal loading, exact resume.
- `datasets.py`: TRAIN-only audited frozen-cache adapter and strict gold-free inference view; generated-only fixture population.
- `backend.py`: independent H-002/gap heads, CTC supervision on fit UIDs only, exact legacy E4 competitors/sequence kernel, strict per-utterance exports.
- `workflow.py`: all3 inner score streams -> one atomic sealed policy -> outer refit -> scores/controls -> official evaluation. No outer-label threshold fitting.
- `pooled.py`: full-key pooled decisions and official EVAL-V3.1 metrics from the frozen outer decisions, with corruption checks.

Each seed has a separate **cumulative2700s active wall budget**, covering population initialization/hash verification, loading, fitting, inference and evaluation. Common Python/library import startup is separately reflected in process wall profiling. Resume does not allocate a new budget. At the cap, no next update/utterance/stage is started; an in-flight Torch/kernel or atomic commit may finish at the next safe point. Exhausted-cap resume cannot silently add time. Missing/malformed budget ledgers in an existing run fail closed instead of resetting used time. A changed allowance requires a separately preregistered new run, not a CLI override. SIGINT/SIGTERM stop requests preserve safe checkpoints and release locks.

Epoch archives retain model, optimizer, Python/NumPy/Torch RNG, sampler/permutation/cursor, completed epoch, initial-model hash and source/config/split/scaler identities. A rolling safe-point checkpoint handles interrupted epochs. Every completed epoch is indexed; historical index loss or archive mutation fails closed. Only a just-completed archive commit can be repaired, after exact full-state equality with the latest checkpoint. Inference reloads the declared terminal epoch, never a metric-selected checkpoint.

## Verification and limits

`preflight/orchestration_v2/` contains26 new tests, source freezes, independent review/fixes, the actual safe CLI plan/rehearsal/authorization-denial receipts and generated keyed outputs. Combined with existing tests: **91 passed**. New-backend E4 competitors and S/O LLRs numerically match the verified MB07 producer within1e-12 on generated inputs. CLI rehearsal is **synthetic/tiny models**, not model quality, real-data generalization, or evidence of completing a full formal seed within45minutes. Selected real head architectures also exercise the new engine on tiny generated-input unit fixtures. Previous real TRAIN smoke and eligible mixed-source joint-policy receipts remain unchanged.

Audio controls operate on cached acoustic features with intended references/topology fixed; no encoder waveform rerun is claimed. Shared-trunk/staged models, encoder fine-tuning, prompt-text-only lineage, deployment safety and full formal runs remain outside this evidence.

Authoritative current decision: `reports/TRAIN_PHASE_01_GO_GATE.md`. Prior conditional reports/manifest are preserved under `preflight/orchestration_v2/prior_reports/` and `PRIOR_EVIDENCE_MANIFEST_SHA256.json`.
