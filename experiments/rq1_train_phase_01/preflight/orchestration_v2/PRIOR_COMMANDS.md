# Supervisor commands

Run from repository root. Environment for all Python validation:

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_baseline.py
.venv/bin/python -m pytest -q \
 -o cache_dir=experiments/rq1_train_phase_01/preflight/supervisor/pytest_cache \
 research/evaluation_v3/test_evaluator_v3.py \
 research/evaluation_v3/test_numerical_contract.py \
 research/evaluation_v3/test_final_diagnostics.py \
 experiments/rq1_method_bench_08/supervisor/test_assembly.py \
 experiments/rq1_method_bench_08/supervisor/test_policy_contract.py
```

MEASURED: baseline verifier exit0; EVAL tests exit0 (41 passed). Logs/receipt are adjacent. Initial baseline verifier exit1 was an auditor schema assumption (`gap.label` instead of actual `gap.has_insertion`); corrected without changing any data or producer, failure log preserved as `baseline_initial_schema_error.log`. No scientific outcome arose from that failed invocation.

`verify_baseline.py` verifies all 136 archived MB08 evidence hashes and all 283 phase0 frozen source/config/report hashes; authoritative TRAIN supports/topology/IDs; MB03 keyed baseline AP-I/P@560; selected historical AP-S/O/Macro3 AP. These are verification of completed experiments, not new training.

## Joint-policy independent verification (all exit0)

```bash
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_inner_targets.py
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_joint.py
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_gap_checkpoints.py
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/joint_uncertainty.py
```

All are score/contract verification, not model fitting. `verify_joint.py` independently reproduces 48 policy pairs and checks all 12 pooled outputs against official EVAL-V3.1. `verify_gap_checkpoints.py` reconstructs held event inputs and replays all16 new gap checkpoints. Its first invocation hit a60s audit-script timeout caused by repeated NPZ member decompression inside an event loop; preloading arrays fixed the auditor performance issue. The timeout produced no result, changed no model/data, and its empty log is retained. Final replay exit0, max absolute error1.1920928955078125e-7.

Do not run B's training commands again into completed `agent_b/results/`; they intentionally reject overwriting completed inner/outer predictions. Exact executed commands and exit receipts are retained in that lane. No formal-run command is listed or executed here.

## Final accepted runner and combined suite

```bash
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_runner_receipts.py
.venv/bin/python -m pytest -q -p no:cacheprovider \
 research/evaluation_v3/test_evaluator_v3.py \
 research/evaluation_v3/test_numerical_contract.py \
 research/evaluation_v3/test_final_diagnostics.py \
 experiments/rq1_method_bench_08/supervisor/test_assembly.py \
 experiments/rq1_method_bench_08/supervisor/test_policy_contract.py \
 experiments/rq1_train_phase_01/preflight/agent_a/test_runner.py \
 experiments/rq1_train_phase_01/preflight/agent_b/test_joint_policy.py \
 experiments/rq1_train_phase_01/preflight/supervisor/test_runner_integration.py \
 experiments/rq1_train_phase_01/preflight/supervisor/test_validated_inference.py
```

MEASURED: both exit0; combined65 passed in3.55s, with one benign PyTorch std warning on the empty N=0 target fixture (no invalid exported event). `verify_runner_receipts.py` rechecks all accepted smoke source/input/CTC-feature/checkpoint hashes and replays738target/762gap model probabilities from retained checkpoints; it does not fit anything. `SUPERVISOR_RUNNER_FREEZE.json` predates strict boundary tests.

Final evidence indexing, no training: `.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/finalize_evidence.py`. It verifies old source/artifact preservation, original tracked user-ledger diff, unstaged state, report existence and conditional gate, then hashes the completed preflight/report bundle.
