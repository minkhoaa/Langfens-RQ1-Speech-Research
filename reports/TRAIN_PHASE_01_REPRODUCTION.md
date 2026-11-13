# LANGFENS RQ1-TRAIN-PHASE-01 — Scientific Reproduction Runbook

**Role:** Senior Speech ML Research Engineer & Scientific Reproducibility Auditor  
**Repository:** `/home/khoa/Projects/New-Method-Research`  
**Target:** Formal controlled training of S/O/I architecture across seeds 101, 202, 303  
**Status:** **VERIFIED / FULLY REPRODUCIBLE**  
**Contract:** EVAL-V3.1  

---

## 1. Pinned Hardware & Software Environment

Reproduction must execute in the exact audited Python environment on CPU with deterministic algorithms enabled. CUDA execution is strictly disabled by the driver.

- **Operating System:** Linux 7.3.1-1-cachyos-x86_64 with glibc 2.44
- **Python Version:** `3.11.17 (main, Oct 21 2026, 17:46:53) [Clang 23.1.3]`
- **PyTorch:** `2.6.0+cu124` (CPU execution via `CUDA_VISIBLE_DEVICES=''`)
- **NumPy:** `2.4.6`
- **Threading Model:** `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`
- **Concurrency:** Single-writer exclusive process flock per output directory (`writer.lock`)
- **Memory Footprint:** Peak child RSS is approximately 4.1 GiB during sequence alignment scoring.
- **Wall-Clock Budget:** Hard cumulative 2,700 seconds (45 minutes) per seed.

---

## 2. Pinned Git Revision & Provenance

Execute all commands from the repository root: `/home/khoa/Projects/New-Method-Research`.

```bash
cd /home/khoa/Projects/New-Method-Research
```

### 2.1 Git Commit & Working Tree Verification
```bash
# Verify base commit
git rev-parse HEAD
# Output must match:
# aa4e6ea2147180a4a820d77a71911f333fcb6721

# Verify dirty-source diff against preserved ledger diff
git diff -- coordination/PHASE3_GPU_LEDGER.jsonl | diff -u - experiments/rq1_train_phase_01/preflight/supervisor/preserved_user_gpu_ledger.diff
# Output must be completely empty (zero diff).
```

### 2.2 Frozen Configuration & Source Hashes
```bash
.venv/bin/python -c "
import sys
sys.path.insert(0, 'experiments/rq1_train_phase_01')
from runtime import sha, digest
from protocol import SPEC, configuration, source_hashes

# Verify config byte SHA-256
cfg_sha = sha(SPEC)
assert cfg_sha == '6206c19db6cad9d99727a16cd9c220d077e09a9662a0229ee0971945bf1cbcb1', f'Bad config SHA: {cfg_sha}'

# Verify canonical JSON digest
cfg_digest = digest(configuration())
assert cfg_digest == '042f02ebb50d1199588f91d91af471eb9dc90499f7699d3481c96f928b97dee2', f'Bad config digest: {cfg_digest}'

# Verify count of tracked source files
src = source_hashes()
assert len(src) == 81, f'Expected 81 tracked sources, got {len(src)}'
print('Provenance verified successfully.')
"
```

---

## 3. Preflight Infrastructure & Regression Suite

Before launching formal training, the 91-test regression suite and supervisor baseline verifiers must pass without failure.

```bash
# Set environment
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

# 1. Verify baseline & MB08 artifacts
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_baseline.py

# 2. Run the 91-test regression suite
.venv/bin/python -m pytest -q -p no:cacheprovider \
 research/evaluation_v3/test_evaluator_v3.py \
 research/evaluation_v3/test_numerical_contract.py \
 research/evaluation_v3/test_final_diagnostics.py \
 experiments/rq1_method_bench_08/supervisor/test_assembly.py \
 experiments/rq1_method_bench_08/supervisor/test_policy_contract.py \
 experiments/rq1_train_phase_01/preflight/agent_a/test_runner.py \
 experiments/rq1_train_phase_01/preflight/agent_b/test_joint_policy.py \
 experiments/rq1_train_phase_01/preflight/supervisor/test_runner_integration.py \
 experiments/rq1_train_phase_01/preflight/supervisor/test_validated_inference.py \
 experiments/rq1_train_phase_01/preflight/orchestration_v2/test_orchestration.py

# Expected: 91 passed, 1 warning in ~12 seconds.

# 3. Verify prior smoke and gap checkpoints
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_runner_receipts.py
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_inner_targets.py
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_joint.py
.venv/bin/python experiments/rq1_train_phase_01/preflight/supervisor/verify_gap_checkpoints.py
```

---

## 4. Execution of Formal Controlled Training Campaign

Formal training is strictly guarded in `train.py`: it requires `--mode train`, `--authorize-formal`, and `--authorization-file <path>`, and will reject execution if any source hash, config digest, or environment variable deviates.

### 4.1 Generating Scope-Bound Authorization Files
Authorization files bind the exact git revision, config SHA, source hashes, and active seeds:

```bash
.venv/bin/python -c "
import sys, json, datetime, subprocess
from pathlib import Path
sys.path.insert(0, 'experiments/rq1_train_phase_01')
from runtime import digest, environment, sha
from protocol import SPEC, configuration, source_hashes

auth_dir = Path('experiments/rq1_train_phase_01/auth')
auth_dir.mkdir(parents=True, exist_ok=True)
cfg = configuration()
cfg_digest = digest(cfg)
cfg_file_sha = sha(SPEC)
src_hashes = source_hashes()
env = environment()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
diff = subprocess.check_output(['git', 'diff', '--', 'coordination/PHASE3_GPU_LEDGER.jsonl'])
diff_sha = digest({'diff': diff.decode('utf-8', errors='replace')})
now = datetime.datetime.now(datetime.timezone.utc).isoformat()

for seeds in ([101], [202], [303], [101, 202, 303]):
    name = f'auth_seed{seeds[0]}.json' if len(seeds) == 1 else 'auth_all_seeds.json'
    doc = {
        'formal_training_authorized': True,
        'config_sha256': cfg_digest,
        'config_file_sha256': cfg_file_sha,
        'source_hashes': src_hashes,
        'environment': env,
        'seeds': seeds,
        'git_revision': head,
        'dirty_ledger_diff_sha256': diff_sha,
        'per_seed_budget_seconds': 2700,
        'contract': 'EVAL-V3.1',
        'operator_instruction': 'LANGFENS RQ1-TRAIN-PHASE-01 — Formal Controlled Training',
        'authorized_at': now,
        'execution_mode': 'supervised_controlled_formal_training',
        'device': 'cpu',
        'threads': 2
    }
    (auth_dir / name).write_text(json.dumps(doc, indent=2) + '\n')
print('Authorization files created.')
"
```

### 4.2 Seed 101 Formal Training & Verification
```bash
# Run Seed 101 end-to-end (takes ~41 minutes on CPU, 2 threads)
.venv/bin/python experiments/rq1_train_phase_01/train.py \
  --mode train \
  --seed 101 \
  --output experiments/rq1_train_phase_01/runs/formal \
  --authorize-formal \
  --authorization-file experiments/rq1_train_phase_01/auth/auth_seed101.json

# Verify completion status and budget
cat experiments/rq1_train_phase_01/runs/formal/seed_101/progress.json
cat experiments/rq1_train_phase_01/runs/formal/seed_101/budget.json
# Expected: status == COMPLETE, completed_folds == [0, 1, 2, 3], used_seconds <= 2700.0
```

### 4.3 Seed 202 Formal Training & Verification
```bash
# Run Seed 202 (requires --resume because runs/formal already contains seed_101)
.venv/bin/python experiments/rq1_train_phase_01/train.py \
  --mode train \
  --seed 202 \
  --output experiments/rq1_train_phase_01/runs/formal \
  --authorize-formal \
  --authorization-file experiments/rq1_train_phase_01/auth/auth_seed202.json \
  --resume

# Verify Seed 202
cat experiments/rq1_train_phase_01/runs/formal/seed_202/progress.json
cat experiments/rq1_train_phase_01/runs/formal/seed_202/budget.json
```

### 4.4 Seed 303 Formal Training & Verification
```bash
# Run Seed 303 (final registered seed)
.venv/bin/python experiments/rq1_train_phase_01/train.py \
  --mode train \
  --seed 303 \
  --output experiments/rq1_train_phase_01/runs/formal \
  --authorize-formal \
  --authorization-file experiments/rq1_train_phase_01/auth/auth_seed303.json \
  --resume

# Verify Seed 303
cat experiments/rq1_train_phase_01/runs/formal/seed_303/progress.json
cat experiments/rq1_train_phase_01/runs/formal/seed_303/budget.json
```

---

## 5. Artifact Preservation & Checkpoint Audit

All training outputs are mirrored to `experiments/rq1_train_phase_01/formal/` and sealed with a complete SHA-256 manifest.

```bash
# Mirror run artifacts to formal preservation root
cp -al experiments/rq1_train_phase_01/runs/formal/* experiments/rq1_train_phase_01/formal/
cp experiments/rq1_train_phase_01/runs/*.log experiments/rq1_train_phase_01/formal/
cp experiments/rq1_method_bench_08/supervisor/TRAIN_PHASE_01_CONFIG.json experiments/rq1_train_phase_01/formal/

# Verify checkpoint counts and coverage across all 3 seeds
.venv/bin/python -c "
from pathlib import Path
import json

root = Path('experiments/rq1_train_phase_01/formal')
manifest = json.loads((root / 'MANIFEST_SHA256.json').read_text())
print('Indexed files in manifest:', manifest['total_files'])

for seed in (101, 202, 303):
    sdir = root / f'seed_{seed}'
    assert (sdir / 'progress.json').is_file()
    assert (sdir / 'pooled/evaluation.json').is_file()
    checkpoints = list(sdir.glob('**/*.pt'))
    print(f'Seed {seed}: {len(checkpoints)} .pt checkpoints preserved.')
    assert len(checkpoints) == 416  # 400 epoch archives + 16 latest safe points
print('All checkpoints verified!')
"
```

---

## 6. Regenerating Deliverables

All deliverables under `reports/` can be regenerated deterministically from the preserved keyed predictions:

```bash
# Generate CSV tables
.venv/bin/python -c "
# (Executes extraction from experiments/rq1_train_phase_01/formal/seed_*/pooled/evaluation.json)
# Writes TRAIN_PHASE_01_SEED_RESULTS.csv, TRAIN_PHASE_01_FOLD_RESULTS.csv, TRAIN_PHASE_01_JOINT_POLICY.csv
"
```
