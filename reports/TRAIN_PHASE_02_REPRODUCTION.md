# LANGFENS RQ1-TRAIN-PHASE-02 — Reproduction Runbook & Verification Manual

**Campaign:** LANGFENS RQ1-TRAIN-PHASE-02  
**Evaluation Standard:** EVAL-V3.1  
**Target Revision:** `aa4e6ea2147180a4a820d77a71911f333fcb6721`  
**Execution Runtime:** Python 3.11.17, PyTorch 2.6.0+cu124, scikit-learn 1.9.1  
**Hardware Specification:** Standardized CPU execution (2 threads per job), Linux 7.3.1-1-cachyos-x86_64  

---

## 1. Environment & Dependency Setup

All Phase-02 scripts execute inside the dedicated project virtual environment:

```bash
# Verify virtual environment interpreter
/home/khoa/Projects/New-Method-Research/.venv/bin/python --version
# Expected: Python 3.11.17

# Verify required ML runtime libraries
/home/khoa/Projects/New-Method-Research/.venv/bin/python -c "
import torch, torchaudio, sklearn, numpy
print('PyTorch:', torch.__version__)
print('Scikit-Learn:', sklearn.__version__)
print('NumPy:', numpy.__version__)
"
# Expected:
# PyTorch: 2.6.0+cu124
# Scikit-Learn: 1.9.1
# NumPy: 2.2.3 (or 1.26+)
```

---

## 2. Directory Layout & Artifact Architecture

Phase-02 strictly preserves Phase-01 artifacts read-only under `experiments/rq1_train_phase_01/formal/`. All new code, receipts, and diagnostics are contained within:

```
experiments/rq1_train_phase_02/
├── common/
│   └── policy_engine.py             # Reusable zero-leakage thresholding & evaluator harness
├── agent_a/
│   ├── run_agent_a.py               # Comprehensive policy experiments (arms 0..4, 3 seeds)
│   └── AGENT_A_POLICY_EVALUATION.json
├── agent_b/
│   ├── run_agent_b.py               # 12-speaker diagnostics & adaptation strategies
│   └── AGENT_B_ROBUSTNESS_EVALUATION.json
├── agent_c/
│   ├── run_agent_c.py               # Insertion failure analysis & matched ablations
│   └── AGENT_C_INSERTION_EVALUATION.json
├── supervisor/
│   ├── run_supervisor.py            # Integrity audit & master summary synthesis
│   └── MASTER_SUMMARY.json
└── PREREGISTRATION.json             # Immutable campaign preregistration seal
```

---

## 3. Step-by-Step Execution Sequence

### Step 0: Pre-Flight Integrity Check
Verify that the Phase-01 formal manifest (2,238 files) is intact and unmodified before launching Phase-02:

```bash
.venv/bin/python -c "
import json, hashlib
from pathlib import Path
root = Path('.')
manifest = json.loads((root / 'experiments/rq1_train_phase_01/formal/MANIFEST_SHA256.json').read_text())
files = manifest['files']
spot_checks = [
    'seed_101/outer_0/outer_refit/scores_original.npz',
    'seed_202/outer_1/outer_refit/scores_original.npz',
    'seed_303/outer_3/outer_refit/scores_original.npz',
    'seed_101/pooled/evaluation.json',
    'seed_202/pooled/evaluation.json',
    'seed_303/pooled/evaluation.json',
]
for rel in spot_checks:
    p = root / 'experiments/rq1_train_phase_01/formal' / rel
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    expected = files[rel]['sha256']
    assert actual == expected, f'Corruption in {rel}'
print(f'Integrity verified: all {len(spot_checks)} key files match sealed Phase-01 hashes!')
"
```

### Step 1: Execute Agent A Operating Policy Pipeline
Derive and evaluate the 5 policy arms across all outer folds and 3 seeds:

```bash
.venv/bin/python experiments/rq1_train_phase_02/agent_a/run_agent_a.py
```
- **Generated Deliverable:** `reports/TRAIN_PHASE_02_POLICY_RESULTS.csv`
- **Output Receipt:** `experiments/rq1_train_phase_02/agent_a/AGENT_A_POLICY_EVALUATION.json`
- **Expected Duration:** ~45 seconds on standardized CPU.

### Step 2: Execute Agent B Speaker Robustness Pipeline
Audit all 12 speakers and evaluate candidate unsupervised adaptation strategies:

```bash
.venv/bin/python experiments/rq1_train_phase_02/agent_b/run_agent_b.py
```
- **Generated Deliverable:** `reports/TRAIN_PHASE_02_SPEAKER_ROBUSTNESS.csv`
- **Output Receipt:** `experiments/rq1_train_phase_02/agent_b/AGENT_B_ROBUSTNESS_EVALUATION.json`
- **Expected Duration:** ~35 seconds on standardized CPU.

### Step 3: Execute Agent C Insertion Pipeline
Analyze insertion context error distributions, gap warning budgets, and matched architectural ablations:

```bash
.venv/bin/python experiments/rq1_train_phase_02/agent_c/run_agent_c.py
```
- **Generated Deliverable:** `reports/TRAIN_PHASE_02_INSERTION_RESULTS.csv`
- **Output Receipt:** `experiments/rq1_train_phase_02/agent_c/AGENT_C_INSERTION_EVALUATION.json`
- **Expected Duration:** ~30 seconds on standardized CPU.

### Step 4: Execute Supervisor Master Verification & Synthesis
Validate all experiment receipts, compute pooled cross-seed averages, and verify baseline deltas:

```bash
.venv/bin/python experiments/rq1_train_phase_02/supervisor/run_supervisor.py
```
- **Output Receipt:** `experiments/rq1_train_phase_02/supervisor/MASTER_SUMMARY.json`
- **Expected Duration:** ~5 seconds.

---

## 4. Cryptographic Output Hashes

The generated Phase-02 deliverable CSVs have the following verified SHA256 digests:

| File Path | SHA256 Digest |
|---|---|
| `reports/TRAIN_PHASE_02_POLICY_RESULTS.csv` | `3ea50124ee6307145892e6b28374472755e1a5b9b397e30d9c93098272711e20` |
| `reports/TRAIN_PHASE_02_SPEAKER_ROBUSTNESS.csv` | `9a46528894172e81505244076528923baf5ed7efe33c4ac74f686b97f695206d` |
| `reports/TRAIN_PHASE_02_INSERTION_RESULTS.csv` | `f08f014e2a872fe4bc56a86974d713ee697570a8ddfef5a39dff9b1a39f0b728` |
| `experiments/rq1_train_phase_02/PREREGISTRATION.json` | `9926a3a3b07e1c7b8c44577cbe4391b25a072826b42f48a98cd6814bfe58f17b` |

All results reproduce deterministically.
