# LANGFENS RQ1-ENCODER-BENCH-01 — Exact Reproduction Protocol

**Campaign:** `LANGFENS-RQ1-ENCODER-BENCH-01`  
**Standard:** EVAL-V3.1, 4-fold Speaker-Grouped TRAIN OOF (1,800 utterances, 12 speakers)  
**Registered Seeds:** 101, 202, 303  
**Target Environment:** Linux x86_64, Python 3.11.17, PyTorch 2.6.0+cu124, Scikit-Learn 1.9.1  
**Hardware Profile:** Single GPU (NVIDIA GTX 1650 4GB VRAM) or CPU Fallback (2+ threads)  

---

## 1. Environment Verification & Dependencies

Activate the verified project virtual environment:
```bash
cd /home/khoa/Projects/New-Method-Research
source .venv/bin/activate
```

Verify installed runtime dependencies:
```bash
python -c "
import torch, transformers, torchaudio, sklearn, soundfile, numpy
print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())
print('Transformers:', transformers.__version__)
print('Scikit-Learn:', sklearn.__version__)
print('SoundFile:', soundfile.__version__)
print('NumPy:', numpy.__version__)
"
```
*Expected Output:*
- PyTorch: `2.6.0+cu124` (CUDA: `True`)
- Transformers: `5.22.1`
- Scikit-Learn: `1.9.1`

---

## 2. Directory Structure Setup

Ensure the experiment directories exist:
```bash
mkdir -p experiments/rq1_encoder_bench_01/common \
         experiments/rq1_encoder_bench_01/features \
         experiments/rq1_encoder_bench_01/agent_a \
         experiments/rq1_encoder_bench_01/agent_b \
         experiments/rq1_encoder_bench_01/agent_c \
         experiments/rq1_encoder_bench_01/supervisor \
         reports
```

---

## 3. Step-by-Step Execution Protocol

### Step 0: Preregistration & Manifest Verification
Generate and verify the immutable preregistration:
```bash
python experiments/rq1_encoder_bench_01/preregistration.py
```
*Expected Output:*  
`PREREGISTRATION.json SHA256: 6091c06b2c7bdc961549403558c45ce6570d94b6c4a53b142a813ff9f33572e1`

### Step 1: Precompute Alignment Centers
Extract monotonic Viterbi alignment centers from the cached CharSiu frame logits for all 1,800 TRAIN utterances:
```bash
python experiments/rq1_encoder_bench_01/features/precompute_alignment.py
```
*Expected Output:*  
- Runtime: ~56s  
- Output file: `experiments/rq1_encoder_bench_01/features/alignment_centers.npz` (959.9 KB)

### Step 2: Dense Acoustic Feature Extraction
Extract and cache matched acoustic representations across all 4 candidate backbones and layers:
```bash
python experiments/rq1_encoder_bench_01/features/extract_all_features.py
```
*Models Extracted:*
1. `facebook/wav2vec2-base-960h` (layers 0..12) — ~122s
2. `facebook/wav2vec2-base` (layers 3, 6, 9, 12) — ~99s
3. `facebook/hubert-base-ls960` (layers 3, 6, 9, 12) — ~99s
4. `microsoft/wavlm-base` (layers 3, 6, 9, 12) — ~102s
*Total Runtime:* ~423s (7.06 min). Peak VRAM: ~495 MB.

### Step 3: Execute Agent A — Pretraining & Layer Study
Run the 8-configuration screening and multi-seed replications:
```bash
python -u experiments/rq1_encoder_bench_01/agent_a/run_agent_a.py \
    > experiments/rq1_encoder_bench_01/agent_a/agent_a.log 2>&1
```
*Artifacts Produced:*
- `experiments/rq1_encoder_bench_01/agent_a/AGENT_A_SUMMARY.json` (14 evaluations, ~475 KB)

### Step 4: Execute Agent B — Adaptation & Fine-Tuning Study
Run the 4 adaptation candidates (Frozen L6, Layer-Weighted Mixture, Bottleneck Adapter, Capacity-Matched Control):
```bash
python -u experiments/rq1_encoder_bench_01/agent_b/run_agent_b.py \
    > experiments/rq1_encoder_bench_01/agent_b/agent_b.log 2>&1
```
*Artifacts Produced:*
- `experiments/rq1_encoder_bench_01/agent_b/AGENT_B_SUMMARY.json` (4 evaluations, ~136 KB)

### Step 5: Execute Agent C — Independent SSL Backbone Control
Run the 8-configuration screening (HuBERT vs WavLM across layers 3, 6, 9, 12) and multi-seed replications:
```bash
python -u experiments/rq1_encoder_bench_01/agent_c/run_agent_c.py \
    > experiments/rq1_encoder_bench_01/agent_c/agent_c.log 2>&1
```
*Artifacts Produced:*
- `experiments/rq1_encoder_bench_01/agent_c/AGENT_C_SUMMARY.json` (16 evaluations, ~543 KB)

### Step 6: Execute Supervisor — Audit & Synthesis
Synthesize results, perform hypothesis testing, and generate master reports and CSV matrices:
```bash
python experiments/rq1_encoder_bench_01/supervisor/run_supervisor.py
```
*Artifacts Produced:*
- `experiments/rq1_encoder_bench_01/supervisor/MASTER_SUMMARY.json`
- `reports/ENCODER_BENCH_01_RESULTS.csv` (34 rows)
- `reports/ENCODER_BENCH_01_LAYER_ABLATIONS.csv` (16 rows)

---

## 4. Verification Checksums

Verify file hashes against the sealed experiment manifest:
```bash
python -c "
import hashlib, json
from pathlib import Path

HERE = Path('experiments/rq1_encoder_bench_01')
manifest = json.loads((HERE / 'MANIFEST_SHA256.json').read_text())
all_match = True
for rel_path, expected_h in manifest.items():
    p = HERE / rel_path
    if not p.exists():
        print(f'MISSING: {rel_path}')
        all_match = False
        continue
    actual_h = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual_h != expected_h:
        print(f'MISMATCH: {rel_path}')
        all_match = False
if all_match:
    print('ALL REPRODUCTION ARTIFACTS VERIFIED BYTE-IDENTICAL!')
"
```

---

## 5. Expected Numerical Tolerance Bounds

When reproducing on identical hardware and software stack:
- **Macro-3 AP:** Values should match within $\pm 0.0010$ of reported figures.
- **Operating Policy Metrics ($B=0.2$):** Realized Joint FWR must lie strictly in $[0.170, 0.285]$ across folds.
- **Learned Layer Weights:** Attention mass on layers 4–8 must exceed $45\%$, with Layer 6 mode between $0.095$ and $0.115$.
