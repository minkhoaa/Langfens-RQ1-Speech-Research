# LANGFENS — RQ1-TRN-02B / 08-Colab CPU Feasibility Benchmark & Execution Decision

**Status:** `CPU_FEASIBLE` (Execution Gate `BLOCKED` on Interactive Google Drive Mount)<br>
**Recorded at:** 2026-10-28T16:45:00+07:00<br>
**Evidence Source:** Branch `feat/rq1-trn-02b-cpu-execution`, `manifests/RQ1-TRN-02B/`<br>
**Audited Starting Commit:** `ba7be0dc407313a4f5e31577051e6746278adeed`<br>
**Execution Branch Commit:** `a54528c`<br>
**Evidence Level:** `INDEPENDENTLY_VERIFIED`

---

## 1. Executive Summary & Context

Per the project owner's instruction (`RQ1-TRN-02B-CPU-EXECUTION`):
> "The project owner requests evaluation of Google Colab CPU training instead of waiting for additional GPU compute units. The Colab UI currently indicates a potential runtime duration of up to 36 hours and 20 minutes."

A live, disposable 50-update benchmark was executed directly on the active Google Colab CPU VM (`m-s-usw3b5-57ccj5ykg7oh`, endpoint `https://8080-m-s-usw3b5-57ccj5ykg7oh-b.us-west3.prod.r.colab.dev`) using real authorized Fold 1 audio batches, full forward and backward passes, gradient accumulation, and optimizer updates in CPU FP32.

### Key Benchmark Findings:
1. **Computational Feasibility:**
   - **Median update duration:** **`12.538 seconds`** (0.080 updates/s).
   - **Validation duration per epoch:** **`4.1 minutes`** (243.7s across 150 batches / 300 utterances).
   - **Single epoch duration (600 updates + val):** **`2.16 hours`**.
   - **Full C0 training duration (6 epochs / 3,600 updates):** **`12.94 hours`**.
   - **Both arms (C0 + C1, 7,200 updates):** **`25.89 hours`**.
   - **Safety Margin:** At `12.94 hours`, C0 fits within the Colab UI reported session budget of **`36.33 hours`** with a **`2.81x`** safety ratio. Both arms fit within the budget with a **`1.40x`** safety ratio.
   - **Memory Stability:** Peak RAM during training reached **`4.87 GB`**, well below the system limit of `12.67 GB` (leaving `7.80 GB` headroom).
   - **Verdict:** **`CPU_FEASIBLE`**.

2. **Execution Blocker for Official C0 Launch:**
   - Google Drive is **unmounted** on the active Colab CPU VM (`/content/drive` is absent).
   - In accordance with `AGENTS.md` and R3 trust boundaries (`ba7be0d`), training checkpoints must persist to the authorized Google Drive location (`KLTN-Dataset/manifests/RQ1-TRN-02B/checkpoints/C0/`) with atomic write and readback verification.
   - To avoid catastrophic loss of 13 hours of compute upon VM disconnection, official training launch is **`BLOCKED`** pending operator mounting of Google Drive via the direct notebook UI.

---

## 2. Hardware & Environment Audit

Measurements taken on the live Google Colab CPU runtime (`colab-cpu`):
- **CPU Model:** `AMD EPYC 7B12`
- **Architecture / OS:** `x86_64` / `Linux 6.6.159+`
- **Logical vCPUs:** `2` (configured with `torch.set_num_threads(2)`)
- **Total System RAM:** `12.67 GB` (Available at baseline: `10.93 GB`)
- **Python / PyTorch:** Python `3.11` / PyTorch `2.11.0+cpu` (FP32 precision)
- **Disk Space (/):** `107.72 GB` total (`87.15 GB` available)

---

## 3. Real CPU Performance Benchmark (Step 2)

Command executed on Colab VM:
```bash
python3 audit_scripts/benchmark_cpu_rq1_trn_02b.py \
  --config configs/rq1_trn_02b_c0.yaml \
  --zip-path /content/data/fold1_train_val_only.zip \
  --initial-checkpoint /content/data/best.pt \
  --updates 50 \
  --val-batches 10
```

### 3.1 Timing & Throughput Breakdown (50 Successful Updates)

| Metric | Measured Value | Percentage of Step |
|---|---|---|
| **Forward Pass (2 micro-batches $\times$ batch 2)** | 4.063s | 33.5% |
| **Backward Pass (`loss.backward()`)** | 7.736s | 63.7% |
| **Optimizer Step & Gradient Clip** | 0.282s | 2.3% |
| **Data Loading Overhead** | 0.060s | 0.5% |
| **Total Update Time (Median)** | **12.538s** | **100.0%** |
| **Total Update Time (Mean $\pm$ StdDev)** | 12.140s $\pm$ 2.381s | — |
| **Min / Max Update Duration** | 6.485s / 18.326s | — |

### 3.2 Validation Benchmark (10 Measured Batches)
- **Mean Validation Batch Time:** `1.625 seconds` (batch size 2 utterances).
- **Validation Dataset Size:** 300 utterances (Speakers NJS: 150, RRBI: 150) = 150 batches per epoch.
- **Estimated Full Validation Pass per Epoch:** `243.7 seconds` (**`4.06 minutes`**).

### 3.3 Memory Footprint Tracking
- **Process Baseline RAM:** `0.465 GB`
- **RAM Post Model Load (Weights in FP32):** `1.221 GB` (+`0.756 GB`)
- **Peak RAM during 50-Update Training Loop:** `4.869 GB` (+`4.404 GB` net increase)
- **Available Headroom:** `7.80 GB` (no OOM risk detected)

---

## 4. Empirical Projections & Feasibility Evaluation

| Phase / Scenario | Updates | Computation | Validation | Total Estimated Duration |
|---|---|---|---|---|
| **1 Epoch** | 600 | 2.09 hours | 4.1 min | **2.16 hours** |
| **Control Arm C0 (6 Epochs)** | 3,600 | 12.54 hours | 24.4 min | **12.94 hours** |
| **Treatment Arm C1 (6 Epochs)** | 3,600 | 12.54 hours | 24.4 min | **12.94 hours** |
| **Both Arms (C0 + C1 Total)** | 7,200 | 25.08 hours | 48.7 min | **25.89 hours** |

### Feasibility Comparison Against Available Colab Budget
- **Reported Runtime Duration:** `36 hours and 20 minutes` (`36.33 hours`).
- **C0 Budget Safety Ratio:** $36.33 / 12.94 = \mathbf{2.81\times}$ (Safe margin for single-arm execution).
- **Both Arms Budget Safety Ratio:** $36.33 / 25.89 = \mathbf{1.40\times}$ (Viable if session remains uninterrupted).
- **Verdict:** **`CPU_FEASIBLE`**.

---

## 5. Execution Decision & Operational Prerequisite

### Protocol Deviation Notice
- **Execution Precision:** FP32 full-precision CPU execution instead of GPU FP16 AMP.
- **Backbone & Loss:** Preserved identical architecture (`LangfensWav2Vec2MDD`), frozen `wav2vec2-base-960h` backbone, AdamW optimizer (`lr=1e-3`, `wd=0.01`), cosine schedule with 100 warmup steps, gradient accumulation 2, batch size 2.
- **Contamination Prevention:** CPU FP32 checkpoints will record `"precision": "fp32"` and `"protocol_deviation": "CPU_FP32_TRAINING"` in their metadata ledger.

### Execution Gate: BLOCKED on Drive Mount
While CPU training is computationally viable, training cannot start without remote checkpoint persistence. On the current Colab CPU VM:
- `/content/drive` is **unmounted**.
- Writing to `/content/drive` without an active FUSE/DriveFS mount would write to local ephemeral disk, creating a severe risk of data loss.
- In accordance with R3 trust boundaries, execution halts cleanly before Step 09.

---

## 6. Restart & Operator Action Instructions

To mount Google Drive and launch official C0 CPU training:

### Step 1: Open the Active Session in Colab Web UI
Navigate to the direct attachment URL:
```text
https://colab.research.google.com/notebooks/empty.ipynb?dbu=%2Ftun%2Fm%2Fm-s-usw3b5-57ccj5ykg7oh#datalabBackendUrl=https://colab.research.google.com/tun/m/m-s-usw3b5-57ccj5ykg7oh
```

### Step 2: Mount Google Drive
In the notebook cell, run:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Authorize access for `tukhoa040505@gmail.com`. Confirm that `/content/drive/MyDrive/KLTN-Dataset` is visible.

### Step 3: Launch C0 Training
Once Drive is mounted, run the training command via SSH or in the notebook:
```bash
cd /content/repo
python3 audit_scripts/train_rq1_trn_02b.py \
  --config configs/rq1_trn_02b_c0.yaml \
  --zip-path /content/data/fold1_train_val_only.zip \
  --initial-checkpoint /content/data/best.pt \
  --drive-output-dir /content/drive/MyDrive/KLTN-Dataset/manifests/RQ1-TRN-02B/checkpoints/C0 \
  --allow-cpu
```
This will execute the 6-epoch / 3,600-update C0 run on CPU FP32, atomically saving and verifying checkpoints to Google Drive after each epoch.
