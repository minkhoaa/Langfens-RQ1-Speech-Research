# LANGFENS — RQ1-PREP-01: CPU Micro-Benchmarking & Audio Pipeline Profiling Report

**Execution Gate:** RQ1-PREP-01 — Phase F CPU Micro-Benchmarking  
**Corpus Targets:** L2-ARCTIC Release v5.0 and SpeechOcean762  
**Deliverable Associated Path:** `manifests/RQ1-PREP-01/AUDIO_BENCHMARK_REPORT.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/`  
**Gate Status:** `PERFORMANCE_PROFILED: PASS`

---

## 1. Executive Summary & Host Hardware Specifications

This benchmark report characterizes the empirical I/O, resampling, collation, and DataLoader throughput of Project Langfens on the native host architecture.

### 1.1 Host Execution Environment
- **Host OS / Kernel:** Linux x86_64 (`Linux-7.2.11-1-cachyos-x86_64-with-glibc2.44`)
- **Processor (CPU):** 12th Gen Intel(R) Core(TM) i5-12450H
- **Topology:** 8 Physical Cores (4 Performance Cores, 4 Efficiency Cores), 12 Logical vCPUs
- **System Memory (RAM):** 31.01 GiB Total (15.24 GiB Available at benchmark time)
- **Software Runtime:**
  - Python: `3.14.8`
  - PyTorch: `2.14.1+cpu`
  - Torchaudio: `2.11.0+cpu`
  - SoundFile: `0.14.0`

---

## 2. Empirical Benchmark Measurements

All benchmarks were executed using 5 warm-up cycles followed by 50 timed repetitions on native audio samples.

### 2.1 Audio Access Latency: Direct Archive vs. NVMe Staging
We compared random-access reading of WAV files directly from nested in-memory ZIP handles against uncompressed local NVMe filesystem reads:

| Storage Access Mode | Median Latency (ms) | 95th Percentile ($p_{95}$) Latency (ms) | Observed Speedup |
| :--- | :---: | :---: | :---: |
| **Direct Nested ZIP Archive Extraction** | **1.5042 ms** | 2.2020 ms | 1.00× (Baseline) |
| **Staged Uncompressed NVMe File Read** | **0.0583 ms** | 0.0854 ms | **25.82× faster** |

*Finding:* Direct streaming from the 7.5 GB ZIP archive adds $\approx 1.45 \text{ ms}$ of decompression overhead per utterance compared to raw NVMe reads. However, it requires zero disk expansion and retains 100% immutable provenance.

### 2.2 Sinc Bandlimited Resampling Latency (44.1 kHz $\to$ 16 kHz)
Using `torchaudio.functional.resample` (polyphase sinc interpolation filter):
- **Sample Count:** 50 utterances
- **Median Latency:** **0.9921 ms** per utterance
- **95th Percentile ($p_{95}$):** **1.6528 ms** per utterance
- **Mean Latency:** **1.0491 ms** per utterance

### 2.3 Variable-Length Batch Collation Latency
Measuring padding of waveforms, binary label sequences (padding $-100$), CTC targets (padding $0$), and length mask tensors across different batch sizes:

| Batch Size | Median Collation Latency (ms) | 95th Percentile ($p_{95}$) Latency (ms) |
| :---: | :---: | :---: |
| **4** | 0.2387 ms | 0.2719 ms |
| **8** | 0.3723 ms | 0.4179 ms |
| **16** | 0.6833 ms | 0.7817 ms |

### 2.4 End-to-End PyTorch DataLoader Throughput
End-to-end evaluation including archive read, TextGrid parsing, sinc resampling (44.1k $\to$ 16k), and variable-length batch collation (Batch Size = 8, 100 samples per evaluation):

| Worker Configuration | Total Elapsed Time (s) | Effective Throughput (utts/sec) | Latency per Utterance (ms) | Peak Process RSS Memory (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **0 Workers (Main Process)** | **0.397 s** | **251.80 utts/sec** | **3.97 ms / utt** | **1,002.57 MB** |
| **2 Workers (Multiprocessing)** | 1.632 s | 61.27 utts/sec | 16.32 ms / utt | 1,003.79 MB |
| **4 Workers (Multiprocessing)** | 1.825 s | 54.80 utts/sec | 18.25 ms / utt | 1,004.07 MB |

---

## 3. Critical Analysis & Disambiguation of Prior Claims

### 3.1 Evaluation of Prior Claims
Prior informal project notes cited hypothetical baselines of "1.68 ms/utterance" and "2,380 utterances/second".
- **Scientific Audit Finding:** Those numbers are **NOT VERIFIED**. They do not represent real end-to-end CPU ingestion (decompression + parsing + bandlimited sinc resampling + collation).
- Achieving 2,380 utts/sec would require an end-to-end latency of $0.42 \text{ ms}$ per utterance, which is physically impossible on a standard CPU when high-quality sinc resampling alone requires $\approx 0.99 \text{ ms}$.

### 3.2 Multiprocessing Bottleneck Analysis
In Python 3.14 on Linux, the default multiprocessing start method is `forkserver`. When `num_workers > 0`:
1. Inter-process communication (IPC) serialization of variable-length tensors over shared-memory queues incurs substantial overhead.
2. The single-process reader (`num_workers = 0`) runs at **251.8 utts/sec** ($\approx 3.97 \text{ ms/utt}$), which is more than sufficient for CPU-based data preparation and pipeline validation.

### 3.3 Recommended Strategy for Model Training (RQ1-TRN-01)
For GPU training phases where input pipeline saturation is required:
1. Use `AudioStagingManager` to extract an ephemeral staging cache of 16 kHz resampled uncompressed WAVs onto NVMe storage.
2. This decouples dataset workers from the ZIP archive reader and reduces file read latency to $0.058 \text{ ms}$, raising potential I/O throughput by $\mathbf{25.8\times}$.
3. Raw archives remain completely untouched and immutable.
