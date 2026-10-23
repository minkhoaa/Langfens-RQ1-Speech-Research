# LANGFENS — RQ1-TRN-00: Dependency & Runtime Verification Report

**Execution Gate:** RQ1-TRN-00 — Modeling Readiness & Reproducibility Preflight  
**Focus Area:** Task B — Dependency Inventory, Host Runtime, GPU Acceleration & Dataset Digest Audit  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Reproducibility Lead  
**Deliverable Associated Path:** `manifests/RQ1-TRN-00/DEPENDENCY_ENVIRONMENT_REPORT.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-00/`  
**Gate Evaluation Status:** `ENVIRONMENT_CONSISTENT: PASS` (CPU & Preflight Ready; GPU Container Plan Defined)  

---

## 1. Executive Summary & Host Hardware Specifications

This report documents the exact execution environment, hardware resources, GPU acceleration status, package dependency registry, and dataset archive digests for Project Langfens.

### 1.1 Host Hardware Topology
- **Host OS / Kernel:** CachyOS Linux x86_64 (`Linux khoa 7.2.11-1-cachyos #1 SMP PREEMPT_DYNAMIC Sun, 18 Oct 2026 14:54:49 +0000 x86_64`)
- **Base Distribution:** Arch Linux rolling derivative
- **Processor (CPU):** 12th Gen Intel(R) Core(TM) i5-12450H
- **Topology:** 8 Physical Cores (4 Performance Cores, 4 Efficiency Cores), 12 Logical vCPUs
- **Frequencies:** Base $400 \text{ MHz}$, Max Turbo $4,400 \text{ MHz}$
- **Caches:** L1d $320 \text{ KiB}$, L1i $384 \text{ KiB}$, L2 $7 \text{ MiB}$, L3 $12 \text{ MiB}$
- **System Memory (RAM):** 31.01 GiB Total ($18.0 \text{ GiB}$ used, $12.0 \text{ GiB}$ available, $35.0 \text{ GiB}$ swap configured)
- **Local NVMe Storage:** Partition `/dev/nvme0n1p3` mounted on `/home`
  - Total Size: $240 \text{ GiB}$
  - Used: $137 \text{ GiB}$ ($58\%$)
  - Free Available: **$101 \text{ GiB}$**

---

## 2. GPU & Acceleration Hardware Verification

### 2.1 Hardware Specification
- **GPU Model:** NVIDIA GeForce GTX 1650 Mobile / Max-Q
- **PCI Bus ID:** `00000000:01:00.0`
- **Total VRAM:** $4,096 \text{ MiB}$ ($4.0 \text{ GiB}$) GDDR6
- **NVIDIA Driver Version:** `615.78.08`
- **Host CUDA Driver / UMD Version:** `13.4`
- **Active Processes:** Single display/QEMU context ($120 \text{ MiB}$ VRAM consumed; $3,976 \text{ MiB}$ unallocated)

### 2.2 PyTorch CUDA Runtime Analysis & Strategic Findings
An empirical query of the active Python virtual environment yields:
```python
>>> import torch
>>> torch.__version__
'2.14.1+cpu'
>>> torch.cuda.is_available()
False
```

#### Root Cause Analysis:
1. The execution host runs bleeding-edge `Python 3.14.8` on CachyOS Linux.
2. Official upstream PyTorch binary wheels (`torch`, `torchaudio`) with CUDA acceleration currently support up to `Python 3.12` (with preview/nightly builds supporting `Python 3.13`). No pre-compiled PyTorch CUDA wheels exist for `Python 3.14`.
3. Consequently, the local `.venv` environment was provisioned with the CPU build `2.14.1+cpu`.

#### Impact on Experimental Phases:
- **Phase RQ1-TRN-00 (Current Preflight):** **ZERO IMPACT.** All preflight tasks (data preparation, annotation validation, TextGrid census reconciliation, loss contract testing, CPU throughput benchmarking) execute with 100% validity on CPU.
- **Phase RQ1-TRN-01+ (Acoustic Modeling & GPU Fine-Tuning):** Running full neural backbones (e.g. Wav2Vec 2.0 with $95\text{M}$ parameters) on GPU requires CUDA acceleration. For subsequent training phases, the team must provision a standard `Python 3.11` or `3.12` environment (or OCI container) equipped with `torch==2.x+cu12x` to harness the GTX 1650 GPU or cloud compute.

---

## 3. Python Runtime & Complete Package Inventory

The local virtual environment (`.venv`) is powered by Python 3.14.8. The exact installed package inventory (exported via `pip list` and locked in `requirements-lock.txt`) is cataloged below:

| Package | Installed Version | Role in Langfens Pipeline |
| :--- | :--- | :--- |
| `annotated-types` | `0.8.0` | Typing constraints for Pydantic models |
| `cffi` | `2.1.1` | Foreign function interface (backend for soundfile) |
| `filelock` | `3.32.3` | Multiprocess filesystem lock coordination |
| `fsspec` | `2026.7.0` | Filesystem abstraction layer |
| `iniconfig` | `2.3.1` | INI configuration parsing for pytest |
| `Jinja2` | `3.1.6` | Template engine (PyTorch dependency) |
| `MarkupSafe` | `3.0.3` | String escaping for Jinja2 |
| `mpmath` | `1.3.0` | Arbitrary-precision floating-point arithmetic |
| `networkx` | `3.6.1` | Graph algorithms (PyTorch torch.fx dependency) |
| `numpy` | `2.5.4` | Numerical array operations & evaluation matrices |
| `packaging` | `26.3` | Python version and package specification parsing |
| `pip` | `26.2.2` | Package installer |
| `pluggy` | `1.7.0` | Plugin management for pytest |
| `psutil` | `7.2.2` | Process monitoring & memory profiling |
| `pycparser` | `3.12` | C parser in Python |
| `pydantic` | `2.14.1` | Data contract schema definition & runtime validation |
| `pydantic_core` | `2.50.1` | High-performance Rust backend for Pydantic |
| `Pygments` | `2.21.0` | Syntax highlighting |
| `pytest` | `9.1.1` | Test framework for regression and contract suites |
| `scipy` | `1.18.1` | Scientific computing, signal processing, correlation metrics |
| `setuptools` | `78.1.0` | Build system backend |
| `soundfile` | `0.14.0` | Audio reading/writing (libsndfile binding) |
| `sympy` | `1.14.0` | Symbolic mathematics (PyTorch TorchDynamo dependency) |
| `torch` | `2.14.1+cpu` | Deep learning tensor framework & neural network modules |
| `torchaudio` | `2.11.0+cpu` | Speech audio I/O & polyphase sinc bandlimited resampling |
| `typing-inspection` | `0.4.4` | Runtime type analysis |
| `typing_extensions` | `4.16.0` | Backported standard typing constructs |

---

## 4. Cryptographic Provenance & Dataset Digest Verification

Before certifying environment readiness, all raw corpus archives and frozen split manifests were re-hashed via SHA-256 in chunked binary mode ($1 \text{ MB}$ chunks) and compared against their certified hashes:

### 4.1 Raw Corpus Archives

| Raw Dataset Archive | Expected SHA-256 Digest | Empirical Local Digest | Status |
| :--- | :--- | :--- | :---: |
| `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip` | `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda` | `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda` | **VERIFIED MATCH** |
| `datasets/SpeechOcean762/original/speechocean762.tar.gz` | `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5` | `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5` | **VERIFIED MATCH** |

### 4.2 Frozen Partition Split Manifests

| Split Manifest Path | Expected SHA-256 Digest | Empirical Local Digest | Status |
| :--- | :--- | :--- | :---: |
| `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | **VERIFIED MATCH** |
| `manifests/RQ1-SPLIT-01/l2arctic_loso_vietnamese_splits.json` | `7a59b63ad73c7616af2ef583760b50253c973090c4c7c76aadc46029cfe69118` | `7a59b63ad73c7616af2ef583760b50253c973090c4c7c76aadc46029cfe69118` | **VERIFIED MATCH** |
| `manifests/RQ1-SPLIT-01/speechocean762_official_splits.json` | `bc3f9ec97b5077402b3c67d89c857b9efe694649bef6e5d43776f07b528bad17` | `bc3f9ec97b5077402b3c67d89c857b9efe694649bef6e5d43776f07b528bad17` | **VERIFIED MATCH** |

---

## 5. Environment Reproducibility Determination

1. **Package Declarations:** Declared in standard `pyproject.toml` and pinned in `requirements-lock.txt`.
2. **Import Verification:** All 25 required libraries import with zero errors or warnings.
3. **Data Integrity:** 100% cryptographic concordance with zero byte tampering.
4. **Hardware Suitability:** CPU, RAM ($31\text{ GiB}$), and NVMe ($101\text{ GiB}$ free) are more than adequate for all preflight and data preparation operations.

**Phase Gate Evaluation:** **`ENVIRONMENT_CONSISTENT: PASS`**
