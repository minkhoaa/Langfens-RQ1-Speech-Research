# Project Langfens — RQ1 Speech Research

**Phoneme-Level Mispronunciation Detection & Diagnosis (MDD) Pipeline, Reproducibility Audit, and Modeling Readiness**

[![CI / Regression Tests](https://img.shields.io/badge/pytest-passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange)](requirements-lock.txt)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY--NC%204.0-lightgrey)](references/licenses/CC-BY-NC-4.0.txt)

---

## 1. Overview & Research Objectives

Project Langfens investigates phoneme-level **Mispronunciation Detection & Diagnosis (MDD)** for Computer-Assisted Pronunciation Training (CAPT), specializing in Second Language (L2) English acquisition with high-fidelity diagnosis for Vietnamese and international learners.

Research Focus:
- **Corpus Target:** L2-ARCTIC v5.0 (24 non-native speakers across 6 L1s: Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese) and SpeechOcean762 (continuous scoring benchmark).
- **Primary Metric:** Strict Binary **Precision-Recall AUC (PR-AUC)** evaluated over the strictly invariant denominator of **$N_{\text{canonical}} = 119,151$** prompt phoneme tokens ($C = 101,633$, $E_{\text{strict}} = 17,518$).
- **Phonemic Error Breakdown:**
  - Categorical Substitutions ($S = 14,098$, with $S_{\text{eval}} = \mathbf{13,840}$ evaluable 39-class substitutions, masking $258$ unalignable `'err'` distortions).
  - Phonemic Deletions ($D = 3,420$, including mandatory tracking of the Vietnamese cohort $D_{\text{Viet}} = \mathbf{1,566}$; $45.79\%$ of corpus deletions).
  - Epenthetic Additions ($I = 1,092$, evaluated via 50 ms acoustic window matching).
- **Core Modeling Paradigm:** End-to-end multi-task Wav2Vec 2.0 acoustic modeling combining:
  1. Realized phoneme CTC sequence recognition,
  2. Token-level canonical binary mispronunciation detection,
  3. 39-class substitution diagnosis.

---

## 2. Repository Architecture

```text
Langfens-RQ1-Speech-Research/
├── configs/                  # Hugging Face registry & experiment YAML configurations
│   ├── huggingface_registry.yaml
│   └── huggingface_model_card.md
├── src/langfens_prep/        # Certified data preparation & model architecture package
│   ├── __init__.py           # Package namespace
│   ├── provenance.py         # Cryptographic SHA-256 verification of raw archives and splits
│   ├── phoneme_normalizer.py # 39-ARPAbet mapping, index dictionaries, reversible normalization
│   ├── textgrid_parser.py    # Praat TextGrid interval parser, strict/lenient policy parsing
│   ├── speechocean_parser.py # SpeechOcean762 scoring parser
│   ├── audio_loader.py       # Polyphase sinc bandlimited resampling (16 kHz), staging manager
│   ├── dataset.py            # PyTorch Datasets with blinded test mode
│   ├── collate.py            # Variable-length batch collator (-100 padding, explicit binary masks)
│   ├── losses.py             # MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, CTC validators
│   ├── model.py              # LangfensWav2Vec2MDD multi-task acoustic architecture
│   ├── validator.py          # Full 3,599-utterance census runner
│   └── benchmark.py          # Real CPU I/O and DataLoader benchmarking suite
├── tests/                    # Comprehensive unit, integration, contract & smoke tests
├── manifests/                # Audit manifests, data contracts, and preflight reports
│   ├── RQ1-AUD-01/           # Corpus integrity & phonetic inventory audits
│   ├── RQ1-STD-01/           # Label accounting & standardization specifications
│   ├── RQ1-SPLIT-01/         # Frozen 4-fold and Vietnamese LOSO/zero-shot manifests
│   ├── RQ1-PREP-01/          # Data preparation reports & CPU profiling benchmarks
│   ├── RQ1-TRN-00/           # Preflight readiness & experimental protocol freeze
│   └── RQ1-TRN-01A/          # Training contract hardening & GPU smoke test reports
├── pyproject.toml            # PEP 517/621 project configuration
├── requirements.txt          # Core dependencies specification
├── requirements-lock.txt     # Pinned reference environment lockfile
└── .gitignore                # Protection against large datasets, audio, and checkpoints
```

---

## 3. Dual Execution Environments

To balance bleeding-edge host compatibility with GPU training stability, the repository supports two verified environments:

| Environment | Python | PyTorch | Device | Primary Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **`.venv`** | `3.14.8` | `2.14.1+cpu` | CPU | Clean host development, data preparation, fast static checks |
| **`.venv-cuda`** | `3.11.17` | `2.14.1+cu130` | GPU (GTX 1650) | CUDA tensor operations, AMP fp16, gradient checkpointing, model training |

### Setup Instructions:
```bash
# 1. CPU Reference Environment (Python 3.14)
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e .

# 2. GPU Training Environment (Python 3.11 with CUDA 12/13)
# Using uv for non-destructive isolated provisioning:
.venv/bin/uv venv --python 3.11 .venv-cuda
.venv/bin/uv pip install --python .venv-cuda/bin/python torch torchaudio numpy scipy soundfile pydantic pytest psutil
```

### Running Test Suites:
```bash
# Run all tests in CPU reference environment
.venv/bin/pytest -v

# Run all tests in CUDA-enabled GPU environment
.venv-cuda/bin/pytest -v
```

---

## 4. Model Registry & Released Artifacts

Model artifacts, checkpoints, and release manifests are managed across Hugging Face Hub, Google Drive, and GitHub:

- **Hugging Face Private Model Repository:** [`https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r`](https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r)
  - **Visibility:** **Private**
  - **Release Version:** `baseline-r-v0.1.0` (Commit: `6c31f7196c2e03a6766c5525833de465a1a9345d`)
  - **Format:** Native lossless FP32 `model.safetensors` (SHA-256: `a5b9d41f1431ad7a8b9718b4d84a7886e39fe5aff45fe742fbe572819f8babc1`)
  - **Status:** **`RELEASED (RESEARCH BASELINE)`** — Multi-task CTC Recognition + Binary MDD (Fold 1 Baseline R)
  - **Validation Performance:** Binary AP **0.316062** (2.72× prevalence), Greedy CTC PER **23.64%**
- **Google Drive Baseline Archive:** [`https://drive.google.com/drive/folders/11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx`](https://drive.google.com/drive/folders/11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx)
  - Stored under project root folder ID `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK` (`KLTN-Dataset`)
  - Remote manifest: [`BASELINE_R_ARTIFACT_MANIFEST.json`](https://drive.google.com/file/d/1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP/view) (SHA-256: `6222c92b63ea58f584267b39fb3ffc51394562086f90de21bbab8c8c1abe1e22`)
  - Checkpoint: `best.pt` (SHA-256: `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`)
- **Baseline Release Manifest:** [`manifests/RQ1-ARTIFACT-01/BASELINE_RELEASE_MANIFEST.md`](manifests/RQ1-ARTIFACT-01/BASELINE_RELEASE_MANIFEST.md)
  - Detailed cross-platform cryptographic checksums, CPU parity verification, and sealed partition guarantees
- **Base Architecture:** `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, 94.4M parameters, frozen)
- **Official Test Status:** Strictly sealed (900 utterances held out across 6 speakers; zero test access)

---

## 5. Licensing & Governance Distinction

This project operates under a strict dual-licensing structure:

1. **Source Code & Tooling:**
   Licensed under the permissive **MIT License**. All source files under `src/langfens_prep/`, `tests/`, and build configuration files may be reused in academic and open-source software.
2. **L2-ARCTIC Speech Corpus & Annotations:**
   The underlying L2-ARCTIC v5.0 corpus is copyrighted by its creators (Zhao et al., 2018) and licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. Derived acoustic models, phoneme embeddings, and diagnostic checkpoints are strictly restricted to **non-commercial academic and educational research**.
3. **SpeechOcean762 Dataset:**
   SpeechOcean762 is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
