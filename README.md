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

## 4. Model Registry: `langfens-wav2vec2-mdd`

Model artifacts, checkpoints, and evaluations are managed via Hugging Face Hub under:
- **Registry ID:** `langfens-wav2vec2-mdd`
- **Visibility:** **Private**
- **Status:** **`PLANNED / NOT TRAINED`** (Architecture & Contract Smoke Testing Phase `RQ1-TRN-01A`)
- **Base Architecture:** `facebook/wav2vec2-base-960h` (95M parameters)
- **Deployment & Export:** Scheduled for Phase `RQ1-TRN-01B+` following authorized 4-fold cross-validation.

---

## 5. Licensing & Governance Distinction

This project operates under a strict dual-licensing structure:

1. **Source Code & Tooling:**
   Licensed under the permissive **MIT License**. All source files under `src/langfens_prep/`, `tests/`, and build configuration files may be reused in academic and open-source software.
2. **L2-ARCTIC Speech Corpus & Annotations:**
   The underlying L2-ARCTIC v5.0 corpus is copyrighted by its creators (Zhao et al., 2018) and licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. Derived acoustic models, phoneme embeddings, and diagnostic checkpoints are strictly restricted to **non-commercial academic and educational research**.
3. **SpeechOcean762 Dataset:**
   SpeechOcean762 is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
