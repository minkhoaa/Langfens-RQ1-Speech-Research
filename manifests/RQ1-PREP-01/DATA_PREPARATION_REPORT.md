# LANGFENS — RQ1-PREP-01: Non-Destructive Data Preparation & Pipeline Validation Report

**Execution Gate:** RQ1-PREP-01 — Pipeline Implementation & Data Preparation  
**Focus Area:** Production Implementation, Validation & CPU Micro-Benchmarking  
**Author / Engineering Lead:** Senior Speech ML Engineer & Research Reproducibility Lead  
**Deliverable Associated Path:** `manifests/RQ1-PREP-01/DATA_PREPARATION_REPORT.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/`  
**Overall Phase Status:** `RQ1-PREP-01: PASS`

---

## 1. Executive Summary & Objective

Phase `RQ1-PREP-01` delivers the production-ready, read-only data ingestion and preparation pipeline for Project Langfens (Phoneme-level Mispronunciation Detection & Diagnosis). 

In accordance with strict reproducibility mandates:
1. **Scope:** **DATA PREPARATION ONLY.** No acoustic model training, fine-tuning, GOP inference, forced alignment, or deployment was initiated.
2. **Immutability:** Raw archive files (`l2arctic_release_v5.0.zip` and `speechocean762.tar.gz`) and frozen split manifests (`manifests/RQ1-SPLIT-01/*.json`) remained 100% untouched.
3. **Census Reconciliation:** All 3,599 gold manual Praat TextGrids of L2-ARCTIC v5.0 (135,890 total intervals, 119,151 canonical phonemes) were exhaustively parsed and reconciled against `manifests/RQ1-STD-01/reconciliation/CORRECTED_LABEL_ACCOUNTING.md` with **zero unexplained variance**.
4. **SpeechOcean762 Compliance:** The 245 unmanifested audio files were strictly excluded from supervised ingestion; continuous scoring metadata across 5,000 utterances was preserved without discrete category fabrication.
5. **Acoustic Engineering:** Safe, bounded-memory audio access was implemented with float32 conversion, polyphase sinc bandlimited resampling (44.1 kHz $\to$ 16 kHz), and variable-length batch collation.
6. **Empirical Benchmarks:** Real CPU throughput on the execution host was profiled at **251.8 utterances/sec** (3.97 ms/utt), establishing an honest scientific baseline.

---

## 2. Software Architecture & Implemented Modules

All data preparation code is organized into a modular, version-controlled package under `src/langfens_prep/`:

```text
src/langfens_prep/
├── __init__.py               # Package metadata and namespace initialization
├── provenance.py             # Cryptographic split and raw archive SHA-256 verification
├── phoneme_normalizer.py     # 39-ARPAbet mapping, stress stripping, AX folding, typo & IPA handling
├── textgrid_parser.py        # Robust Praat TextGrid parser, interval taxonomy, strict/lenient logic
├── speechocean_parser.py     # SpeechOcean762 scores.json & Kaldi metadata parser
├── audio_loader.py           # Safe nested ZIP/tar streaming, 16kHz resampling, staging abstraction
├── dataset.py                # Lightweight PyTorch Dataset adapters with test-label blindness
├── collate.py                # Variable-length batch collator (-100 padding, CTC 0-blank padding)
├── validator.py              # Full 3,599-utterance census runner and split isolation auditor
└── benchmark.py              # CPU micro-benchmarking suite for I/O, resampling, and collation
```

---

## 3. Cryptographic Provenance & Manifest Integrity

Every input artifact was verified via SHA-256 before pipeline execution:

| Artifact Path | Expected SHA-256 | Empirical SHA-256 | Verification Status |
| :--- | :--- | :--- | :---: |
| `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip` | `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda` | `490d5a43...` | **MATCH** |
| `datasets/SpeechOcean762/original/speechocean762.tar.gz` | `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5` | `876c34b8...` | **MATCH** |
| `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | `7d6bfdf6...` | **MATCH** |
| `manifests/RQ1-SPLIT-01/l2arctic_loso_vietnamese_splits.json` | `7a59b63ad73c7616af2ef583760b50253c973090c4c7c76aadc46029cfe69118` | `7a59b63a...` | **MATCH** |
| `manifests/RQ1-SPLIT-01/speechocean762_official_splits.json` | `bc3f9ec97b5077402b3c67d89c857b9efe694649bef6e5d43776f07b528bad17` | `bc3f9ec9...` | **MATCH** |

### Split Provenance Erratum Adjudication
- **Finding:** Manifest metadata header recorded `"generation_timestamp_utc": "2026-10-23T13:30:00Z"`, whereas Google Drive sync occurred at `2026-10-23 13:26:52 +07:00` (`06:26:52Z` UTC).
- **Cause:** `audit_scripts/generate_rq1_splits.py:28` statically assigned `13:30` (local planning time) with literal `Z` without host timezone (+07:00) offset conversion.
- **Resolution:** Frozen split manifests remain untouched. Historical UTC header timestamp is marked **`NOT_VERIFIED`** in `SPLIT_PROVENANCE_TIME_ERRATUM.md`, while data payload digests (`payload_sha256`) are certified as **`VERIFIED_MATCH`**.

---

## 4. Full Annotation Census & Reconciled Label Accounting

Validation across all 3,599 manual gold TextGrids yielded an exact match against the theoretical accounting model:

```text
Global Praat TextGrid Census (135,890 Total Intervals)
├── 1. Non-Canonical Segments (16,739 intervals)
│   ├── Acoustic Silence (sil, sp, spn, ""): 15,647 intervals (11.5145%)
│   └── Epenthetic Additions (sil,PPL,a): 1,092 intervals (0.8036%)
│
└── 2. Canonical Prompt Tokens (N = 119,151 intervals; 100.0000%)
    ├── Correct Phone Productions (C): 101,633 tokens (85.2976%)
    ├── Phonemic Deletions (D): 3,420 tokens (2.8703%)
    └── Categorical Substitutions (S): 14,098 tokens (11.8320%)
        ├── Clean Substitutions (Diff Base): 12,364 tokens (87.7004% of S)
        ├── Clean Substitutions (Same Base / Stress / Typo): 4 tokens (0.0284% of S)
        ├── Base-Preserving Asterisk Allophones (CPL,CPL*,s): 1,112 tokens (7.8876% of S)
        ├── Category-Shifting Accented Substitutions (CPL,PPL*,s): 360 tokens (2.5536% of S)
        └── Distorted / Uncertain Realizations (CPL,err,s): 258 tokens (1.8300% of S)
```

### Policy Accounting
- **Strict Binary MDD Policy:**
  - $C_{\text{strict}} = 101,633$ ($85.30\%$), $E_{\text{strict}} = 17,518$ ($14.70\%$). Imbalance: $5.80 : 1$.
- **Lenient Pedagogical Sensitivity Policy:**
  - $C_{\text{lenient}} = 101,633 + 1,112 = 102,745$ ($86.23\%$).
  - $E_{\text{lenient}} = (14,098 - 1,112) + 3,420 = 16,406$ ($13.77\%$). Imbalance: $6.26 : 1$.
- **Invariant Denominator:**
  - $N_{\text{canonical}} = C_{\text{strict}} + E_{\text{strict}} = C_{\text{lenient}} + E_{\text{lenient}} = \mathbf{119,151}$ ($\Delta = 0$).

---

## 5. Acoustic Pipeline & Micro-Benchmarking

Empirical profiling on the 12th Gen Intel Core i5-12450H CPU (12 vCPUs, 31 GiB RAM):
- **WAV Ingestion:** Direct ZIP streaming requires $1.50 \text{ ms/utt}$; local NVMe staging requires $0.058 \text{ ms/utt}$ ($25.8\times$ faster).
- **Polyphase Sinc Resampling (44.1k $\to$ 16k):** Median $0.99 \text{ ms/utt}$, $p_{95} = 1.65 \text{ ms/utt}$.
- **Batch Collation:** Variable-length padding for batch size 8 takes $0.37 \text{ ms}$.
- **PyTorch DataLoader:** Single-process throughput achieves **251.80 utts/sec** ($3.97 \text{ ms/utt}$), with stable peak memory of $1,002.6 \text{ MB}$.

---

## 6. Formal Acceptance Gate Evaluation

| Gate ID | Evaluation Standard | Status | Scientific Evidence |
| :--- | :--- | :---: | :--- |
| `ANNOTATION_PARSER_VALIDATED` | Full TextGrid & scores.json parsing | **PASS** | 3,599 TextGrids & 5,000 score records parsed with zero parse failures. |
| `LABEL_SEMANTICS_VALIDATED` | Strict/Lenient MDD & error taxonomy | **PASS** | $N=119,151$ invariant; exact match on all 14,098 substitutions and 3,420 deletions. |
| `AUDIO_PIPELINE_VALIDATED` | Sample rate, channels, float32, sinc resampling | **PASS** | High-quality 16 kHz mono conversion; zero NaNs/Infs; safe archive streaming. |
| `MANIFEST_ISOLATION_VALIDATED` | Partition boundaries & zero test leakage | **PASS** | 0 cross-partition overlap; 245 unmanifested files excluded; blinded test mode. |
| `DATALOADER_VALIDATED` | PyTorch Dataset & batch collator contracts | **PASS** | Padded waveforms, -100 label padding, CTC 0-blank padding pass integration tests. |
| `PERFORMANCE_PROFILED` | Real CPU micro-benchmarks documented | **PASS** | Profiling completed (251.8 utts/s); unverified baselines disproved. |
| `PROVENANCE_DOCUMENTED` | Checksums, timestamp erratum & audit trail | **PASS** | SHA-256 verified; erratum documented with UTC+7 root cause analysis. |

**Final Phase Determination:** **`RQ1-PREP-01: PASS`**
