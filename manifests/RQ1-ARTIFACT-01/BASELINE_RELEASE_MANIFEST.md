# BASELINE RELEASE MANIFEST — LANGFENS RQ1-ARTIFACT-01

**Artifact ID:** `RQ1-ARTIFACT-01`  
**Experiment / Arm:** `RQ1-TRN-02A` (Fold 1 Baseline R)  
**Release Version:** `baseline-r-v0.1.0`  
**Release Date:** 2026-10-27  
**Status:** RELEASED (PRIVATE) & FULLY VERIFIED  
**Auditor / Role:** Senior Machine Learning Engineer, MLOps Release Engineer & Scientific Reproducibility Auditor

---

## 1. Executive Summary

This manifest records the formal archival, serialization, private publication, and three-platform cross-verification of **LANGFENS RQ1 Fold 1 Baseline R**. 

Baseline R serves as the multi-task neural foundation for the LANGFENS research program. It establishes empirical acoustic conditioning performance prior to Task C prompt-dropout ablations (RQ1-TRN-02B). All binaries and metadata have been synchronized across Google Drive, Hugging Face, and GitHub with verified byte-for-byte cryptographic checksums.

---

## 2. Pinned Three-Platform References

| Platform | Resource Type | Identifier / URL | Access / Visibility |
|---|---|---|---|
| **Hugging Face** | Model Repository | [`https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r`](https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r) | **PRIVATE** (`baseline-r-v0.1.0`) |
| **Google Drive** | Baseline Archive Folder | [`https://drive.google.com/drive/folders/11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx`](https://drive.google.com/drive/folders/11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx) | Authorized Workspace |
| **Google Drive** | Project Root Folder | [`https://drive.google.com/drive/folders/1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`](https://drive.google.com/drive/folders/1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK) | `KLTN-Dataset` |
| **Google Drive** | Baseline Artifact Manifest | [`https://drive.google.com/file/d/1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP/view`](https://drive.google.com/file/d/1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP/view) | `BASELINE_R_ARTIFACT_MANIFEST.json` |
| **GitHub** | Repository HEAD | `minkhoaa/Langfens-RQ1-Speech-Research` | Branch: `docs/rq1-artifact-01-baseline-release` |
| **GitHub** | Reviewed Execution Source | Commit SHA: `8486cbb3596a0f5d2603dcdf46fd891b5c849840` | Clean Reference Base |
| **GitHub** | Training Run Source | Commit SHA: `caa56783113d40eac438e5f458cdb313b8932218` | Recorded in `best.pt` metadata |

---

## 3. Cryptographic Checksum Inventory

### Source Training Checkpoints (Google Drive: `manifests/RQ1-TRN-02A/checkpoints/`)

| File | File Size (Bytes) | MD5 Digest | SHA-256 Digest | Google Drive ID |
|---|---|---|---|---|
| `best.pt` | 412,366,029 | `70c16eff09235e85b4cebc056113994a` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | `10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I` |
| `last.pt` | 412,367,181 | `fa52f658103942bb36bbb530f60b2ca2` | `bf46fab366363d46082082cf7f088b218f6d8d12851f5a9de706276b789efcb6` | `1VdlhoQHfd9MqjzIkMUEqdqMxhLlJiO-s` |
| `history.json` | 8,819 | `6ef71d8b85a8137d971e0729c7e41ade` | `270dab59626bf665c0e52964b6f3c5a90e46fbaede856df987046c641dbceff5` | `128mkF6EFTHl7ztiVVy56rUEwmoqlKy-M` |
| `EVALUATION_REPORT.json` | 5,611 | `2d87afc5ee889b0d8e5a0fb327ddedb3` | `e4d093dd48bcfa9ecc0c7a2f452a5db9a86ff92b7e6702099cc504c85611778f` | `1vehhJp8Q23uIFY7hIEtjrIZ_czFki1wV` |
| `BASELINE_COMPARISON.json` | 1,227 | `106547eac8779477220fae3c6d107ce1` | `e08f4328115e968b8dd2f18979e660ef5a89d884b2e3e6449ef6b050597f11f9` | `1PR5HLpyCJMNYM-TblNqTOPnpg0EQaNmQ` |
| `BASELINE_R_ARTIFACT_MANIFEST.json` | 7,486 | `b7be3b64c7fe0a4f5f5fe844ebf32b85` | `6222c92b63ea58f584267b39fb3ffc51394562086f90de21bbab8c8c1abe1e22` | `1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP` |

### Authorized Dataset & Manifest Inventory

| File | File Size (Bytes) | MD5 Digest | SHA-256 Digest | Location / ID |
|---|---|---|---|---|
| `l2arctic_4fold_splits.json` | 448,900 | `06eb421389eb544c005090114eb6a6d0` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | Drive `1vlQtqZqGAQUdEY0vQxGLZW3U7C9cdfbQ` |
| `fold1_train_val_only.zip` | 761,922,918 | `fec058e897d0da739127ba79bfef521a` | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | Drive `1xdypDWopd585OjuQeHjxlWOJfU2oiJiY` |

### Exported Hugging Face Release Package (`minkhoaa/langfens-rq1-fold1-baseline-r`)

| File | File Size (Bytes) | SHA-256 Digest | Status |
|---|---|---|---|
| `model.safetensors` | 389,650,424 | `a5b9d41f1431ad7a8b9718b4d84a7886e39fe5aff45fe742fbe572819f8babc1` | Verified via download & parity |
| `config.json` | 1,395 | `c4fb33bd9d8acd335064fff4e1347e1089c38df8c10ac92de4ac7b1049f372c7` | Verified |
| `training_metadata.json` | 2,099 | `c95cdd393bf1c3c12ee7eb57b641e2091b00e46fceb3e3c915008ae2fa3d4c6f` | Verified |
| `evaluation_results.json` | 2,246 | `27e587fa60d76b2c236efcc648e15b120633c4a9489949ab54c88b826f3f58c8` | Verified |
| `phone_inventory.json` | 3,629 | `942f9c4fcb3a1d7c8f3b8f3fbbb6805fea411800bbbb008a66da81b7fe1ec818` | Verified |
| `example_inference.py` | 5,703 | `89173fa53ca2e169fb85570dff014ec8d1b2f83600a7ac1bc6b7ba4ac1adb18c` | Verified runnable on CPU |
| `README.md` | 9,630 | `385be1c2ab667a75cd8cfa20de911a51c587b69cc17da5f24dc1ae7d1eb3bc0e` | Verified model card |
| `ARTIFACT_MANIFEST.json` | 1,452 | `4e832c25b648d44d3d4883a2ceac3524c7bc0a824ad563c38049f32d358cf1a6` | Verified root manifest |

---

## 4. Model Architecture & Head Configuration

- **Class:** `LangfensWav2Vec2MDD` (`src/langfens_prep/model.py`)
- **Pretrained Backbone:** `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, 94,396,320 parameters). The backbone feature encoder and transformer context network remained frozen (`requires_grad=False`) throughout training.
- **Alignment Module:** `CanonicalAcousticCrossAttention` (4 attention heads, 128 hidden dim) mapping variable-length frame representations to prompt phonemes without TextGrid gold timestamps.
- **Trained Heads:**
  1. `ctc_head`: Linear projection ($768 \to 40$ classes: index 0 `<BLANK>`, 1..39 standard ARPAbet phonemes).
  2. `binary_head`: Multi-layer perceptron with GELU activations ($768 + 128 \to 256 \to 1$ logit) predicting phone-level mispronunciation probability.
- **Untrained / Frozen Head:**
  - `diagnosis_head`: Linear projection ($768 \to 256 \to 39$ classes). Kept frozen during Baseline R.

---

## 5. Verified Performance Benchmarks (Protocol O)

Evaluation was performed on the full Fold 1 validation partition (300 utterances, 9,956 phonemes across non-native speakers `NJS` and `RRBI`):

- **Validation Binary Average Precision (AP):** **0.316062**
- **Validation Prevalence Baseline:** **0.116211** (AP Ratio: **2.72×**)
- **Empirical Canonical Prior AP:** **0.303063** (prompt-only lookup without acoustic features)
- **Acoustic AP Delta over Prior:** **+0.012999** (statistically authentic acoustic dependence)
- **Optimal F1 Operating Point ($\tau = 0.20$):** Precision 0.2911, Recall 0.5367, **F1 0.3775**
- **Auxiliary CTC Greedy Phoneme Error Rate (PER):** **0.236413 (23.64%)**

---

## 6. Numerical Parity Verification

Forward passes were executed on all 300 validation utterances comparing the source PyTorch checkpoint (`best.pt`) against the exported native `model.safetensors` on CPU:

- **Maximum Absolute CTC Logit Difference:** `0.000000e+00`
- **Maximum Absolute Binary Logit Difference:** `0.000000e+00`
- **Missing Parameters:** `set()` (Zero)
- **Unexpected Parameters:** `[]` (Zero)
- **Status:** **PASS (100% BIT-FOR-BIT EQUIVALENCE)**

---

## 7. Safety, Sealing & Governance Conformance

1. **Official Test Sealing:** The official 900-utterance test partition was strictly sealed. Zero test audio files or labels were staged or accessed.
2. **Execution Boundary:** All model training occurred on Google Colab GPU runtimes. No training was executed locally.
3. **Immutability:** Original Google Drive checkpoints (`best.pt`, `last.pt`) remain completely untouched.
4. **Visibility:** Hugging Face repository visibility is verified **PRIVATE**.
