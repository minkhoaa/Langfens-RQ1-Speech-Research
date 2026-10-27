# HUGGING FACE RELEASE REPORT — LANGFENS RQ1-ARTIFACT-01

**Date:** 2026-10-27  
**Artifact ID:** `RQ1-ARTIFACT-01`  
**Hugging Face Repository:** [`minkhoaa/langfens-rq1-fold1-baseline-r`](https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r)  
**Visibility:** **PRIVATE** (Verified)  
**Release Tag:** `baseline-r-v0.1.0`  
**Commit SHA:** `6c31f7196c2e03a6766c5525833de465a1a9345d`  
**Status:** PASS — FULL RELEASE & INFERENCE REPRODUCIBILITY VERIFIED

---

## 1. Authentication & Namespace Verification

- **Authenticated Namespace:** `minkhoaa` (Minh Khoa)
- **Account Type:** User (`type: 'user'`, ID: `69ae5099442941751f4e0ead`)
- **Authentication Method:** OAuth / Token via official `huggingface_hub` and `hf` CLI
- **Repository Visibility:** Strictly configured and verified as **PRIVATE** (`private=True`)

---

## 2. Remote Repository File Inventory

The repository contains the following verified artifacts at commit `6c31f7196c2e03a6766c5525833de465a1a9345d`:

| File Name | File Size (Bytes) | SHA-256 Digest | Role / Purpose |
|---|---|---|---|
| `model.safetensors` | 389,650,424 | `a5b9d41f1431ad7a8b9718b4d84a7886e39fe5aff45fe742fbe572819f8babc1` | Native lossless FP32 model weights |
| `config.json` | 1,395 | `c4fb33bd9d8acd335064fff4e1347e1089c38df8c10ac92de4ac7b1049f372c7` | Architecture hyperparameters & head metadata |
| `training_metadata.json` | 2,099 | `c95cdd393bf1c3c12ee7eb57b641e2091b00e46fceb3e3c915008ae2fa3d4c6f` | Training provenance, seed, optimizer, updates |
| `evaluation_results.json` | 2,246 | `27e587fa60d76b2c236efcc648e15b120633c4a9489949ab54c88b826f3f58c8` | Verified validation metrics (Binary AP, PER) |
| `phone_inventory.json` | 3,629 | `942f9c4fcb3a1d7c8f3b8f3fbbb6805fea411800bbbb008a66da81b7fe1ec818` | Standardized 39 ARPAbet and CTC phone indices |
| `example_inference.py` | 5,703 | `89173fa53ca2e169fb85570dff014ec8d1b2f83600a7ac1bc6b7ba4ac1adb18c` | Self-contained, runnable CPU inference loader |
| `README.md` | 9,630 | `385be1c2ab667a75cd8cfa20de911a51c587b69cc17da5f24dc1ae7d1eb3bc0e` | Complete Hugging Face Model Card |
| `ARTIFACT_MANIFEST.json` | 1,452 | `4e832c25b648d44d3d4883a2ceac3524c7bc0a824ad563c38049f32d358cf1a6` | Checksum inventory of all release files |

*Excluded Files Audit:* Zero raw waveform archives, zero TextGrids, zero user credentials, zero optimizer state histories, and zero test split labels were staged or uploaded.

---

## 3. Post-Upload Verification Protocol

To satisfy release requirements, verification was conducted beyond CLI exit codes:

### Step 1: Remote Metadata Query
- Queried repository info via `huggingface_hub.HfApi().repo_info()`.
- Verified `private == True`.
- Verified commit SHA `6c31f7196c2e03a6766c5525833de465a1a9345d`.
- Verified git tag `baseline-r-v0.1.0`.

### Step 2: Fresh Temporary Download & Checksum Validation
- Downloaded `model.safetensors` using `hf_hub_download` into a clean isolated temporary directory.
- Re-computed SHA-256 digest:
  ```
  Expected:   a5b9d41f1431ad7a8b9718b4d84a7886e39fe5aff45fe742fbe572819f8babc1
  Downloaded: a5b9d41f1431ad7a8b9718b4d84a7886e39fe5aff45fe742fbe572819f8babc1
  Status:     MATCH
  ```

### Step 3: Architecture Reconstruction & Strict Weight Loading
- Instantiated `LangfensWav2Vec2MDD` with frozen backbone (`facebook/wav2vec2-base-960h` @ `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`).
- Loaded downloaded `model.safetensors` via `safetensors.torch.load_model(model, path, strict=True)`.
- Verified load result:
  ```
  Missing Keys:    set() (0 missing)
  Unexpected Keys: []    (0 unexpected)
  ```

### Step 4: CPU Inference Execution
- Executed `example_inference.py` using CPU forward inference.
- Verified greedy CTC collapse and canonical phoneme mispronunciation probability calculation.
- Confirmed error-free forward execution.

### Step 5: Full Validation Numerical Parity
- Re-evaluated all 300 Fold 1 validation utterances (75 batches) comparing original `best.pt` with exported `model.safetensors`:
  ```
  Maximum Absolute CTC Logit Difference:    0.000000e+00
  Maximum Absolute Binary Logit Difference: 0.000000e+00
  Parity Gate: PASS
  ```

---

## 4. Release Status Conclusion

- Hugging Face Repository Visibility: **PRIVATE**
- Exported Weights Checksum: **VERIFIED**
- CPU Inference Loader: **PASS**
- Numerical Parity: **PASS (100% BIT-FOR-BIT EQUIVALENCE)**
- Release Revision: **`baseline-r-v0.1.0`**
