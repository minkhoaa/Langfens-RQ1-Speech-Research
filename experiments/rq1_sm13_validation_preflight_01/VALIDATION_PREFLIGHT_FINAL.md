# LANGFENS RQ1-SM13-VALIDATION-PREFLIGHT-01 — Final Validation Readiness Audit Report

**Role:** Independent Speech ML Validation Lead & Scientific Auditor  
**Campaign ID:** `LANGFENS-RQ1-SM13-VALIDATION-PREFLIGHT-01`  
**Target Candidate Artifact:** `LANGFENS-SM13-FUSION-V1` (Primary Seed 101, Replication Seeds 202/303)  
**Governing Standard:** EVAL-V3.1 ($N$ targets, $N+1$ exact insertion gaps)  
**Execution Environment:** x86_64 Linux Host, NVIDIA GeForce GTX 1650 (4096 MiB VRAM), PyTorch 2.6.0+cu124, CUDA 12.4  
**Date of Audit Certification:** November 11, 2026  

---

## 1. Executive Summary & Final Authorization Review

An exhaustive, independent scientific and engineering preflight audit was performed on `LANGFENS-SM13-FUSION-V1` artifacts formally trained under campaign `RQ1-SM13-FORMAL-TRAIN-01`.

The audit was executed across three independent validation lanes:
- **Phase 0:** Cryptographic artifact recovery, checkpoint loading, and source lineage verification.
- **Lane A:** Operational warning policy transfer, score distribution divergence, and FWR discrepancy audit.
- **Lane B:** Authoritative split definitions, speaker disjointness, `LXC` identity overlap, and prior exposure analysis.
- **Lane C:** Complete end-to-end inference benchmarking from raw WAV without oracle features or precomputed alignments.

### Authoritative Preflight Verdicts:

```
========================================================================================
FINAL AUDIT DECISION MATRIX:
----------------------------------------------------------------------------------------
1. ARTIFACT_INTEGRITY:                 [ PASS ]
   - 100% bit-exact SHA-256 match across 159 audited files (Phase 0, 02R, Formal 01).
   - Clean deserialization of Target Heads, Gap Heads, and LinearCTC state dicts.

2. POLICY_TRANSFER:                     [ BLOCKED ]
   - Severe discrepancy diagnosed: in-sample FWR values (0.776 / 0.497 / 0.550) were
     caused by a silent file lookup error defaulting to dummy thresholds (0.70 / 0.50).
   - No deployable calibrated policy currently exists for the full-TRAIN models.
   - External FWR control remains strictly UNCONFIRMED.

3. SPEAKER_ISOLATION:                   [ PASS (With Strict Policy Lock) ]
   - Zero speaker overlap in L2_ARCTIC_SPLIT.yaml between TRAIN and TEST_SEALED.
   - Confirmed no LXC_test alias exists in TEST_SEALED.
   - Any evaluation using speaker LXC (from TRAIN) is permanently barred as contamination.
   - DEV_SEARCH is classified as development-exposed, not a fresh blind holdout.

4. INFERENCE_COMPLETENESS:              [ PASS (Contract Clarified & Omissions Corrected) ]
   - Full WAV -> CharSiu -> DP -> Wav2Vec2 -> Slicing -> MLP -> CTC/E4 -> Policy verified.
   - Published latency (74.03 ms) corrected to true complete latency: 223.09 ms (GPU).
   - Real-time deployability confirmed (RTF = 0.0549 on GPU, 0.2130 on CPU).
   - Raw-text-to-phone G2P explicitly excluded from verified scope.

5. INDEPENDENT_VALIDATION_READINESS:    [ NO_GO ]
   - Proceeding to TEST_SEALED is BLOCKED until DEV_CAL calibration is authorized,
     executed, and cryptographically frozen.
========================================================================================
```

---

## 2. Lane A — Warning Policy Transfer Audit

### 2.1 Resolution of the FWR Discrepancy
The audit investigated the massive divergence between:
- **CV Nominal-$B=0.20$ Realized FWR:** $0.235370$ false warnings per utterance across 4 outer folds.
- **Full-TRAIN In-Sample FWR:** $0.776111$ (Seed 101), $0.496667$ (Seed 202), and $0.550000$ (Seed 303).
- **Nominal Target FWR:** $0.200000$.

| Condition | Seed | Macro-3 AP | Macro-3 F1 ($B=0.2$) | Joint FWR ($B=0.2$) | False Alarms Total | Gold Correct FP | Wrong-Type Warnings | Gap False Insertions |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CV Outer OOF (4 Folds)** | 101 | 0.324357 | 0.184115 | 0.234444 | 422 | 207 | 52 | 163 |
| **CV Outer OOF (4 Folds)** | 202 | 0.320912 | 0.196812 | 0.281667 | 507 | 348 | 81 | 78 |
| **CV Outer OOF (4 Folds)** | 303 | 0.336630 | 0.182595 | 0.190000 | 342 | 222 | 45 | 75 |
| **CV OOF Mean** | **Pooled** | **0.327300** | **0.187841** | **0.235370** | **423.7** | **259.0** | **59.3** | **105.3** |
| | | | | | | | | |
| **Full-TRAIN In-Sample** | 101 | 0.580630 | 0.417067 | **0.776111** | 1,397 | 437 | 15 | 945 |
| **Full-TRAIN In-Sample** | 202 | 0.593743 | 0.459214 | **0.496667** | 894 | 533 | 15 | 346 |
| **Full-TRAIN In-Sample** | 303 | 0.589308 | 0.449221 | **0.550000** | 990 | 465 | 25 | 500 |
| **Full-TRAIN Mean** | **Pooled** | **0.587894** | **0.441834** | **0.607593** | **1,093.7** | **478.3** | **18.3** | **597.0** |

### 2.2 Root-Cause Forensic Findings:
1. **Silent Path Bug in `train_sm13_formal.py`:**
   Line 315 attempted to load `.../outer_0/inner_0/policy.json`. That file did not exist (the actual file was `outer_0/policies.json`). The code silently fell back to hardcoded dummy constants:
   $$\tau_T = 0.70, \quad \tau_G = 0.50$$
2. **Avalanche of False Insertions:**
   In non-native non-inserted gaps ($y_G = 0$), evaluating with $\tau_G = 0.50$ triggered **945 false alarms** on Seed 101, accounting for an FWR overshoot of $+0.525$ on its own.
3. **Threshold Calibration Contrast:**
   The actual calibrated thresholds in CV were $\bar{\tau}_T \approx 0.828$ and $\bar{\tau}_G \approx 0.740$. Applying these calibrated thresholds to full-TRAIN predictions yields $\text{FWR} = 0.2517$, fully dispelling the apparent $0.776$ anomaly.
4. **Distribution Shift Under Full Fitting:**
   Refitting on all 1,800 utterances (12 speakers) produced sharper upper-tail confidences ($P_{95}$ target confidence rose from $0.777$ to $0.835$). Consequently, direct transfer of 9-speaker OOF thresholds to full-data models without independent calibration cannot guarantee nominal budget adherence.
5. **Strict Label Tuning Prohibition:**
   The auditor strictly refuses to tune thresholds on full-TRAIN labels. Such practice constitutes in-sample label leakage and invalidates generalization.
6. **DEV_CAL Preregistration:**
   A formal calibration protocol on `DEV_CAL` (300 utterances, 2 unseen speakers: `ERMS, THV`) is preregistered in `POLICY_DISTRIBUTION_AUDIT.md` to establish frozen, deployable operating thresholds before independent testing.

---

## 3. Lane B — Split & Generalization Integrity Audit

### 3.1 Forensic Audit of Speaker Registry & `LXC` Overlap
1. **Authoritative Sealed Test Composition:**
   In `data/split_specs/L2_ARCTIC_SPLIT.yaml`, `TEST_SEALED` contains exactly 4 speakers:
   $$\text{TEST\_SEALED} = \{\text{ZHAA} \text{ (Arabic F)}, \text{TNI} \text{ (Hindi M)}, \text{MBMPS} \text{ (Spanish M)}, \text{TLV} \text{ (Vietnamese M)}\}$$
   There is **no Chinese speaker** in `TEST_SEALED`. Speaker `LXC` (Chinese F) is strictly in `TRAIN`.
2. **Investigation of `LXC_test`:**
   Exhaustive regex and git history audits confirm that **no file, record, or alias named `LXC_test` exists in the repository**.
3. **Historical Suitcase Corpus Mischaracterization:**
   The audit uncovered that in `reports/upr01/WAVE_B_GATE_REVIEW.md`, the "L2-ARCTIC Suitcase Corpus" was erroneously claimed to be an "Untouched Independent Cohort" despite containing `LXC`, `EBVS`, and `HQTV` (all from `TRAIN`!).
4. **Speaker Disjointness Principle:**
   Speaker disjointness is defined by **biological human identity**, not filename suffixes. Neural encoders memorizing speaker `LXC` during training will overperform on `LXC` speech regardless of whether prompts are novel or labeled `LXC_test`. Any future evaluation containing `LXC` is **PERMANENTLY PROHIBITED**.
5. **Status of `DEV_SEARCH`:**
   `DEV_SEARCH` (speakers `YBAA, NCC, RRBI, YDCK`) has been repeatedly exposed across 109 historical experiments. It is formally classified as **DEVELOPMENT-EXPOSED** and disqualified from blind holdout claims.
6. **Demographic Disclosure:**
   `TEST_SEALED` evaluates prospective generalization on Arabic, Hindi, Spanish, and Vietnamese speakers. Chinese and Korean generalization remains unassessed until external authorized benchmarks are unlocked.

---

## 4. Lane C — Complete End-to-End Inference Audit

### 4.1 Audited True Latency vs Prior Published Sub-Graph
The previous report (`RUNTIME_BENCHMARK.md`) published a GPU latency of $74.03\text{ ms}$. Independent code audit revealed that this benchmark:
- Loaded pre-cached alignment centers from disk (omitting CharSiu and Viterbi DP).
- Passed all-zero dummy vectors for phoneme embeddings.
- Ran a linear CTC projection in $0.08\text{ ms}$, but omitted CTC/E4 sequence likelihood scoring.

The auditor implemented and profiled the **true, complete end-to-end pipeline** from native WAV audio files:

| Pipeline Stage Operation | Prior Benchmark | Audited GPU Mean | Audited CPU Mean | Omission Status |
|---|:---:|:---:|:---:|:---:|
| **1. WAV Decode & Audio Load** | Included | 4.71 ms | 0.47 ms | Included |
| **2. CharSiu Acoustic Forward (`en_w2v2_fc_10ms`)** | **OMITTED** | **74.32 ms** | **455.43 ms** | **Omitted ($33.3\%$ of GPU runtime)** |
| **3. Viterbi Trellis Alignment DP** | **OMITTED** | **29.60 ms** | **27.16 ms** | **Omitted ($13.3\%$ of GPU runtime)** |
| **4. Wav2Vec2-base-960h (13 Layers)** | Included | 50.60 ms | 297.67 ms | Included |
| **5. Acoustic Crop Feature Slicing (13L)** | Included | 14.32 ms | 16.05 ms | Included |
| **6. Reference Phone 1-Hot Encoding** | **OMITTED (Used 0s)**| 0.03 ms | 0.03 ms | Corrected to Real One-Hot |
| **7. SM13 Target Mixing Head** | Included | 0.70 ms | 1.09 ms | Included |
| **8. SM13 Gap Mixing Head** | Included | 0.37 ms | 0.97 ms | Included |
| **9. CTC Layer 6 Projection** | Included | 0.11 ms | 0.16 ms | Included |
| **10. CTC/E4 Sequence Scoring (C++ Kernel)** | **OMITTED** | **47.29 ms** | **41.51 ms** | **Omitted ($21.2\%$ of GPU runtime)** |
| **11. Fixed 50/50 Fusion & Policy Engine** | Included | 0.03 ms | 0.03 ms | Included |
| **COMPLETE END-TO-END PIPELINE** | **$74.03\text{ ms}$ (Reported)** | **$223.09\text{ ms}$** | **$840.66\text{ ms}$** | **$3.15\times$ Underreporting Corrected** |

### 4.2 Runtime Summary & Resource Profile
- **Warm Latency Distribution (GPU):**
  - $\text{P50} = 194.97\text{ ms}$, $\text{P90} = 378.72\text{ ms}$, $\text{P95} = 444.62\text{ ms}$, $\text{P99} = 553.65\text{ ms}$.
  - Cold Start Latency: $396.40\text{ ms}$.
- **Real-Time Factor (RTF):**
  - GPU RTF: **$0.0549$** ($\mathbf{18.2\times}$ faster than real time; mean audio duration: $4.09\text{ s}$).
  - CPU RTF: **$0.2130$** ($\mathbf{4.7\times}$ faster than real time).
  - Both platforms strictly satisfy interactive tutoring requirements ($\text{RTF} < 0.25$).
- **Memory Footprint:**
  - Peak GPU VRAM: **$855.9\text{ MB}$** (corrected from $274.9\text{ MB}$; leaves $> 79\%$ free headroom on a 4GB GTX 1650).
  - Peak Host System RAM: $1,205.0\text{ MB}$ (GPU) / $2,065.9\text{ MB}$ (CPU).
- **Contract Verification:**
  - Output cardinality strictly confirmed: $N$ target simplexes and $N+1$ gap simplexes across arbitrary utterance lengths.
  - OOV / unknown reference phone fallback verified (graceful zero-embedding acoustic degradation).
  - Stable coordinate IDs verified: `{uid}::{pos}` and `{uid}::gap_{pos}`.
- **Inference Scope Limitation:**
  - The verified deployable contract is strictly defined as **(Native 16 kHz WAV Audio, Canonical Intended Phone Sequence)**.
  - Raw-text-to-phone generation via CMUDict G2P is unverified, lacks homograph and prosodic disambiguation, and is **formally excluded from the artifact scope**.

---

## 5. Artifact Manifest & Cryptographic Lineage

All evidence, benchmarks, scripts, and reports generated during this preflight audit are preserved under `experiments/rq1_sm13_validation_preflight_01/`:

| Artifact Path | Description | Cryptographic Integrity |
|---|---|:---:|
| `VALIDATION_PREFLIGHT_FINAL.md` | Authoritative Preflight Synthesis Report | Audit Master Lock |
| `POLICY_DISTRIBUTION_AUDIT.md` | Warning Policy Forensic Audit & DEV_CAL Preregistration | Audited & Certified |
| `SPEAKER_IDENTITY_AND_SPLIT_AUDIT.md` | Speaker Registry, LXC Overlap & Disjointness Audit | Audited & Certified |
| `END_TO_END_RUNTIME_AUDIT.md` | Complete Inference Benchmark & Omission Correction | Audited & Certified |
| `INDEPENDENT_VALIDATION_PROTOCOL.md` | Pre-Registered Two-Phase Prospective Protocol | Pre-Registered Lock |
| `BLOCKERS_AND_NEXT_ACTIONS.md` | Critical Discrepancy Breakdown & Unblocking Roadmap | Operator Gate Lock |
| `benchmark_complete_inference.py` | Complete End-to-End Auditing & Benchmarking Engine | Verified Reproducible |
| `COMPLETE_INFERENCE_BENCHMARK_RECEIPT.json` | Raw Machine Execution Latency & Memory Telemetry | Verified Deterministic |
| `MANIFEST_SHA256.json` | Bitwise SHA-256 Manifest of Audit Directory | Crytographically Sealed |

---

## 6. Auditor Final Directive & Next Steps

1. **Independent Validation Gate Remains Closed:**
   Access to `TEST_SEALED` audio, labels, and evaluation scripts is strictly **PROHIBITED** at this stage.
2. **Required Action by Lead Operator:**
   Review `BLOCKERS_AND_NEXT_ACTIONS.md` and provide formal authorization to execute **Phase 1 Calibration on `DEV_CAL`** only.
3. **Condition for Final Unsealing:**
   Only after operating policy thresholds $(\tau_T^*, \tau_G^*)$ are empirically derived on `DEV_CAL` and cryptographically frozen into `FROZEN_CALIBRATED_POLICY.json` shall the gate to `TEST_SEALED` be unlocked.

**AUDIT CONCLUSION: PREFLIGHT AUDIT COMPLETE — ARTIFACTS FROZEN — AWAITING OPERATOR GATE AUTHORIZATION.**
