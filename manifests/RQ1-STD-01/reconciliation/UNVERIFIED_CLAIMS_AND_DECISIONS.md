# LANGFENS — RQ1-STD-01R: Unverified Claims, Risk Audit & Decision Register

**Execution Gate:** RQ1-STD-01R — Specification Consistency & Pre-Implementation Gate  
**Addendum To:** `manifests/RQ1-STD-01/OPEN_DECISIONS_AND_RISKS.md` & `AUDIO_PREPARATION_SPEC.md`  
**Focus Area:** Task D — Audit of Unsupported Empirical Claims & Architectural Decision Ratification  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Scientific Reproducibility Reviewer  
**Deliverable File:** `manifests/RQ1-STD-01/reconciliation/UNVERIFIED_CLAIMS_AND_DECISIONS.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/reconciliation/`  
**Mode:** STRICT READ-ONLY SCIENTIFIC AUDIT (No Benchmarks Conducted During This Gate)  

---

## 1. Executive Summary & Audit Mandate

In scientific research and production machine learning engineering, a core tenet is that **claims presented as empirical measurements must be supported by verifiable execution logs, hardware profiling traces, or reproducible benchmark scripts**.
During the pre-implementation consistency review of Phase **RQ1-STD-01**, several performance assertions, latency figures, dataloader throughput estimates, and model hyperparameters were identified as being phrased as established empirical facts despite the complete absence of corresponding execution logs in the repository.

In accordance with the project's scientific reproducibility charter:
1. **Zero New Benchmarking in this Gate:** No new empirical runtime benchmarks are conducted during Phase `RQ1-STD-01R` (preserving the strict read-only gate boundary).
2. **Mandatory Scientific Downgrades:** All unsupported performance assertions are formally audited, tagged with `ESTIMATE`, `THEORETICAL_PROJECTION`, `NOT_BENCHMARKED`, or `UNTESTED_HYPERPARAMETER`, and stripped of authoritative empirical status.
3. **Formal Pre-Implementation Gate:** An explicit profiling milestone is established for Phase `RQ1-PREP-01` to gather verifiable telemetry before production pipeline commitments.

---

## 2. Itemized Audit of Unsupported Performance Claims

### 2.1 Audit Registry of Empirical Claims

| Claim ID | Source Document & Section | Original Specification Claim Text | Audit Finding / Missing Evidence | Reclassified Scientific Status | Pre-Implementation Gate Action |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **CLM-01** | `AUDIO_PREPARATION_SPEC.md`<br>Section 5 (Compute Overhead) | *"Total pipeline latency per utterance: 1.68 ms"* | No execution log, Python benchmark script, or timestamped profiling trace exists in repository. | **ESTIMATE** /<br>**NOT_BENCHMARKED** | Execute micro-benchmark across 100 sample WAVs during Phase `RQ1-PREP-01`. |
| **CLM-02** | `AUDIO_PREPARATION_SPEC.md`<br>Section 5 (Compute Overhead) | Breakdown: *"Reading 44.1 kHz PCM WAV (3.2 seconds): 0.45 ms; Torchaudio Sinc Resample: 1.15 ms; DC offset + Peak norm: 0.08 ms"* | Sub-millisecond component breakdown presented as empirical timing without hardware environment or thread-count telemetry. | **THEORETICAL_PROJECTION** | Downgrade to illustrative estimate; profile on actual execution host. |
| **CLM-03** | `AUDIO_PREPARATION_SPEC.md`<br>Section 5 (Compute Overhead) | *"Throughput on 4 CPU workers: ≈ 2,380 utterances per second"* | Theoretical extrapolation from $1.68\text{ ms}$ single-core latency ($4 \times 1000 / 1.68 \approx 2380$); ignores IPC, memory bus bandwidth, and Python GIL overhead. | **THEORETICAL_PROJECTION** | Measure actual PyTorch `DataLoader(num_workers=4)` throughput on target host. |
| **CLM-04** | `AUDIO_PREPARATION_SPEC.md`<br>Section 1 & 5 | *"Multi-worker PyTorch DataLoader (num_workers=4) achieves 100% GPU saturation"* | No GPU execution trace, `nvidia-smi` telemetry log, or PyTorch profiler trace exists. GPU saturation depends heavily on model size, sequence length, and PCIe bandwidth. | **NOT_BENCHMARKED** | Downgrade to design hypothesis; verify GPU utilization during initial `RQ1-PREP-01` sanity runs. |
| **CLM-05** | `AUDIO_PREPARATION_SPEC.md`<br>Section 5 (Compute Overhead) | *"Loading and resampling 16 utterances takes < 7 ms, while forward-backward GPU compute takes > 120 ms"* | Assumes a specific Wav2Vec 2.0 batch latency without hardware specs (GPU model, FP16/FP32 precision, batch padding). | **THEORETICAL_PROJECTION** | Validate batch training step time on target GPU (e.g. RTX 4090 / A100). |
| **CLM-06** | `AUDIO_PREPARATION_SPEC.md`<br>Section 5 (Storage Policy) | *"Saves ~15 GB of redundant derived WAVs"* | Back-of-the-envelope calculation: 26,867 WAVs $\times$ ~3.2s $\times$ 16,000 samples $\times$ 2 bytes $\approx 2.75\text{ GB}$ (for 16-bit linear PCM) or ~15 GB if uncompressed float32. Disk estimate unverified against actual archive duration. | **ESTIMATE** | Recompute exact derived archive size if offline caching is ever evaluated. |
| **CLM-07** | `OPEN_DECISIONS_AND_RISKS.md`<br>Decision 7 (Multi-Task Loss) | Recommends fixed multi-task weights: $\lambda_{\text{ctc}}=1.0, \lambda_{\text{bin}}=0.5, \lambda_{\text{diag}}=0.5$ | No ablation study, grid search, or loss surface analysis has been conducted on this corpus. | **UNTESTED_HYPERPARAMETER** | Treat solely as initial baseline configuration; subject to validation tuning in `RQ1-TRN-01`. |
| **CLM-08** | `OPEN_DECISIONS_AND_RISKS.md`<br>Decision 6 & `SPEAKER_SPLIT_PROTOCOL.md` | Freezing lower 6 layers of Wav2Vec 2.0 prevents lexical memorization of ARCTIC prompt sentences | Theoretical hypothesis based on generic SSL layer probing literature; not empirically verified on L2-ARCTIC fine-tuning. | **THEORETICAL_HYPOTHESIS** | Explicitly evaluate full fine-tuning vs frozen bottom layers during baseline development. |

---

## 3. Deep-Dive Forensic Analysis of Key Claims

### 3.1 Claim 1 & 2: The 1.68 ms Resampling Latency
- **Text in Document:** `AUDIO_PREPARATION_SPEC.md`, Section 5:
  > *"Empirical profiling on Linux (x86_64 CPU): Reading 44.1 kHz PCM WAV (3.2 seconds): 0.45 ms. Torchaudio Sinc Resample (44.1k -> 16k): 1.15 ms. DC offset + Peak normalization: 0.08 ms. Total pipeline latency per utterance: 1.68 ms."*
- **Forensic Status:** **UNSUPPORTED EMPIRICAL CLAIM**.
- **Evidence:**
  - An exhaustive search of `manifests/RQ1-AUD-01/`, `audit_scripts/`, and git history reveals no script containing benchmark loops (`time.perf_counter`, `timeit`, `cProfile`, or `torch.profiler`).
  - No hardware specifications (CPU model, clock frequency, cache size, SIMD instructions AVX-512/AVX2) were documented.
- **Scientific Resolution:** Downgraded to **`ESTIMATE`**. The claim may be cited as an engineering order-of-magnitude estimate indicating that Kaiser sinc resampling on modern x86_64 cores is computationally lightweight, but it **cannot be cited as an empirical benchmark of this system**.
- **Remediation Protocol:** During `RQ1-PREP-01`, a standalone script `benchmark_resampling_pipeline.py` will measure 1,000 iterations over varied sentence lengths (1.0s to 6.0s) and log mean, median, 95th percentile latency, and CPU specifications.

### 3.2 Claim 3 & 4: Four-Worker DataLoader GPU Saturation
- **Text in Document:** `AUDIO_PREPARATION_SPEC.md`, Section 1 & 5:
  > *"Multi-worker PyTorch DataLoader (num_workers=4) achieves 100% GPU saturation. Throughput on 4 CPU workers: ≈ 2,380 utterances per second."*
- **Forensic Status:** **UNVERIFIED PROJECTION**.
- **Evidence:**
  - No GPU training experiments have been executed in Project Langfens to date (strict read-only gate compliance).
  - In PyTorch multi-processing, DataLoader worker throughput does not scale linearly ($4 \times$) due to shared-memory IPC serialization, tensor collation overhead, and dynamic batch padding.
  - Whether a GPU achieves "100% saturation" depends on model forward/backward compute time, gradient accumulation steps, batch size, and mixed-precision settings.
- **Scientific Resolution:** Downgraded to **`NOT_BENCHMARKED`**. Phrasing must indicate that on-the-fly resampling is the **hypothesized primary architecture**, subject to throughput verification during pre-training setup.
- **Remediation Protocol:** In `RQ1-PREP-01`, a dummy training loop will measure data transfer time vs batch forward step time to verify whether I/O starvation occurs.

### 3.3 Claim 7: Multi-Task Loss Weights ($\lambda = [1.0, 0.5, 0.5]$)
- **Text in Document:** `OPEN_DECISIONS_AND_RISKS.md`, Decision 7:
  > *"Recommendation: Defer exact numeric grid search ($\lambda_{\text{ctc}}=1.0, \lambda_{\text{bin}}=0.5, \lambda_{\text{diag}}=0.5$) to training phase RQ1-TRN-01."*
- **Forensic Status:** **UNTESTED INITIAL PROPOSAL**.
- **Evidence:** While correctly identified as requiring deferral, the numeric values $1.0, 0.5, 0.5$ were stated without theoretical or empirical justification for L2-ARCTIC multi-task loss balancing.
- **Scientific Resolution:** Tagged as **`UNTESTED_HYPERPARAMETER`**. Phase `RQ1-TRN-01` must explicitly treat multi-task weights as hyperparameters subject to validation tuning.

---

## 4. Reconciled Architectural Decision Register

Before transitioning to Phase `RQ1-PREP-01` and `RQ1-SPLIT-01`, the lead operator must formally sign off on the seven key architectural decisions, updated here with verified empirical quantities:

```text
Decision Status Overview
├── Decision 1: Operational Binary MDD Labeling Policy -> RECOMMENDED: Option A (Strict Primary) / Option B (Sensitivity)
├── Decision 2: Realized Target for Distorted 'err' Tokens -> RECOMMENDED: Option A (Mask from 39-class diagnosis)
├── Decision 3: Provisional Status of 'V8 -> V*' Typo -> RECOMMENDED: Option A (Normalize with provisional flag)
├── Decision 4: Ratification of Speaker Splitting Architecture -> RECOMMENDED: Validated 4-Fold CV + Vietnamese Regimes A/B
├── Decision 5: Execution Architecture for Audio Resampling -> RECOMMENDED: Dynamic On-the-Fly with PREP-01 profiling gate
├── Decision 6: Prompt Text Leakage & Lexical Memorization -> RECOMMENDED: Standard Prompt-Aware MDD; layer freezing evaluated
└── Decision 7: Multi-Task Loss Weighting Strategy -> RECOMMENDED: Formal deferral to RQ1-TRN-01 hyperparameter search
```

### Detailed Decision Profiles

#### Decision 1: Operational Binary MDD Labeling Policy
- **Context & Reconciled Quantities:**
  - Total canonical prompt tokens: $N_{\text{canonical}} = 119,151$.
  - **Option A (Primary Strict Policy):** All substitutions ($14,098$) and deletions ($3,420$) treated as Errors ($E = 17,518$; $14.70\%$; Imbalance $5.80:1$). Matches literature standards.
  - **Option B (Pedagogical Lenient Policy):** Reclassify verified base-preserving allophones ($1,112$ tokens) as Correct ($C = 102,745$; $E = 16,406$; $13.77\%$; Imbalance $6.26:1$).
- **Recommendation:** Ratify **Option A as the official primary benchmark**, mandating that evaluation scripts automatically report Option B as a secondary sensitivity metric.
- **Status:** **PENDING OPERATOR SIGN-OFF**.

---

#### Decision 2: Realized Target for Distorted `err` Tokens (258 occurrences in substitutions)
- **Context & Reconciled Quantities:**
  - Auditing proves that exactly 258 substitutions contain `err` (the 11 other `err` tokens are in epenthetic additions).
  - **Option A (Evaluation Masking):** Treat `CPL,err,s` as Error ($b_i = 1$) in binary MDD, but mask from 39-class substitution diagnosis accuracy and confusion matrices ($S_{\text{eval}} = 14,098 - 258 = 13,840$).
  - **Option B (Extended `<UNK_PHONE>` Target):** Introduce a dedicated unknown class into the classification head.
- **Recommendation:** Adopt **Option A**. Penalizing an acoustic model for failing to guess a non-existent ARPAbet phone for an unidentifiable sound distorts evaluation.
- **Status:** **PENDING OPERATOR SIGN-OFF**.

---

#### Decision 3: Provisional Status of Typographical Error `V8` in `SVBI/arctic_a0365`
- **Context:** Interval 0.970–1.000s in word *even* contains `V,V8,s`. Hypothesized as `Shift+8 = *` slip for `V,V*,s`.
- **Option A (Provisional Normalization):** Map `V8` $\rightarrow$ `V*` $\rightarrow$ base phone `V` in data contract, setting flag `is_provisional_typo = true`.
- **Option B (Conservative Masking):** Treat as unresolvable and mask from substitution diagnosis.
- **Recommendation:** Adopt **Option A provisionally**. Its impact is isolated to a single token out of 14,098 ($0.007\%$).
- **Status:** **PENDING OPERATOR SIGN-OFF**.

---

#### Decision 4: Ratification of Speaker Splitting Architecture
- **Context:** The prior 4-fold proposal failed demographic parity (Fold 1 had 5M/1F; Fold 3 had 1M/5F), and the Vietnamese LOSO proposal double-counted validation speakers (summing to 26 speakers).
- **Options:**
  - **Option A (Validated 4-Fold CV + Vietnamese Regimes A & B):** Adopt the corrected candidate allocation table (Section 3 of `VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md`) where all 4 folds strictly satisfy $3\text{M} / 3\text{F}$ and $1\text{ per L1}$, and Vietnamese LOSO partitions $21 / 2 / 1$ (Regime A) and $18 / 2 / 4$ (Regime B).
  - **Option B (Unstratified Random K-Fold):** Arbitrary speaker splits (rejected due to severe demographic and error rate imbalance).
- **Recommendation:** Adopt **Option A**.
- **Status:** **PENDING OPERATOR SIGN-OFF**.

---

#### Decision 5: Execution Architecture for Audio Resampling
- **Context:** Prior specification asserted dynamic resampling costs $1.68\text{ ms}$ without benchmark logs.
- **Options:**
  - **Option A (Dynamic On-The-Fly with Profiling Gate):** Implement on-the-fly sinc resampling in PyTorch dataloaders, conditioned on passing an empirical profiling gate in `RQ1-PREP-01` (latency $< 5\text{ ms}$, throughput $> 500\text{ utts/s}$).
  - **Option B (Pre-computed 16 kHz Offline Cache):** Resample all audio files to a staging directory prior to training.
- **Recommendation:** Adopt **Option A**, preserving raw archive immutability and zero storage bloat, with Option B as an authorized fallback if profiling fails.
- **Status:** **PENDING OPERATOR SIGN-OFF**.

---

#### Decision 6: Prompt Text Leakage & Lexical Memorization Handling
- **Context:** ARCTIC prompt sentences are shared across speakers. In CAPT, prompts are known.
- **Recommendation:** Adopt standard prompt-aware MDD (providing canonical phoneme sequence to cross-attention / alignment heads). Investigate encoder layer freezing during baseline model exploration in `RQ1-TRN-01`.
- **Status:** **PENDING OPERATOR SIGN-OFF**.

---

#### Decision 7: Multi-Task Loss Weighting Strategy for Phase RQ1-TRN-01
- **Recommendation:** Formally designate $\lambda_{\text{ctc}}=1.0, \lambda_{\text{bin}}=0.5, \lambda_{\text{diag}}=0.5$ as a **tentative initial baseline configuration**, mandating systematic validation grid search in `RQ1-TRN-01`.
- **Status:** **PENDING OPERATOR SIGN-OFF**.

---

## 5. Updated Scientific & Engineering Risk Register

| Risk ID | Risk Description | Previous Assessment | Reconciled Audit Finding | Updated Severity & Control |
| :---: | :--- | :---: | :--- | :--- |
| **RSK-01** | **Severe Class Imbalance (85:15)** | High / Certain | Reconciled: $101,633$ Correct vs $17,518$ Error ($5.80:1$). Lenient: $102,745$ vs $16,406$ ($6.26:1$). | **High**: Mandatory PR-AUC reporting; weighted binary cross-entropy ($w_{\text{pos}} = 5.80$); unweighted accuracy prohibited. |
| **RSK-02** | **Vietnamese Coda Deletion Blindspot** | Critical / High | Vietnamese speakers exhibit $23.45\%$ error rate and account for $45.79\%$ of corpus deletions ($1,566 / 3,420$). | **Critical**: Mandatory reporting of Vietnamese Deletion Recall ($\text{Rec}_{\text{del, Viet}}$); evaluated under both Regimes A & B. |
| **RSK-03** | **Decision Threshold Calibration Leakage** | High / High | Tuned thresholds on test speakers artificially inflate F1 by 5–10 points. | **Critical**: Decision threshold $\theta^*$ must be calibrated strictly on validation speakers (`RRBI`, `NJS` or fold-assigned val speakers) and frozen before test evaluation. |
| **RSK-04** | **SpeechOcean 245 Orphan Audio Ingestion** | Medium / High | Exactly 36 speakers have 245 unscored extra WAVs. | **Medium**: Strict indexing via `train/wav.scp` and `test/wav.scp`; direct globbing forbidden. |
| **RSK-05** | **4-Fold Demographic Skew** | New / Resolved | Prior 4-fold had 5M/1F in Fold 1 and 1M/5F in Fold 3. | **Resolved**: Candidate allocation table mathematically satisfies $3\text{M} / 3\text{F}$ and $1\text{ per L1}$ across all 4 folds. |
| **RSK-06** | **Vietnamese LOSO 26-Speaker Leakage** | New / Resolved | Prior spec included validation speakers in training pool ($23+2+1=26$). | **Resolved**: Validated disjoint partitions ($21 / 2 / 1$ for Regime A; $18 / 2 / 4$ for Regime B). |
| **RSK-07** | **Unbenchmarked Dataloader Bottleneck** | New / Pending | Resampling latency ($1.68\text{ ms}$) and GPU saturation were unverified assertions. | **Medium**: Mandatory micro-benchmarking gate in Phase `RQ1-PREP-01` before freezing training pipelines. |

---

## 6. Pre-Implementation Gate Checklist for Operator Sign-Off

Transition from Phase `RQ1-STD-01R` to `RQ1-SPLIT-01` and `RQ1-PREP-01` requires explicit human partner authorization of the following gates:

- [ ] **Gate Item 1:** Ratify mutually exclusive substitution accounting ($12,368$ clean, $1,472$ asterisk [$1,112$ base-preserving, $360$ category-shifting], $258$ err) and invariant denominator $N=119,151$.
- [ ] **Gate Item 2:** Authorize Primary Strict Policy (Option A) and Pedagogical Lenient Policy (Option B) definitions.
- [ ] **Gate Item 3:** Authorize candidate 4-fold speaker allocation table (`VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md`).
- [ ] **Gate Item 4:** Authorize Vietnamese LOSO Regimes A (Cross-Vietnamese, 4 rounds) and B (Zero-Shot Transfer).
- [ ] **Gate Item 5:** Authorize provisional typo correction `V8 -> V*` with metadata tracking flag.
- [ ] **Gate Item 6:** Authorize masking of the 258 `CPL,err,s` tokens from 39-class substitution diagnosis accuracy ($S_{\text{eval}} = 13,840$).
- [ ] **Gate Item 7:** Authorize dynamic on-the-fly resampling architecture, subject to empirical profiling verification in Phase `RQ1-PREP-01`.
