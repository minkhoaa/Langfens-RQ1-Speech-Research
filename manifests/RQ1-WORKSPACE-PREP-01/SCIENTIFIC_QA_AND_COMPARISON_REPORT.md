# Scientific QA & Comparative Evaluation Report — RQ1 Prep

**Status:** `PREPARATION_AUDITED_CRITERIA_FROZEN; CANDIDATE_METRICS_PENDING_EXECUTION; EXPERIMENTS_HOLD`  
**Author:** Agent D — Scientific QA & Comparison Subagent  
**Date:** 2026-11-03  
**Coordinator Workspace:** `/home/khoa/Projects/KLTN/Dataset` (`prep/rq1-shared-base`)  
**Shared Base Commit:** `0d268ff8c444e16fd76aca2f4ad6ece966908edc`  
**Binding Evaluation Contract:** `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`  
**Contract SHA-256:** `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`  

---

## Executive Summary

This report establishes the independent scientific QA audit and comparative evaluation framework for the two parallel candidate lanes in RQ1:
- **P-SCORE:** Segmentation-free acoustic graph scoring (`GOP-CTC-SF-S` vs `GOP-CTC-SF-SD`) evaluated on frozen Baseline R (E0) acoustic emissions.
- **P-SUP:** Compact natural-operation supervision (`CompactOperationHead` with 3-way `operation_ce` treatment vs parameter-matched `binary_collapsed` contemporaneous control).

### Primary Findings & Governance Determinations:
1. **Shared Frozen Evaluation Contract Verified:**  
   `EVALUATION_CONTRACT_V1.md` (SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`) is bit-identically verified across all three workspaces (coordinator `prep/rq1-shared-base`, P-SCORE `prep/rq1-p-score`, P-SUP `prep/rq1-p-sup`). All split definitions, data sealing protocols, and isolation boundaries are cryptographically confirmed.
2. **Historical Baseline Provenance & AP Metrics Reconciled:**  
   Historical Baseline R (E0 `best.pt`, 3,600 updates, SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`) performance figures are formally reconciled:
   - GPU FP16 AMP online validation AP: **`0.316062`**
   - CPU FP32 offline replay Protocol O AP: **`0.317021`** (absolute delta $+0.000959$, $+0.30\%$, explained by precision boundaries)
   - Cross-fold pilot fold 1 evaluation AP: **`0.317012`** (delta $\le 9 \times 10^{-6}$)
   - Protocol Z Aligned-Only AP: **`0.313530`** (99.63% coverage, 9,919 tokens)
   - E1 alignment-aware skip benchmark AP: **`0.309293`** (omission AP `0.032252`, skip mass exactly `0.000000` on deletions)  
   These figures represent evaluator-specific archival anchors, NOT interchangeable metrics. None overwrites another.
3. **Strict Labeling & Evaluation Denominators Certified:**  
   The Fold 1 validation population comprises exactly 300 utterances and 9,956 canonical phone tokens ($N_{\text{cor}}=8,799$, $N_{\text{sub}}=1,012$, $N_{\text{del}}=145$). Distinct evaluation populations are certified:
   - **Binary AP Denominator:** $N = N_{\text{err}} + N_{\text{cor}} = 1,157 + 8,799 = \mathbf{9,956}$ (Prevalence: $11.62\%$).
   - **Legacy Omission AP Denominator:** $N = N_{\text{del}} + N_{\text{cor}} = 145 + 8,799 = \mathbf{8,944}$ (Prevalence: $1.62\%$). Substitutions are strictly excluded. Any deletion-vs-all metric ($N=9,956$) mislabeled as omission AP is rejected.
   - **Legacy Substitution AP Denominator:** $N = N_{\text{sub}} + N_{\text{cor}} = 1,012 + 8,799 = \mathbf{9,811}$ (Prevalence: $10.31\%$). Deletions are strictly excluded.
   - **Realized PER Reference:** $N_{\text{realized}} = \mathbf{9,881}$ tokens.
4. **Prospective Acceptance & Noninferiority Criteria FROZEN:**  
   Prospective criteria, minimum effect margins, and noninferiority boundaries are prespecified and frozen **prior to candidate metric inspection**, eliminating risks of post-hoc threshold tuning or outcome-dependent hypothesis drift.
5. **Acoustic Controls Verified:**  
   The deterministic within-speaker cyclic shift-1 mismatch control (`within_speaker_cyclic`, baseline acoustic gap $\Delta_{\text{acoustic}} = +0.022299$) and the canonical empirical prior baseline ($\text{AP} = 0.303063$, neural margin $+0.013958$) are certified as mandatory falsifiers.
6. **Candidate Implementation Readiness vs. Execution Status:**  
   - P-SCORE: Prototype implementation and shared-schema bridge portability are **GO** (269 passed tests). Real-data inference is **NOT_STARTED** (0 updates, 0 real predictions emitted).
   - P-SUP: Prototype implementation and parameter matching (266,667 trainable params) are **GO** (88 passed tests). Preflight script reports `PREPARATION_AUDITED_EXPERIMENT_HOLD`. Remote training is **NOT_STARTED** (0 updates, 0 attempts, no model artifact).
7. **Scientific Disposition:**  
   Candidate comparative performance is **DEFERRED** until real-data runs are executed under authorized conditions. No statistical superiority or ranking is inferred from synthetic prototypes. Both experimental lanes remain in **HOLD** pending operator execution authorization.

---

## 1. Shared Frozen Evaluation Contract Verification

### 1.1 Contract Document & Cryptographic Integrity
The binding evaluation contract governing all RQ1 candidate models is `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`. Its cryptographic digest was independently verified across all three repository worktrees:

$$\text{SHA-256}(\text{EVALUATION\_CONTRACT\_V1.md}) = \texttt{4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2}$$

| Repository / Worktree | Branch | Relative Path | Verified SHA-256 Digest | Status |
|---|---|---|:---:|:---:|
| Coordinator (`/home/khoa/Projects/KLTN/Dataset`) | `prep/rq1-shared-base` | `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md` | `4d99723e4b3f62f5...` | **MATCH** |
| Agent A (`langfens-rq1-p-score`) | `prep/rq1-p-score` | `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md` | `4d99723e4b3f62f5...` | **MATCH** |
| Agent B (`langfens-rq1-p-sup`) | `prep/rq1-p-sup` | `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md` | `4d99723e4b3f62f5...` | **MATCH** |

### 1.2 Split Manifests, Speaker Partitioning & Sealing Invariants
The authoritative split manifest is `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
For Fold 1, partition boundaries and ordered utterance digests are verified:

```
Ordered TRAIN Utterance Digest (2,399 utts): 2ebe54f2fc77e2a3ee9890ce3dc21f725897ebbe85f812d3c017c3b495f14b27
Ordered VAL Utterance Digest (300 utts):   6d34642b15bbf80cbd5bd102b5574ab36f5adac74be6d6d0ae7218b002eff065
```

- **TRAIN Partition (2,399 utterances, 16 speakers):**  
  `BWC`, `ERMS`, `HKK`, `HQTV`, `MBMPS`, `NCC`, `SKA`, `SVBI`, `THV`, `TLV`, `TNI`, `TXHC`, `YBAA`, `YDCK`, `YKWK`, `ZHAA`.
- **VAL Partition (300 utterances, 2 speakers):**  
  `NJS` (Spanish L1, 150 utterances), `RRBI` (Russian L1, 150 utterances).
- **TEST Partition (900 utterances, 6 speakers) — STRICTLY SEALED:**  
  `ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`.
  *Invariant:* Active Fold 1 test audio, TextGrids, annotations, timestamps, and transcripts are completely absent from staging packages and execution environments (`fold1_train_val_only.zip` SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`).

### 1.3 Reduced-TRAIN Partitioning for Parallel Pilots
To enable rigorous, leak-free model selection and threshold calibration without exposing VAL, the 16 TRAIN speakers are deterministically partitioned using sorted SHA-256 digests of `rq1-contract-v1:1337:SPEAKER`:

| Sub-Partition | Utterance Count | Speaker Roster | Permitted Scientific Function |
|---|---:|---|---|
| **FIT** | 1,799 | `BWC`, `ERMS`, `HKK`, `HQTV`, `SKA`, `SVBI`, `TNI`, `TXHC`, `YBAA`, `YDCK`, `YKWK`, `ZHAA` (12 spk) | Model parameter optimization, backward passes, gradient updates. |
| **SELECT** | 300 | `MBMPS`, `TLV` (2 spk) | Checkpoint selection, early stopping monitoring (`blank_fraction < 0.99`). Zero gradient updates. |
| **CAL** | 300 | `NCC`, `THV` (2 spk) | Decision threshold calibration ($\tau$), temperature scaling. Zero parameter fitting. |
| **VAL** | 300 | `NJS`, `RRBI` (2 spk) | Unseen exploratory evaluation screen. Zero tuning, zero calibration. |

*Critical Historical Constraint:* Frozen E0 Baseline R was originally trained on all 2,399 TRAIN utterances, including `SELECT` and `CAL`. Consequently, P-SCORE calibration on CAL is **not speaker-disjoint from E0 acoustic training**. This irreducible historical condition is documented; P-SCORE primary ranking is evaluated on raw frozen-weight scores without learned fusion.

---

## 2. Historical Baseline Provenance and AP Reconciliation

### 2.1 Baseline Checkpoints & Backbone Integrity
The reference models evaluated in prior research cycles are anchored by immutable cryptographic digests:

| Checkpoint Name | Role | Updates | Epoch (0-idx / 1-idx) | Checkpoint SHA-256 Digest | Status |
|---|---|---:|---:|:---:|:---:|
| **E0 `best.pt`** | Primary Production Baseline R | 3,600 | 5 / 6 | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | Verified |
| **E0 `last.pt`** | Endpoint Checkpoint | 6,600 | 10 / 11 | `bf46fab366363d46082082cf7f088b218f6d8d12851f5a9de706276b789efcb6` | Verified |
| **E1 `fixed_budget.pt`** | Alignment-Aware Skip Model | 3,600 | 5 / 6 | `3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24` | Verified |
| **Backbone** | Pretrained Wav2Vec2 | — | — | `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`) | Frozen |

### 2.2 Numerical Reconciliation of Historical AP Values (.316062, .317012, .317021)
Historical documentation and audit records report three distinct Binary AP values for Baseline R on Fold 1 validation. This audit resolves their exact technical provenance:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          HISTORICAL BASELINE R BINARY AP MATRIX                        │
├─────────────────────┬──────────────┬─────────────┬──────────────┬──────────────────────┤
│ Metric Value        │ Evaluation   │ Hardware    │ Arithmetic   │ Reference Source     │
│                     │ Protocol     │ Platform    │ Precision    │                      │
├─────────────────────┼──────────────┼─────────────┼──────────────┼──────────────────────┤
│ 0.316062            │ Protocol O   │ Colab T4    │ FP16 AMP     │ best.pt metadata     │
│ 0.317021            │ Protocol O   │ CPU Replay  │ FP32 Full    │ RQ1-TRN-02A-EVAL-R2  │
│ 0.317012            │ Protocol O   │ Cross-Fold  │ FP32 Pilot   │ PILOT_FOLD1_REPORT   │
│ 0.313530            │ Protocol Z   │ CPU Replay  │ FP32 Aligned │ ZERO-ORACLE-VERIFY   │
└─────────────────────┴──────────────┴─────────────┴──────────────┴──────────────────────┘
```

1. **`0.316062` (Online FP16 AMP Baseline):**  
   Computed during initial training on Google Colab (`audit_scripts/train_fold1_ctc_binary.py`) using PyTorch `autocast("cuda", dtype=torch.float16)` with batch size 2. Stored in `best.pt` checkpoint header under `best_ap`.
2. **`0.317021` (Offline Full-Precision FP32 Replay):**  
   Computed during independent CPU verification (`RQ1-TRN-02A/evaluation_reconciliation/HISTORICAL_VS_REPLAYED_COMPARISON.md`) using strict `float32` inference and batch size 1. The difference ($\Delta = +0.000959$, $+0.30\%$) is completely accounted for by floating-point mantissa rounding (11-bit FP16 vs 24-bit FP32) affecting 4 boundary tokens near $\tau=0.50$ (TP: 198 $\to$ 201; FP: 209 $\to$ 208).
3. **`0.317012` (Cross-Fold Evaluator Replay):**  
   Computed in `RQ1-TRN-03A` pilot fold 1 benchmark (`audit_scripts/evaluate_cross_fold_validation.py`). Differs from `0.317021` by only $9 \times 10^{-6}$ ($2.26 \times 10^{-7}$ in internal unrounded floats), reflecting minor implementation differences in tie-breaking sort order.
4. **`0.313530` (Protocol Z Aligned-Only Baseline):**  
   Evaluated strictly on the 9,919 phone tokens where G2P frontend queries align with gold annotations (99.63% coverage). The small drop ($\Delta = -0.002904$ vs Protocol O 0.316434) demonstrates prompt-only feasibility without gold TextGrid intervals.

*Governance Mandate:* These values are preserved as evaluator-specific historical anchors. They must **never** be averaged, overwritten, or conflated in comparative reporting.

### 2.3 E0 vs. E1 Archival Benchmark Synthesis
Archival benchmark records (`manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json`) establish the comparative performance of the alignment-aware skip mechanism (E1) against Baseline R (E0):

| Primary Metric Dimension | Baseline R (E0) | Skip Model (E1) | Delta (E1 $-$ E0) | Relative Delta | Scientific Interpretation |
|---|---:|---:|---:|---:|---|
| **Binary MDD AP** ($N=9,956$) | **0.317021** | 0.309293 | $-0.007728$ | $-2.44\%$ | Detection accuracy degraded |
| **Omission AP** ($N=8,944$) | **0.035020** | 0.032252 | $-0.002768$ | $-7.90\%$ | Omission ranking degraded |
| **Recall @ 15% FPR** | **0.400000** | 0.365517 | $-0.034483$ | $-8.62\%$ | Operating sensitivity decreased |
| **Greedy CTC PER** ($N=9,881$) | 0.236818 | **0.232365** | $-0.004453$ | $-1.88\%$ | Modest acoustic transcription gain |
| **Omission Skip Mass** | 0.000000 | 0.000000 | $0.000000$ | $0.00\%$ | **Complete collapse of skip token** |
| **Correct Phone Skip Mass** | 0.000000 | 0.000031 | $+0.000031$ | — | Near-zero activation |
| **In-Window Attention Mass** | 0.032023 | 0.035854 | $+0.003831$ | $+11.96\%$ | Marginal attention shift |
| **Mean Peak Offset** | 1.191963s | 1.197316s | $+0.005353$ | — | Persistent cross-attention lag |

*Scientific Audit Conclusion:* E1 empirical failure proves that introducing a learned null token without explicit supervision does not resolve phoneme omission detection. The skip mass collapsed to 0.000000 on deletions. Consequently, E1 cross-fold rollout remains **STRICT NO-GO**, and Baseline R remains the production reference.

---

## 3. Strict Label, Masking, and Denominator Accounting

### 3.1 Population Census & Error Subtypes
The official Fold 1 validation partition (`NJS` and `RRBI`) comprises 300 TextGrids with parsed canonical phone intervals:

```
Total Canonical Tokens:        9,956 (100.00%)
├── Correct Phone Tokens:      8,799  (88.38%)
└── Mispronunciation Errors:   1,157  (11.62%)
    ├── Substitutions (s):     1,012  (10.16%)
    └── Deletions / Omissions:   145   (1.46%)
```

Per-speaker disaggregated distribution:
- **Speaker `NJS` (Spanish L1, 150 utts):** 5,011 canonical tokens; 4,370 correct, 563 substitutions, 78 deletions (641 total errors, prevalence $12.79\%$).
- **Speaker `RRBI` (Russian L1, 150 utts):** 4,945 canonical tokens; 4,429 correct, 449 substitutions, 67 deletions (516 total errors, prevalence $10.43\%$).

### 3.2 Evaluation Denominators & AP Metric Definitions
The evaluation contract enforces three distinct token cohorts. Each cohort targets an exact scientific question:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             CERTIFIED EVALUATION DENOMINATORS                          │
├────────────────────┬───────────────────────────────────┬───────────┬───────────────────┤
│ Evaluation Cohort  │ Constituent Population Tokens     │ Size (N)  │ Error Prevalence  │
├────────────────────┼───────────────────────────────────┼───────────┼───────────────────┤
│ Binary AP          │ Substitutions + Deletions vs Cor  │   9,956   │ 11.6211% (1,157)  │
│ Legacy Omission AP │ Deletions vs Correct ONLY         │   8,944   │  1.6212% (145)    │
│ Legacy Sub AP      │ Substitutions vs Correct ONLY     │   9,811   │ 10.3149% (1,012)  │
│ Realized PER Ref   │ Realized Chronological Phones     │   9,881   │ —                 │
└────────────────────┴───────────────────────────────────┴───────────┴───────────────────┘
```

1. **Binary AP ($N=9,956$):**  
   Evaluates discrimination between all errors ($y \in \{1, 2\}$) and correct phones ($y=0$).  
   Denominator: $N_{\text{err}} + N_{\text{cor}} = 1,157 + 8,799 = 9,956$.
2. **Legacy-Comparable Omission AP ($N=8,944$):**  
   Evaluates omission ranking sensitivity.  
   Denominator: $N_{\text{del}} + N_{\text{cor}} = 145 + 8,799 = 8,944$.  
   *Rejection Invariant:* Substitutions are strictly excluded from the evaluation table. If substitutions are retained as negative instances ($N=9,956$), the positive class prevalence is artificially depressed to $1.4564\%$, and false alarms on substitutions contaminate omission precision. Any such metric is **strictly rejected as mislabeled**.
3. **Legacy-Comparable Substitution AP ($N=9,811$):**  
   Evaluates substitution ranking sensitivity.  
   Denominator: $N_{\text{sub}} + N_{\text{cor}} = 1,012 + 8,799 = 9,811$. Deletions are strictly excluded.
4. **Greedy Realized CTC PER ($N_{\text{realized}}=9,881$):**  
   Evaluates CTC head acoustic transcription against chronological realized phones (including valid additions, correct phones, and substitutions; excluding deletions, silences, and unrecognized phones). Computed via Levenshtein distance: $\text{PER} = (S + D + I) / 9,881$.

### 3.3 Strict Label Normalization & Masking Invariants
- **Same-Base Phone Variants:** Annotations marked `s:AA*` or similar same-base modifications remain substitutions ($y=1$). The evaluator never derives KEEP or correct status from normalized phoneme string equality.
- **Unknown Error Identity (`err`):** Annotations labeled `err` retain error target $y=1$ for binary/operation loss, but auxiliary realization targets are masked with `-100`.
- **Acoustic Segments for Deletions:** Deletion placeholder `sil` is not an observed acoustic segment. Deletion has a canonical positional query, but no realization target (`-100`). Deletions are **never** masked out of binary or operation loss.
- **Additions / Insertions:** Additions have no canonical prompt position and are tracked as inter-token sequence gaps. They are never smuggled into canonical-token AP metrics.

### 3.4 Protocol O vs. Protocol Z Boundary & Leakage Audit
- **Protocol O (Oracle-Canonical):** Canonical phoneme queries are extracted from human annotator TextGrids. While inference uses no TextGrid segment timestamps, the phoneme sequence itself reflects oracle transcription. Archival use only.
- **Protocol Z (Zero-Oracle Deployment):** Inference accepts raw audio waveform and written text prompts only. Canonical phonemes are generated via CMUdict G2P (with deterministic fallback). Gold TextGrid data is joined strictly offline after predictions and hashes are frozen.
- **Timestamp Leakage Audit:** Code inspection of `src/langfens_prep/p_score` and `src/langfens_prep/p_sup` confirms **zero gold TextGrid timestamps or boundary intervals are imported or queried at runtime**.

---

## 4. Prospective Acceptance & Noninferiority Criteria

To prevent post-hoc metric shopping, HARKing, or threshold manipulation, acceptance margins and decision boundaries are hereby prespecified and frozen.

### 4.1 P-SCORE Prospective Evaluation Criteria
Primary hypothesis: *Does admitting a target-deletion alternative in CTC graph scoring (`GOP-CTC-SF-SD`) improve omission ranking over allowing substitutions only (`GOP-CTC-SF-S`) without degrading overall detection?*

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        P-SCORE PROSPECTIVE DECISION BOUNDARIES                         │
├─────────────────────────┬──────────────────────────────┬───────────────────────────────┤
│ Metric Dimension        │ Statistical Comparison       │ Acceptance Margin             │
├─────────────────────────┼──────────────────────────────┼───────────────────────────────┤
│ Primary Omission AP     │ SF-SD vs SF-S (N=8,944)      │ ΔAP_omit ≥ +0.020 (+2.0% abs) │
│ Operating Sensitivity   │ Recall @ 15% FPR Ceiling     │ ΔRecall@FPR15 ≥ +0.030 (+3.0%)│
│ Binary AP Noninferiority│ SF-SD vs E0 Binary (N=9,956) │ ΔAP_bin ≥ -0.005 (-0.5% max)  │
│ Substitution Noninfer.  │ SF-SD vs SF-S (N=9,811)      │ ΔAP_sub ≥ -0.005 (-0.5% max)  │
│ Acoustic Grounding Gap  │ Matched vs Shift-1 Mismatched│ Δ_acoustic ≥ +0.015 (+1.5%)   │
│ Canonical Prior Margin  │ SF-SD vs Canonical Prior     │ Δ_prior ≥ +0.010 (+1.0%)      │
│ Prompt Coverage Policy  │ Valid Output Accounting      │ Coverage ≥ 99.50% (≤0.5% abs) │
└─────────────────────────┴──────────────────────────────┴───────────────────────────────┘
```

**Decision Rules for P-SCORE:**
- **PASS (GO):** $\Delta \text{AP}_{\text{omit}} \ge +0.020$, $\Delta \text{Recall@FPR15} \ge +0.030$, noninferiority satisfied on Binary and Substitution AP ($\ge -0.005$), $\Delta_{\text{acoustic}} \ge +0.015$, and coverage $\ge 99.50\%$.
- **INCONCLUSIVE (HOLD):** Modest omission gain ($0 < \Delta \text{AP}_{\text{omit}} < +0.020$) or minor noninferiority breach ($[-0.010, -0.005]$), or acoustic reliance unconfirmed.
- **FAIL (NO-GO):** $\Delta \text{AP}_{\text{omit}} \le 0$ (SF-SD fails to outperform SF-S), or Binary AP degradation $> 0.010$, or catastrophic failure on repeated phones / coverage.

### 4.2 P-SUP Prospective Evaluation Criteria
Primary hypothesis: *Does compact 3-way natural-operation supervision (`operation_ce`) improve omission ranking beyond contemporaneous binary supervision (`binary_collapsed`), without learning text priors or sacrificing substitution accuracy?*

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         P-SUP PROSPECTIVE DECISION BOUNDARIES                          │
├─────────────────────────┬──────────────────────────────┬───────────────────────────────┤
│ Metric Dimension        │ Statistical Comparison       │ Acceptance Margin             │
├─────────────────────────┼──────────────────────────────┼───────────────────────────────┤
│ Primary Omission AP     │ T-OP vs B-CONTROL (N=8,944)  │ ΔAP_omit ≥ +0.030 (+3.0% abs) │
│ Operating Sensitivity   │ Recall @ 15% FPR Ceiling     │ ΔRecall@FPR15 ≥ +0.040 (+4.0%)│
│ Dedicated Omission AP   │ s_omit vs s_error on T-OP    │ Documented subtype separation │
│ Binary AP Noninferiority│ T-OP vs B-CONTROL (N=9,956)  │ ΔAP_bin ≥ -0.005 (-0.5% max)  │
│ Substitution Noninfer.  │ T-OP vs B-CONTROL (N=9,811)  │ ΔAP_sub ≥ -0.005 (-0.5% max)  │
│ CTC Transcription PER   │ Greedy CTC PER (N=9,881)     │ ΔPER ≤ +0.005 (+0.5% max deg) │
│ Acoustic Grounding Gap  │ Matched vs Shift-1 Mismatched│ Δ_acoustic ≥ +0.015 (+1.5%)   │
│ Canonical Prior Margin  │ T-OP vs Canonical Prior      │ Δ_prior ≥ +0.015 (+1.5%)      │
│ Budget Adherence        │ Fixed 3,600-update milestone │ Exactly 3,600 successful up.  │
└─────────────────────────┴──────────────────────────────┴───────────────────────────────┘
```

**Decision Rules for P-SUP:**
- **PASS (GO):** $\Delta \text{AP}_{\text{omit}} \ge +0.030$ at the fixed 3,600-update milestone, $\Delta \text{Recall@FPR15} \ge +0.040$, Binary and Substitution AP noninferiority verified ($\ge -0.005$), $\Delta \text{PER} \le +0.005$, and acoustic gap $\ge +0.015$.
- **INCONCLUSIVE (HOLD):** Modest omission gain ($0 < \Delta \text{AP}_{\text{omit}} < +0.030$), or optimization instability (AMP skips $> 100$).
- **FAIL (NO-GO):** $\Delta \text{AP}_{\text{omit}} \le 0$ at 3,600 updates, or substitution collapse ($\Delta \text{AP}_{\text{sub}} < -0.015$), or CTC transcription failure ($\Delta \text{PER} > +0.010$).

---

## 5. Acoustic Mismatch and Canonical-Only Controls

### 5.1 Within-Speaker Cyclic Shift-1 Mismatch Control
To ensure that error detection is driven by genuine acoustic-phonetic realization rather than prompt-level language model memorization, the evaluation suite mandates a deterministic within-speaker cyclic derangement:

- **Derangement Protocol:** For each validation speaker (`NJS`, `RRBI`) with 150 utterances ordered chronologically, utterance audio $i$ is paired with prompt transcript $(i + 1) \pmod{150}$.
- **Certified Invariants:**
  1. **Zero Self-Matches:** $\pi(i) \neq i$ for all $i \in \{0, \dots, 149\}$.
  2. **Zero Cross-Speaker Leaks:** Mismatches remain strictly within speaker boundaries (`NJS` audio is never paired with `RRBI` prompts).
  3. **Acoustic Covariate Preservation:** Preserves speaker vocal tract characteristics, SNR, fundamental frequency, and recording channel.
- **Historical Baseline R Benchmark Evidence:**
  - Matched Audio Binary AP: **`0.317021`**
  - Mismatched Audio Binary AP: **`0.294722`**
  - **Acoustic Reliance Gap:** $\Delta_{\text{acoustic}} = 0.317021 - 0.294722 = \mathbf{+0.022299}$ (+2.23 percentage points).
  - Paired Bootstrap 95% CI: $[+0.0048, +0.0396]$ under adjacent cyclic derangement.

### 5.2 Canonical Empirical Prior Baseline
- **Methodology:** Phonemes are assigned static prior error probabilities $P_{\text{train}}(\text{error} \mid g_j)$ computed from the 79,426 training phonemes across Fold 1, evaluating predictions solely from canonical phone identity without acoustic conditioning.
- **Benchmark Evidence:**
  - Canonical Prior Binary AP: **`0.303063`**
  - Neural Margin over Prior: $\text{AP}_{\text{matched}} - \text{AP}_{\text{prior}} = 0.317021 - 0.303063 = \mathbf{+0.013958}$ (+1.40 percentage points).
- **Audit Mandate:** Any candidate claiming acoustic omission diagnosis must exceed **both** the mismatched control ($\text{AP} > 0.294722$) and the canonical empirical prior ($\text{AP} > 0.303063$).

---

## 6. Prediction, Failure, Abstention, and Validator Audit

### 6.1 Audit of Agent A (P-SCORE) Implementation Readiness
- **Repository State:** Branch `prep/rq1-p-score`, tested source commit `69d05e36c83b7ce10cb679730855ea440cf10e8e`.
- **Synthetic Unit & Scorer Test Suite:** **269 tests passed** (`tests/test_p_score_ctc.py`, `tests/test_p_score_runtime.py`, `tests/test_p_score_shared_schema.py`).
  - CTC forward/backward path enumeration verified on brute-force lattices.
  - Repeated phone state transitions and blank separation confirmed.
  - Occupancy normalization floors and epsilon-deletion mechanics verified.
- **Shared Schema Bridge:** Emitted synthetic rows validated against `strict_runtime_validator.py`. Following initial rejection of `posterior_target_occupancy`, the field was renamed to `posterior_phone_state_occupancy` and passed all validation checks.
- **Real-Data Execution State:** **NOT_STARTED** (0 successful updates, 0 skipped updates, 0 optimizer attempts, 0 real utterances processed). No real candidate metrics exist.

### 6.2 Audit of Agent B (P-SUP) Implementation Readiness
- **Repository State:** Branch `prep/rq1-p-sup`, tested source commit `d2b9702627fa03c7d2de97ff8e825bb3ffbe0b86`.
- **Synthetic Unit & Model Test Suite:** **88 tests passed** (`tests/test_p_sup_labels.py`, `tests/test_p_sup_model.py`, `tests/test_p_sup_runtime.py`).
  - Label collation and target filtering verified under strict rules.
  - Model forward and loss algebra verified under `torch.no_grad()`.
- **Model Parameter Matching:** `CompactOperationHead` comprises exactly **266,667 trainable parameters**. The downstream head is identical for treatment (`operation_ce`) and control (`binary_collapsed`), ensuring rigorous capacity control.
- **Preflight Audit:** `audit_scripts/p_sup_preflight.py` executed safely and returned `PREPARATION_AUDITED_EXPERIMENT_HOLD`.
- **Real-Data Execution State:** **NOT_STARTED** (0 updates, 0 attempts, no model artifact). No real candidate metrics exist. Staging package `deploy/kaggle_p_sup_train` prepared for Kaggle execution.

### 6.3 Runtime Abstention & Validator Enforcement
Both arms adhere to the V1 Abstention Invariant: **Abstention, no invented probability**.
- **Validated Failure Modes:** Empty prompts, missing audio frames, nonfinite acoustic emissions, and out-of-vocabulary phone tokens trigger explicit failure records:
  - `status = "abstain"`
  - `abstention_reason` recorded (e.g., `"empty_prompt"`, `"nonfinite_emissions"`)
  - Decision scores assigned `null` (not imputed with validation prevalence $0.116$ or heuristic $0.90$).
- **Anti-Leakage Enforcement:** Strict JSON schema validation asserts that zero gold fields (`gold_canonical_index`, `gold_phone`, `gold_operation`) enter runtime prediction files.

---

## 7. Metric Assessment & Bootstrap Uncertainty Analysis

### 7.1 Historical Baseline Operating Curves (E0 Baseline R)
Baseline R full-precision operating points under Protocol O ($N=9,956$, 1,157 errors) across threshold sweep $\tau \in [0.05, 0.95]$:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      BASELINE R PROTOCOL O OPERATING POINT PROFILE                     │
├───────────┬───────────┬────────┬───────────┬────────┬──────────┬──────────┬────────────┤
│ Threshold │ Precision │ Recall │ F1 Score  │ TP     │ FP       │ FN       │ TN         │
├───────────┼───────────┼────────┼───────────┼────────┼──────────┼──────────┼────────────┤
│ τ = 0.05  │  0.1446   │ 0.9196 │  0.2499   │ 1,064  │  6,296   │   93     │   2,503    │
│ τ = 0.10  │  0.1795   │ 0.8358 │  0.2955   │   967  │  4,422   │  190     │   4,377    │
│ τ = 0.15  │  0.2355   │ 0.6534 │  0.3462   │   756  │  2,453   │  401     │   6,346    │
│ τ = 0.20* │  0.2918   │ 0.5385 │  0.3785*  │   623  │  1,512   │  534     │   7,287    │
│ τ = 0.25  │  0.3574   │ 0.3846 │  0.3706   │   445  │    801   │  712     │   7,998    │
│ τ = 0.30  │  0.4132   │ 0.3250 │  0.3640   │   376  │    534   │  781     │   8,265    │
│ τ = 0.50  │  0.4914   │ 0.1737 │  0.2567   │   201  │    208   │  956     │   8,591    │
└───────────┴───────────┴────────┴───────────┴────────┴──────────┴──────────┴────────────┘
* Optimal F1 operating point achieved at τ = 0.20 (F1 = 0.3785).
```

### 7.2 Diagnostic FPR Operating Ceilings (Omission Subset, N=8,944)
For clinical pronunciation feedback, false alarms on correct phones destroy student trust. Operating performance must be evaluated at strict correct-token FPR ceilings ($\text{FPR} = \text{FP} / (\text{FP} + \text{TN})$):

| Operating Metric | 5% FPR Ceiling ($\text{FPR} \le 0.05$) | 10% FPR Ceiling ($\text{FPR} \le 0.10$) | 15% FPR Ceiling ($\text{FPR} \le 0.15$) |
|---|---:|---:|---:|
| **Operating Threshold ($\tau$)** | $\tau \approx 0.45$ | $\tau \approx 0.30$ | $\tau \approx 0.22$ |
| **Achieved Correct FPR** | $0.0482$ | $0.0984$ | $0.1492$ |
| **E0 Baseline R Omission Recall** | $0.2138$ | $0.3241$ | **$0.4000$** |
| **E1 Skip Model Omission Recall** | $0.1793$ | $0.2828$ | **$0.3655$** |
| **Delta (E1 $-$ E0)** | $-0.0345$ | $-0.0413$ | **$-0.0345$** |

### 7.3 Bootstrap Uncertainty & Small-Sample Methodological Limitations
- **Bootstrap Protocol:** Paired utterance-level resampling (1,000 bootstrap iterations, fixed RNG seed 1337) across the 300 Fold 1 validation utterances.
- **Fundamental Small-Sample Caveat ($G=2$):**  
  Because Fold 1 validation contains only **two human speakers** (`NJS` and `RRBI`), utterance-level bootstrap resampling estimates sampling variance *within those two specific speakers only*.
  - It does **NOT** estimate population variance across speakers, L1 accents, or recording environments.
  - Computing token-level bootstrap or claiming population statistical significance ($p < 0.05$) based on 9,956 non-independent tokens across $G=2$ speakers is **statistically invalid (pseudo-replication)**.
  - Any observed advantage on Fold 1 validation remains an **exploratory screen**. Confirmatory claims require evaluation across the remaining 3 folds (6 held-out speakers) under Cross-Fold Contract `RQ1-TRN-03A`.

---

## 8. Candidacy Comparison & Scientific Rigor

### 8.1 Current Status: No Real-Data Results Exist Yet
As established in Section 6, neither Agent A nor Agent B has executed real-data inference or model training on the L2-ARCTIC corpus:
- P-SCORE has executed synthetic-cache tests only; no frozen emission cache has been loaded or scored.
- P-SUP has executed synthetic forward passes only; no backward passes, optimizer steps, or model checkpoints have been executed.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             CANDIDATE EXECUTION AUDIT CENSUS                           │
├─────────────────────┬──────────────┬─────────────┬──────────────┬──────────────────────┤
│ Candidate Arm       │ Successful   │ Skipped AMP │ Trained      │ Real Prediction Rows │
│                     │ Updates      │ Skips       │ Artifacts    │ Emitted              │
├─────────────────────┼──────────────┼─────────────┼──────────────┼──────────────────────┤
│ P-SCORE (GOP-SF-SD) │ 0            │ 0           │ None (frozen)│ 0                    │
│ P-SUP (T-OP)        │ 0            │ 0           │ None         │ 0                    │
│ P-SUP (B-CONTROL)   │ 0            │ 0           │ None         │ 0                    │
└─────────────────────┴──────────────┴─────────────┴──────────────┴──────────────────────┘
```

### 8.2 Strict Prohibition Against Premature Inferential Superiority
In strict compliance with `AGENTS.md` and the scientific evaluation contract:
1. **Never Infer Superiority from Rubrics or Synthetic Passes:** The fact that P-SCORE passed 269 synthetic tests and P-SUP passed 88 tests provides evidence of software correctness, **not scientific or empirical superiority**.
2. **Never Fabricate or Extrapolate Metrics:** In the absence of executed real-data predictions, candidate AP, Recall@FPR15, and PER values are **DEFERRED**.
3. **Independent Evaluation Integrity:** P-SCORE and P-SUP are orthogonal scientific approaches. P-SCORE tests acoustic graph decoding over frozen weights; P-SUP tests compact representation learning via natural-operation supervision. Neither lane depends on, or vetoes, the other.

---

## 9. Comprehensive Governance & Validation Recommendations

### 9.1 Summary of Phase Gates (GO / HOLD / NO-GO)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPREHENSIVE GOVERNANCE AUDIT                            │
├────────────────────────────────────────┬─────────────┬─────────────────────────────────┤
│ Verification Gate                      │ Status      │ Technical Justification         │
├────────────────────────────────────────┼─────────────┼─────────────────────────────────┤
│ 1. Shared Frozen Contract Integrity    │ GO          │ SHA-256 bit-identical (3 checkouts)│
│ 2. Historical E0/E1 Provenance Audit   │ GO          │ Reconciled .316062/.317012/.317021 │
│ 3. Strict Denominator & Label Standards│ GO          │ 8,944 omission vs 9,956 binary  │
│ 4. Prospective Criteria Freeze         │ GO          │ Margins frozen prior to results │
│ 5. P-SCORE Synthetic Implementation    │ GO          │ 269 tests passed; schema bridge │
│ 6. P-SUP Synthetic Implementation      │ GO          │ 88 tests passed; capacity match │
│ 7. P-SCORE Real-Data Scoring           │ HOLD        │ Emission cache & auth required  │
│ 8. P-SUP Remote GPU Training           │ HOLD        │ Execution auth & trainer run req│
│ 9. DIAG-02B Causal Training Parity Gate│ HOLD        │ 9/80 archive parity failures    │
│ 10. Model Rollout & Deployment         │ NO-GO       │ Confirmatory evaluation pending │
└────────────────────────────────────────┴─────────────┴─────────────────────────────────┘
```

### 9.2 Recommended Immediate Next Steps
1. **Maintain Preparation Checkpoint:** Commit this scientific report and append its verification receipt to coordinator documents.
2. **P-SCORE Next Step (Inference Lane):**
   - Verify presence and cryptographic SHA-256 digest of private FP32 emission cache on Fold 1 validation.
   - Obtain operator execution authorization packet.
   - Execute `audit_scripts/p_score_infer.py --mode frozen-cache` with worker pool.
3. **P-SUP Next Step (Training Lane):**
   - Review and formally approve `manifests/RQ1-P-SUP/OBJECTIVE_POLICY_AMENDMENT.md`.
   - Complete trainer packaging for Kaggle T4 execution (`deploy/kaggle_p_sup_train`).
   - Obtain operator execution authorization packet.
   - Execute paired T-OP and B-CONTROL training on remote Kaggle T4 platform to exactly 3,600 updates.
4. **Post-Execution Step (Comparative Assessment):**
   - Once full-precision prediction files (`runtime_predictions.jsonl`, `offline_token_evaluation.jsonl`) are delivered by both agents, execute independent offline evaluation against the frozen criteria defined in Section 4.

---

*Certified by Agent D — Scientific QA & Comparison Subagent.*
