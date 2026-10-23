# LANGFENS — RQ1-STD-01R: Specification Consistency & Pre-Implementation Gate Report

**Execution Gate:** RQ1-STD-01R — Specification Consistency & Pre-Implementation Gate  
**Gate Lead / Evaluator:** Senior Speech ML Research Engineer & Scientific Reproducibility Reviewer  
**Audit Target:** All Eight Documents in `manifests/RQ1-STD-01/`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/reconciliation/`  
**Associated Addenda Deliverables:**
1. `manifests/RQ1-STD-01/reconciliation/CORRECTED_LABEL_ACCOUNTING.md`
2. `manifests/RQ1-STD-01/reconciliation/VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md`
3. `manifests/RQ1-STD-01/reconciliation/UNVERIFIED_CLAIMS_AND_DECISIONS.md`
**Host & Runtime Baseline:** Linux khoa 7.2.11-1-cachyos (x86_64), Python 3.14.8, Timezone: Asia/Ho_Chi_Minh (+07:00)  
**Corpora Audited:** L2-ARCTIC Release v5.0 (3,599 Gold TextGrids) & SpeechOcean762 (OpenSLR SLR101, 5,000 Scored Utterances)  
**Execution Mode:** STRICT READ-ONLY SCIENTIFIC AUDIT (Zero Dataset Mutations, Zero Training Runs, Zero Premature Manifest Freezes)  

---

## 1. Executive Summary & Gate Verdict

This review (**RQ1-STD-01R**) constitutes the formal scientific pre-implementation gate for Project Langfens.
Before authorizing the creation of production dataset splits (`RQ1-SPLIT-01`), preprocessing pipelines (`RQ1-PREP-01`), or acoustic model fine-tuning (`RQ1-TRN-01`), the eight core specification documents authored under Phase **RQ1-STD-01** were subjected to rigorous cross-examination against raw empirical evidence, cryptographic checksums, mathematical proofs, and literature precedents.

Operating strictly under a read-only mandate, this audit:
1. Reconciled all phonemic substitution accounting discrepancies, discovering the physical ground-truth decomposition ($14,098 = 12,368 \text{ clean} + 1,472 \text{ asterisk} + 258 \text{ err}$) and mathematically identifying the source of the previously reported 1,011 discrepancy.
2. Disproved the gender-balance claim in the prior 4-fold test split proposal (which assigned 5M/1F to Fold 1 and 1M/5F to Fold 3), proved combinatorial feasibility, and constructed a candidate allocation table that simultaneously satisfies 3M/3F, 1 per L1, and zero leakage across all four folds.
3. Identified and resolved a 26-speaker contradiction in the previous Vietnamese Leave-One-Speaker-Out (LOSO) specification (where validation speakers were double-counted in training), formalizing two provably disjoint regimes: Cross-Vietnamese Generalization (21 train / 2 val / 1 test) and Zero-Shot Vietnamese Transfer (18 train / 2 val / 4 test).
4. Audited all unverified performance claims (including the 1.68 ms CPU resampling latency and 4-worker DataLoader GPU saturation assertions), formally downgrading them to `ESTIMATE`, `THEORETICAL_PROJECTION`, or `NOT_BENCHMARKED`, and instituting a mandatory micro-profiling gate in `RQ1-PREP-01`.
5. Cross-checked all eight documents across nine critical methodological domains to verify complete mutual consistency.

### Official Gate Verdict

$$\mathbf{RQ1\text{-}STD\text{-}01R: PASS}$$

*Conditions for Passage Met:*
1. All annotation counts across L2-ARCTIC reconcile with zero unexplained variance ($\Delta = 0$).
2. Asterisk totals are disambiguated by physical counting scope ($1,472$ in substitutions; $1,496$ in `phones` tier; $1,550$ in entire TextGrid corpus).
3. Candidate 4-fold assignments satisfy all six stated demographic and structural constraints programmatically.
4. Vietnamese LOSO partitions are proven to be strictly disjoint with zero speaker leakage and calibration isolation.
5. All unsupported performance claims are formally cataloged and downgraded.
6. The open decisions requiring human operator sign-off are explicitly formulated.

---

## 2. Cross-Document Methodology Consistency Audit (Task E)

The eight documents in `manifests/RQ1-STD-01/` were evaluated across nine critical scientific dimensions. Below is the systematic consistency ledger:

### 2.1 The Nine Methodological Consistency Checks

| Domain | Specification Documents Checked | Identified Ambiguity / Prior Contradiction | Reconciled Ground Truth & Resolution | Consistency Status |
| :--- | :--- | :--- | :--- | :---: |
| **1. `PPL*` Label Policies** | `MDD_LABEL_SCHEMA.md`<br>`EVALUATION_PROTOCOL.md`<br>`OPEN_DECISIONS_AND_RISKS.md` | Prior draft (`MDD_LABEL_ACCOUNTING_SPEC.md`, Sec. 2) defined $b_i=0 \iff$ single phone or `PPL*`, implying all asterisks are correct, contradicting Sec. 3. | Reconciled: Policy 1 (Strict) treats all 1,472 asterisk substitutions as Errors ($E=17,518$; $14.70\%$). Policy 2 (Lenient) reclassifies 1,112 base-preserving allophones as Correct ($C=102,745$; $E=16,406$; $13.77\%$). Both policies have strictly invariant denominator $N=119,151$. | **CONSISTENT** |
| **2. `err` Masking** | `MDD_LABEL_SCHEMA.md`<br>`EVALUATION_PROTOCOL.md`<br>`PHONEME_MAPPING_SPEC.md` | Previous spec cited 269 `err` tags subtracted from substitution denominator ($14,098 - 269 = 13,829$). | Reconciled: Exactly 258 `err` tags occur in substitutions (11 occur in epenthetic additions). Evaluated 39-class diagnosis denominator is strictly $S_{\text{eval}} = 14,098 - 258 = \mathbf{13,840}$. In binary MDD, `err` is always Error ($b_i=1$). | **CONSISTENT** |
| **3. Provisional `V8` Handling** | `MDD_LABEL_SCHEMA.md`<br>`DATA_CONTRACT_SPEC.md`<br>`OPEN_DECISIONS_AND_RISKS.md` | `SVBI/arctic_a0365` contains `V,V8,s` in word *even*. Keyboard slip hypothesis (`Shift+8 = *`) could be assumed as fact. | Reconciled: Explicitly designated as **PROVISIONAL**. Data contract exposes `contains_provisional_v8` boolean flag. In binary MDD, it is Error ($b_i=1$) under all interpretations; in diagnosis, mapped to `V` under Option A, masked under Option B. | **CONSISTENT** |
| **4. ARPAbet / AX / IPA Mappings** | `PHONEME_MAPPING_SPEC.md`<br>`MDD_LABEL_SCHEMA.md`<br>`DATA_CONTRACT_SPEC.md` | Risk of symbol collision between L2-ARCTIC `AX` (schwa) and CMUDict/SO762 `AH0`. | Reconciled: Multi-layer architecture: Layer 0 preserves raw `AX`; Layer 1 maps `AX` $\rightarrow$ `AH`; Layer 3 retains `AX` as diagnostic CTC token (Index 3). Complete 39-phone IPA table verified. | **CONSISTENT** |
| **5. CTC Blank vs. Silence** | `PHONEME_MAPPING_SPEC.md`<br>`DATA_CONTRACT_SPEC.md` | Potential conflation of CTC blank with acoustic pause or phone deletion. | Reconciled: CTC `<BLANK>` (Index 0) is an alignment non-emission token. Acoustic silence `<SIL>` (Index 1) represents acoustic pauses. Phone deletions (`CPL,sil,d`) are canonical prompt tokens with acoustic absence. | **CONSISTENT** |
| **6. Insertion Event Denominators** | `MDD_LABEL_SCHEMA.md`<br>`EVALUATION_PROTOCOL.md`<br>`STANDARDIZATION_POLICY.md` | Prior audit reports summed insertions into PER ($15.62\%$) and binary error, producing $100.92\%$ total. | Reconciled: Canonical denominator is strictly $N_{\text{canonical}} = C + S + D = 119,151$. Insertions ($I = 1,092$) occur between tokens over silence and are evaluated strictly as an independent continuous acoustic event detection task (Task 4) with False Alarm Rate (FAR) metric. | **CONSISTENT** |
| **7. Speaker Leakage Prevention** | `SPEAKER_SPLIT_PROTOCOL.md`<br>`OPEN_DECISIONS_AND_RISKS.md` | Prior 4-fold test splits had 5M/1F in Fold 1 and 1M/5F in Fold 3; Vietnamese LOSO had 26 speakers. | Reconciled: Candidate 4-fold allocation satisfies $3\text{M} / 3\text{F}$ and $1\text{ per L1}$ across all 4 folds. Vietnamese LOSO partitioned into 21 train / 2 val / 1 test (Regime A) and 18 train / 2 val / 4 test (Regime B). Decision threshold $\theta^*$ calibrated strictly on validation. | **CONSISTENT** |
| **8. Bootstrap Confidence Intervals** | `EVALUATION_PROTOCOL.md` | Risk of token-level i.i.d. assumption inflating statistical significance. | Reconciled: Non-parametric bootstrap ($B=1,000$) clustered at the utterance level to capture intra-speaker and intra-sentence phonotactic correlations. | **CONSISTENT** |
| **9. Licensing & Provenance** | `STANDARDIZATION_POLICY.md`<br>`DATA_CONTRACT_SPEC.md` | Multi-dataset license compliance and raw archive mutability. | Reconciled: L2-ARCTIC (CC BY 4.0), SpeechOcean762 (CC BY 4.0 via OpenSLR), CMUDict (BSD). Raw archives in `datasets/*/original/` remain 100% immutable and read-only. Provenance verified via SHA-256. | **CONSISTENT** |

---

## 3. Detailed Task A Findings: Substitution Accounting Reconciliation

### 3.1 The 1,011 Discrepancy Reconciliation Ledger
The table below documents the exact transition from the erroneous draft values in `MDD_LABEL_SCHEMA.md` to the physically verified counts:

| Metric / Category | Previous Draft Citation | Reconciled Physical Count | Difference ($\Delta$) | Forensic Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **Total Phonemic Substitutions ($S$)** | 14,098 | **14,098** | 0 | Ground truth confirmed across all 3,599 TextGrids. |
| **Standard Substitutions** | 11,048 | **12,368** | +1,320 | Previous draft subtracted base-preserving allophones twice ($14,098 - 1,770 - 1,281 = 11,047$). Correct count is $14,098 - 1,472 - 258 = 12,368$. |
| **Asterisk Substitutions** | 1,770 | **1,472** | -298 | 1,770 was an unverified draft estimate. Physical census finds 1,472 in substitutions and 24 in additions (1,496 total in `phones` tier). |
| • *Base-Preserving Allophones* | 1,281 | **1,112** | -169 | Exact count of substitution intervals where $\text{base}(C) = \text{base}(P)$ with `*`. |
| • *Category-Shifting Asterisk* | 489 | **360** | -129 | Exact count of substitution intervals where $\text{base}(C) \ne \text{base}(P)$ with `*`. |
| **Uncertain `err` Substitutions** | 269 | **258** | -11 | 269 was the global count of `err` across all error types. Exactly 11 `err` tags occur in epenthetic additions (`sil,err,a`), leaving 258 in substitutions. |
| **Sum of Reported Subcategories** | **13,087** | **14,098** | **+1,011** | $12,368 + 1,472 + 258 = \mathbf{14,098}$. Discrepancy completely eliminated. |

### 3.2 Formal Accounting Proof
Let $S$ denote the set of all intervals in the `phones` tier with `err_type == 's'`:
$$|S| = 14,098$$
We partition $S$ based on the mutually exclusive predicates:
$$\begin{aligned}
S_{\text{err}} &= \{i \in S \mid \text{'err'} \in \text{label}_i\} \\
S_{\text{star, same}} &= \{i \in S \setminus S_{\text{err}} \mid \text{'*'} \in \text{label}_i \land \text{base}(C_i) = \text{base}(P_i)\} \\
S_{\text{star, diff}} &= \{i \in S \setminus S_{\text{err}} \mid \text{'*'} \in \text{label}_i \land \text{base}(C_i) \ne \text{base}(P_i)\} \\
S_{\text{clean}} &= S \setminus (S_{\text{err}} \cup S_{\text{star, same}} \cup S_{\text{star, diff}})
\end{aligned}$$

By construction, these four sets are pairwise disjoint:
$$\forall A, B \in \{S_{\text{err}}, S_{\text{star, same}}, S_{\text{star, diff}}, S_{\text{clean}}\}, \ A \ne B \implies A \cap B = \emptyset$$
Their physical cardinalities from exhaustive re-parsing are:
$$|S_{\text{err}}| = 258, \quad |S_{\text{star, same}}| = 1,112, \quad |S_{\text{star, diff}}| = 360, \quad |S_{\text{clean}}| = 12,368$$
Summing the cardinalities:
$$|S_{\text{err}}| + |S_{\text{star, same}}| + |S_{\text{star, diff}}| + |S_{\text{clean}}| = 258 + 1,112 + 360 + 12,368 = \mathbf{14,098} = |S|$$
$\blacksquare$ *The partition is exact, complete, and verified.*

---

## 4. Detailed Task B Findings: 4-Fold Evaluation Rectification

### 4.1 Diagnosis of Prior 4-Fold Defect
The previous table in `SPEAKER_SPLIT_PROTOCOL.md` (Section 4) exhibited extreme biological sex distortion:
- **Fold 1:** 5 Males (`ABA`, `BWC`, `ASI`, `EBVS`, `HQTV`), 1 Female (`HJK`).
- **Fold 3:** 1 Male (`YBAA`), 5 Females (`NCC`, `SVBI`, `YDCK`, `MBMPS`, `THV`).
This would have invalidated cross-fold benchmarking because acoustic performance differences between folds would reflect pitch ($F_0$) and formant scale variation rather than phonological error detection.

### 4.2 Validated Candidate Allocation Summary
The corrected candidate allocation table (detailed in `VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md`) achieves perfect demographic stratification:
- **Fold 1 Test:** `ABA` (Arab, M), `LXC` (Chin, F), `ASI` (Hind, M), `HJK` (Kor, F), `EBVS` (Span, M), `PNV` (Viet, F) $\rightarrow$ **3M / 3F**.
- **Fold 2 Test:** `SKA` (Arab, F), `BWC` (Chin, M), `SVBI` (Hind, F), `HKK` (Kor, M), `MBMPS` (Span, F), `HQTV` (Viet, M) $\rightarrow$ **3M / 3F**.
- **Fold 3 Test:** `YBAA` (Arab, M), `NCC` (Chin, F), `RRBI` (Hind, M), `YDCK` (Kor, F), `ERMS` (Span, M), `THV` (Viet, F) $\rightarrow$ **3M / 3F**.
- **Fold 4 Test:** `ZHAA` (Arab, F), `TXHC` (Chin, M), `TNI` (Hind, F), `YKWK` (Kor, M), `NJS` (Span, F), `TLV` (Viet, M) $\rightarrow$ **3M / 3F**.

In each fold:
- **Test:** Exactly 6 speakers (3M, 3F, 1 per L1).
- **Val:** Exactly 2 speakers (1M, 1F, distinct L1s).
- **Train:** Exactly 16 speakers (8M, 8F, all 6 L1s).
- **Zero Leakage:** $S_{\text{train}} \cap S_{\text{val}} \cap S_{\text{test}} = \emptyset$.
- Total test tokens per fold are remarkably balanced: 29,769 (Fold 1), 29,798 (Fold 2), 29,798 (Fold 3), 29,786 (Fold 4), summing exactly to $119,151$ canonical tokens.

---

## 5. Detailed Task C Findings: Vietnamese LOSO Rectification

### 5.1 Resolution of the 26-Speaker Contradiction
The previous specification stated:
$$\text{Train (23)} + \text{Val (2)} + \text{Test (1)} = \mathbf{26 \text{ speakers}}$$
In a 24-speaker corpus, this required reusing the 2 validation speakers in the training pool.

### 5.2 The Two Validated Vietnamese Regimes
We formalize two mutually disjoint protocols:

#### Regime A: Cross-Vietnamese Generalization (Intra-L1 LOSO; 4 Rounds)
- **Objective:** Evaluate generalization to a new Vietnamese speaker when other Vietnamese learners are present during training.
- **Cardinals per Round:**
  - **Train:** $21$ speakers (18 non-Vietnamese + 3 remaining Vietnamese speakers).
  - **Val:** $2$ non-Vietnamese speakers (`RRBI` [M, Hindi], `NJS` [F, Spanish]).
  - **Test:** $1$ held-out Vietnamese speaker (`HQTV`, `PNV`, `THV`, or `TLV`).
  - Total: $21 + 2 + 1 = \mathbf{24 \text{ speakers}}$ ($3,599$ utterances).
- **Calibration Isolation:** Threshold $\theta^*$ is calibrated strictly on the validation speakers; the held-out Vietnamese speaker is evaluated only at inference.

#### Regime B: Zero-Shot Vietnamese-L1 Transfer
- **Objective:** Evaluate generalization to Vietnamese speech when the model has never heard any Vietnamese speaker during training or validation.
- **Cardinals:**
  - **Train:** $18$ non-Vietnamese speakers ($2,699$ utterances; $89,059$ tokens; $11,640$ errors; $13.07\%$).
  - **Val:** $2$ non-Vietnamese speakers (`RRBI`, `NJS`; $300$ utterances; $9,956$ tokens; $1,157$ errors).
  - **Test:** All $4$ Vietnamese speakers (`HQTV`, `PNV`, `THV`, `TLV`; $600$ utterances; $20,136$ tokens; $4,721$ errors; $23.45\%$).
  - Total: $18 + 2 + 4 = \mathbf{24 \text{ speakers}}$ ($3,599$ utterances).

Both regimes are mathematically proven disjoint: $S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$.

---

## 6. Detailed Task D Findings: Downgrading Unsupported Performance Claims

Every performance claim lacking reproducible logs has been audited and downgraded in `UNVERIFIED_CLAIMS_AND_DECISIONS.md`:
1. **1.68 ms Resampling Latency (`AUDIO_PREPARATION_SPEC.md`, Sec. 5):**
   - Lacks CPU profiling script and timing log.
   - **Downgraded to:** `ESTIMATE` / `NOT_BENCHMARKED`.
2. **Four-Worker DataLoader GPU Saturation (`AUDIO_PREPARATION_SPEC.md`, Sec. 1 & 5):**
   - Asserts 100% GPU saturation (~2,380 utts/s) without GPU telemetry or profiling traces.
   - **Downgraded to:** `THEORETICAL_PROJECTION` / `NOT_BENCHMARKED`.
3. **15 GB Derived WAV Storage Savings (`AUDIO_PREPARATION_SPEC.md`, Sec. 5):**
   - Unverified back-of-the-envelope calculation.
   - **Downgraded to:** `ESTIMATE`.
4. **Multi-Task Loss Weights ($\lambda = [1.0, 0.5, 0.5]$) (`OPEN_DECISIONS_AND_RISKS.md`, Dec. 7):**
   - Stated as recommendation without validation sweep.
   - **Downgraded to:** `UNTESTED_HYPERPARAMETER`.
5. **Lower 6 Layers Freezing (`OPEN_DECISIONS_AND_RISKS.md`, Dec. 6):**
   - Theoretical conjecture based on generic SSL literature.
   - **Downgraded to:** `THEORETICAL_HYPOTHESIS`.

A mandatory micro-benchmarking gate is established for Phase `RQ1-PREP-01` to gather verifiable measurements before freezing pipeline parameters.

---

## 7. Master Specification Consistency Matrix

| Specification Document | Original Function | Reconciled Status in RQ1-STD-01R | Addendum Superseding Document |
| :--- | :--- | :---: | :--- |
| `STANDARDIZATION_POLICY.md` | Core governance & principles | **RATIFIED** | Fully aligned with reconciled denominators and policies. |
| `MDD_LABEL_SCHEMA.md` | Label taxonomy & denominators | **RECONCILED** | Superseded in part by `CORRECTED_LABEL_ACCOUNTING.md` (exact subcategory counts, 1,472 asterisk tokens, 258 err tokens, $S_{\text{eval}}=13,840$). |
| `PHONEME_MAPPING_SPEC.md` | Phonetic inventories & CTC | **RATIFIED** | 39-phone core, Layer 0–3 spaces, AX mapping, and 70-token MFA dictionary fully verified. |
| `AUDIO_PREPARATION_SPEC.md` | Acoustic DSP & resampling | **RECONCILED** | Signal math verified; performance claims downgraded in `UNVERIFIED_CLAIMS_AND_DECISIONS.md`. Dynamic resampling subject to PREP-01 profiling gate. |
| `SPEAKER_SPLIT_PROTOCOL.md` | Partitioning architectures | **RECONCILED** | Superseded in part by `VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md` (validated 3M/3F 4-fold table and corrected Vietnamese LOSO Regimes A/B). |
| `EVALUATION_PROTOCOL.md` | Metrics & CAPT rubrics | **RATIFIED** | Evaluated substitution diagnosis denominator updated to $S_{\text{eval}} = 13,840$; all metrics mathematically aligned. |
| `DATA_CONTRACT_SPEC.md` | Dataloader record schema | **RATIFIED** | JSON/Pydantic schemas incorporate provisional typo flag, strict/lenient binary labels, and provenance hashes. |
| `OPEN_DECISIONS_AND_RISKS.md` | Decisions & risk register | **RECONCILED** | Augmented by `UNVERIFIED_CLAIMS_AND_DECISIONS.md` with reconciled quantities for operator sign-off. |

---

## 8. Pre-Implementation Gate Sign-Off & Transition Boundaries

### Permitted Actions in Subsequent Phases
Upon operator review and approval of the four addenda in `manifests/RQ1-STD-01/reconciliation/`:
1. **Phase RQ1-SPLIT-01:** Deterministic generation and freezing of JSON split manifests based on the validated candidate allocation table (`VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md`).
2. **Phase RQ1-PREP-01:** Implementation of PyTorch dataloaders, execution of the audio resampling micro-benchmark gate, and generation of verified example records complying with `DATA_CONTRACT_SPEC.md`.

### Strictly Prohibited Actions (Still Active)
- No training runs or model fine-tuning (`RQ1-TRN-01`).
- No mutation of raw archives or in-place file modifications.
- No inference or evaluation runs.

$$\mathbf{RQ1\text{-}STD\text{-}01R: COMPLETE\ AND\ PASSED}$$
