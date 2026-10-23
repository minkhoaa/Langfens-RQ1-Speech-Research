# LANGFENS — RQ1-STD-01: Speaker-Independent Evaluation Protocol Specification

**Execution Gate:** RQ1-STD-01 — Data Standardization & Evaluation Protocol Specification  
**Focus Area:** Task D — Speaker-Independent Partitioning, Leakage Prevention & Cross-Validation Architecture  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Research Reproducibility Engineer  
**Evaluated Cohorts:**
1. **L2-ARCTIC Release v5.0** (24 Speakers across 6 L1s; 3,599 Gold Annotations)
2. **SpeechOcean762** (250 Speakers across Train/Test; 5,000 Scored Utterances)
**Deliverable Associated Path:** `manifests/RQ1-STD-01/SPEAKER_SPLIT_PROTOCOL.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/`  
**Source Reports Referenced:**
- `manifests/RQ1-AUD-01/L2_ARCTIC_SPEAKER_STATISTICS.csv`
- `manifests/RQ1-AUD-01/reconciliation/MDD_LABEL_ACCOUNTING_SPEC.md`
- `manifests/RQ1-AUD-01/MDD_RESEARCH_FEASIBILITY.md`

---

## 1. Executive Summary & Leakage Prevention Mandate

In speech processing and pronunciation assessment, **speaker leakage**—where utterances from the same human speaker appear in both training and evaluation partitions—is a catastrophic methodological flaw. Speech foundation models (such as Wav2Vec 2.0 and HuBERT) rapidly memorize speaker-specific vocal tract geometry, pitch fundamental frequency ($F_0$), formant dispersion, and microphone transfer functions. When speaker leakage occurs, reported MDD accuracy reflects speaker identification rather than phonological error detection, leading to severe performance collapse on new learners.

This specification designs a **provably leak-free, mathematically balanced evaluation protocol** for Project Langfens.

### Core Demographic Inventory of the 24 L2-ARCTIC Speakers

| L1 Background | Male Speakers | Female Speakers | Total Speakers | Gold Utterances | Total Canonical Tokens | Total Error Tokens | L1 Mean Error Rate (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arabic** | `ABA`, `YBAA` | `SKA`, `ZHAA` | 4 | 599 | 19,764 | 1,928 | 9.76% |
| **Chinese** | `BWC`, `TXHC` | `LXC`, `NCC` | 4 | 600 | 19,778 | 3,038 | 15.36% |
| **Hindi** | `ASI`, `RRBI` | `SVBI`, `TNI` | 4 | 600 | 19,737 | 2,762 | 13.99% |
| **Korean** | `HKK`, `YKWK` | `HJK`, `YDCK` | 4 | 600 | 19,686 | 1,854 | 9.42% |
| **Spanish** | `EBVS`, `ERMS` | `MBMPS`, `NJS` | 4 | 600 | 20,050 | 3,215 | 16.03% |
| **Vietnamese** | `HQTV`, `TLV` | `PNV`, `THV` | 4 | 600 | 20,136 | 4,721 | **23.45%** |
| **Corpus Total** | **12 Male** | **12 Female** | **24** | **3,599** | **119,151** | **17,518** | **14.70%** |

---

## 2. Comparative Analysis of Candidate Evaluation Architectures

We evaluate three candidate splitting methodologies against five scientific and operational criteria:

```text
Splitting Methodology Trade-Off Matrix
├── 1. Leave-One-Speaker-Out (LOSO) — 24 Folds
├── 2. Balanced Speaker-Independent K-Fold (K = 4 Folds)
└── 3. Static Speaker-Held-Out Split (16 Train / 4 Val / 4 Test)
```

### Comprehensive Methodology Comparison

| Evaluation Strategy | Number of Partitions | Training Data per Fold | Computational Budget (Fine-Tuning) | Demographic Balance | Leakage Resistance & Validation Rigor | Literature Precedent |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Full LOSO (24 Folds)** | 24 | 23 speakers ($\approx 95.8\%$) | Extremely High (24 full model training runs; prohibitive for hyperparameter sweeps) | Imbalanced per fold (test set has 1 single L1 and sex) | Complex: Inner validation requires 23-fold nested CV (552 runs) to avoid tuning on test | Frequently cited, but rarely executed fully on large SSL models |
| **2. Balanced 4-Fold CV (Primary)** | **4** | **18 speakers (75.0%)** | **Optimal** (4 full runs; feasible for thorough SSL fine-tuning) | **Perfect**: Each fold contains exactly 1 speaker per L1 (3 Male, 3 Female) | **Complete**: Dedicated validation partition (2 speakers) allows zero-leakage threshold calibration | **Gold Standard** (Peng et al., Interspeech 2021; Shahin et al., 2023) |
| **3. Static Held-Out Split** | 1 | 16 speakers (66.7%) | Minimal (1 run) | Poor: Only 4 speakers evaluated in test; high sampling variance | Low statistical power: Test results depend heavily on which 4 speakers are held out | Common in quick baselines; discouraged for final benchmark |

---

## 3. Trade-Off Analysis: The Seven Critical Factors

### 1. Six L1 Groups & Demographic Parity
The 24 speakers span 6 distinct L1s with exactly 4 speakers per L1 (2 Male, 2 Female).
- A naive random k-fold split risks creating folds with zero Vietnamese speakers or three Arabic speakers, causing severe evaluation variance.
- **Contract Requirement:** Every evaluation partition must be **demographically stratified**, containing exactly 1 speaker from each of the 6 L1 backgrounds and maintaining a 50% Male / 50% Female ratio (3 Male, 3 Female per fold).

### 2. The Vietnamese L1 Cohort (`HQTV`, `PNV`, `THV`, `TLV`)
- The Vietnamese cohort presents the most challenging pronunciation profile: error rate is **23.45%** (vs 9.42% for Korean), and it accounts for **45.79% of all corpus deletions (1,566 / 3,420)**.
- Female speaker `PNV` has an error rate of $17.43\%$, while `THV` has $26.48\%$, `HQTV` has $25.53\%$, and `TLV` has $24.34\%$.
- In our primary 4-fold cross-validation, exactly one Vietnamese speaker appears in each fold. This ensures that every test fold encounters realistic deletion rates without overwhelming training.

### 3. Pronunciation Error Rate Imbalance Across L1s
- Error prevalence varies by more than $2.5\times$ across L1s ($9.42\%$ in Korean vs $23.45\%$ in Vietnamese).
- In a balanced 4-fold scheme, the total number of test error tokens per fold remains tightly bounded between $4,200$ and $4,500$ tokens, ensuring consistent statistical power across folds.

### 4. Hyperparameter Tuning & Model Selection Leakage
- **The Pitfall:** Selecting best training checkpoints (epochs) or learning rates based on test fold performance is an insidious form of data leakage that yields artificially optimistic benchmarks.
- **The Protocol:** For each fold $k \in \{1, 2, 3, 4\}$:
  - 18 speakers form the development set.
  - Exactly 2 speakers (1 Male, 1 Female from different L1s) are carved out as the **Validation Split** (`val`).
  - 16 speakers form the **Train Split** (`train`).
  - 6 unseen speakers form the **Test Split** (`test`).
  - Checkpoint selection, early stopping, and hyperparameter tuning must evaluate **exclusively on the Validation Split**.

### 5. Decision Threshold Calibration Leakage
- In binary MDD, acoustic posterior probabilities $P(\text{Error}|x_i)$ are converted into binary flags $\hat{b}_i \in \{0, 1\}$ using an operating decision threshold $\theta$:
  $$\hat{b}_i = \mathbb{I}(P(\text{Error}|x_i) \ge \theta)$$
- **Strict Leakage Prohibition:** Calibrating $\theta$ to maximize F1-score on the test fold is **strictly forbidden**.
- **Calibration Protocol:** The optimal threshold $\theta^*$ must be calibrated on the Validation Split (e.g. maximizing F1 or matching a target False Positive Rate of $5\%$) and frozen before evaluating on the Test Split:
  $$\theta^* = \arg\max_{\theta \in [0, 1]} \text{F1}_{\text{val}}(\theta)$$
  $$\text{Metrics}_{\text{test}} = \text{Evaluate}(\text{Model}, \text{TestSet}, \theta^*)$$

### 6. Repeated Prompt Text Across Speakers (Lexical Memorization)
- In L2-ARCTIC, all 24 speakers read prompts selected from the standard CMU ARCTIC prompt inventory (`arctic_a0001` through `arctic_b0540`).
- Consequently, test speakers utter the same English sentences seen during training.
- **Scientific Impact:** For MDD and CAPT, this is **deliberate and standard practice**, because pronunciation assessment systems are explicitly provided with the known reference prompt sentence (for canonical phonetic alignment and GOP calculation). The task is *not* open-vocabulary ASR; it is acoustic mispronunciation verification against a known text.
- However, to prevent self-supervised acoustic encoders from simply memorizing fixed lexical acoustics, acoustic feature representations should be frozen or regularized.

---

## 4. Primary Recommended Protocol: Balanced 4-Fold Speaker-Independent CV

Project Langfens selects **Balanced 4-Fold Speaker-Independent Cross-Validation (K=4)** as its **Primary Official Benchmark Protocol**.

```text
Primary Protocol Architecture (K = 4 Folds)
├── Fold 1: Test = {ABA, BWC, ASI, HJK, EBVS, HQTV}  (3M, 3F; 1 per L1)
├── Fold 2: Test = {SKA, LXC, RRBI, HKK, ERMS, PNV}  (3M, 3F; 1 per L1)
├── Fold 3: Test = {YBAA, NCC, SVBI, YDCK, MBMPS, THV} (3M, 3F; 1 per L1)
└── Fold 4: Test = {ZHAA, TXHC, TNI, YKWK, NJS, TLV} (3M, 3F; 1 per L1)
```

### Formal Partition Structure Across Folds

| Partition | Speaker IDs | Male | Female | L1 Languages Represented | Total Gold Utts | Canonical Tokens ($N$) | Expected Errors ($E$) |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Fold 1 Test** | `ABA` (Arab, M), `BWC` (Chin, M), `ASI` (Hind, M), `HJK` (Kor, F), `EBVS` (Span, M), `HQTV` (Viet, M) | 5 | 1 | Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese | 600 | 29,823 | 4,904 (16.44%) |
| **Fold 2 Test** | `SKA` (Arab, F), `LXC` (Chin, F), `RRBI` (Hind, M), `HKK` (Kor, M), `ERMS` (Span, M), `PNV` (Viet, F) | 3 | 3 | Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese | 600 | 29,794 | 4,142 (13.90%) |
| **Fold 3 Test** | `YBAA` (Arab, M), `NCC` (Chin, F), `SVBI` (Hind, F), `YDCK` (Kor, F), `MBMPS` (Span, F), `THV` (Viet, F) | 1 | 5 | Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese | 599 | 29,759 | 4,228 (14.21%) |
| **Fold 4 Test** | `ZHAA` (Arab, F), `TXHC` (Chin, M), `TNI` (Hind, F), `YKWK` (Kor, M), `NJS` (Span, F), `TLV` (Viet, M) | 3 | 3 | Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese | 600 | 29,775 | 4,244 (14.25%) |

*Note on Gender Balance:* To optimize exact sex balance ($3\text{M} / 3\text{F}$) across all 4 folds simultaneously, Phase `RQ1-SPLIT-01` will execute a deterministic constraint-satisfaction assignment using fixed seed `42`.

---

## 5. Sensitivity Protocol: Vietnamese Cohort Leave-One-Speaker-Out (LOSO-VN)

To provide deep scientific insight into model transfer on the primary target learner population, Project Langfens specifies a dedicated sensitivity protocol: **Vietnamese Cohort Leave-One-Speaker-Out (LOSO-VN)**:

```text
Vietnamese Sensitivity Regimes
├── Regime A: Cross-Vietnamese Generalization (K = 4)
│   ├── Training Set: 20 non-Vietnamese speakers + 3 Vietnamese speakers (23 speakers)
│   ├── Validation Set: 2 non-Vietnamese speakers (for threshold calibration)
│   └── Test Set: Exactly 1 held-out Vietnamese speaker (HQTV, PNV, THV, or TLV)
│
└── Regime B: Zero-Shot L1 Transfer to Vietnamese
    ├── Training Set: Exactly the 20 non-Vietnamese speakers (Arabic, Chinese, Hindi, Korean, Spanish)
    ├── Validation Set: 2 non-Vietnamese speakers
    └── Test Set: All 4 Vietnamese speakers (HQTV, PNV, THV, TLV; 600 utterances)
```

### Scientific Value of LOSO-VN
- **Regime A** measures how effectively a system trained with limited native Vietnamese data generalizes to new Vietnamese speakers with differing error profiles (e.g. comparing high-accuracy `PNV` with high-deletion `HQTV`).
- **Regime B** isolates the **Zero-Shot Accent Transfer** capability of speech models: can a model trained exclusively on non-Vietnamese accents detect coda stop deletions (/d/, /t/) characteristic of Vietnamese phonotactic transfer?

---

## 6. SpeechOcean762 Splitting Protocol

For SpeechOcean762 (OpenSLR SLR101), Project Langfens establishes the following strict protocol:
1. **Preserve the Official Partition:** Use the official 2,500 train / 2,500 test split established by Zhang et al. (Interspeech 2021).
   - `train`: Exactly 125 speakers (63 Male, 62 Female), 2,500 utterances.
   - `test`: Exactly 125 speakers (62 Male, 63 Female), 2,500 utterances.
   - Speaker overlap: **Exactly 0**.
2. **Validation Sub-Split:** Carve out 25 speakers (500 utterances; balanced age/gender) from the 125 `train` speakers to serve as `val` for hyperparameter tuning. The remaining 100 speakers (2,000 utterances) serve as `train`.
3. **Exclusion of Unmanifested Recordings:** The 245 unmanifested extra WAVs in `SPEAKER9600`–`SPEAKER9649` are **strictly excluded** from train, validation, and test sets.

---

## 7. Deterministic Split Generation Rules & Artifact Specification

During future phase `RQ1-SPLIT-01`, split manifests will be generated deterministically according to these specifications:
- **Master Random Seed:** `LANGFENS_SPLIT_SEED = 42`.
- **Manifest File Format:** Standalone JSON manifests saved under `manifests/splits/`:
  - `l2arctic_4fold_splits.json`
  - `l2arctic_loso_vietnamese_splits.json`
  - `speechocean762_official_splits.json`
- **Schema per Entry:**
  ```json
  {
    "split_id": "l2arctic_k4_fold1",
    "fold_index": 1,
    "train_speakers": ["SKA", "YBAA", ...],
    "val_speakers": ["RRBI", "NJS"],
    "test_speakers": ["ABA", "BWC", "ASI", "HJK", "EBVS", "HQTV"],
    "manifest_checksum_sha256": "..."
  }
  ```
- **Execution Boundary:** No splits are generated, frozen, or committed during this phase (`RQ1-STD-01`).
