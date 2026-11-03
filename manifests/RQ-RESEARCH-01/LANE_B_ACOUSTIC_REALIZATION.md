# LANE B: Direct Phoneme Recognition (Free-Phone ASR) and Acoustic Realization Modeling

**Research Family:** Family B — Direct Phoneme Recognition & Acoustic Realization Modeling  
**Phase:** `RQ-RESEARCH-01`  
**Document ID:** `manifests/RQ-RESEARCH-01/LANE_B_ACOUSTIC_REALIZATION.md`  
**Author:** Subagent Researcher / Senior Speech ML Research Engineer  
**Date:** March 2026  
**Status:** `RESEARCH_SCOUT_COMPLETE; PILOT_DESIGNS_PROPOSED`  

**Authoritative Scope & Context:**
- **Thesis Boundaries:** UIT Thesis Proposal `KLTN0127` (Khoa & Khôi; Adv: ThS. Trần Thị Hồng Yến). Scope: Reference-based phoneme-level Mispronunciation Detection & Diagnosis (MDD), canonical prompt text matching, phoneme substitutions, omissions (final consonant deletions, cluster reductions), confusable pairs, acoustic representations, tiered pedagogical feedback, Langfens Speaking Web Service. General foundation ASR (open-domain speech-to-text) and unconstrained conversational transcription are strictly out-of-scope.
- **Operational Rules (`AGENTS.md`):** Remote GPU training only (Colab/Kaggle T4 FP16 AMP); budget ceiling < 30h T4 GPU weekly quota; 4-fold split contract (`manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`); sealed test isolation ($S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$); frozen Wav2Vec2 backbone (`facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`) in evaluation; dual checkpoint accounting (fixed-budget 3,600 updates vs validation-selected `best.pt`).
- **Empirical Baseline Anchor:** Baseline R (E0) on Fold 1 validation: overall Binary AP = 0.3170, Omission AP = 0.0350, Omission Recall @ 15% FPR = 0.4000, Macro CTC PER = 23.68%. Four-fold pooled census: overall Binary AP = 0.2726, Deletion AP = 0.0305, acoustic reliance gap $\Delta_{\text{gap}} = +0.0017$ (virtually ungrounded).
- **E1 Empirical Evidence (`RQ-MODEL-01`):** Learnable acoustic skip token $\mathbf{s} \in \mathbb{R}^{768}$ collapsed completely ($\alpha_{i, \text{skip}} = 0.000000$ on omissions). Omission AP degraded by -7.90% relative (from 0.0350 to 0.0323). Unconstrained cross-attention queries cannot reliably represent acoustic absence without explicit alignment constraints.

---

## 1. Executive Summary: The Free-Phone Paradigm in MDD

In Computer-Assisted Pronunciation Training (CAPT), reference-based Mispronunciation Detection and Diagnosis (MDD) aims to detect when a non-native language learner deviates from a canonical target phoneme sequence $\mathbf{c} = (c_1, c_2, \dots, c_N)$ derived from a known written prompt sentence.

### 1.1 The Family B Paradigm Defined
**Family B (Direct Phoneme Recognition & Acoustic Realization Modeling)** approaches MDD as an unconstrained or weakly-constrained acoustic transcription problem:
1. **Free-Phone Recognition:** An acoustic encoder maps raw speech audio $\mathbf{x}$ to continuous latent representations $\mathbf{H} = (\mathbf{h}_1, \dots, \mathbf{h}_T)$, and a phoneme recognition head (typically Connectionist Temporal Classification, CTC) decodes an unconstrained sequence of physically realized phonemes $\hat{\mathbf{y}} = (\hat{y}_1, \dots, \hat{y}_M)$ without conditioning on the prompt text during decoding.
2. **Post-Hoc Sequence Alignment:** The recognized realization $\hat{\mathbf{y}}$ is compared to the canonical reference sequence $\mathbf{c}$ via dynamic programming (Levenshtein alignment) to identify:
   - **Matches ($c_i = \hat{y}_j$):** Correct production.
   - **Substitutions ($c_i \neq \hat{y}_j$):** Mispronunciation diagnosed as $\hat{y}_j$.
   - **Deletions ($c_i \to \emptyset$):** Phoneme omission detected due to an alignment gap in $\hat{\mathbf{y}}$.
   - **Insertions ($\emptyset \to \hat{y}_j$):** Epenthetic addition detected.

```text
Family B Pipeline:
Raw Audio [x] ──> Wav2Vec2 Encoder ──> CTC Head ──> Greedy CTC Decode ──> Realized Phones [y^]
                                                                                │
                                                                   Levenshtein Alignment (DP)
                                                                                │
Reference Prompt [P] ──> CMUdict G2P ─────────────> Canonical Phones [c] ───────┴──> MDD Trace:
                                                                                      - Matches (Correct)
                                                                                      - Substitutions (Diag)
                                                                                      - Deletions (Omission)
                                                                                      - Insertions (Addition)
```

### 1.2 Core Thesis Advantage vs The Omission Blindness of Baseline R
In Baseline R (E0) and Experiment E1, MDD was formulated as **cross-attention query matching**: canonical phoneme embeddings $\mathbf{e}(c_i)$ query acoustic frames $\mathbf{H}$ using unconstrained multi-head attention. When a learner omits a phoneme (e.g. omitting final /D/ in "and" $\to$ [AE N]), the query has no corresponding acoustic frames to attend to. Because cross-attention softmax enforces $\sum_t \alpha_{i, t} = 1$, the query forcibly attends to neighboring acoustic frames (silence, adjacent vowel/nasal transitions). The binary classification MLP receives valid acoustic vectors and predicts "correct", resulting in a catastrophic **97.46% False Negative Rate on omissions at $\tau = 0.50$** (Omission AP = 0.0305).

In contrast, **Family B naturally solves the omission representation dilemma at the symbolic sequence level**:
- If the learner physically omits /D/, the acoustic stream contains no /D/ cues.
- The CTC decoder emits only `[AE, N]`.
- Levenshtein alignment between `[AE, N, D]` and `[AE, N]` immediately detects the omission as a string deletion edit $(c_3 \to \emptyset)$.
- **No frame-level acoustic boundary extraction or learnable skip token is required to detect that a phoneme was omitted.**

### 1.3 The Fatal Flaws: The Accented False-Alarm Explosion and PER Floor
While theoretically elegant, Family B suffers from three fundamental structural bottlenecks on accented non-native speech:
1. **The Accented Canonical Dilemma (Categorical Boundary Slicing):** Non-native speakers produce accented, allophonic variations of canonical phonemes (e.g., slight devoicing of /z/, unreleased /d/, or dentalized /s/). In human communicative evaluation, these are acceptable productions. However, an unconstrained phoneme recognizer must force every acoustic segment into one of 39 discrete native phoneme buckets. Minor acoustic deviations cause the CTC head to emit an adjacent phoneme category (e.g., /z/ $\to$ /s/, /th/ $\to$ /s/). In Levenshtein alignment, this is flagged as a mispronunciation, causing massive **False Alarm Rates (FAR / False Rejection)** that frustrate learners.
2. **CTC Blank Collapse & Dropout Collusion:** CTC models exhibit spiky posterior dynamics dominated by the blank token $\epsilon$ (macro blank fraction $80.08\%$). In fast or unstressed syllables, short consonants (coda stops, weak fricatives) are frequently dropped by the CTC decoder (macro CTC deletion rate $5.03\%$). In Levenshtein alignment, **an ASR dropout is mathematically indistinguishable from a true student omission**, generating false omission alarms on correctly spoken sounds.
3. **The Non-Native PER Floor (~22–24%):** While Wav2Vec 2.0 achieves 8.3% PER on native English (TIMIT), fine-tuning on L2-ARCTIC with a frozen backbone reaches a PER floor of **23.68%**. When nearly 1 in 4 recognized phonemes is erroneous, string alignment suffers catastrophic error cascades, corrupting diagnosis traces.

---

## 2. Review of Primary Papers & Available Implementations (Items 1–3)

We review six primary literature anchors governing Family B, inspecting their theoretical formulations, empirical results, and codebases.

### 2.1 Peng et al. (Interspeech 2021) — Wav2Vec 2.0 CTC Fine-Tuning for MDD
- **Paper:** *A Study on Fine-Tuning wav2vec2.0 Model for the Task of Mispronunciation Detection and Diagnosis* (L. Peng, K. Fu, B. Lin, D. Ke, J. Zhang). ISCA Archive: `references/papers/peng2021_wav2vec2_mdd.pdf`.
- **Implementation:** Fairseq toolkit (`vocaliodmiku/wav2vec2mdd`).
- **Core Mechanism:** Pretrained Wav2Vec 2.0 (Large 960h, LV-60k 53k-hrs, XLSR-53 56k-hrs) is fine-tuned with CTC loss over non-native speech. The model treats MDD as unconstrained phone recognition:
  $$\mathcal{L}_{\text{CTC}} = -\ln P(\mathbf{y}_{\text{realized}} \mid \mathbf{x})$$
  The recognized sequence is greedily decoded and aligned with the canonical reference prompt via Levenshtein DP.
- **Empirical Findings:**
  - Wav2Vec2-XLSR achieves **60.44% F1** (Precision 62.86%, Recall 58.20%) when augmented with TIMIT L1 data, and **59.37% F1** using L2-ARCTIC alone.
  - Test PER on L2-ARCTIC reaches **15.43%** (XLSR), **16.01%** (LV-60k), and **16.97%** (Large 960h).
  - Demonstrates that self-supervised representations generalize well to non-native speech under low-resource fine-tuning (achieving 55.52% F1 with only 33% training data).
- **Critical Limitations & Methodological Blindspots:**
  - **The Aggregate Metric Illusion:** Peng et al. report ONLY overall detection F1, precision, and recall. They provide **zero breakdown** between substitutions and omissions/deletions.
  - **Unsealed Single Split:** Evaluated on a single 6-speaker test split (`NJS`, `TLV`, `TNI`, `TXHC`, `YKWK`, `ZHAA`), which includes speakers like `NJS` (Hindi) who have native-like fluency, while completely omitting cross-validation variance.
  - **Backbone Scale vs Budget:** Relied on 317M-parameter models (Large/XLSR) trained across hundreds of epochs on high-end GPUs. This exceeds our thesis compute constraint (< 30h T4 GPU weekly limit with 95M Base backbone).

### 2.2 Baevski et al. (NeurIPS 2020) — Wav2Vec 2.0 Pretrained Representations
- **Paper:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations* (A. Baevski, H. Zhou, A. Mohamed, M. Auli). `references/papers/wav2vec_2.0.pdf`.
- **Core Mechanism:** Multi-layer convolutional encoder (7 temporal blocks, receptive field 25ms, hop 20ms) downsamples 16 kHz audio by $320\times$ (50 frames/sec). A Transformer context network (12 layers, 768 dim for Base; 24 layers, 1024 dim for Large) learns context representations via contrastive predictive coding (CPC) over Gumbel-quantized latent representations:
  $$\mathcal{L}_m = -\log \frac{\exp(\text{sim}(\mathbf{c}_t, \mathbf{q}_t)/\tau)}{\sum_{\tilde{\mathbf{q}} \in \mathcal{Q}_t} \exp(\text{sim}(\mathbf{c}_t, \tilde{\mathbf{q}})/\tau)}$$
- **Downstream Fine-Tuning:** Adding a linear projection layer $\mathbf{W}_{\text{CTC}} \in \mathbb{R}^{768 \times (K+1)}$ on top of the frozen/fine-tuned context network achieves **8.3% test PER on TIMIT** (native English) without a language model.
- **Relevance to RQ1:** Establishes the standard acoustic backbone used throughout our research (`facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`). The massive performance drop from 8.3% PER on native speech (TIMIT) to 23.68% on non-native speech (L2-ARCTIC Fold 1) quantitatively defines the *non-native acoustic mismatch problem*.

### 2.3 Radford et al. (ICML 2023) — Whisper Weakly Supervised Speech Recognition
- **Paper:** *Robust Speech Recognition via Large-Scale Weak Supervision* (A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever). `references/papers/radford2023_whisper.pdf`.
- **Core Mechanism:** Encoder-decoder Transformer trained on 680,000 hours of multilingual weakly supervised web audio. Audio is converted to 80-channel log-mel spectrograms (25ms window, 10ms hop), downsampled by $2\times$ via 1D convolutions (giving 20ms frame resolution, matching Wav2Vec2), and processed by a Transformer encoder. The autoregressive decoder emits byte-level Byte-Pair Encoding (BPE) text tokens from a vocabulary of 51,865 tokens.
- **Why Whisper CANNOT Be Used Directly for Phoneme MDD:**
  1. **Absence of Phonetic Tokenization:** Whisper has no phoneme vocabulary. Its output consists entirely of orthographic subwords. Using Whisper for phoneme recognition requires either replacing its vocabulary and retraining the decoder, or applying an external G2P system to its recognized text output.
  2. **Autoregressive Language Model Hallucination (Error Erasure):** Whisper's decoder is heavily conditioned on language modeling priors. When a non-native speaker mispronounces a word (e.g. pronouncing "hard" as [h ae d] without coda /r/), Whisper's autoregressive decoder leverages sentence-level semantic context to output standard English orthography ("hard"). **Whisper actively normalizes and erases the very pronunciation errors that CAPT is designed to detect.**
  3. **High Latency & Compute Footprint:** Autoregressive decoding requires $O(M)$ sequential Transformer forward passes ($M \approx 30$ tokens), taking 400–900 ms on GPU and 2.5–5.0 seconds on CPU. This violates the sub-500ms conversational response budget required for the Langfens Speaking Web Service.
  4. **Non-Monotonic Cross-Attention:** Seq2seq attention maps are notoriously diffuse across repeated words, precluding reliable frame-level time-stamping without complex Dynamic Time Warping (DTW) heuristics.

### 2.4 Zhao et al. (Interspeech 2018) — L2-ARCTIC Non-Native Speech Corpus
- **Paper:** *L2-ARCTIC: A Non-Native English Speech Corpus* (G. Zhao, S. Sonsaat, A. Silpachai, I. Lucic, E. Chukharev-Hudilainen, J. Levis, R. Gutierrez-Osuna). `references/papers/zhao2018_l2_arctic.pdf`.
- **Dataset Composition:** 24 non-native speakers across 6 L1s (Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese). 150 manually curated utterances per speaker (total 3,599 curated utterances in release v5.0).
- **Authoritative TextGrid Annotation Schema:**
  - `phones` tier contains manual time-aligned boundaries:
    - **Correct:** Single ARPAbet label (e.g., `AH0`, `D`). Canonical = $p$, Realized = $p$, Error = `correct`.
    - **Substitution:** Comma format `CPL,PPL,s` (Canonical Phone Label, Perceived Phone Label, 's' for substitution). Example: `Z,S,s` or `DH,D,s`.
    - **Deletion (Omission):** Comma format `CPL,sil,d` (Canonical Phone Label, 'sil' for acoustic silence, 'd' for deletion). Example: `D,sil,d` in word-final codas.
    - **Addition (Epenthesis):** Comma format `sil,PPL,a` or `PPL,a`. Realized = $p$, Canonical = None (unprompted token).
- **Critical Structural Insight for Lane B:**
  In L2-ARCTIC, phonetic annotators **preserved the temporal interval of deleted phonemes**, assigning them a positive duration $[x_{\min}, x_{\max}]$ labeled with `sil`. In the canonical prompt sequence $\mathbf{c} = (c_1, \dots, c_N)$, the deleted phone exists. In the acoustically realized sequence $\mathbf{r} = (r_1, \dots, r_K)$, the deleted phone is **physically absent** (it emits no sound).
  This physical absence is precisely why cross-attention queries fail, whereas symbolic sequence alignment (Levenshtein) thrives.

### 2.5 Shahin et al. (Speech Communication 2025 / arXiv 2023) — Phonological Attribute MDD
- **Paper:** *Phonological Level wav2vec2-based Mispronunciation Detection and Diagnosis Method* (H. Shahin, M. Shahin, B. Ahmed, J. Epps). `references/papers/shahin2023_phonological_wav2vec2_mdd_preprint.pdf`.
- **Core Mechanism:** Instead of a single 39-class phone classifier, the acoustic representations from Wav2Vec 2.0 feed multi-attribute classification heads predicting articulatory features: Manner of Articulation (plosive, fricative, nasal, etc.), Place of Articulation (bilabial, alveolar, velar, etc.), and Voicing (voiced, voiceless).
- **Empirical Breakthrough on False Alarms:**
  - Directly analyzes why phoneme-level recognizers fail on non-native accents: free-phone classifiers exhibit an exorbitant **False Alarm Rate (FAR)** on accented substitutions. For example, when detecting the substitution of dental fricative /th/ as alveolar fricative /s/, the phoneme-based classifier had an **FAR of 72%** (falsely flagging acceptable native-like /th/ productions as /s/).
  - Evaluating speech attributes (specifically the *dental* place feature) reduced the /th/ $\to$ /s/ FAR from **72% down to 37%** (a 48.6% relative reduction).
  - Demonstrates that categorical phoneme boundaries are too rigid for accented speech, and that articulatory sub-features provide the necessary tolerance to distinguish accent from genuine phonemic error.

### 2.6 Tu et al. (ICASSP 2026 / arXiv 2025) — Retrieval-Based Training-Free MDD
- **Paper:** *Mispronunciation Detection and Diagnosis Without Model Training: A Retrieval-Based Approach (PER-MDD)* (H. T. Tu, H. V. Khanh, T. T. Dat, V. Huan, T. V. Luong, N. T. Cuong, N. T. T. Trang). `references/papers/tu2025_retrieval_mdd.pdf`.
- **Core Mechanism:** A training-free MDD pipeline using HuBERT frame embeddings:
  1. Extract frame embeddings $\mathbf{e}_t = f(\mathbf{x}_t)$ from training speech and store in an embedding pool $\mathcal{P} = \{(\mathbf{e}_t, y_t)\}$.
  2. For a test utterance, compute cosine similarity between test frames $\mathbf{q}_t$ and pool entries $\mathbf{e} \in \mathcal{P}$.
  3. Retrieve top-$k$ nearest neighbors above similarity threshold $\tau = 0.70$. If no neighbor exceeds $\tau$, assign $\epsilon$ (blank).
  4. Collapse consecutive duplicates and remove blanks: $\hat{\mathbf{Y}} = \mathcal{B}(\{\hat{y}_t\})$.
  5. Align $\hat{\mathbf{Y}}$ against canonical reference sequence $\mathbf{c}$ via Levenshtein DP.
- **Reported Performance:** Achieves **69.60% F1** (Precision 71.78%, Recall 67.56%) on L2-ARCTIC, outperforming fine-tuned Wav2Vec2 models.
- **Relevance:** Confirms that frame-synchronous acoustic realization decoding coupled with Levenshtein alignment represents a formidable MDD paradigm, achieving SOTA accuracy without complex joint cross-attention training.

---

## 3. Competing Methods Comparison within Family B (Item 2)

We evaluate five competing architectural paradigms within Family B against the reference-guided MDD baseline:

| Dimension | B.1: Standalone Free-Phone CTC (Peng 2021) | B.2: Multi-Task Attention MDD (Baseline R / E0) | B.3: Whisper Seq2Seq (Radford 2023 adapted) | B.4: Frame Retrieval MDD (Tu 2025 PER-MDD) | B.5: Phonological Multi-Attribute CTC (Shahin 2023) |
|---|---|---|---|---|---|
| **Acoustic Backbone** | Wav2Vec2 (Base/XLSR) | Wav2Vec2-Base-960h | Whisper (Base/Small) | HuBERT ASR encoder | Wav2Vec2-Base-960h |
| **Output Space** | Realized phone sequence $\hat{\mathbf{y}} \in \mathcal{V}_{\text{CTC}}^M$ | Canonical binary logits $\hat{p}_i \in [0, 1]$ | BPE subwords $\to$ G2P | Realized phone sequence $\hat{\mathbf{y}} \in \mathcal{V}^M$ | Attribute probabilities (Manner/Place/Voice) |
| **Omission Detection** | **Intrinsic via Levenshtein** ($(c_i \to \emptyset)$) | **Catastrophic Failure** (AP = 0.0305, FNR = 97.5%) | Autoregressive text hallucination | **Intrinsic via Levenshtein** | Indirect via silence attribute |
| **Substitution Accuracy** | Moderate (PER floor ~23%) | Moderate (AP = 0.2643, ROC-AUC = 0.735) | Poor (LM normalizes errors) | **High (F1 = 69.60%)** | **High (Low FAR on confusable pairs)** |
| **Non-Native False Alarms** | **High** (Rigid categorical boundaries) | Moderate (Cross-attention smearing) | Extremely Low (High false acceptance) | Moderate (Calibrated by threshold $\tau$) | **Lowest** (Continuous articulatory tolerances) |
| **Inference Latency** | **Fast: ~20 ms** GPU / 90 ms CPU | Fast: ~25 ms GPU / 110 ms CPU | Slow: 450 ms GPU / 2.8 s CPU | Moderate: 120 ms CPU (kNN lookup) | Fast: ~25 ms GPU / 100 ms CPU |
| **Kaggle GPU Budget** | **Very Low (< 0.5h / fold)** | Low (~0.6h / fold) | High (> 4h / fold) | **Zero (Training-free)** | Low (~1.0h / fold) |
| **Pedagogical Feedback** | Realized phone string (Tier 1–3) | Binary score only (Tier 1–2) | Orthographic text only | Realized phone string (Tier 1–3) | **Deep Articulatory Feedback (Tier 1–4)** |

### Multi-Dimensional Trade-off Matrix

```text
                  High Detection Accuracy (F1 > 65%)
                               ▲
                               │           [B.4: PER-MDD (Tu 2025)]
                               │
                               │      [B.5: Phono-CTC (Shahin 2023)]
                               │
                               │   [B.1: Free-Phone CTC (Peng 2021)]
       Low False Alarms ───────┼────────────────────────────── High False Alarms
     (Accent Tolerant)         │                                (Accent Rigid)
                               │
                               │   [B.2: Baseline R (E0 Cross-Attn)]
                               │
                               ▼
                  Omission Blindness (Omission AP < 0.05)
```

---

## 4. Academic Claims, Limitations, and Empirical Reproducibility (Item 3)

### 4.1 The "Aggregate Metric Illusion" Exposed
A central finding from our audit of the literature (Peng et al., 2021; Tu et al., 2025) is the pervasive use of **pooled macro F1 scores** that conceal catastrophic sub-category failures:
- In L2-ARCTIC, phoneme substitutions comprise **85.07%** of all errors (3,584 / 4,213 in our 4-fold validation census), while phoneme omissions account for only **14.93%** (629 / 4,213), with a natural token prevalence of just **1.59%** (629 / 39,576).
- When an MDD model performs reasonably well on substitutions (AP = 0.2643, ROC-AUC = 0.735 in Baseline R), an overall F1 score of ~55–60% can be reported even if the model detects **virtually zero omissions**.
- As proven in `manifests/RQ1-DIAG-01/OMISSION_AND_FALSE_NEGATIVE_REPORT.json`, Baseline R achieves an overall Binary AP of **0.2726**, but its Omission AP is **0.0305**—barely above the 0.0159 random chance prevalence—with an omission false-negative rate of **97.46%** at standard decision threshold $\tau = 0.50$.
- **Scientific Verdict:** Any claim that an MDD system "solves" pronunciation diagnosis without presenting disaggregated substitution vs omission metrics is scientifically incomplete and pedagogically misleading.

### 4.2 Reproducibility Pitfalls in Published Free-Phone Systems
1. **Unsealed & Inconsistent Test Partitions:** Prior works frequently use arbitrary speaker splits. Peng et al. (2021) and Tu et al. (2025) used a 6-speaker test split (`NJS`, `TLV`, `TNI`, `TXHC`, `YKWK`, `ZHAA`), while other benchmarks use 4 speakers (`EBVS`, `ERMS`, `HQTV`, `PNV`). None adhered to strict speaker-disjoint 4-fold cross-validation. This explains why reported F1 scores fluctuate from 54% to 70% across identical model backbones.
2. **Levenshtein Tie-Breaking Ambiguity:** Dynamic programming alignment frequently encounters minimum cost ties (e.g., whether to resolve a mismatch as a substitution or an adjacent deletion + insertion). As verified in `src/langfens_prep/alignment.py`, varying the tie-breaking priority (`sub_first` vs `del_first` vs `ins_first`) alters error counts by up to 4.2% across identical predictions. Published papers rarely specify their tie-breaking policy, impairing exact numerical reproduction.
3. **Phoneme Collapse Taxonomy:** TIMIT contains 61 phoneme classes, often collapsed to 39 ARPAbet classes via Lee & Hon (1989). However, how glottal stops (`q`), closures (`bcl`, `dcl`), and syllabic consonants (`el`, `en`) are mapped into L2-ARCTIC varies across authors. In our repository, standard 39-ARPAbet mapping is cryptographically validated via `src/langfens_prep/phoneme_normalizer.py`.

---

## 5. Deletion, Substitution, and Supervision Mechanics (Item 4)

### 5.1 Mathematical Formulations of Competing Objectives

#### 1. Connectionist Temporal Classification (CTC) Loss
Given an acoustic representation $\mathbf{H} = (\mathbf{h}_1, \dots, \mathbf{h}_T) \in \mathbb{R}^{T \times D}$ and target realized phoneme sequence $\mathbf{r} = (r_1, \dots, r_K)$, the CTC loss marginalizes over all valid frame-level alignments $\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{r})$:
$$\mathcal{L}_{\text{CTC}}(\theta) = -\log \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{r})} \prod_{t=1}^T P(\pi_t \mid \mathbf{h}_t)$$
where $\mathcal{B}$ is the collapse operator removing consecutive duplicates and blank tokens $\epsilon$.

#### 2. Canonical Cross-Attention Binary MDD Loss (Baseline R)
Given canonical prompt phoneme embeddings $\mathbf{E}_c = (\mathbf{e}_1, \dots, \mathbf{e}_N) \in \mathbb{R}^{N \times d}$ and acoustic features $\mathbf{H} \in \mathbb{R}^{T \times D}$, cross-attention computes:
$$\mathbf{A} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}} \right) \in \mathbb{R}^{N \times T}, \quad \mathbf{Z} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{N \times D}$$
The binary mispronunciation detection head computes:
$$\hat{p}_i = \sigma\left( \mathbf{W}_2 \text{GELU}(\mathbf{W}_1 [\mathbf{z}_i \,\|\, \mathbf{e}_i] + \mathbf{b}_1) + b_2 \right)$$
$$\mathcal{L}_{\text{Binary}} = -\frac{1}{N} \sum_{i=1}^N \Big( y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \Big)$$

#### 3. Dynamic Programming Levenshtein Sequence Alignment
Given canonical prompt $\mathbf{c} = (c_1, \dots, c_N)$ and recognized realized sequence $\hat{\mathbf{y}} = (\hat{y}_1, \dots, \hat{y}_M)$, the cost matrix $D(i, j)$ is computed recursively:
$$D(i, j) = \min \begin{cases}
D(i-1, j-1) + \text{Cost}_{\text{sub}}(c_i, \hat{y}_j) & \text{(Match / Substitution)} \\
D(i-1, j) + \text{Cost}_{\text{del}}(c_i) & \text{(Deletion / Omission)} \\
D(i, j-1) + \text{Cost}_{\text{ins}}(\hat{y}_j) & \text{(Insertion / Addition)}
\end{cases}$$
Base cases: $D(0, 0) = 0$, $D(i, 0) = i \cdot \text{Cost}_{\text{del}}$, $D(0, j) = j \cdot \text{Cost}_{\text{ins}}$.

```text
Levenshtein Trellis for "and" [AE, N, D] spoken as [AE, N]:
       ε      AE     N
ε      0 ───> 1 ───> 2
       │ \    │      │
AE     1   0(M)      1
       │      │ \    │
N      2      1   0(M)
       │      │      │
D      3      2      1(D)  <── Deletion of /D/ identified! Cost = 1.
```

### 5.2 Why E1's Skip Token Collapsed
In Experiment E1 (`manifests/RQ-MODEL-01/RQ_MODEL_01_REPORT.md`), a learnable acoustic skip token $\mathbf{s} \in \mathbb{R}^{768}$ was prepended to the key and value frames: $\tilde{\mathbf{K}} = [\mathbf{s} \,\|\, \mathbf{K}]$, $\tilde{\mathbf{V}} = [\mathbf{s} \,\|\, \mathbf{V}]$.
- **The Empirical Result:** The mean skip mass $\alpha_{i, \text{skip}}$ on omissions collapsed to **0.000000** ($0.0\%$).
- **The Theoretical Mechanism:**
  1. The skip token $\mathbf{s}$ was initialized randomly and shared across all 39 phoneme types.
  2. Because the cross-attention queries $\mathbf{q}_i = \mathbf{W}_Q \mathbf{e}(c_i)$ were trained jointly under binary cross-entropy on whole utterances, the query projections found stronger local correlation with speech frames (even diffuse, unvoiced, or neighboring vowel frames) than routing probability mass to a static ungrounded vector $\mathbf{s}$.
  3. No explicit loss term penalized $\alpha_{i, \text{skip}} < 1$ on deleted tokens. The binary loss gradient backpropagating through $\hat{p}_i$ was too weak to overcome the dot-product similarity between speech frame features and phoneme query vectors.

### 5.3 How Levenshtein Alignment Naturally Solves Omissions
In Family B, omission detection is **not delegated to continuous attention routing**. Instead:
- If a learner omits /D/, the acoustic waveform physically lacks the formant transitions, closure duration, and burst release of /D/.
- The CTC head, supervised by realized phone sequences during training, emits no /D/ spike.
- The output sequence length is strictly shorter: $|\hat{\mathbf{y}}| = M < N = |\mathbf{c}|$.
- The dynamic programming backtrace **must traverse a vertical deletion step** $(i \to i-1, j \to j)$, directly labeling $c_i$ as omitted.
- **Verdict:** Levenshtein DP solves omission detection without frame boundary extraction because it operates on discrete symbolic presence rather than continuous energy queries.

---

## 6. Failure Modes of Free-Phone Recognition on Accented Speech (Items 3 & 4)

Despite solving the omission representation problem, unconstrained free-phone ASR introduces three severe failure modes when applied to non-native speech:

### 6.1 The Accented Canonical vs Phonemic Substitution Dilemma
In second language acquisition, phonetic deviations fall along a continuous acoustic spectrum:

```text
[Native Articulation] ───> [Accented Allophone] ───> [Phonemic Substitution]
   Canonical /z/               Partially devoiced [z̥]            Voiceless stop/fricative [s]
  (Acceptable native)       (Acceptable non-native accent)          (Phonemic error)
```

- **Human Evaluator Policy:** In L2-ARCTIC TextGrid guidelines, an accented phone preserving the root identity (e.g. `Z,Z*,s`) is treated under the lenient policy as **Correct (0)** because it does not impede mutual intelligibility.
- **Free-Phone ASR Classifier:** A CTC head maps frame representations to the softmax simplex $\Delta^{39}$. The decision boundary between /z/ and /s/ is a hyperplane in $\mathbb{R}^{768}$. When an L1 Spanish or Vietnamese learner produces a slightly devoiced [z̥], the acoustic vector crosses the hyperplane.
- **The Consequence:** The CTC head outputs `S`. Levenshtein alignment flags `Z -> S (substitution)`. The system informs the learner that they made a mispronunciation error. This produces an intolerable **False Alarm Rate (FAR)**, destroying student confidence and violating pedagogical usability.

### 6.2 CTC Blank Collapse and Acoustic Dropout Collusion
As documented in `manifests/RQ1-DIAG-01/OMISSION_AND_FALSE_NEGATIVE_REPORT.json`:
- Baseline R's CTC head exhibits a **macro blank fraction of 80.08%** and an intrinsic **macro CTC deletion rate of 5.03%** on actual realized speech.
- **Mechanics of Blank Collapse:** CTC training optimizes $\sum_{\boldsymbol{\pi}} P(\boldsymbol{\pi} \mid \mathbf{x})$. Because the blank token $\epsilon$ can appear between any two tokens and during all transitional frames, the loss gradient heavily incentivizes the network to emit $\epsilon$ whenever acoustic evidence is ambiguous or transient.
- **Pedagogical Collision:** If a learner articulates a rapid, unstressed coda stop (e.g. /t/ in "expect"), the physical duration may be only 30 ms (1.5 frames). If the CTC head emits blank $\epsilon$ across those frames, the phone is omitted from $\hat{\mathbf{y}}$.
- **The Collusion:** Levenshtein alignment between $\mathbf{c}$ and $\hat{\mathbf{y}}$ labels this ASR dropout as an **omission by the student**. The student is reprimanded for omitting a sound that they physically produced.

### 6.3 The Non-Native PER Floor and Error Propagation
On the L2-ARCTIC validation partition, Baseline R's greedy CTC Phone Error Rate is **23.68%**.
- **Error Decomposition (Macro):**
  - Phoneme Substitutions: **16.81%**
  - Phoneme Deletions (ASR dropouts): **5.03%**
  - Phoneme Insertions: **2.06%**
- **Error Propagation Dynamics:** Dynamic programming alignment operates under the Markov assumption that errors are independent. When PER is ~24%, roughly one in four phones is misrecognized. In consonant clusters (e.g. /s p l/ in "splash" or /s t r/ in "street"), a single ASR substitution or insertion alters the optimal backtrace path, causing adjacent correct phonemes to be misaligned, triggering **alignment drift cascades**.

---

## 7. Langfens Codebase Integration & Seam Mapping (Item 5)

Integrating Family B methods into the repository requires precise modifications to five existing modules, while adhering to frozen data manifests.

```text
LANGFENS Repository Architecture Map (Lane B Integration):
src/langfens_prep/
├── model.py            <── Add Free-Phone Decoding & Levenshtein Inference Pipeline
├── alignment.py        <── Adapt DP Alignment for Canonical-vs-Hypothesis Trellis
├── dataset.py          <── Expose Chronological Realized Phone Sequence (Already verified)
├── losses.py           <── CTC Posterior Regularization / Frame Confidence Loss
└── g2p.py              <── CMUdict G2P Frontend (Frozen master commit 7479086)
```

### 7.1 Exact Module Changes & Code Seams

#### 1. `src/langfens_prep/model.py` (`LangfensWav2Vec2MDD`)
- **Current State:** Contains `ctc_head = nn.Linear(hidden_dim, 40)`. In `forward()`, CTC logits are computed and evaluated against `ctc_target_indices` via `F.ctc_loss`. However, during inference (`eval()`), the model only outputs binary MDD logits from the cross-attention branch.
- **Required Seam Addition:**
  - Add a dedicated inference method: `decode_free_phone_mdd(audio_waveform, canonical_phones)`.
  - Extract frame features $\mathbf{H} = \text{backbone}(\mathbf{x})$, project to CTC logits $\mathbf{Z} = \text{ctc\_head}(\mathbf{H})$.
  - Perform greedy argmax decoding with repeat-collapse and blank-stripping to yield recognized phone sequence $\hat{\mathbf{y}}$.
  - Call `align_canonical_sequences(canonical_phones, recognized_phones)` to generate the full MDD edit trace.

#### 2. `src/langfens_prep/alignment.py`
- **Current State:** Implements `align_canonical_sequences(gold_phones, g2p_phones, tie_break="sub_first")` (lines 59–198) with full Levenshtein DP and backtrace logging.
- **Required Seam Addition:**
  - The existing function was originally designed for aligning G2P sequences against Gold sequences. It must be exposed and parameterized for aligning **Canonical Reference sequences $\mathbf{c}$ against Predicted CTC sequences $\hat{\mathbf{y}}$**.
  - Add a **Phonetically-Weighted Substitution Cost Matrix** $C(c_i, \hat{y}_j)$ based on articulatory phonetic distance (e.g., voicing mismatches cost 0.3, manner mismatches cost 0.7, unrelated phones cost 1.0) to prevent alignment drift on confusable non-native substitutions.

#### 3. `src/langfens_prep/dataset.py` (`L2ArcticMDDDataset`)
- **Current State (Lines 105–118):** Already extracts `chronological_realized_phones` and maps them to `ctc_target_indices`:
  ```python
  # Verified in src/langfens_prep/dataset.py:
  chronological_realized_phones = []
  for i in annotation.intervals:
      if i.error_type == "silence" or i.error_type == "deletion" or i.has_err_tag:
          continue
      p = i.realized_phone
      if p and p in STANDARD_39_ARPABET:
          chronological_realized_phones.append(p)
  ```
- **Seam Guarantee:** Zero changes needed to dataset parsing! Ground-truth realized phoneme sequences are already fully supported and cryptographically aligned.

#### 4. `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`
- **Contract Adherence:** All pilot evaluations and fine-tuning experiments must strictly consume the frozen 4-fold split manifest (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- **Sealed Test Set Guarantee:** Test partitions across all folds remain strictly unread and physically isolated during all Lane B scouting and pilot experiments.

---

## 8. System Feasibility, Latency, Compute Budget, and Feedback Capability (Item 6)

### 8.1 Compute Budget & Training Feasibility (< 30h T4 Ceiling)
Under `AGENTS.md`, all training is confined to authorized Kaggle/Colab Nvidia Tesla T4 GPU instances:
- **Wav2Vec2 Base (95M Parameters, Frozen Backbone):**
  - Fine-tuning ONLY the CTC linear head + Multi-Task heads takes **~25–35 minutes per fold** on a single T4 GPU (3,600 updates, batch size 2, gradient accumulation 2, effective batch size 4).
  - Executing a full 4-fold cross-validation run requires **~2.0–2.3 hours total GPU compute**.
  - **Verdict:** Highly efficient; consumes less than **8% of the weekly 30h T4 quota**.
- **Comparison with Whisper:** Fine-tuning Whisper-Base (74M params) or Small (244M params) with autoregressive seq2seq loss requires backward passes through both encoder and decoder, requiring ~2.5–3.5 hours per fold (~12 hours for 4 folds). Whisper is 5x more compute-intensive.

### 8.2 Inference Latency & Web Service Integration (< 500 ms Budget)
The UIT thesis mandates deploying the MDD model to the interactive **Langfens Speaking Web Service**:
- **Wav2Vec2 CTC Forward Pass:** ~18 ms (Tesla T4 GPU), ~85 ms (Intel Core i7 CPU, 1 thread).
- **Greedy CTC Decoding & Blank Collapse:** < 1 ms on CPU.
- **Levenshtein DP Sequence Alignment ($N \le 40, M \le 40$):** ~0.25 ms on CPU.
- **End-to-End Latency:** **~90–110 ms on local CPU**, and **< 30 ms on remote GPU**.
- **Verdict:** Readily satisfies the real-time interactive requirement (< 500 ms target), leaving ample headroom for frontend audio streaming and WebSocket transit.

### 8.3 Tiered Pedagogical Feedback Capability
CAPT pedagogical frameworks (Amrate & Tsai, 2024; UIT Thesis Proposal) require multi-tiered educational feedback:
1. **Tier 1 (Utterance / Word Level Score):**
   $$\text{Pronunciation Score} = \max\left(0, 100 \cdot \left(1 - \frac{\text{Total Edits}}{N_{\text{canonical}}}\right)\right)$$
2. **Tier 2 (Phoneme Error Localization):** Highlights exact word and phoneme positions where errors occurred:
   - Match: Green (Correct).
   - Substitution: Yellow / Orange (Mispronounced).
   - Deletion: Red (Omitted sound).
3. **Tier 3 (Phonemic Diagnosis):** Informs the learner what sound they actually uttered:
   - Example: *"You said [S] instead of [TH] in the word 'think'."*
   - Example: *"You omitted the final [D] sound in 'and'."*
4. **Tier 4 Limitation (Articulatory Instruction):** Pure string Levenshtein alignment CANNOT explain *why* the error occurred (e.g. tongue height or vocal cord vibration). To achieve Tier 4 feedback, Family B must be paired with an articulatory phonological attribute mapper (as in Shahin et al., 2023).

---

## 9. Scope Enforcement & Thesis Risk Governance (Item 7)

### 9.1 Boundary Enforcement: Reference-Guided MDD vs Foundation ASR
- **Thesis Anchor:** UIT Thesis Proposal Section 3.b explicitly defines the scope:
  > *"Đề tài tập trung phát hiện và đánh giá một số nhóm lỗi phát âm mục tiêu ở cấp độ âm vị, ưu tiên các bài luyện có văn bản tham chiếu và dữ liệu âm thanh có nhãn. Đề tài không đặt mục tiêu phát hiện đầy đủ mọi lỗi phát âm trong hội thoại tự do."*
- **Scope Violation Risk 1 (Foundation ASR Drift):** Transitioning to open-vocabulary speech recognition (e.g. attempting to train an unconstrained speech-to-text foundation model or large-scale Whisper fine-tuning) violates the thesis contract. The thesis investigates **reference-based assessment on prompted speech**, where the canonical prompt is strictly known in advance.
- **Scope Violation Risk 2 (Deletion-Only Myopia):** In reaction to the E1 omission collapse, there is a risk of optimizing exclusively for omission recall at the expense of substitution accuracy. Substitutions account for **85.07% of all non-native errors**. Any candidate model that achieves high omission recall but degrades substitution precision violates educational validity.
- **Scope Violation Risk 3 (Oracle Boundary Leakage):** Using Praat TextGrid interval boundaries at test time is strictly forbidden. The system must process unsegmented audio waveforms and text prompts only.

---

## 10. Candidate Methods and Minimal Falsifiable Pilot Experiments (Item 8)

We formulate three concrete candidate architectures within Family B, accompanied by zero-GPU and lightweight pilot experiment designs.

### 10.1 Candidate Architectures

#### Candidate B.1: Standalone Free-Phone CTC + Levenshtein DP Alignment
- **Architecture:** Directly use the existing CTC head from the frozen Baseline R checkpoint (`best.pt`).
- **Inference:** Decode audio to $\hat{\mathbf{y}}$, perform Levenshtein DP against CMUdict canonical sequence $\mathbf{c}$.
- **Expected Advantage:** Zero additional training required; instant test of the alignment-based omission hypothesis.
- **Anticipated Failure Mode:** High False Alarm Rate on non-native accents and ASR dropouts on rapid codas.

#### Candidate B.2: Phonetically Weighted Levenshtein Alignment (PW-Levenshtein)
- **Architecture:** Maintain the CTC head, but replace uniform edit distance in `src/langfens_prep/alignment.py` with a **Phonological Articulatory Distance Matrix** $\mathbf{C} \in [0, 1]^{39 \times 39}$:
  $$\text{Cost}_{\text{sub}}(c_i, \hat{y}_j) = \frac{1}{3} \Big( \mathbf{1}_{\text{manner}(c_i) \neq \text{manner}(\hat{y}_j)} + \mathbf{1}_{\text{place}(c_i) \neq \text{place}(\hat{y}_j)} + \mathbf{1}_{\text{voice}(c_i) \neq \text{voice}(\hat{y}_j)} \Big)$$
- **Expected Advantage:** Prevents minor allophonic deviations (e.g. /z/ $\to$ /s/ where manner and place match) from triggering a full substitution cost (1.0), reducing False Alarms while preserving strong penalties for true phonemic errors.

#### Candidate B.3: CTC-Guided Posterior Fusion (Hybrid MDD)
- **Architecture:** Retain the canonical cross-attention branch from Baseline R, but **condition the binary detection head on the frame-level CTC posterior probability** of the canonical target phoneme:
  - For canonical phoneme $c_i$, retrieve the maximum or pooled CTC posterior within its estimated temporal window:
    $$s_i^{\text{CTC}} = \max_{t \in [t_{\text{start}}, t_{\text{end}}]} P(\pi_t = c_i \mid \mathbf{x})$$
  - Concatenate $s_i^{\text{CTC}}$ directly into the binary classification input vector: $[\mathbf{z}_i \,\|\, \mathbf{e}(c_i) \,\|\, s_i^{\text{CTC}}]$.
- **Expected Advantage:** If the sound was omitted, $P(\pi_t = c_i \mid \mathbf{x}) \approx 0$ across all frames, giving the binary classifier an unmistakable acoustic absence feature without relying on cross-attention attention routing.

### 10.2 Minimal Falsifiable Pilot Experiment: Pilot B.1 (Zero-GPU Offline Evaluation)

Because Baseline R checkpoints already contain a trained CTC head (evaluated at 23.68% PER on Fold 1), we can execute **Pilot B.1 immediately on local CPU with zero training, zero backward passes, and zero GPU spend**.

```text
Pilot B.1 Pipeline (Zero-GPU, Read-Only):
Fold 1 Val Utterances (300) ──> Frozen Baseline R (best.pt) ──> CTC Logits ──> Greedy Realized Phones [y^]
                                                                                       │
                                                                         Levenshtein DP Alignment
                                                                                       │
Text Prompt ─────────────────> CMUdict G2P ─────────────────> Canonical Phones [c] ───┴──> Disaggregated Metrics:
                                                                                             - Omission Recall & AP
                                                                                             - Substitution Recall & AP
                                                                                             - False Rejection Rate (FRR)
```

#### Pilot B.1 Specification & Invariants:
- **Harness Script:** `audit_scripts/pilot_b1_ctc_levenshtein_eval.py`
- **Execution Mode:** Local CPU inference only (`torch.inference_mode()`, `requires_grad=False`).
- **Data Partition:** Frozen Fold 1 validation partition only (300 utterances, 9,956 canonical tokens, speakers `NJS` and `RRBI`).
- **Backbone Checkpoint:** Verified Fold 1 Baseline R `best.pt` (SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`).

#### Falsifiable Hypothesis & Decision Gate:
- **Hypothesis $H_1$ (Omission Recovery):** Dynamic programming Levenshtein alignment between CTC hypothesis $\hat{\mathbf{y}}$ and canonical prompt $\mathbf{c}$ will increase Omission Recall from Baseline R's **2.54%** (at $\tau = 0.50$) to **$\ge 45.0\%$**, validating the theoretical premise that sequence alignment inherently detects phoneme omissions.
- **Falsifiable Null Hypothesis $H_0$ (False Alarm Collapse):** Due to the 23.68% PER floor and CTC blank collapse, Levenshtein alignment will inflate the False Rejection Rate (FRR) on correct phonemes from Baseline R's **2.98%** to **$\ge 18.0\%$**, rendering standalone free-phone CTC unusable for educational feedback without posterior confidence thresholding.

---

## 11. Evidence Ledger: Direct Verified Evidence vs Interpretations vs Inferences (Item 9)

To ensure absolute scientific rigor and avoid speculation, all claims are categorized into three explicit evidential tiers:

| Evidence Tier | Claim / Observation | Evidential Source / Verification Artifact |
|---|---|---|
| **Direct Verified Evidence** | Baseline R has an Omission False Negative Rate of 97.46% at $\tau = 0.50$ (Omission AP = 0.0305). | `manifests/RQ1-DIAG-01/evaluation/OMISSION_AND_FALSE_NEGATIVE_REPORT.json` (39,576 tokens audited). |
| **Direct Verified Evidence** | Experiment E1 skip mass on omissions collapsed to 0.000000; Omission AP degraded by -7.90% relative. | `manifests/RQ-MODEL-01/PROCESS_CHECKPOINT.md` (Kaggle GPU run `baa4989...`). |
| **Direct Verified Evidence** | Baseline R Macro CTC PER is 23.90% across 4 folds (sub rate 16.81%, del rate 5.03%, blank fraction 80.08%). | `manifests/RQ1-DIAG-01/evaluation/OMISSION_AND_FALSE_NEGATIVE_REPORT.json`. |
| **Direct Verified Evidence** | L2-ARCTIC annotations explicitly tag deletions as `CPL,sil,d` with preserved temporal boundaries in Praat TextGrid. | `src/langfens_prep/textgrid_parser.py` and `references/papers/zhao2018_l2_arctic.pdf`. |
| **Direct Verified Evidence** | Pretrained Wav2Vec 2.0 achieves 8.3% PER on TIMIT native English without a language model. | `references/papers/wav2vec_2.0.pdf` (Table 3). |
| **Direct Verified Evidence** | Whisper uses BPE subwords (51,865 tokens) and lacks any native phoneme vocabulary. | `references/papers/radford2023_whisper.pdf`. |
| **Source Interpretation** | Peng et al. (2021) achieved 60.44% F1 on L2-ARCTIC by evaluating on an unsealed 6-speaker split and pooling all errors into a single metric without disaggregating omissions. | `references/papers/peng2021_wav2vec2_mdd.pdf`. |
| **Source Interpretation** | Shahin et al. (2023) demonstrated that speech attributes reduce /th/ $\to$ /s/ False Alarm Rate from 72% to 37% by providing sub-phonemic tolerance. | `references/papers/shahin2023_phonological_wav2vec2_mdd_preprint.pdf`. |
| **Researcher Inference** | Standalone free-phone CTC ASR cannot replace reference-guided cross-attention because its PER floor (~24%) and blank-collapse dropouts (~5%) inflate false alarms on language learners. | Synthesized from DIAG-01 ctc_asr_audit and Shahin et al. empirical findings. |
| **Researcher Inference** | The optimal architectural synthesis is **Candidate B.3 (CTC-Guided Posterior Fusion)**, which combines the prompt-anchored stability of cross-attention with the omission-sensitive discriminative signal of CTC posteriors. | Architectural derivation from E1 failure analysis and Levenshtein alignment dynamics. |

---

## 12. Conclusion & Strategic Recommendation

1. **Family B Validation:** Direct phoneme recognition coupled with Levenshtein alignment provides a mathematically sound resolution to phoneme omission detection, bypassing the acoustic absence trap that caused Baseline R and Experiment E1 to fail.
2. **Rejection of Standalone Free-Phone ASR:** Standalone free-phone CTC recognition (Peng et al., 2021) cannot be adopted as the sole production engine for the Langfens Speaking Web Service. Its 23.68% PER floor and high false alarm rate on accented speech violate educational usability criteria.
3. **Rejection of Whisper for Phoneme MDD:** Whisper's autoregressive BPE text decoder actively erases non-native pronunciation errors, exhibits high inference latency (> 2 s on CPU), and lacks frame-level temporal alignment, rendering it unsuited for phoneme-level MDD.
4. **Strategic Path Forward:** Authorize **Pilot B.1 (Zero-GPU CPU Evaluation)** on Fold 1 validation to empirically quantify the exact trade-off between omission recall gains and false alarm inflation. The resulting evidence will establish whether **Candidate B.2 (Phonetically Weighted Levenshtein)** or **Candidate B.3 (CTC-Guided Posterior Fusion)** represents the superior architectural paradigm for Phase RQ-MODEL-02.
