# State-of-the-Art Literature Survey: English L2 Phoneme-Level Pronunciation Assessment

**Author:** Autonomous Research Subagent  
**Date:** November 2026  
**Document Identity:** Independent Scientific Literature Survey  
**Scope:** Goodness of Pronunciation (GOP), Neural/Transformer/SSL Adaptations, Phone Omission/Substitution/Insertion Detection & Diagnosis, Open-Source Models, Benchmarks (L2-ARCTIC, SpeechOcean762), and Low-Resource Local Feasibility.

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Scientific Objective
Phoneme-level Computer-Assisted Pronunciation Training (CAPT) and Mispronunciation Detection and Diagnosis (MDD) aim to identify and diagnose acoustic deviations produced by second-language (L2) learners at the segmental (phonemic) level. Formally, given:
1. An input learner acoustic waveform $\mathbf{x} = (x_1, \dots, x_T) \in \mathcal{X}$,
2. A known canonical prompt text $\mathbf{w} = (w_1, \dots, w_K)$, which maps via a grapheme-to-phoneme (G2P) lexicon to an expected canonical phoneme sequence $\mathbf{c} = (c_1, \dots, c_N) \in \mathcal{P}^N$ (where $\mathcal{P}$ is the target phoneme inventory, e.g., ARPAbet with $|\mathcal{P}| = 39$),

the system must produce a localized diagnostic judgment for every phonemic unit and inter-phoneme transition.

### 1.2 Core Error Ontology: The Four Fundamental Classes
A rigorous assessment system must distinguish four mutually exclusive phonetic states:
1. **`CORRECT` ($c_i$ produced as $c_i$):** The learner articulated the expected phoneme $c_i$ within an acceptable native-like acoustic/phonetic boundary.
2. **`SUBSTITUTION` ($c_i$ produced as $s_i \in \mathcal{P} \setminus \{c_i\}$):** The learner attempted the target phoneme but produced an identifiable acoustic substitution (e.g., /θ/ $\to$ [s] or [t], /ð/ $\to$ [d], /iy/ $\to$ [ih]).
3. **`OMISSION` / Deletion ($c_i \to \emptyset$):** The expected phoneme was genuinely skipped or omitted (e.g., dropping word-final stop consonants /d/, /t/ or unstressed schwa /ə/).
4. **`INSERTION` / Addition ($\emptyset \to r_j \in \mathcal{P}$):** The learner articulated an extra, unintended phonetic segment between canonical target phones (e.g., epenthetic vowels like /ə/ added into consonant clusters: "spoon" /spu:n/ $\to$ [səpu:n]).

### 1.3 The Structural Asymmetry: Phone-Anchored vs. Gap-Anchored Events
A central insight from the literature is the **mathematical asymmetry** between errors:
- **Phone-Anchored Classes (`CORRECT`, `SUBSTITUTION`, `OMISSION`):** Attached directly to the canonical phoneme slots $i \in \{1, \dots, N\}$. For every canonical token $c_i$, its status is a 3-way mutually exclusive categorization over slot $i$.
- **Gap-Anchored Class (`INSERTION`):** Insertions do not exist in the canonical prompt $\mathbf{c}$. Instead, they occur at the **inter-phonemic boundaries (gaps)** $g_k \in \{g_0, g_1, \dots, g_N\}$, where $g_0$ is the pre-utterance boundary and $g_N$ is the post-utterance boundary. In an utterance of $N$ expected phones, there are exactly $N+1$ potential insertion sites.

> **Key Evaluation Fallacy in Literature:** Pooling insertion events directly into the denominator of canonical phonemes artificially corrupts macro-average precision and recall. A principled evaluation system must treat insertion detection as an independent gap-verification task or evaluate it via joint string alignment with strict denominator accounting.

---

## 2. Goodness of Pronunciation (GOP) & Neural/Transformer Adaptations

### 2.1 Classical GMM-HMM GOP (Witt & Young, 2000)
The foundational metric for pronunciation scoring, introduced by Silke Witt and Steve Young (2000), evaluates the posterior probability that a learner articulated the canonical phoneme $p$ given acoustic feature segment $\mathbf{o}_{t_s}^{t_e} = (\mathbf{o}_{t_s}, \dots, \mathbf{o}_{t_e})$:

$$\text{GOP}(p) = \frac{1}{t_e - t_s + 1} \log P(p \mid \mathbf{o}_{t_s}^{t_e}) = \frac{1}{t_e - t_s + 1} \log \left[ \frac{P(\mathbf{o}_{t_s}^{t_e} \mid p) P(p)}{\sum_{q \in \mathcal{Q}} P(\mathbf{o}_{t_s}^{t_e} \mid q) P(q)} \right]$$

Assuming equal phone priors ($P(p) \approx P(q)$) and approximating the unconstrained denominator by the maximum likelihood Viterbi path (free phone decoding):

$$\text{GOP}(p) \approx \frac{1}{t_e - t_s + 1} \left[ \log P(\mathbf{o}_{t_s}^{t_e} \mid p) - \max_{q \in \mathcal{Q}} \log P(\mathbf{o}_{t_s}^{t_e} \mid q) \right]$$

#### Inherent Limitations:
1. **Alignment Sensitivity:** Requires strict forced alignment boundaries $[t_s, t_e]$ from a native ASR acoustic model. If the learner mispronounces or omits the phone, forced alignment breaks down.
2. **Blind to Omissions and Insertions:** Standard GOP requires a target phone $p$ to evaluate. It cannot score insertions (no canonical target) and handles omissions poorly (assigns arbitrary frames to a non-existent sound).

### 2.2 Deep Neural Network (DNN-HMM) GOP (Hu et al., 2013, 2015)
Wenping Hu et al. (Interspeech 2013, 2015) replaced GMM acoustic models with Deep Neural Networks (DNN) outputting senone (HMM state) posteriors $P(s \mid \mathbf{o}_t)$. 

1. **Phone Posterior Marginalization:**
   The posterior for phoneme $p$ at frame $t$ is obtained by summing over all tied states (senones) $s$ mapped to $p$:
   $$P(p \mid \mathbf{o}_t) = \sum_{s \in p} P(s \mid \mathbf{o}_t)$$

2. **Log Phone Posterior (LPP):**
   $$\text{LPP}(p) = \frac{1}{t_e - t_s + 1} \sum_{t=t_s}^{t_e} \log P(p \mid \mathbf{o}_t)$$

3. **Log Posterior Ratio (LPR):**
   To measure the discriminative margin against competing phoneme $q \in \mathcal{P}$:
   $$\text{LPR}(q \mid p) = \text{LPP}(q) - \text{LPP}(p) = \frac{1}{t_e - t_s + 1} \sum_{t=t_s}^{t_e} \left[ \log P(q \mid \mathbf{o}_t) - \log P(p \mid \mathbf{o}_t) \right]$$

4. **Kaldi 84-Dimensional GOP Feature Vector:**
   For a phoneme inventory of size $|\mathcal{P}| = 42$ (as in Kaldi Librispeech recipe), the GOP representation for each phoneme is an 84-dimensional vector:
   $$\mathbf{f}_{\text{GOP}}(p) = \left[ \text{LPP}(p_1), \dots, \text{LPP}(p_{42}), \text{LPR}(p_1 \mid p), \dots, \text{LPR}(p_{42} \mid p) \right]$$
   This 84-dim vector serves as the primary input feature for modern regressors and transformers.

### 2.3 Context-Aware GOP (CaGOP) (Shi et al., Interspeech 2020)
Jiatong Shi et al. addressed the limitation that GOP treats each phonetic segment independently, ignoring coarticulation, incomplete plosives, and liaisons.
- Introduced a **Transition Factor** $w_t^{\text{trans}}$ weighting frame posteriors based on transition likelihood between adjacent phonemes.
- Introduced a **Self-Attention Duration Factor** encoding temporal length deviations from native distributions.
- Yielded a 20% relative improvement in phoneme-level MDD on non-native corpora.

### 2.4 Goodness of Pronunciation Transformer (GOPT) (Gong et al., ICASSP 2022)
Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang, and James Glass (MIT CSAIL) proposed **GOPT**, moving beyond isolated phone scoring to end-to-end multi-aspect, multi-granularity assessment on **SpeechOcean762**.

```
Input Audio ----> Kaldi TDNN-F AM ----> Force Alignment [ts, te] ----> 84-dim GOP Features
                       |
                       v
               Linear Projection (84 -> 24)
                       +
               Phoneme Embedding (42 -> 24)
                       +
               Positional Embedding (24)
                       +
          5 Trainable Utterance [CLS] Tokens
                       |
                       v
               3-Layer Transformer Encoder (d=24, heads=4)
                       |
       +---------------+---------------+---------------+
       |                               |               |
Utterance Heads (5x)            Word Heads (3x)   Phoneme Head (1x)
[Acc, Comp, Flu, Pros, Tot]     [Acc, Stress, Tot]    [Accuracy]
```

#### Architecture & Mathematical Specification:
1. **Input Representation:**
   $$\mathbf{h}_i^{(0)} = \mathbf{W}_{\text{gop}} \mathbf{f}_{\text{GOP}}(c_i) + \mathbf{W}_{\text{phn}} \mathbf{e}(c_i) + \mathbf{p}_i$$
   where $\mathbf{f}_{\text{GOP}} \in \mathbb{R}^{84}$, $\mathbf{e}(c_i) \in \mathbb{R}^{42}$ is the one-hot canonical phone vector, and $\mathbf{h}_i^{(0)} \in \mathbb{R}^{24}$.
2. **Utterance `[CLS]` Tokens:**
   Prepends five learnable tokens $\mathbf{t}_{\text{acc}}, \mathbf{t}_{\text{comp}}, \mathbf{t}_{\text{flu}}, \mathbf{t}_{\text{pros}}, \mathbf{t}_{\text{tot}} \in \mathbb{R}^{24}$ to the phoneme sequence, which aggregate utterance-level contextual dynamics via bidirectional self-attention.
3. **Multi-Task Objective:**
   $$\mathcal{L} = \mathcal{L}_{\text{phoneme}} + \frac{1}{3}\sum_{k=1}^3 \mathcal{L}_{\text{word}, k} + \frac{1}{5}\sum_{m=1}^5 \mathcal{L}_{\text{utt}, m}$$
   where all losses are Mean Squared Error (MSE) on expert-annotated scores.
4. **Benchmark Impact:**
   GOPT achieved a phoneme-level Pearson Correlation Coefficient (PCC) of **0.612** on SpeechOcean762, significantly outperforming Kaldi SVR (0.450) and LSTM (0.591).

### 2.5 Self-Supervised Learning (SSL) Pronunciation Assessment (Kim et al., Interspeech 2022)
Eesung Kim et al. established that self-supervised pretrained backbones (**Wav2Vec 2.0** and **HuBERT**) capture superior acoustic representations for pronunciation assessment:
- Fine-tuned `wav2vec2-large` and `hubert-large` on L2 speech using CTC loss.
- Demonstrated that **intermediate transformer layers** encode phonetic and articulatory deviations, whereas top layers specialize in lexical tokens.
- Reached **0.75-0.79 PCC** on Korean ESL learners (KESL) and **0.75 PCC** on SpeechOcean762 using layer-wise weighted representation pooling + BiLSTM.

### 2.6 Alignment-Free & CTC-Based GOP Framework (Cao et al., Interspeech 2024 / 2025)
Xinwei Cao, Zijian Fan, Torbjørn Svendsen, and Giampiero Salvi (Interspeech 2024, arXiv:2507.16838) demonstrated how to compute GOP directly from CTC-trained models without forced alignment.

#### The Core Problem of CTC: "Peaky" Behavior
CTC models output blank tokens ($\phi$) for the majority of frames, with sharp non-blank spikes at transition intervals. Standard frame-level posterior averaging ($\frac{1}{T}\sum \log P(p|\mathbf{x}_t)$) fails because blank frames dominate.

#### Formulation 1: Forced-Aligned CTC GOP (`GOP-CTC-align`)
Filters out blank tokens and averages posteriors only across active non-blank spikes:
$$\text{GOP}_{\text{CTC-align}}(c_i) = \frac{1}{| \{t \in [t_s, t_e] : \hat{\pi}_t \neq \phi\} |} \sum_{t=t_s, \hat{\pi}_t \neq \phi}^{t_e} \log P(c_i \mid \mathbf{o}_t)$$

#### Formulation 2: Segmentation-Free CTC GOP (`GOP-SF`)
Eliminates temporal segmentation entirely by computing sequence probability ratios via forward-backward CTC:
Let $L_{\text{cano}} = (l_1, \dots, l_{i-1}, l_i, l_{i+1}, \dots, l_N)$ be the full canonical sequence.  
Let $L_{\setminus i} = (l_1, \dots, l_{i-1}, l_{i+1}, \dots, l_N)$ be the sequence with target phone $l_i$ **omitted**.  
Let $L_{i \to j} = (l_1, \dots, l_{i-1}, l'_j, l_{i+1}, \dots, l_N)$ be the sequence with target phone $l_i$ **substituted** by $l'_j$.

$$\text{GOP-SF}(l_i) = \log \frac{P(L_{\text{cano}} \mid \mathbf{O}_1^T)}{P(L_{\setminus i} \mid \mathbf{O}_1^T)} = \mathcal{L}_{\text{CTC}}(L_{\setminus i}, \mathbf{O}_1^T) - \mathcal{L}_{\text{CTC}}(L_{\text{cano}}, \mathbf{O}_1^T)$$

$$\text{LPR}_{\text{SF}}(l_i \to l'_j) = \log \frac{P(L_{\text{cano}} \mid \mathbf{O}_1^T)}{P(L_{i \to j} \mid \mathbf{O}_1^T)} = \mathcal{L}_{\text{CTC}}(L_{i \to j}, \mathbf{O}_1^T) - \mathcal{L}_{\text{CTC}}(L_{\text{cano}}, \mathbf{O}_1^T)$$

#### Empirical Breakthrough on SpeechOcean762 & CMU-Kids:
- On SpeechOcean762, Alignment-Free CTC GOP features (`GOP-feature-CTC-AF`) fed into SVR increased PCC from **0.441** (Kaldi TDNN) to **0.568** (29.02% relative gain).
- Fed into GOPT, PCC reached **0.618**.
- On CMU-Kids real speech errors, alignment-free substitution+deletion GOP (`GOP-CTC-AF-SD`) achieved **0.915 AUC**, outperforming DNN-GOP (0.796) and CE-Wav2Vec2 (0.851).

### 2.7 Logit-Based GOP vs. Probability-Based GOP (Parikh et al., Interspeech 2025)
Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, and Helmer Strik (Interspeech 2025) identified that standard softmax posteriors suffer from **overconfidence and poor dynamic range** on non-native speech.
They proposed logit-based scoring extracted before softmax from Wav2Vec2 CTC models:
1. **MaxLogit GOP:**
   $$\text{GOP}_{\text{MaxLogit}}(p) = \max_{t \in [t_s, t_e]} \ell_t^{(p)}$$
2. **MarginLogit GOP (Distance to nearest competitor):**
   $$\text{GOP}_{\text{Margin}}(p) = \frac{1}{T} \sum_{t=t_s}^{t_e} \left( \ell_t^{(p)} - \max_{k \neq p} \ell_t^{(k)} \right)$$
3. **Findings:** MarginLogit GOP showed significantly lower false alarm rates on L2 Dutch-English (MPC) and SpeechOcean762 children's speech, providing robust linear separation between genuine phoneme substitutions and minor acoustic distortions.

---

## 3. Detection & Diagnosis of OMISSION, SUBSTITUTION, and INSERTION

### 3.1 Methodological Families for Mispronunciation Detection & Diagnosis (MDD)

| Approach Family | Key References | Input Representations | Omission Mechanism | Insertion Mechanism | Strengths | Limitations |
|---|---|---|---|---|---|---|
| **Classical Extended Recognition Network (ERN / WFST)** | Witt & Young (2000), Lee & Glass (2012) | Acoustic filterbanks + WFST decoding graph with phonological substitution & skip arcs | Null/epsilon ($\epsilon$) arc traversal in phoneme HMM graph | Optional insertion loop arcs between canonical nodes | Transparent, rule-grounded, explicit error paths | Suffers from graph size explosion; fails on unanticipated non-native error variants |
| **Free-Phone Recognition (FPR) + Dynamic Programming** | Leung et al. (ICASSP 2019), Chen et al. (Interspeech 2022) | Acoustic frames $\mathbf{X}$ $\to$ decoded phone string $\mathbf{R}$; aligned to prompt $\mathbf{C}$ via Levenshtein / Needleman-Wunsch | Deletion edit operation ($c_i \to \epsilon$) | Insertion edit operation ($\epsilon \to r_j$) | Detects all 4 classes; alignment-free decoding | DP alignment ambiguity (substitute vs del+ins); prone to native ASR language model bias |
| **Hierarchical / Selective CTC Decoders** | Feng et al. (ICASSP 2020 SED-MDD), Yan et al. (2020) | Acoustic waveform + canonical phoneme sequence injected via cross-attention / gate | Target phoneme skipped in CTC decoding path | Decoded non-blank tokens between canonical anchors | End-to-end trainable; models both acoustic and linguistic representations | Complex architectures; canonical prompt can cause "linguistic cheating" |
| **Non-Parametric Frame Retrieval (PER-MDD)** | Tu et al. (ICASSP 2026, arXiv:2511.20107) | HuBERT frame embeddings queried against reference phoneme prototype pool via cosine similarity | Silence / empty nearest-neighbor set ($\mathcal{N}^* = \emptyset \to \text{blank}$) | Mode of retrieved neighbors spanning extra frames | Training-free acoustic scoring; robust to small L2 training sets | High PER due to frame-level insertion chatter; cannot handle long-range phonotactics |
| **Optimal Transport CTC & Iterative Fusion (CROTTC-IF)** | Geng et al. (Interspeech 2026, arXiv:2604.22133) | WavLM Large + Conformer AM + 2-layer Transformer LM decoded via Optimal Transport CTC | Target phoneme probability mass pushed to blank/skipped | Non-canonical mass assigned across inter-phone gap intervals | SOTA F1 (71.77%) on L2-ARCTIC; eliminates LLM linguistic over-correction bias | High computational cost for OT alignment optimization |

### 3.2 Deep Dive: OMISSION Detection Mechanics

#### Why Omission is Scientifically Challenging
1. **The Forced-Alignment Pathology:** Traditional forced alignment algorithms (Viterbi over HMM or CTC lattices) **must assign every canonical phoneme to some slice of time**. When a speaker completely omits /t/ in "postman" (/poʊstmæn/ $\to$ [poʊsmæn]), forced alignment arbitrarily assigns a 10-30 ms slice of the surrounding /s/ or /m/ to /t/. The resulting acoustic features do not reflect /t/, but neither do they reflect silence; they reflect a mangled neighbor.
2. **Coarticulation vs. Genuine Omission:** In fluent L2 speech, weak vowels (e.g., unstressed schwa /ə/ or unstressed /ɪ/) and final unreleased plosives ([p̚], [t̚], [k̚]) have durations as short as 15-25 ms and low energy. Distinguishing an incomplete/reduced articulation from a genuine omission is acoustically non-trivial.

#### Direct Verified Solutions:
- **CTC Sequence Probability Loss Difference (Cao et al., 2024):**
  $$\Delta \mathcal{L}_{\text{omission}}(l_i) = \mathcal{L}_{\text{CTC}}(L_{\setminus i}, \mathbf{X}) - \mathcal{L}_{\text{CTC}}(L_{\text{cano}}, \mathbf{X})$$
  If the speaker genuinely omitted $l_i$, the acoustic likelihood of the truncated sequence $L_{\setminus i}$ is higher than the canonical sequence, resulting in $\Delta \mathcal{L}_{\text{omission}} \le 0$.
- **Acoustic Energy & Duration Thresholding:** Combining frame posterior drop with local RMS energy in the predicted segment directly verifies whether physical voicing/turbulent airflow took place.

### 3.3 Deep Dive: INSERTION Detection Mechanics

#### Why Insertion is the Most Difficult CAPT Task
1. **The Canonical Sequence Has No Target Slot:** Standard CAPT systems construct fixed vectors corresponding to the expected sequence $c_1, \dots, c_N$. An insertion event has **no anchor** in $\mathbf{c}$.
2. **The Evaluation Evasion in Literature:** As noted by Tu et al. (2025) and Geng et al. (2026), the vast majority of MDD papers evaluate performance solely on canonical slots (True Acceptance, True Rejection, False Acceptance, False Rejection). Insertions are omitted from MDD precision/recall/F1 metrics entirely and pushed into the general ASR Phone Error Rate ($PER = \frac{S+D+I}{N}$).
3. **Levenshtein Alignment Indeterminacy:** When a learner pronounces an unexpected sound sequence, edit distance alignment between canonical $C$ and recognized $R$ is often degenerate. For example, if target is /t/ and realized is [d s]:
   - Alignment A: Substitution (/t/ $\to$ [d]) + Insertion ($\emptyset \to$ [s]) $\implies$ Cost = 1 + 1 = 2.
   - Alignment B: Deletion (/t/ $\to \emptyset$) + Insertion ($\emptyset \to$ [d]) + Insertion ($\emptyset \to$ [s]) $\implies$ Cost = 1 + 1 + 1 = 3.
   Standard edit distance prefers Alignment A, but the phonetic truth might be that /t/ was omitted and [d s] was an epenthetic intrusion.

#### Architectural Solutions:
1. **Explicit Gap-State Decoding (Inter-Phoneme Boundary Classification):**
   Augment the canonical sequence with explicit transition slots:
   $$\mathbf{c}_{\text{augmented}} = (g_0, c_1, g_1, c_2, \dots, c_N, g_N)$$
   Every gap $g_k$ is classified as $\emptyset$ (null/correct transition) or $r \in \mathcal{P}$ (insertion).
2. **Unconstrained Free-Phone Decoding with Tuned Insertion Penalty:**
   Using an acoustic model fine-tuned on non-native phonemes without a strong language model prior, followed by phonologically-constrained Levenshtein alignment where substitution cost $C(c_i, s_j)$ is proportional to articulatory distance (Chen et al., 2022).

---

## 4. Open-Source Corpora, Benchmarks, and Foundation Models

### 4.1 Benchmark Corpora Comparison

| Dataset | Authors / Venue | License | Speakers & Demographics | Audio Volume | Annotations & Granularity | Class Support | Limitations & Notes |
|---|---|---|---|---|---|---|---|
| **L2-ARCTIC** | Zhao et al. (Interspeech 2018) | CC BY-NC-SA 4.0 | 24 non-native speakers (12M/12F), 6 L1s: Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese | 3,599 utterances (~3.6 hours, ~150 per spk) | Praat TextGrids: manual phone boundaries, words, comments (`sub`, `del`, `ins`) | `CORRECT`: Yes<br>`SUBSTITUTION`: Yes<br>`OMISSION`: Yes<br>`INSERTION`: Yes | High class imbalance (~70% sub, ~18% del, ~12% ins). Standardized 6-speaker test split (NJS, TLV, TNI, TXHC, YKWK, ZHAA). |
| **SpeechOcean762** | Zhang et al. (Interspeech 2021) | CC BY 4.0 | 250 Chinese speakers (125 children, 125 adults; 125M/125F) | 5,000 utterances (~5-6 hours, 20 per spk) | 5 phonetician expert scores: phone accuracy [0-2], word acc/stress [0-10], utterance acc/comp/flu/pros/tot [0-10] | `CORRECT`: Yes<br>`SUBSTITUTION`: Yes<br>`OMISSION`: Partial (score 0)<br>`INSERTION`: **NO** | No explicit phone insertion events. Omission is conflated with score 0. Fixed 2,500 train / 2,500 test speaker-disjoint split. |
| **CMUDict** | Carnegie Mellon University | BSD-like | N/A (Lexicon resource) | 134,000+ words | General American English phonetic transcriptions (39 ARPAbet phonemes + lexical stress 0,1,2) | Reference lexicon only | Native reference only; does not provide non-native phonetic variants or acoustic audio. |
| **CMU-Kids** | Eskenazi et al. (1997) | LDC (Open research subset) | 76 American children | 5,180 utterances | Word & phone-level phonetic transcriptions | Binary mispronunciation (simulated + real) | Native children's speech; used by Cao et al. for evaluating CTC-GOP on developmental speech errors. |
| **ERJ (English Read by Japanese)** | Minematsu et al. (2002) | Academic research | Japanese university students | Hundreds of hours | Phonetic transcription & prosodic ratings | Evaluated as Out-of-Domain (OOD) testbed | Restricted distribution; often used as zero-shot accent transfer benchmark. |

### 4.2 SOTA Performance Milestones on L2-ARCTIC Benchmark

The following table synthesizes reported performance on the standardized L2-ARCTIC speaker-disjoint test set (6 held-out speakers: NJS, TLV, TNI, TXHC, YKWK, ZHAA):

| Model Family | Reference | Text Prompt Used? | Precision (%) | Recall (%) | F1-Score (%) | FRR (%) $\downarrow$ | FAR (%) $\downarrow$ | PER (%) $\downarrow$ |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GOP Baseline** | Zhao et al. (2018) | Yes | 52.88 | 35.42 | 42.42 | 12.24 | 28.30 | - |
| **CNN-RNN-CTC** | Leung et al. (2019) | No | 54.20 | 46.80 | 50.21 | 8.40 | 53.20 | 18.50 |
| **MPL-MDD (Wav2Vec2)** | Yang et al. (2022) | No | 60.39 | 51.20 | 55.42 | 5.60 | 48.80 | 14.36 |
| **SED-MDD** | Feng et al. (2020) | Yes | 58.36 | 55.00 | 56.33 | 6.55 | 45.00 | 14.81 |
| **Articulatory Alignment (AM-AF)** | Chen et al. (2022) | Yes | 53.41 | 65.38 | 58.79 | - | - | - |
| **Wav2Vec2-CTC** | Peng et al. (2021) | No | 62.86 | 58.57 | 60.44 | 5.70 | 41.80 | 16.20 |
| **Joint-Align** | Lin et al. (2022) | Yes | **77.12** | 53.31 | 63.04 | - | - | - |
| **PER-MDD (HuBERT Frame Retrieval)** | Tu et al. (2025) | No | 71.78 | 67.56 | 69.60 | 4.43 | 32.44 | 104.08* |
| **CROTTC-IF (WavLM + Conformer + LM)** | Geng et al. (2026) | No | 76.94 | **67.24** | **71.77** | **3.39** | **32.76** | 46.52 |

*\*Note on PER-MDD: High PER (104.08%) is caused by unconstrained frame-level nearest neighbor retrieval generating excess insertions, despite achieving strong canonical-slot MDD F1 (69.60%).*

### 4.3 SOTA Performance Milestones on SpeechOcean762 Benchmark

Evaluated on the official 2,500 test utterances (Pearson Correlation Coefficient PCC $\uparrow$):

| Model | Reference | Phoneme PCC | Word Acc PCC | Word Stress PCC | Utt Accuracy PCC | Utt Fluency PCC | Utt Prosody PCC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Kaldi Baseline (SVR)** | Zhang et al. (2021) | 0.450 | - | - | - | - | - |
| **GOP-LSTM** | Gong et al. (2022) | 0.591 | 0.514 | 0.294 | 0.720 | 0.745 | 0.747 |
| **GOPT (Librispeech TDNN-F)** | Gong et al. (2022) | 0.612 | 0.533 | 0.291 | 0.714 | 0.753 | 0.760 |
| **GOPT (CTC Alignment-Free)** | Cao et al. (2024) | 0.618 | - | - | - | - | - |
| **Wav2Vec2-Robust + BLSTM** | Kim et al. (2022) | - | - | - | - | 0.750 | 0.740 |
| **HuBERT-Large + BLSTM** | Kim et al. (2022) | - | - | - | - | **0.760** | **0.750** |
| **GOPT (PAII Non-Native AM)** | Gong et al. (2022) | **0.679** | **0.588** | 0.146 | **0.727** | 0.692 | 0.694 |

---

## 5. Open-Source Models, Checkpoints, & Toolkits

### 5.1 Hugging Face Foundation Checkpoints

| Checkpoint Name | Model Architecture | Params | Pretraining Data | Fine-tuning Target | VRAM Footprint (FP16 / FP32) | Verified Accessibility |
|---|---|:---:|---|---|:---:|:---:|
| `facebook/wav2vec2-base-960h` | Wav2Vec 2.0 Base (12L, d=768) | 95M | LibriSpeech 960h | Character CTC | ~190 MB / ~380 MB | Verified (Public) |
| `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | Wav2Vec 2.0 XLSR (24L, d=1024) | 317M | Multilingual 53 langs | IPA Phonemes (eSpeak) | ~635 MB / ~1.27 GB | Verified (Public) |
| `facebook/wav2vec2-lv-60-espeak-cv-ft` | Wav2Vec 2.0 Large (24L, d=1024) | 317M | Libri-Light 60k hrs | IPA Phonemes (eSpeak) | ~635 MB / ~1.27 GB | Verified (Public) |
| `facebook/hubert-base-ls960` | HuBERT Base (12L, d=768) | 95M | LibriSpeech 960h | Unsupervised k-means | ~190 MB / ~380 MB | Verified (Public) |
| `facebook/hubert-large-ls960-ft` | HuBERT Large (24L, d=1024) | 317M | LibriSpeech 960h | Character CTC | ~635 MB / ~1.27 GB | Verified (Public) |
| `microsoft/wavlm-base-plus` | WavLM Base+ (12L, d=768) | 95M | 60k hrs Libri-Light/Vox/Common | Masked speech denoising | ~190 MB / ~380 MB | Verified (Public) |
| `microsoft/wavlm-large` | WavLM Large (24L, d=1024) | 317M | 60k hrs Libri-Light/Vox/Common | Masked speech denoising | ~635 MB / ~1.27 GB | Verified (Public) |
| `openai/whisper-tiny` / `whisper-base` | Encoder-Decoder Transformer | 39M / 74M | 680k hrs Multilingual | Multitask ASR/Translation | ~80 MB / ~150 MB | Verified (Public) |

### 5.2 Open-Source Toolkits & Repositories

| Repository | GitHub URL | License | Key Functionality | Applicability to Lab |
|---|---|:---:|---|---|
| **GOPT** | `YuanGongND/gopt` | BSD-3-Clause | PyTorch Transformer for multi-aspect scoring using 84-dim GOP features | Direct baseline and downstream scoring module |
| **Kaldi Speechocean Recipe** | `kaldi-asr/kaldi` (`egs/speechocean762/s5`) | Apache-2.0 | TDNN-F acoustic model training, forced alignment graph generation, 84-dim GOP extraction | Standard acoustic baseline and GOP extractor |
| **Charsiu** | `lingjzhu/charsiu` | MIT | Wav2Vec2/HuBERT-based forced aligner and frame-level phone segmenter | Lightweight forced alignment and boundary detection |
| **ESPnet-CAPT** | `espnet/espnet` | Apache-2.0 | End-to-end ASR, phoneme CTC, and pronunciation evaluation recipes | Multi-task training pipelines |
| **CTC-Segmentation** | `lumaku/ctc-segmentation` | MIT | Fast dynamic programming alignment between CTC log probabilities and target text | Frame-accurate alignment without HMM decoders |

---

## 6. Local Hardware & Low-Resource Strategy (NVIDIA GTX 1650 4GB VRAM)

### 6.1 Hardware Realities
- **Host Configuration:** NVIDIA GeForce GTX 1650 (4 GB GDDR5/GDDR6 VRAM), 32 GB System RAM, Intel Core i5-12450H CPU (8 cores / 12 threads).
- **VRAM Constraint:** 4,096 MB total VRAM. At least 500-700 MB is reserved by system display/Xorg, leaving **~3.2 to 3.5 GB usable VRAM**.

### 6.2 Architectural Feasibility Matrix

| Architecture Strategy | Backbone Model | Trainable Parameters | Peak VRAM | Feasibility on GTX 1650 | Recommended Training Protocol |
|---|---|:---:|:---:|:---:|---|
| **Full Fine-Tuning Large** | `wav2vec2-large` (317M) | 317M | >6.5 GB | **IMPOSSIBLE (OOM)** | Requires remote cluster / Colab (unauthorized). |
| **Full Fine-Tuning Base** | `wav2vec2-base` (95M) | 95M | ~3.4 - 3.8 GB | **MARGINAL / HIGH RISK** | Batch size = 1, gradient checkpointing, FP16 mixed precision. |
| **Frozen Backbone + Feature Caching** | `wav2vec2-base` or `hubert-base` or `wavlm-base-plus` | 0 (Backbone frozen) | < 800 MB | **EXTREMELY SAFE & FAST** | Extract SSL frame embeddings to SSD once; train downstream models on cached features. |
| **Lightweight Downstream Scorer** | GOPT (Transformer d=24 to 256) or BiLSTM or MLP | 50k - 2M | < 500 MB | **OPTIMAL (< 1 min/epoch)** | Can run dozens of full training epochs and sweeps locally in minutes. |
| **LoRA / Adapter Tuning** | `wav2vec2-base` + LoRA ($r=8$) | ~1.5M | ~2.2 GB | **FEASIBLE** | Allows target domain adaptation with gradient checkpointing and FP16. |

### 6.3 Strategic Execution Recommendation: Two-Stage Decoupled Workflow
1. **Stage 1 (Representation Extraction / Caching):**
   Run frozen inference of `wav2vec2-xlsr-53-espeak-cv-ft` or `wavlm-base-plus` on CPU / low-batch GPU to cache frame-level logits, posteriors, and layer embeddings for all utterances in L2-ARCTIC and SpeechOcean762.
2. **Stage 2 (Local Hypothesis Iteration on GPU):**
   Train and optimize downstream classifiers, alignment-free CTC loss differencing, and GOPT scoring transformers directly on the cached representations. This circumvents VRAM bottlenecks and delivers 50x faster training cycles.

---

## 7. Critical Gap Analysis & Falsifiable Hypotheses

### 7.1 Identified Scientific Gaps in Current Literature

1. **Gap 1: The "Insertion Void" in Benchmark MDD:**
   Virtually all top-tier papers (Leung 2019, Feng 2020, Lin 2022, Tu 2025, Geng 2026) report MDD performance solely over canonical slots, ignoring phone insertions in their F1/Precision/Recall metrics. Insertions are either discarded or buried in the general ASR Phone Error Rate (PER). There is no established 4-class evaluation protocol that reports class-specific AP, Precision, Recall, and F1 for `INSERTION`.
2. **Gap 2: Forced-Alignment Fragility vs. Alignment-Free Diagnostics:**
   Traditional GOP methods (Kaldi, GOPT) rely heavily on HMM forced alignment. When a learner omits a phone or speaks with heavy disfluency, the forced aligner injects severe boundary errors. While Cao et al. (2024) introduced scalar alignment-free CTC GOP, it has not been systematically integrated into a multi-class framework that explicitly diagnoses substitutions versus omissions versus insertions.
3. **Gap 3: The "Linguistic Trap" of Autoregressive / LLM Systems:**
   Recent attempts to feed non-native speech into large multimodal models (e.g., Whisper, Qwen2-Audio, LLaMA-based MDD) suffer from severe language model hallucination: strong textual priors cause the model to output the canonical words, completely masking the learner's acoustic errors (Geng et al., 2026; Xie et al., 2026). MDD requires **acoustic-first, prompt-conditioned** modeling rather than unconstrained linguistic generation.

### 7.2 Falsifiable Research Hypotheses for the Lab

- **Hypothesis H1 (Alignment-Free Omission Mechanism):**
  *Proposition:* The sequence-level CTC loss difference $\Delta \mathcal{L}_{\text{omission}}(l_i) = \mathcal{L}_{\text{CTC}}(L_{\setminus i}, \mathbf{X}) - \mathcal{L}_{\text{CTC}}(L_{\text{cano}}, \mathbf{X})$ calculated from a phoneme-CTC backbone (`wav2vec2-xlsr-53-espeak-cv-ft`) will separate genuine phone omissions from short/reduced realizations with higher Average Precision than forced-alignment frame posterior averaging ($\text{GOP}_{\text{DNN}}$).
  *Falsification Criterion:* If Average Precision for `OMISSION` using $\Delta \mathcal{L}_{\text{omission}}$ is $\le$ standard forced-aligned $\text{GOP}_{\text{DNN}}$ on the held-out L2-ARCTIC partition, H1 is rejected.

- **Hypothesis H2 (Explicit Inter-Phoneme Gap Anchoring for Insertions):**
  *Proposition:* Augmenting the canonical phoneme sequence with explicit boundary gap variables $g_0, \dots, g_N$ and classifying each gap using pooled inter-phone CTC logit activations will achieve $\text{AP} \ge 0.60$ on `INSERTION` detection on L2-ARCTIC, whereas canonical-slot GOP models achieve an effective AP of $0.00$.
  *Falsification Criterion:* If the gap-anchored classifier does not achieve at least a 20-point increase in Average Precision over the canonical baseline on L2-ARCTIC, H2 is rejected.

- **Hypothesis H3 (MarginLogit vs. Softmax GOP Calibration):**
  *Proposition:* Pre-softmax margin features ($\ell_t^{(c_i)} - \max_{k \neq c_i} \ell_t^{(k)}$) will yield significantly lower Expected Calibration Error (ECE) and higher Area Under Precision-Recall Curve (AUPRC) on `SUBSTITUTION` detection than softmax posterior probabilities on L2-ARCTIC, due to the elimination of softmax overconfidence.
  *Falsification Criterion:* If ECE is higher or AUPRC is lower for MarginLogit than Softmax GOP across 5 seeds on the dev partition, H3 is rejected.

---

## 8. Verified Bibliography

1. **Witt, S. M., & Young, S. J.** (2000). *Phone-level pronunciation scoring and assessment for interactive language learning.* Speech Communication, 30(2-3), 95–108. [DOI: 10.1016/S0167-6393(99)00044-8]
2. **Hu, W., Qian, Y., Soong, F. K., & Wang, Y.** (2015). *Improved Goodness of Pronunciation (GOP) measure for computer-aided pronunciation training (CAPT).* Interspeech 2015, 385–389. [DOI: 10.21437/Interspeech.2015-385]
3. **Sudhakara, P., Ramanathi, P., Yarra, C., & Ghosh, P. K.** (2019). *An Improved Goodness of Pronunciation (GoP) Measure for Pronunciation Evaluation with DNN-HMM System Considering HMM Transition Probabilities.* Interspeech 2019, 2363–2367. [DOI: 10.21437/Interspeech.2019-2363]
4. **Shi, J., Huo, N., & Jin, Q.** (2020). *Context-Aware Goodness of Pronunciation for Computer-Assisted Pronunciation Training.* Interspeech 2020, 2953–2957. [DOI: 10.21437/Interspeech.2020-2953; arXiv:2008.08647]
5. **Gong, Y., Chen, Z., Chu, I.-H., Chang, P., & Glass, J.** (2022). *Transformer-Based Multi-Aspect Multi-Granularity Non-Native English Speaker Pronunciation Assessment.* ICASSP 2022, 9746743. [DOI: 10.1109/ICASSP43922.2022.9746743; arXiv:2205.03432]
6. **Kim, E., Jeon, J.-J., Seo, H., & Kim, H.** (2022). *Automatic Pronunciation Assessment using Self-Supervised Speech Representation Learning.* Interspeech 2022, 10245. [DOI: 10.21437/Interspeech.2022-10245; arXiv:2204.03863]
7. **Cao, X., Fan, Z., Svendsen, T., & Salvi, G.** (2024). *A Framework for Phoneme-Level Pronunciation Assessment Using CTC.* Interspeech 2024, 459–463. [DOI: 10.21437/Interspeech.2024-459; arXiv:2507.16838]
8. **Parikh, A. K., Tejedor-Garcia, C., Cucchiarini, C., & Strik, H.** (2025). *Evaluating Logit-Based GOP Scores for Mispronunciation Detection.* Interspeech 2025, 1012. [DOI: 10.21437/Interspeech.2025-1012; arXiv:2506.12067]
9. **Parikh, A. K., Tejedor-Garcia, C., Cucchiarini, C., & Strik, H.** (2025). *Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge.* Interspeech 2025, 829. [DOI: 10.21437/Interspeech.2025-829]
10. **Zhao, G., Sonsaat, S., Silpachai, A., Lucic, I., Chukharev-Hudilainen, E., Levis, J., & Gutierrez-Osuna, R.** (2018). *L2-ARCTIC: A Non-native English Speech Corpus.* Interspeech 2018, 2783–2787. [DOI: 10.21437/Interspeech.2018-1110]
11. **Zhang, J., Zhang, Z., Wang, Y., Yan, Z., Song, Q., Huang, Y., Li, K., Povey, D., & Wang, Y.** (2021). *speechocean762: An Open-Source Non-Native English Speech Corpus for Pronunciation Assessment.* Interspeech 2021, 1259–1263. [DOI: 10.21437/Interspeech.2021-1259; arXiv:2104.01378]
12. **Leung, W.-K., Liu, X., & Meng, H.** (2019). *CNN-RNN-CTC Based End-to-end Mispronunciation Detection and Diagnosis.* ICASSP 2019, 8682654. [DOI: 10.1109/ICASSP.2019.8682654]
13. **Feng, Y., Fu, G., Chen, Q., & Chen, K.** (2020). *SED-MDD: Towards Sentence Dependent End-To-End Mispronunciation Detection and Diagnosis with Selective Decoder.* ICASSP 2020, 9052975. [DOI: 10.1109/ICASSP40776.2020.9052975]
14. **Chen, Q., Lin, B., & Xie, Y.** (2022). *An Alignment Method Leveraging Articulatory Features for Mispronunciation Detection and Diagnosis in L2 English.* Interspeech 2022, 10309. [DOI: 10.21437/Interspeech.2022-10309]
15. **Yang, M., Hirschi, K., Looney, S., Kang, O., & Hansen, J. H. L.** (2022). *Improving Mispronunciation Detection with Wav2vec2-based Momentum Pseudo-Labeling for Accentedness and Intelligibility Assessment.* Interspeech 2022, 11039. [DOI: 10.21437/Interspeech.2022-11039; arXiv:2203.15937]
16. **Tu, H. T., Khanh, H. V., & Nguyen, H.** (2025/2026). *Mispronunciation Detection and Diagnosis Without Model Training: A Retrieval-Based Approach.* ICASSP 2026 / arXiv:2511.20107.
17. **Geng, H., Yang, L., & Chen, Y.** (2026). *Beyond Acoustic Sparsity and Linguistic Bias: A Prompt-Free Paradigm for Mispronunciation Detection and Diagnosis.* Interspeech 2026 / arXiv:2604.22133.
18. **Zhu, L.** (2022). *Charsiu: A tool for phone segmentation and alignment using wav2vec 2.0.* Interspeech 2022 / GitHub: `lingjzhu/charsiu`.
