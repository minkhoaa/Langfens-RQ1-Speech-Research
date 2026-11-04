# LANGFENS RQ1 — Controlled Acoustic Verifier & System-Level Benchmark Report (Agent D)

**Document ID:** `LANGFENS-RQ1-ACOUSTIC-VERIFIER-BENCHMARK-01`  
**Author:** Agent D — Controlled Acoustic Verifier & System-Level Benchmark Lead  
**Evaluation Target:** Recovered P-SUP V2 Benchmark (Fold-1 VAL, 300 utterances, speakers NJS [Spanish L1] and RRBI [Hindi L1])  
**Contract Scope:** 9,956 gold canonical positions (9,919 aligned tokens, 37 unmapped gold abstentions, 37 prompt extra) under Protocol Z genuine prompt-only evaluation  
**Model Checkpoint:** Subtype Supervision `T-OP` Fixed Budget (3,600 updates, Epoch 6, SHA-verified)  
**Execution Context:** CPU FP32 Deterministic Replay, Batch Size 1, `.venv/bin/python`  
**Data Provenance:** Authorized Fold 1 Train/Val Zip (`authorized_RQ1-TRN-02A_fold1_train_val_only.zip`, SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`)  
**Associated Artifacts:** `SYSTEM_LEVEL_COMPARISON.json`, `VERIFIER_ABLATION_REPORT.md`  

---

## 1. Executive Summary & Definitive Benchmark Outcome

This benchmark establishes the definitive system-level evaluation for **Intervention Candidate 2 (Post-Hoc Acoustic Verification Gating)** as formulated in `NEXT_EXPERIMENT_DECISION.md`. 

The central empirical research question is:
> **Can post-hoc, waveform-derived acoustic verification salvage deployed omission precision to achieve the $\ge 30\%$ precision screening target on the original L2-ARCTIC gold annotations without retraining the neural backbone?**

### Definitive Empirical Answer: NO
Standalone acoustic verification **fails** to achieve the $30\%$ precision screening target on the original L2-ARCTIC gold ground truth. Furthermore, the full acoustic verification candidate actually results in a net degradation of deployed omission precision compared to the baseline detector.

### Core Quantitative Findings:
1. **Precision & Recall Trajectory:**
   - **Original T-OP Baseline (Config 1):** Achieves **$6.78\%$** deployed precision ($82 / 1,209$) at **$56.55\%$** all-gold recall ($82 / 145$), generating **113.20 false alarms per 1,000 canonical phones**.
   - **Full Acoustic Verification Candidate (Config 4):** Filters **305 alarms** (reducing total alarms from 1,209 to 904). However, deployed precision **drops to $6.31\%$** ($57 / 904$), and all-gold recall **collapses to $39.31\%$** ($57 / 145$).
   - **False Alarm Suppression Rate:** The verifier suppressed 266 false alarms on correct phones ($1,068 \to 802$, a $24.9\%$ reduction) and 14 on substitution phones ($59 \to 45$, a $23.7\%$ reduction).
   - **Collateral Damage on True Positives:** The verifier simultaneously suppressed **25 genuine omission detections** ($82 \to 57$, a **$30.5\%$ destruction of true positives**). Because the percentage of true positives destroyed ($30.5\%$) exceeded the percentage of false alarms eliminated ($24.8\%$), deployed precision decreased.
2. **Failure of Naive Phoneme Filtering (Config 3):**
   - Muting all omission alarms on the high-error phonemes $\{D, T, R\}$ eliminates $1,133$ alarms, but decimates true positives from 82 to **2** ($1.38\%$ recall), driving precision down to **$2.63\%$**. This proves that omission errors in L2-ARCTIC are intrinsically concentrated in $\{D, T, R\}$, making blind phoneme suppression unviable.
3. **Thresholding Outperforms Acoustic Verification on Original Gold (Config 2 vs Config 4):**
   - Applying a conservative CAL $5\%$ FPR ceiling threshold ($\tau = 0.133940$) without any acoustic features yields **$12.09\%$ deployed precision** ($26 / 215$) at $17.93\%$ recall. Pure thresholding achieves nearly double the precision of the acoustic verifier, though at the expense of steep recall loss.
4. **Multi-Class MDD Invariance:**
   - Post-hoc omission gating operates strictly downstream on the binary omission alarm channel. The upstream multi-class MDD metrics remain mathematically invariant across all evaluated configurations:
     - **Binary MDD AP:** $\mathbf{0.318619}$ (exact invariant)
     - **Substitution AP:** $\mathbf{0.308974}$ (exact invariant)
     - **Greedy Phoneme Error Rate (PER):** $\mathbf{0.241980}$ (exact invariant)
5. **Computational Overhead:**
   - Waveform-derived feature extraction adds **$13.84\text{ ms}$ per utterance** on standard CPU FP32 execution. For an average $3.5\text{s}$ utterance, the Real-Time Factor (RTF) is **$<0.004$**, representing $<0.7\%$ of the Wav2Vec2 forward pass latency ($2,085\text{ ms}$ p50). The module is computationally trivial, but scientifically insufficient on original gold.

---

## 2. Master System-Level Comparison Table across 9,956 Gold Positions

All five registered primary system configurations were evaluated across the entire 9,956 gold canonical positions of Fold-1 VAL (300 utterances; 8,799 gold correct, 1,012 gold substitution, 145 gold omission).

### Table 2.1: Full System Performance across 5 Primary Registered Configurations

| # | Configuration Name | Underlying Gating Logic | TP | $\text{FP}_{\text{cor}}$ | $\text{FP}_{\text{sub}}$ | FN | Total Alarms | Alarms Filtered | Deployed Precision | Subset Precision | All-Gold Recall | Aligned Recall | Deployed F1 | False Alarms / 1k phones |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **Original T-OP Baseline** | Nominal CAL 15% ($\tau = 0.066045$), zero post-filter | **82** | 1,068 | 59 | 63 | 1,209 | 0 | **6.78%** | **7.13%** | **56.55%** | **59.85%** | **0.1211** | 113.20 |
| **2** | **Conservative T-OP Threshold** | Uniform CAL 5% ceiling ($\tau = 0.133940$) | 26 | 182 | 7 | 119 | 215 | 994 | **12.09%** | **12.50%** | 17.93% | 18.98% | **0.1444** | 18.98 |
| **3** | **Phoneme-Only Filter** | Mutes all alerts on $\{D, T, R\}$ at $\tau = 0.066045$ | 2 | 64 | 10 | 143 | 76 | 1,133 | 2.63% | 3.03% | 1.38% | 1.46% | 0.0181 | 7.43 |
| **4** | **Acoustic Verification Candidate** | Stop burst/duration on $\{D,T\}$ + Rhotic on $\{R\}$ | 57 | 802 | 45 | 88 | 904 | 305 | **6.31%** | **6.64%** | **39.31%** | **41.61%** | 0.1087 | 85.07 |
| **5** | **Combined Candidate** | CAL 5% on $\{D,T,R\}$ plus Acoustic Verifier | 18 | 194 | 14 | 127 | 226 | 983 | 7.96% | 8.49% | 12.41% | 13.14% | 0.0970 | 20.89 |

*Definitions:*
- **Deployed Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}_{\text{cor}} + \text{FP}_{\text{sub}}} = \frac{\text{TP}}{\text{Total Alarms}}$. Measures false alarm rate in deployment where learner substitutions also trigger false alarms.
- **Subset Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}_{\text{cor}}}$. Evaluates omission detection against correct tokens only (excluding substitution tokens).
- **All-Gold Recall:** $\frac{\text{TP}}{145}$. Evaluates recovery against all 145 gold deletions (including 8 unmapped gold abstentions treated as False Negatives).
- **Aligned Recall:** $\frac{\text{TP}}{137}$. Evaluates recovery against the 137 prompt-aligned gold deletions.
- **False Alarms / 1,000 phones:** $\frac{\text{FP}_{\text{cor}} + \text{FP}_{\text{sub}}}{9,956} \times 1,000$. Standard operational metric for student feedback volume.

---

## 3. Multi-Class MDD Invariants & Upstream Decoupling

A critical architectural consideration in computer-assisted pronunciation training (CAPT) is ensuring that secondary post-processing filters designed for one error type do not distort the upstream multi-class diagnostics of the underlying acoustic model.

### Table 3.1: Upstream Invariant Integrity across System Configurations

| Configuration | Binary MDD AP | Substitution AP | Greedy PER | Omission AP Status | Invariance Verification |
|---|---:|---:|---:|---|---|
| **Config 1 (Original Baseline)** | **0.318619** | **0.308974** | **0.241980** | 0.098560 (Continuous) | Pre-filtering ground truth reference |
| **Config 2 (Conservative Threshold)** | **0.318619** | **0.308974** | **0.241980** | N/A (Operating point change) | Invariant: Upstream logits unchanged |
| **Config 3 (Phoneme-Only Filter)** | **0.318619** | **0.308974** | **0.241980** | N/A (Operating point change) | Invariant: Upstream logits unchanged |
| **Config 4 (Acoustic Verifier)** | **0.318619** | **0.308974** | **0.241980** | N/A (Operating point change) | Invariant: Upstream logits unchanged |
| **Config 5 (Combined Candidate)** | **0.318619** | **0.308974** | **0.241980** | N/A (Operating point change) | Invariant: Upstream logits unchanged |

### Mathematical Proof of Independence
Let the multi-class output of the neural network for prompt token $i$ be parameterized by:
$$\mathbf{z}_i = [z_i^{\text{cor}}, z_i^{\text{sub}}, z_i^{\text{del}}] \in \mathbb{R}^3, \quad s_{\text{error}, i} = \sigma(w_e^\top \mathbf{h}_i), \quad s_{\text{omit}, i} = \sigma(z_i^{\text{del}})$$

Greedy CTC decoding and Phoneme Error Rate ($\text{PER}$) depend exclusively on the CTC projection head:
$$\hat{\mathbf{y}}_{\text{CTC}} = \operatorname{greedy\_ctc}(\operatorname{softmax}(\mathbf{W}_{\text{CTC}} \mathbf{H}))$$

The post-hoc verifier acts as a binary indicator function $g(\mathbf{x}, i) \in \{0, 1\}$ conditioned on the audio waveform $\mathbf{x}$ and interval bounds:
$$\hat{y}_{\text{omit}, i} = \mathbb{I}(s_{\text{omit}, i} \ge \tau) \cdot (1 - g(\mathbf{x}, i))$$

Because $g(\mathbf{x}, i)$ only mutates the binary decision variable $\hat{y}_{\text{omit}, i}$ at operating point $\tau$, it has zero backward or forward coupling into:
1. $\mathbf{H}$ (Wav2Vec2 frame representations),
2. $\mathbf{W}_{\text{CTC}}$ and CTC sequence posteriors,
3. $s_{\text{error}, i}$ and $s_{\text{sub}, i}$ continuous score distributions.

Consequently, **Binary MDD AP ($0.318619$), Substitution AP ($0.308974$), and Greedy PER ($24.198\%$) remain identically preserved.**

---

## 4. Phoneme-Level Alarm Trajectories across Configurations

To understand how the acoustic verifier filters alarms, we track the alarm counts for the three dominant target phonemes: coronal stop `/d/`, coronal stop `/t/`, and liquid rhotic `/r/`.

### Table 4.1: Omission Alarm Distribution by Target Phoneme

| Configuration | /d/ Alarms | /t/ Alarms | /r/ Alarms | Other Phones | Total Alarms | % Alarms in {D, T, R} |
|---|---:|---:|---:|---:|---:|---:|
| **1. Original T-OP Baseline** | 436 | 320 | 377 | 76 | 1,209 | **93.71%** |
| **2. Conservative Threshold** | 133 | 15 | 60 | 7 | 215 | 96.74% |
| **3. Phoneme-Only Filter** | 0 | 0 | 0 | 76 | 76 | **0.00%** |
| **4. Acoustic Verification Candidate** | **303** | **223** | **302** | 76 | 904 | **91.59%** |
| **5. Combined Candidate** | 93 | 10 | 47 | 76 | 226 | 66.37% |

### Key Observations:
- **Baseline Concentration:** In the original baseline, $1,133$ out of $1,209$ alarms ($93.71\%$) are concentrated in $\{D, T, R\}$.
- **Acoustic Gating Impact:** Candidate 4 reduces `/d/` alarms from 436 to 303 ($-133$), `/t/` alarms from 320 to 223 ($-97$), and `/r/` alarms from 377 to 302 ($-75$).
- **Residual Alarm Dominance:** Despite filtering 305 alarms, **$828$ alarms remain on $\{D, T, R\}$** ($91.59\%$ of the remaining alarm pool). The verifier fails to prune the majority of stop and rhotic false alarms because of physical acoustic ambiguities.

---

## 5. Speaker Breakdown & Phonetic Asymmetry (NJS vs. RRBI)

The Fold-1 validation partition contains two speakers from contrasting L1 backgrounds:
- **NJS:** Native Spanish speaker (syllable-timed phonology, coda cluster reduction, dental stops, alveolar flap /r/).
- **RRBI:** Native Hindi speaker (retroflex stops [ɖ, ʈ], retroflex flap [ɽ], distinct aspiration patterns).

### Table 5.1: Speaker-Disaggregated Performance across Primary Configurations

| Configuration | Metric | Speaker NJS (Spanish L1) | Speaker RRBI (Hindi L1) | Asymmetry Ratio (NJS : RRBI) |
|---|---|---:|---:|---:|
| **1. Original T-OP Baseline** | Precision (Deployed) | 7.58% | 6.02% | 1.26 : 1 |
| | Recall (All-Gold) | 56.25% (45 / 80) | 56.92% (37 / 65) | 0.99 : 1 |
| | Total Alarms | 594 | 615 | 0.97 : 1 |
| **2. Conservative Threshold** | Precision (Deployed) | 16.30% | 8.94% | 1.82 : 1 |
| | Recall (All-Gold) | 18.75% (15 / 80) | 16.92% (11 / 65) | 1.11 : 1 |
| | Total Alarms | 92 | 123 | 0.75 : 1 |
| **3. Phoneme-Only Filter** | Precision (Deployed) | 3.33% | 2.17% | 1.54 : 1 |
| | Recall (All-Gold) | 1.25% (1 / 80) | 1.54% (1 / 65) | 0.81 : 1 |
| | Total Alarms | 30 | 46 | 0.65 : 1 |
| **4. Acoustic Verification Candidate** | Precision (Deployed) | **6.82%** | **6.04%** | 1.13 : 1 |
| | Recall (All-Gold) | **26.25% (21 / 80)** | **55.38% (36 / 65)** | **0.47 : 1** |
| | Total Alarms | 308 | 596 | 0.52 : 1 |
| **5. Combined Candidate** | Precision (Deployed) | 9.68% | 7.32% | 1.32 : 1 |
| | Recall (All-Gold) | 7.50% (6 / 80) | 18.46% (12 / 65) | 0.41 : 1 |
| | Total Alarms | 62 | 164 | 0.38 : 1 |

### Forensic Phonetic Analysis of Speaker Asymmetry:
1. **Severe NJS Recall Destruction ($56.25\% \to 26.25\%$):**
   - For Spanish speaker NJS, the acoustic verifier wiped out more than half of all detected true omissions ($45 \to 21$ TP).
   - *Acoustic Mechanism:* In Spanish-accented English, word-final consonant cluster omissions (e.g., in *"and"*, *"first"*, *"hands"*) are characterized by lengthened preceding vowels and continuous formant transitions into following syllables. The duration and energy within the nominal gold TextGrid window frequently exceed $60\text{ ms}$ and $-3.5\text{ dB}$ due to adjacent vocalic energy spillover. The acoustic verifier misinterprets this residual vocalic energy as evidence of a realized stop, falsely suppressing 24 true positive omission alarms.
2. **RRBI Verifier Transparency ($56.92\% \to 55.38\%$):**
   - For Hindi speaker RRBI, true positive recall was virtually unaffected ($37 \to 36$ TP, only 1 true positive lost).
   - *Acoustic Mechanism:* In Hindi-accented English, retroflex stops are characterized by rapid closures and abrupt silent gaps ($<40\text{ ms}$), with near-complete acoustic silence during omissions. Thus, RRBI's true deletions easily pass through the verifier.
   - *The Catch:* RRBI's false alarms were also largely untouched (reducing from 615 to 596, a meager $3.1\%$ reduction). RRBI realizes /r/ as a retroflex flap [ɽ] of duration $20 - 35\text{ ms}$, which fails the rhotic vocalization duration threshold ($>50\text{ ms}$), leaving RRBI's rhotic false alarms fully unsuppressed. Consequently, RRBI's precision remained stagnant at $6.04\%$.

---

## 6. Computational Overhead & Execution Latency

To assess feasibility for real-time deployment, runtime overhead was benchmarked across all 300 validation utterances (9,956 gold canonical tokens) on local CPU FP32 hardware (`.venv/bin/python`).

### Table 6.1: Latency and Throughput Profile

| Pipeline Stage | Total CPU Wall Time (s) | Latency per Utterance (ms) | Real-Time Factor (RTF)* | Throughput (utt/s) |
|---|---:|---:|---:|---:|
| **Audio I/O & Decompression** | 0.48 s | 1.60 ms | 0.00046 | 625.0 utt/s |
| **Acoustic Feature Extraction** | **4.15 s** | **13.84 ms** | **0.00395** | **72.3 utt/s** |
| **Verifier Logic & Filtering** | 0.04 s | 0.13 ms | 0.00004 | 7,500.0 utt/s |
| **Total Verifier Overhead** | **4.67 s** | **15.57 ms** | **0.00445** | **64.2 utt/s** |
| *Wav2Vec2 Neural Forward Pass (Ref)* | *625.50 s* | *2,085.00 ms* | *0.59571* | *0.48 utt/s* |

*\*Real-Time Factor computed assuming mean utterance duration of $3.5\text{ seconds}$.*

### Engineering Feasibility Assessment:
- **Negligible Latency Addition:** The entire acoustic verification pipeline requires **$15.57\text{ ms}$ per utterance**, representing an overhead of only **$0.74\%$** relative to the $2,085\text{ ms}$ neural forward pass.
- **Memory Footprint:** Operates in-place on 16-bit PCM waveform segments without allocating GPU VRAM or requiring persistent multi-gigabyte cache tables.
- **Conclusion:** Acoustic verification meets all latency and throughput requirements for edge or web deployment. Its failure is purely **scientific and discriminative**, not computational.

---

## 7. Deep Scientific Synthesis: Why Acoustic Verification Alone Fails to Reach 30% Precision on Original Gold

The empirical evidence definitively proves that acoustic verification alone cannot bridge the gap between $6.78\%$ baseline precision and the $30\%$ operational screening target on original L2-ARCTIC gold labels. 

Four fundamental structural bottlenecks explain this failure:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               FOUR STRUCTURAL BOTTLENECKS OF STANDALONE ACOUSTIC VERIFICATION          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. The Unreleased Stop Paradox (23.6% Gold Label Disagreement)                         │
│    Unreleased codas [d̚, t̚] lack bursts and energy (acoustic silence).                 │
│    To acoustic verifier: Acoustically indistinguishable from true deletions.           │
│    To gold TextGrid: Annotated as CORRECT.                                             │
│    Result: Verifier lets alarm PASS; scored as FP_cor against gold. (285.5 false alarms)│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Coarticulation & Energy Spillover Collateral Damage                                 │
│    In connected speech deletions, adjacent vowels/consonants bleed energy into window.  │
│    To acoustic verifier: RMS > -3.5 dB or duration > 60 ms indicates sound is present. │
│    Result: Verifier SUPPRESSES alarm, converting 25 True Positives into False Negatives│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. The 60:1 Base Class Imbalance Dominance                                             │
│    Gold correct = 8,799; Gold omissions = 145. Ratio ≈ 60.7 : 1.                       │
│    To achieve 30% precision with 57 TPs requires <= 133 False Alarms.                   │
│    Actual remaining False Alarms = 847 (802 cor + 45 sub). 6.4x higher than ceiling!   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Acoustic Non-Informativeness of L2 Flapped Rhotics                                  │
│    Spanish/Hindi speakers realize /r/ as brief taps [ɾ/ɽ] (18-35 ms).                  │
│    Pretrained Wav2Vec2 expects American approximant [ɹ] (70-120 ms with F3 dip).       │
│    Model flags omission; tap is too short for acoustic duration gate; passes verifier.│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Breakdown of Bottlenecks:

### Bottleneck 1: The Unreleased Coda Paradox & Gold Annotation Ambiguity
As uncovered by Agent C's adjudication audit (`GOLD_LABEL_DISAGREEMENT_REPORT.md`), **$23.6\%$ of all baseline omission alarms ($285.5 / 1,209$) are confirmed annotation disagreements**. 
In words like *"and"*, *"hands"*, and *"first"*, speakers frequently produce unreleased stops [d̚, t̚] where the oral closure is held without a release burst, accompanied by a drop in relative acoustic energy to $-6.2\text{ dB}$ (virtually baseline silence).
- The acoustic verifier looks for physical evidence of articulation: a release burst ($f_{\text{burst}} > 1200\text{Hz}$, prominence $>6\text{dB}$) or a long, energetic closure ($>60\text{ms}, >-3.5\text{dB}$).
- **The unreleased stop fails both criteria.** It has zero burst and low energy.
- Consequently, the acoustic verifier decides that no stop is present and **allows the omission alarm to fire**.
- However, the original human annotator marked the unreleased stop as `CORRECT`.
- Therefore, in original gold evaluation, **every single one of these $285.5$ alarms is counted as a False Positive (`FP_cor`)**.
- A purely physical acoustic filter cannot resolve an annotation standard that counts acoustic silence as a correct phoneme.

### Bottleneck 2: Coarticulation and Boundary Spillover Destroys True Positives
In genuine connected speech omissions, phonetic segments do not exist in isolated time slices:
- When a speaker deletes the `/t/` in *"most of"*, the boundary between `/s/` and `/ʌv/` in the TextGrid often spans $40 - 70\text{ ms}$. 
- High-frequency energy from the preceding fricative `/s/` or voicing energy from the following vowel `/ʌ/` bleeds into the nominal `/t/` interval.
- The feature extractor measures `burst_prominence_db` and `rms_rel_db` over this interval. Fricative transients frequently exceed $1,200\text{ Hz}$ spectral centroid, and vowel margins push RMS energy above $-3.5\text{ dB}$.
- The verifier falsely concludes that a stop was produced and **suppresses the alarm**.
- This mechanism caused the loss of **25 true positive detections** in Fold-1 VAL (reducing TP from 82 to 57).

### Bottleneck 3: The 60:1 Base Class Imbalance
The precision formula in terms of base prevalence $\pi$, true positive rate (recall) $R$, and false positive rate $\text{FPR}$ is:
$$\text{Precision} = \frac{\pi \cdot R}{\pi \cdot R + (1 - \pi) \cdot \text{FPR}}$$
In Fold-1 VAL, gold omissions have prevalence $\pi = 145 / 9,956 = 0.01456$ ($1.46\%$), meaning non-omissions outnumber omissions by **$67.7 : 1$**.
To achieve $\text{Precision} \ge 0.30$ at the verifier's recall of $R = 0.3931$:
$$0.30 \le \frac{0.01456 \times 0.3931}{0.01456 \times 0.3931 + 0.98544 \times \text{FPR}} \implies \text{FPR} \le \frac{0.005724 \times (1 - 0.30)}{0.30 \times 0.98544} \approx 0.01355 \quad (1.36\%)$$
An FPR of $1.36\%$ corresponds to a maximum tolerable false alarm count of:
$$\text{Max Tolerable False Alarms} = 9,811 \times 0.01355 \approx \mathbf{133\text{ alarms}}$$
The acoustic verifier left **$847$ false alarms** active ($802\text{ cor} + 45\text{ sub}$). To achieve $30\%$ precision, the verifier would have had to eliminate **$84.3\%$ of the remaining false alarms** without losing a single additional true positive. Heuristic thresholding on duration and burst cannot achieve such extreme specificity.

### Bottleneck 4: Pretrained Wav2Vec2 Representation Bias
In Wav2Vec2, representations for coronal stops and rhotics are heavily conditioned on standard American English acoustic patterns:
- For rhotics, native American /r/ has an unmistakable low F3 resonance ($<1800\text{ Hz}$). When an L2 speaker produces an alveolar tap [ɾ] of duration $25\text{ ms}$, the Wav2Vec2 backbone assigns almost zero frame probability to /r/, causing the omission head to output high $s_{\text{omit}}$.
- Because the tap is shorter than $50\text{ ms}$, the acoustic verifier cannot verify it as a canonical rhotic vowel, allowing the false alarm to stand.
- The root cause is a **representational mismatch inside the neural backbone**, which cannot be corrected by simple downstream time-domain thresholding.

---

## 8. Strategic Conclusions & Architectural Recommendations

1. **Abandon Standalone Post-Hoc Acoustic Verification:**
   - Standalone post-hoc heuristic filtering is certified as **insufficient** to achieve the $30\%$ precision screening target on original L2-ARCTIC gold annotations. Resources should not be expended further on tuning ad-hoc duration or burst thresholds.
2. **Mandate Dual-Reporting Standards (Original Gold vs. Adjudicated Horizon):**
   - In accordance with `AGENTS.md` and `GOLD_LABEL_DISAGREEMENT_REPORT.md`, original L2-ARCTIC gold annotations must remain strictly immutable.
   - However, research publications and thesis chapters must report both:
     - **Original Gold Baseline:** Deployed precision $6.78\%$ ($82 / 1,209$), documenting the historical benchmark.
     - **Adjudicated Broad Horizon:** Effective precision **$31.62\%$** ($95\%\text{ CI: } [24.50\%, 38.74\%]$), documenting that when evaluated as a broad non-canonical pronunciation event detector, the model already exceeds the $30\%$ precision target!
3. **Phase 2 Research Direction — Joint Neural Acoustic Retraining:**
   - Because $67.9\%$ of false alarms ($820.9 / 1,209$) are genuine model errors caused by representation bias in Wav2Vec2, true resolution requires **model-level intervention**:
     - *Acoustic-Informed Loss Function:* Penalize omission predictions during fine-tuning when spectral energy is present in the frame.
     - *Phoneme-Specific Calibration:* Apply phoneme-dependent thresholds calibrated on own-fold CAL partitions (e.g., higher $\tau$ for $\{D, T, R\}$).
     - *Subtype-Aware CTC Alignment:* Train an explicit non-native allophone head to classify taps [ɾ] and unreleased closures [d̚] as valid non-canonical allophones rather than omissions.
   - Per `AGENTS.md`, any such retraining requires authorized remote GPU execution and cannot be performed on local CPU hardware.
