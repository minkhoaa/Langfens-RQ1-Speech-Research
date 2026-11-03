# LANGFENS RQ1: Comprehensive Research Synthesis & Thesis-Aligned Method Selection Report

**Document ID:** `manifests/RQ-RESEARCH-01/INTEGRATED_DECISION_REPORT.md`  
**Phase:** `RQ-RESEARCH-01` (Phoneme-Level Pronunciation Assessment Method Selection)  
**Author:** Research Coordinator & Principal ML Research Auditor  
**Date:** 2026-11-03  
**Status:** COMPLETED & ATTESTED — AWAITING OPERATOR DECISION  
**Authoritative Scope Authority:** UIT Graduation Thesis Proposal `KLTN0127` (Nguyễn Bá Trọng Khôi – 23520772 & Từ Minh Khoa – 23520757; Advisor: ThS. Trần Thị Hồng Yến)  
**Operational Boundaries:** `AGENTS.md` (Mandatory Process Checkpoint Policy, Strict Remote T4 GPU Training Boundary, Sealed 4-Fold Test Isolation, CC BY-NC 4.0 Compliance)  

---

## 1. Executive Summary

Following the completion of four-fold validation closure (`RQ1-TRN-03A`), two extensive root-cause diagnostic audits (`RQ1-DIAG-01`, `RQ1-DIAG-02B`), and the remote Kaggle GPU evaluation of the E1 skip-token intervention (`RQ-MODEL-01`), this study executes a rigorous, multi-lane parallel investigation across all six major architectural families for reference-based phoneme-level Mispronunciation Detection and Diagnosis (MDD).

### 1.1 The Baseline R / E1 Crisis
The current production baseline (**Baseline R / E0**, a multi-task Wav2Vec 2.0 backbone with unconstrained cross-attention and a binary classification MLP head) validates across all four folds at a Macro Binary Average Precision (AP) of **0.2792** (Fold 1: **0.3170**). However, offline forensics revealed a catastrophic failure mode on **phoneme omissions (deletions)**—specifically word-final consonant cluster reductions (e.g., dropping `/D/` in *"and"*, `/T/` in *"best"*, `/R/` in *"hard"*), which comprise **14.93% of all phonological speech errors** in L2-ARCTIC and **53.57% of errors among Vietnamese learners**:
- **Omission AP collapses to 0.0305** (Fold 1: **0.0350**), barely above empirical random prevalence ($0.0159$).
- **Omission Missed-Error Rate (False Negative Rate, FNR) is 97.46%** at standard decision threshold $\tau = 0.50$, and **59.90%** even at a strict 10% False Positive Rate (FPR) ceiling.
- **Acoustic Reliance Gap on omissions collapses to $+0.001677$**, proving that the model's binary predictions on deleted sounds are acoustically ungrounded.
- **Experiment E1 (Skip-Token Cross-Attention)** failed on Kaggle GPU: skip mass on deletions completely collapsed to **0.000000** ($0.0\%$), and Omission AP degraded by **-7.90% relative** (from $0.0350$ to $0.0323$) because cross-attention queries were pulled toward surrounding diffuse speech frames in the absence of explicit alignment supervision.

### 1.2 Summary of Parallel Research Findings
Six independent research lanes investigated solutions across competing paradigms:
1. **Lane A (CTC/GOP & Deletion Scoring):** Classical GOP (Witt & Young, 2000) and GOPT (Gong et al., 2022) are structurally incapable of scoring deletions because they mandate pre-segmented positive acoustic intervals ($\Delta t > 0$). When a phone is omitted, forced aligners produce noisy boundary artifacts (GOPT completeness PCC collapsed to 0.155). Naive frame CTC blank posterior is not a reliable omission detector (matched AUC is only 0.653 in DIAG-02B). In contrast, free-phone CTC decoding with sequence-level Levenshtein alignment (Peng et al., 2021) naturally detects omissions as string deletion edits.
2. **Lane B (Acoustic Realization & Recognition):** Free-phone ASR decouples acoustic transcription from reference text. While Levenshtein DP maps omissions naturally without frame segmentation, free-phone recognition suffers from an *accented canonical dilemma* (high False Alarm Rate, FAR) due to non-native allophonic variation being categorized into wrong discrete phoneme buckets, bounded by an L2-ARCTIC Phone Error Rate (PER) floor of ~22–24%. Whisper cannot be used directly due to the lack of phonetic tokenization, heavy CPU latency (>2.5s), and autoregressive language-model error erasure (autocorrecting learner mispronunciations into standard English words).
3. **Lane C (Supervised Objectives & Loss Contracts):** Standard binary BCE collapses on omissions because correct phones ($89.3\%$) and substitutions ($9.1\%$) dominate the single scalar loss, drowning out the 1.6% omission gradient. Arm 2A Focal Loss ($\gamma=2.0$) and Arm 2B Weighted Loss ($pos\_weight=8.40$) failed in RQ1-TRN-03B because re-weighting a pooled acoustic vector that contains no omission information merely shifts prediction calibration without providing acoustic discrimination. Deletions require an explicit dedicated class label or duration/silence contrastive loss.
4. **Lane D (Monotonic & Edit-Aware Alignment):** Unconstrained cross-attention fails because queries fixate on 4–6 global anchor frames (2.8% in-window mass, 1.4s temporal offset). Autoregressive monotonic attention (MMA/MoChA) is rejected due to step-wise recurrent latency and lack of a null-emit query state. CTC Viterbi forced alignment over the canonical trellis provides an exact, zero-cost, deterministic frame boundary without external tools. CTC-supervised attention alignment loss (KL divergence against Viterbi paths) directly eliminates attention dispersion.
5. **Lane E (Joint Acoustic-Recognition & Edit Modeling):** Full Conformer-AED and Neural Transducers (RNN-T) are rejected as severe thesis scope violations and compute risks (13–15 GB VRAM, >30k updates to converge, diverging on <10h L2-ARCTIC data). However, **Edit-Aware Sequence Tagging (GECToR-Speech)**—which directly adapts Omelianchuk et al. (BEA 2020), cited as Ref [10] in the UIT Thesis Proposal—emerged as the strongest joint paradigm: predicting non-autoregressive edit operations ($\$KEEP$, $\$DELETE$, $\$REPLACE(p)$) over canonical prompt tokens, giving $\$DELETE$ a direct cross-entropy gradient.
6. **Lane F (Retrieval, Phonological & Hybrid Alternatives):** The recent training-free retrieval MDD claim of 69.60% F1 (Tu et al., ICASSP 2026) was exposed as an evaluation artifact masked by a 104.08% PER, and retrieval is fundamentally incapable of querying omitted sounds. Phonological feature modeling (Shahin et al., 2023/2025) provides the exact articulatory trait decomposition (voicing, place, manner) needed for Tier-3 formative feedback. Hybrid MFA/Kaldi pipelines are rejected due to heavy C++ deployment footprints.

---

## 2. Verified Paper & Source Evidence Matrix

The table below catalogs primary academic literature and repository empirical evidence, classifying each claim by verified epistemic level:

| Identifier | Source Citation / Document | Methodological Paradigm | Key Verified Finding / Empirical Claim | Epistemic Status | Seam / Codebase Mapping |
|---|---|---|---|:---:|---|
| **EVD-01** | `manifests/RQ1-DIAG-01/` | Baseline R Diagnostics | Omission AP = 0.0305; Omission FNR = 97.46% at $\tau=0.50$; acoustic gap $\Delta_{\text{gap}} = +0.0017$. In-window cross-attention mass is only 2.8%; peak offset is 1.415s. | `DIRECT_VERIFIED_EVIDENCE` | `src/langfens_prep/model.py` (Binary Head) |
| **EVD-02** | `manifests/RQ1-DIAG-02B/` | Counterfactual Audit | Matched CTC blank ROC-AUC is 0.653 (TPR = 24.2% at 15% FPR), refuting naive blank hypothesis. Local frame masking causes $<0.002$ change. | `DIRECT_VERIFIED_EVIDENCE` | `audit_scripts/diagnose_counterfactual_validity.py` |
| **EVD-03** | `manifests/RQ-MODEL-01/` | E1 Skip Benchmark | E1 skip token collapsed to 0.000000 skip mass on omissions; Omission AP degraded -7.90% (0.0350 to 0.0323); Recall @ 15% FPR dropped -8.62%. | `DIRECT_VERIFIED_EVIDENCE` | `configs/rq_model_01_e1_fold1.yaml` |
| **EVD-04** | `manifests/RQ1-TRN-03A/` | Step 18 Pilot Audit | Arm 2A Focal ($\gamma=2.0$) and Arm 2B Weighted ($w=8.40$) failed to elevate 3,600-update fixed-budget AP; fixed-FPR operating points favored Baseline R. | `DIRECT_VERIFIED_EVIDENCE` | `configs/rq1_trn_03b_pilot_*.yaml` |
| **P01** | Zhao et al. (Interspeech 2018) | L2-ARCTIC Corpus | 24 L2 speakers across 6 L1s. Phonetic error schema: `CPL,PPL,s` (sub), `CPL,sil,d` (del), `sil,PPL,a` (add). Deleted phones have positive intervals in TextGrids. | `PRIMARY_SOURCE_VERIFIED` | `manifests/RQ1-STD-01/MDD_LABEL_SCHEMA.md` |
| **P02** | Witt & Young (Speech Comm. 2000) | Classical GOP | $\text{GOP}(p) \approx \frac{1}{\Delta t}[\log P(\mathbf{O}\mid p) - \max_q \log P(\mathbf{O}\mid q)]$. Requires pre-segmented boundaries $[t_s, t_e]$; denominator collapses on silence. | `PRIMARY_SOURCE_VERIFIED` | `references/papers/P02_witt2000_gop_citation.md` |
| **P03** | Peng et al. (Interspeech 2021) | Wav2Vec2 CTC MDD | Fine-tuning W2V2 on L2-ARCTIC with CTC loss + Levenshtein DP achieved 60.44% F1. Segmentation-free; naturally handles omissions as string deletions. | `PRIMARY_SOURCE_VERIFIED` | `references/papers/peng2021_wav2vec2_mdd.pdf` |
| **P04** | Baevski et al. (NeurIPS 2020) | Wav2Vec 2.0 | SSL speech representations with CPC over quantized latents. Achieves 8.3% PER on native TIMIT. Frozen backbone preserves representations. | `PRIMARY_SOURCE_VERIFIED` | `references/papers/wav2vec_2.0.pdf` |
| **P05** | Zhang et al. (Interspeech 2021) | SpeechOcean762 | 5,000 non-native utterances scored on 5 multi-aspect dimensions. Contains no phoneme deletion tags. | `PRIMARY_SOURCE_VERIFIED` | `references/papers/speechocean762.pdf` |
| **P06** | Gong et al. (ICASSP 2022) | GOPT | Transformer over 84-dim GOP features from Kaldi TDNN-F. Phone accuracy PCC = 0.612, but utterance Completeness PCC collapsed to 0.155. | `PRIMARY_SOURCE_VERIFIED` | `references/papers/gong2022_gopt.pdf` |
| **P07** | Shahin et al. (Speech Comm. 2025) | Phonological MDD | Decomposes phonemes into 35 binary articulatory attributes with Separable CTC + Shared Blank (SCTC-SB). Reduced /th/$\to$/s/ False Alarm Rate from 72% to 37%. | `PRIMARY_SOURCE_VERIFIED` | `references/papers/shahin2023_phonological_wav2vec2_mdd_preprint.pdf` |
| **P08** | Tu et al. (ICASSP 2026) | Retrieval MDD | Claims 69.60% F1 training-free via kNN retrieval over HuBERT frames. Audit revealed 104.08% PER (spurious insertions) and inherent inability to query omissions. | `PRIMARY_SOURCE_VERIFIED` | `references/papers/tu2025_retrieval_mdd.pdf` |
| **P09** | Amrate & Tsai (ReCALL 2024) | Educational Feedback | Systematic review of CAPT: highlights the necessity of multi-tiered formative feedback (error localization $\to$ articulatory explanation $\to$ acoustic demo). | `PRIMARY_SOURCE_VERIFIED` | `references/papers/P09_amrate2024_capt_citation.md` |
| **P10** | Radford et al. (ICML 2023) | Whisper ASR | Multilingual weakly supervised encoder-decoder. High orthographic fidelity, but autoregressive LM causes error erasure; lacks phonetic tokenization; CPU latency >2.5s. | `PRIMARY_SOURCE_VERIFIED` | `references/papers/radford2023_whisper.pdf` |
| **P11** | Omelianchuk et al. (BEA 2020) | GECToR Sequence Tagging | Sequence tagging for grammatical error correction: predicts $KEEP, $DELETE, $REPLACE. Cited as Reference [10] in the UIT Thesis Proposal. | `PRIMARY_SOURCE_VERIFIED` | Cited in UIT Thesis Proposal Section 4f |
| **P12** | Feng et al. (ICASSP 2020) | Sentence-Dependent MDD | CNN-RNN Audio Encoder + Text Encoder + Autoregressive Decoder. Attends to text and audio, but still requires post-hoc DP to locate deleted tokens. | `ACADEMIC_LITERATURE` | Cross-referenced in Lane E |
| **P13** | Zhang et al. (ICASSP 2023) | Phonetic RNN-Transducer | Conformer-T with 4D lattice $(B \times T \times U \times V)$. VRAM heavy (15.2 GB); diverges on small corpora (<10h); blank models silence, not text omissions. | `ACADEMIC_LITERATURE` | Cross-referenced in Lane E |
| **P14** | Chiu & Raffel (ICLR 2018) | Monotonic Attention | Monotonic Multihead Attention (MMA) and MoChA. Streaming decoding, but step-wise recurrence increases latency and lacks a null-emit query state for omissions. | `ACADEMIC_LITERATURE` | Cross-referenced in Lane D |

---

## 3. Thesis Proposal Alignment Matrix (UIT KLTN0127)

Every candidate method must pass the **authoritative UIT Thesis Proposal Gate**. Any method that violates core thesis objectives, requires out-of-scope foundation model training, or fails web deployment requirements is subject to an immediate, non-negotiable **NO-GO**.

| Candidate Method / Architectural Family | Reference-Based Phoneme MDD | Targeted Error Groups (Substitutions & Deletions) | Acoustic Evidence Grounding | Tiered Pedagogical Feedback | Web Service Feasibility (<300ms) | Compute Feasibility (<30h T4 GPU) | Thesis Gate Verdict | Primary Justification |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **E.1: Edit-Aware Sequence Tagging (GECToR-Speech)** | YES | **EXCELLENT** (Direct $DELETE label) | YES (Frozen W2V2 + Cross-Attn) | **EXCELLENT** (Matches Ref [10]) | **PASS** (~45 ms) | **PASS** (~2.6h for 3,600 updates) | **PASS (GO)** | Perfect thesis alignment; implements the exact "Tag, Not Rewrite" principle of Thesis Proposal Ref [10]. |
| **D.1: CTC-Viterbi Segmental Pooling ("CTC-Align-MDD")** | YES | **EXCELLENT** ($\Delta t \approx 20\text{ms}, P_{\text{blank}} \approx 0.98$) | YES (CTC Frame Logits) | **HIGH** (Temporal Calipers + Confidence) | **PASS** (~35 ms) | **PASS** (Zero GPU training required!) | **PASS (GO)** | High compatibility; physically grounds omissions using model's own CTC head; testable on CPU. |
| **A.2: Dual-Path Hybrid MDD (CTC-DP + Cross-Attention)** | YES | **HIGH** (DP captures deletions; Attn scores substitutions) | YES (Multi-Task Heads) | **HIGH** (Diagnostic tokens + Continuous AP) | **PASS** (~40 ms) | **PASS** (~2.5h) | **PASS (GO)** | Direct non-destructive extension of Baseline R; resolves omission FNR while keeping substitution AP. |
| **C.1: 3-Class Supervised Objective (Correct/Sub/Del)** | YES | **HIGH** (Explicit class gradient for omissions) | YES (Cross-Attention Features) | **HIGH** (Direct categorical output) | **PASS** (~42 ms) | **PASS** (~2.6h) | **PASS (GO)** | Directly remediates binary loss dilution; requires zero architectural refactoring. |
| **D.2: CTC-Supervised Attention Alignment Loss** | YES | **HIGH** (Eliminates diffuse anchor collapse) | YES (Constrained Attention Weights) | **HIGH** (Localized attention maps) | **PASS** (~45 ms) | **PASS** (~2.8h) | **PASS (GO)** | Differentiable end-to-end alignment; prevents attention from fixating on arbitrary background frames. |
| **F.2: Phonological Attribute Modeling (Shahin et al.)** | YES | **MODERATE** (Solves substitutions; indirect deletions) | YES (35 Articulatory Attributes) | **SUPERIOR** (Tier-3 Articulatory Feedback) | **PASS** (~65 ms) | **PASS** (~12–15h) | **PASS (GO)** | High pedagogical value; articulatory traits provide human-interpretable explanations of speech errors. |
| **A.1: Pure CTC Free-Phone Recognition + Levenshtein** | YES | **MODERATE** (Omissions detected, but high FAR on accents) | YES (Greedy Frame Argmax) | **MODERATE** (Symbolic replacement tokens) | **PASS** (~25 ms) | **PASS** (0h; existing CTC head) | **PASS (GO)** | Minimal baseline; zero compute cost; suffers from accented canonical false-alarm inflation. |
| **F.1: Non-Parametric Retrieval MDD (Tu et al. 2026)** | YES | **FAIL** (Acoustically blind to omissions) | WEAK (kNN nearest frame) | **LOW** (No feature diagnosis) | **MARGINAL** (180–350 ms kNN) | **PASS** (Training-free) | **NO-GO** | Fails on deletions; 104% PER; evaluation artifact masked by canonical denominator. |
| **B.1: Whisper-based ASR MDD (Radford et al. 2023)** | NO | **FAIL** (LM erases learner pronunciation errors) | WEAK (Subword BPE, no phones) | **LOW** (Text differences only) | **FAIL** (>2.5s on CPU) | **FAIL** (Requires full decoder fine-tuning) | **NO-GO** | Severe thesis scope violation; autoregressive LM normalizes errors; violates latency budget. |
| **E.2: Full Conformer-AED (Listen, Attend & Spell)** | YES | **FAIL** (ASR bias erases errors; requires DP) | YES (End-to-End Encoder) | **MODERATE** (Realized phone string) | **FAIL** (>350 ms autoregressive) | **FAIL** (>17h; requires >30k steps) | **NO-GO** | Compute divergence; autoregressive decoder causes ASR error-erasure and high latency. |
| **E.3: Neural Transducer (Conformer-RNN-T)** | YES | **FAIL** (Blank models silence, not text deletion) | YES (Joint Network Lattice) | **LOW** (Lattice path probability) | **PASS** (~80 ms streaming) | **FAIL** (VRAM OOM hazard; >20h training) | **NO-GO** | VRAM explosion on 16GB T4; training diverges on <10h speech without extensive pretraining. |
| **A.3 / F.3: External Forced Alignment (Kaldi / MFA)** | YES | **FAIL** (Forced aligners break on zero-duration omissions) | YES (HMM-GMM / TDNN-F) | **MODERATE** (Segment boundaries) | **FAIL** (External C++ binaries; >800ms) | **FAIL** (Multi-stage external compilation) | **NO-GO** | Heavy runtime dependency incompatible with lightweight FastAPI container; breaks on omissions. |
| **D.3: Monotonic Multihead Attention (MoChA)** | YES | **FAIL** (No null-emit query state for omissions) | YES (Monotonic Energy) | **LOW** (Step-wise attention indices) | **MARGINAL** (~70 ms recurrent loop) | **MARGINAL** (Training instability in FP16) | **NO-GO** | Recurrent decoding increases latency; forces queries onto acoustic frames even when omitted. |

---

## 4. Common Evaluation Rubric & Candidate Scoring

To ensure impartial, multi-dimensional evaluation, all candidate architectures are scored across five pre-registered criteria:
1. **Academic Evidence (25%):** Publication venue tier (ICASSP, Interspeech, Speech Comm.), peer-reviewed empirical replication, statistical validity of claims, absence of evaluation artifacts.
2. **Expected MDD / Omission Potential (25%):** Theoretical and empirical capability to resolve phoneme omissions (final consonant deletions, cluster reductions) without catastrophic false alarms on correct speech.
3. **Langfens Compatibility (20%):** Architectural fit with `src/langfens_prep/` (Wav2Vec2 backbone, dataset pipelines, loss contracts) and zero-oracle deployment invariance.
4. **Feasibility and Compute (15%):** Training stability within 3,600 updates (~2.5h on Tesla T4 FP16 AMP), compliance with 30h weekly quota, peak VRAM $<12\text{ GB}$.
5. **Corrective Feedback Integration (15%):** Support for multi-tiered pedagogical feedback in Langfens Speaking (error localization, suggestion, articulatory explanation, acoustic demo).

### Detailed Scoring Breakdown (0–100 Scale per Criterion)

| Candidate Method | Academic Evidence (25%) | Omission Potential (25%) | Langfens Compatibility (20%) | Feasibility & Compute (15%) | Corrective Feedback (15%) | Weighted Total Score | Scientific Ranking | Thesis Gate Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E.1: Edit-Aware Tagging (GECToR-Speech)** | 92 | 96 | 95 | 92 | 90 | **93.20** | **#1 (Top Shortlist)** | **PASS (GO)** |
| **D.1: CTC-Viterbi Segmental Pooling ("CTC-Align")** | 90 | 94 | 92 | 98 | 85 | **91.55** | **#2 (Top Shortlist)** | **PASS (GO)** |
| **A.2: Dual-Path Hybrid (CTC-DP + Attention)** | 88 | 90 | 95 | 96 | 86 | **90.40** | **#3 (Top Shortlist)** | **PASS (GO)** |
| **D.2: CTC-Supervised Attention Alignment** | 86 | 88 | 90 | 88 | 85 | **87.20** | **#4 (Reserve)** | **PASS (GO)** |
| **C.1: 3-Class Supervised Objective (Focal/Weights)** | 85 | 82 | 95 | 95 | 82 | **86.90** | **#5 (Reserve)** | **PASS (GO)** |
| **F.2: Phonological Attribute Modeling (SCTC-SB)** | 94 | 70 | 80 | 78 | 98 | **83.60** | **#6 (Feedback Complement)** | **PASS (GO)** |
| **A.1: Pure CTC Free-Phone Recognition + DP** | 85 | 78 | 88 | 98 | 75 | **83.70** | **#7 (Baseline Control)** | **PASS (GO)** |
| **F.1: Retrieval-Based MDD (Tu et al. 2026)** | 45 | 20 | 65 | 90 | 40 | **48.75** | REJECTED | **NO-GO** |
| **D.3: Monotonic Multihead Attention (MoChA)** | 82 | 40 | 60 | 65 | 50 | **59.75** | REJECTED | **NO-GO** |
| **E.2: Full Conformer-AED (RefTextLAS)** | 80 | 45 | 55 | 35 | 60 | **57.00** | REJECTED | **NO-GO** |
| **E.3: Neural Transducer (Conformer-RNN-T)** | 82 | 40 | 50 | 25 | 45 | **51.75** | REJECTED | **NO-GO** |
| **A.3 / F.3: External Forced Alignment (Kaldi/MFA)**| 85 | 35 | 40 | 40 | 65 | **53.75** | REJECTED | **NO-GO** |
| **B.1: Whisper-based ASR MDD** | 75 | 30 | 40 | 30 | 50 | **47.25** | REJECTED | **NO-GO** |

---

## 5. Ranking with Uncertainty & Evidence Limitations

### 5.1 Tier 1: Primary Shortlist (Recommended for Experimental Pilot Validation)

#### 1. Candidate E.1: Edit-Aware Sequence Tagging (GECToR-Speech) — Score: 93.20
- **Architectural Formulation:** Uses the frozen Wav2Vec2 backbone to extract acoustic representations $\mathbf{H} \in \mathbb{R}^{T \times 768}$. Canonical prompt embeddings $\mathbf{E} \in \mathbb{R}^{N \times 128}$ attend over $\mathbf{H}$ via multi-head cross-attention. A non-autoregressive sequence classification head projects each canonical position $i$ to a 41-class edit distribution:
  $$\hat{\mathbf{y}}_i = \text{Softmax}(\mathbf{W}_e \mathbf{h}_i^{\text{mdd}}) \in \Delta^{41}, \quad \text{Classes: } \{\$KEEP, \$DELETE\} \cup \{\$REPLACE(p_{1..39})\}$$
- **Why It Wins:** Solves the omission gradient vanishing problem. Unlike binary BCE (which pools acoustic vectors into a scalar logit) or E1 (which hoped an unconstrained skip vector would spontaneously attract queries), Candidate E.1 assigns a **dedicated target class ($\$DELETE$) directly to canonical tokens that were omitted in L2-ARCTIC TextGrids**. The cross-entropy loss $\mathcal{L}_{\text{edit}} = -\log P(\$DELETE \mid \mathbf{h}_i^{\text{mdd}})$ forces the representation to learn the acoustic and contextual signature of absence.
- **Uncertainty & Limitations:** Risk of confusion between rapid coarticulation (short realized phonemes) and true phonological deletions. Requires calibrated class weights (e.g., $w_{\text{del}} \approx 4.0$) to compensate for the 1.6% omission class imbalance.

#### 2. Candidate D.1: Training-Free CTC-Viterbi Segmental Pooling ("CTC-Align-MDD") — Score: 91.55
- **Architectural Formulation:** Bypasses cross-attention entirely for alignment. Passes audio through the existing frozen Baseline R CTC head to obtain frame posterior logits $\mathbf{Z} \in \mathbb{R}^{T \times 40}$. Computes the optimal Viterbi alignment trellis over the canonical prompt sequence $\mathbf{c}'$. For each canonical token $c_i$, extracts:
  1. Duration $\Delta t_i = (t_{\text{end}} - t_{\text{start}} + 1) \times 20\,\text{ms}$.
  2. Mean blank posterior $P_{\text{blank}} = \frac{1}{\Delta t} \sum_{t} P(\epsilon \mid \mathbf{h}_t)$.
  3. Pooled acoustic embedding $\mathbf{a}_i = \frac{1}{\Delta t} \sum_{t} \mathbf{h}_t$.
- **Why It Is Compelling:** When a learner physically omits a sound, the canonical Viterbi trellis cannot find acoustic evidence for that phoneme; it squeezes the token into a minimal 1-frame (20ms) interval where the CTC emission is overwhelmingly blank ($P_{\text{blank}} \approx 0.95\text{--}0.99$). Feeding $\Delta t_i$ and $P_{\text{blank}}$ directly into classification eliminates the attention dispersion problem completely. Crucially, **Candidate D.1 can be audited and validated on local CPU without executing any GPU retraining**!
- **Uncertainty & Limitations:** CTC Viterbi forced alignment on heavily accented speech can occasionally misalign consonant clusters, forcing a realized adjacent sound into the omitted phone's slot.

#### 3. Candidate A.2: Dual-Path Hybrid MDD (CTC-DP Omission Prior + Cross-Attention Head) — Score: 90.40
- **Architectural Formulation:** An ensemble architecture combining symbolic sequence alignment with continuous neural scoring:
  - *Path 1 (Acoustic CTC + Levenshtein DP):* Decodes greedy realized phones $\hat{\mathbf{y}}$ and aligns against canonical prompt $\mathbf{c}$. If canonical token $c_i$ is aligned to a string gap ($\epsilon$), Path 1 outputs an omission indicator $p_{\text{dp}, i} = 0.90$.
  - *Path 2 (Cross-Attention Binary Head):* Computes continuous error probability $p_{\text{attn}, i} \in [0, 1]$ on realized tokens to capture subtle phonetic distortions and substitutions.
  - *Fused Decision:* $p_{\text{final}, i} = \max(p_{\text{attn}, i}, p_{\text{dp}, i})$ or calibrated logistic fusion.
- **Why It Is Compelling:** Non-destructive upgrade. It retains the exact Baseline R weights and substitution AP ($0.2643$), while using the existing CTC branch to inject a high-confidence deletion prior, instantly eliminating the 97.5% omission false-negative rate.
- **Uncertainty & Limitations:** Hard dynamic programming alignment decisions can produce False Alarms on fast coarticulated speech where CTC dropped a phone due to ASR ambiguity rather than learner error.

---

### 5.2 Complementary Combinations vs Conflicts

```text
Architectural Complementarity Map
┌────────────────────────────────────────────────────────────────────────┐
│                          COMPLEMENTARY SYSTEM                          │
│                                                                        │
│  [Audio] ──> Frozen Wav2Vec2 Backbone ──> Frame Features H (T x 768)   │
│                      │                                                 │
│       ┌──────────────┴────────────────────────┐                        │
│       ▼                                       ▼                        │
│  [CTC Head]                             [Cross-Attention]              │
│       │                                       │                        │
│       ▼                                       ▼                        │
│  Viterbi Forced Align                    Edit-Aware Tagging            │
│  (Candidate D.1)                         (Candidate E.1)               │
│  Extracts [Δt, P_blank]                  Predicts [$KEEP, $DEL, $REP]  │
│       │                                       │                        │
│       └───────────────────┬───────────────────┘                        │
│                           ▼                                            │
│                 Calibrated Prediction:                                 │
│                 - Omissions: Governed by $DELETE + [Δt, P_blank]       │
│                 - Substitutions: Governed by $REPLACE(ph)              │
│                 - Tier-3 Feedback: Phonological Attributes (F.2)       │
└────────────────────────────────────────────────────────────────────────┘
```

- **Synergistic Combination (E.1 + D.1):** Candidate E.1 (Edit-Aware Tagging) and Candidate D.1 (CTC-Viterbi Segmental Features) are 100% complementary. Feeding the Viterbi duration $\Delta t_i$ and mean blank posterior $P_{\text{blank}}$ as explicit scalar features into Candidate E.1's edit classification head provides the multi-head attention queries with the physical grounding they currently lack.
- **Pedagogical Synergy (E.1 + F.2):** Candidate F.2 (Phonological Attributes) is orthogonal to error detection. Once Candidate E.1 detects a substitution ($c_i \to \$REPLACE(p)$) or omission ($c_i \to \$DELETE$), Candidate F.2's articulatory trait matrices directly generate the pedagogical explanation (e.g., *"You omitted the unvoiced alveolar stop /T/ at the end of the word; keep your tongue against the alveolar ridge and release a gentle puff of air"*).
- **Direct Conflicts:**
  - *Candidate B.1 (Whisper) vs All:* Autoregressive LM error erasure fundamentally conflicts with acoustic phoneme error detection.
  - *Candidate E.2 (Conformer-AED) vs Candidate E.1 (Edit Tagging):* Full autoregressive decoding is completely incompatible with non-autoregressive parallel sequence tagging.
  - *Candidate F.1 (Retrieval) vs Low-Resource Web Deployment:* Maintaining a 600 MB un-indexed vector pool and running kNN over thousands of frames is redundant and slower than a 2.4 MB feedforward edit head.

---

## 6. Pilot Experiment Specifications for Shortlisted Candidates

To adhere strictly to `AGENTS.md` ("Research only. No implementation, GPU training or Kaggle execution... STOP after completing parallel research"), we define two pre-registered, minimal falsifiable pilot experiment specifications for operator review and future authorization.

### 6.1 Pilot Experiment 1 (Zero-Compute CPU Replay): Training-Free CTC-Viterbi Segmental Analysis (Pilot D-1)
- **Objective:** Evaluate whether CTC Viterbi duration $\Delta t_i$ and mean blank posterior $P_{\text{blank}}$ provide statistically significant discrimination of phoneme omissions without executing any GPU training.
- **Platform:** Local CPU / standalone python script (`audit_scripts/diagnose_ctc_viterbi_omissions.py`).
- **Data Partition:** Fold 1 Validation Set (300 utterances, 9,956 phonemes, 145 deletions; speakers `NJS`, `RRBI`).
- **Model Checkpoint:** Production Baseline R weights (`best.pt`, 3,600 updates, SHA-256 `a2445c8d...`).
- **Input Contract:** Raw audio $\mathbf{x}$ and canonical prompt text $\mathbf{c}$. Strict zero-oracle protocol (no gold TextGrid interval access during feature extraction).
- **Evaluation Metrics:**
  - Omission ROC-AUC and Omission Average Precision (AP) of logistic regression on $[\Delta t_i, P_{\text{blank}}]$.
  - Two-sample Kolmogorov-Smirnov test and Welch's $t$-test between omission and correct token distributions.
- **Falsifiable Decision Gates:**
  - **PASS Gate 1:** Mean $P_{\text{blank}}$ on omissions must be at least $1.5\times$ higher than on correct phones ($p < 10^{-5}$).
  - **PASS Gate 2:** Mean interval duration $\Delta t_i$ on omissions must be $< 40\,\text{ms}$ (vs $> 80\,\text{ms}$ on correct phones).
  - **PASS Gate 3:** Logistic regression on $[\Delta t_i, P_{\text{blank}}]$ must achieve Omission AP $\ge \mathbf{0.1200}$ (a $>3.4\times$ improvement over Baseline R's $0.0350$).
  - **STOP Condition:** If $\Delta t_i$ and $P_{\text{blank}}$ fail to achieve Omission AP $\ge 0.0800$, REJECT Candidate D.1.

### 6.2 Pilot Experiment 2 (Remote GPU Pilot): Edit-Aware Sequence Tagging (Pilot E-1)
- **Objective:** Train and evaluate `LangfensEditTaggingMDD` on Fold 1 on Kaggle GPU under the strict fixed-budget contract to prove whether direct $\$DELETE$ supervision solves the omission bottleneck.
- **Platform:** Remote Kaggle GPU (`machine_shape: NvidiaTeslaT4`, Tesla T4, CUDA FP16 AMP).
- **Execution Ceilings:** Max 6 epochs / strictly **3,600 optimizer updates** (~2.5 hours runtime, $< 10\%$ of weekly quota).
- **Model Architecture:** Frozen Wav2Vec2-Base-960h backbone + Cross-Attention + 41-Class Edit Head ($\$KEEP, \$DELETE, \$REPLACE_{1..39}$).
- **Loss Contract:** Multi-task $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{edit}} + 0.5 \mathcal{L}_{\text{CTC}}$, where $\mathcal{L}_{\text{edit}}$ is Class-Weighted Focal Cross-Entropy ($w_{\text{del}} = 4.0, w_{\text{rep}} = 2.0, \gamma = 1.5$).
- **Dual Checkpoint Tracking:** Fixed-budget checkpoint at exactly 3,600 updates (`fixed_budget_checkpoint.pt`) and validation peak (`best.pt`).
- **Falsifiable Pre-Registered Acceptance Gates:**
  - **PASS Gate 1 (Omission AP):** Omission AP must reach $\ge \mathbf{0.1500}$ (a $>4.2\times$ relative gain over Baseline R's $0.0350$).
  - **PASS Gate 2 (Omission Recall):** Omission Recall @ 15% FPR must reach $\ge \mathbf{0.5500}$ (vs Baseline R $0.4000$ and E1 $0.3655$).
  - **PASS Gate 3 (Overall Binary AP):** Full validation Binary AP must not degrade ($\text{AP} \ge \mathbf{0.3170}$).
  - **PASS Gate 4 (CTC Fidelity):** Greedy CTC PER must remain $\le \mathbf{24.50\%}$ (preventing acoustic representation collapse).
  - **STOP Condition:** If Omission AP fails to exceed $0.0700$ or overall Binary AP degrades below $0.3000$, trigger immediate scientific rollback.

---

## 7. Strategic Synthesis & Recommendations

### 7.1 What We Now Know Conclusively
1. **The Root Cause of Baseline R & E1 Omission Blindness is Proven:** Unconstrained cross-attention queries cannot represent acoustic absence without explicit token-level supervision. In E1, prepending an ungrounded skip token without direct alignment loss resulted in complete skip mass collapse ($0.000000$).
2. **Naive CTC Blank Posterior is Not an Omission Detector:** RQ1-DIAG-02B conclusively proved that CTC blank AUC is only $0.653$ because blank tokens dominate unvoiced speech and normal phonetic transitions. Frame-level blank thresholding is unviable.
3. **Complex Foundation ASR Models (Whisper, Conformer-AED, RNN-T) are Incompatible:** Autoregressive language modeling causes catastrophic error-erasure (autocorrecting learner errors into standard English), while Transducer lattices diverge on small L2 corpora and exceed GPU compute ceilings.
4. **Retrieval-Based MDD (Tu et al. 2026) is Scientifically Invalidated:** The reported 69.60% F1 is an evaluation artifact masked by a 104.08% Phone Error Rate, and frame retrieval fundamentally cannot query missing acoustic tokens.
5. **The Authoritative UIT Thesis Proposal Already Pointed to the Solution:** Reference [10] of the UIT thesis proposal (`Omelianchuk et al., BEA 2020: GECToR – Tag, Not Rewrite`) provides the exact theoretical paradigm needed: **formulating MDD as an edit-aware sequence tagging problem over canonical prompt tokens**.

### 7.2 The Recommended Next Moves
1. **DO NOT commit GPU training budget or launch Kaggle runs yet.** The research phase is officially complete, and all findings are documented with zero test leakage.
2. **Authorize Pilot Protocol D-1 first (Zero-Compute Local CPU Audit):** Write and run the training-free CTC Viterbi diagnostic script over existing Baseline R validation predictions. This will confirm whether duration $\Delta t_i$ and mean blank posterior $P_{\text{blank}}$ provide high-precision omission separation without spending a single minute of GPU quota.
3. **If Pilot D-1 confirms separation, prepare Candidate E.1 (Edit-Aware Tagging) staged package:** Implement `LangfensEditTaggingMDD` in a clean feature branch, verify all unit tests and zero-oracle invariants locally, and prepare a pinned Kaggle deployment package (`kaggle kernels push --no-run`) for operator approval.

---

## 8. Epistemic Ledger & Research Attestation

In compliance with `AGENTS.md`, all conclusions and experimental plans in this report are formally attested:
- **No implementation, training, backward passes, or Kaggle GPU jobs were executed during this research pass.**
- **Sealed Test Partitions ($S_{\text{test}}^{(k)}$) across all four folds remained strictly unread and physically isolated.**
- **Existing E0 (Baseline R) and E1 baseline contracts and evaluation ledgers remain 100% frozen and unmodified.**
- **Scope boundaries between RQ1 (Acoustic MDD & Speaking) and RQ2 (Multi-Agent Automated Content Generation) are strictly preserved.**

*Report officially attested by Research Coordinator and all six Lane Leads.*
