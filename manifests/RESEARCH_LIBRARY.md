# LANGFENS — RQ1-ACQ-01: Research Library & Literature Mapping

**Project:** LANGFENS (Phoneme-level Pronunciation Assessment & CAPT)  
**Research Focus:** Phoneme-level Pronunciation Assessment, Mispronunciation Detection & Diagnosis (MDD), Goodness of Pronunciation (GOP), Wav2Vec 2.0 Fine-Tuning, Educational Pronunciation Feedback.  
**Gate:** RQ1-ACQ-01 (Research Resources Acquisition)  
**Date:** 2026-10-22  

---

## 1. Research Papers Taxonomy & Methodological Grouping

The research literature acquired for RQ1 is organized into four core methodological pillars:

```text
LANGFENS Research Library
├── 1. Foundational Acoustic Representations & SSL
│   ├── P04: Baevski et al. (2020) — wav2vec 2.0
│   └── P10: Radford et al. (2023) — Whisper
├── 2. Acoustic Alignment & Pronunciation Scoring (GOP)
│   ├── P02: Witt & Young (2000) — Classical Phone-Level GOP
│   └── P06: Gong et al. (2022) — Transformer GOP (GOPT)
├── 3. Mispronunciation Detection & Diagnosis (MDD)
│   ├── P03: Peng et al. (2021) — wav2vec 2.0 MDD Fine-Tuning
│   ├── P07: Shahin et al. (2023/2025) — Phonological wav2vec 2.0 MDD
│   └── P08: Tu et al. (2025/2026) — Training-Free Retrieval-Based MDD
├── 4. Non-Native Speech Corpora & Pronunciation Benchmarks
│   ├── P01: Zhao et al. (2018) — L2-ARCTIC Speech Corpus
│   └── P05: Zhang et al. (2021) — SpeechOcean762 Pronunciation Corpus
└── 5. Educational Feedback & CAPT Pedagogy
    └── P09: Amrate & Tsai (2024) — CAPT Systematic Review & Feedback Design
```

---

## 2. Detailed Literature Entries

### Pillar 1: Foundational Acoustic Representations & SSL

#### P04 — wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
- **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (Facebook AI)
- **Year / Venue:** NeurIPS 2020 / arXiv:2006.11477v3
- **File:** `references/papers/wav2vec_2.0.pdf` (Alias: `baevski2020_wav2vec2.pdf`)
- **File Size & Checksum:** 740,863 bytes | SHA-256: `4a881708378143547a5c053f62090104afb49782a4c71d78441dde0a818758ff`
- **Verification Status:** `VERIFIED` (Pre-existing on Drive; verified complete 19 pages)
- **Role in RQ1:** Provides the theoretical and architectural backbone for self-supervised speech representation learning, contrastive loss, quantization codebooks, and latent feature extraction for subsequent phoneme recognition and MDD fine-tuning.

#### P10 — Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)
- **Authors:** Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (OpenAI)
- **Year / Venue:** ICML 2023 (PMLR Vol. 202: 28492–28518)
- **File:** `references/papers/radford2023_whisper.pdf`
- **File Size & Checksum:** 823,459 bytes | SHA-256: `84496ed07ac01bf3d00866a1934f81173ddbba99f7488de59bb8ae3b28f0f85e`
- **Verification Status:** `VERIFIED` (Newly acquired Open Access PDF; 27 pages)
- **Role in RQ1:** Optional supporting reference for robust encoder representations, multi-task speech tokenization, cross-accent robustness, and potential zero-shot/few-shot comparative baselines.

---

### Pillar 2: Acoustic Alignment & Pronunciation Scoring (GOP)

#### P02 — Phone-level Pronunciation Scoring and Assessment for Interactive Language Learning
- **Authors:** Silke M. Witt, Steve J. Young (Cambridge University)
- **Year / Venue:** 2000, *Speech Communication*, Elsevier, Vol. 30, No. 2–3, pp. 95–108
- **DOI:** [10.1016/S0167-6393(99)00044-8](https://doi.org/10.1016/S0167-6393(99)00044-8)
- **File:** `references/papers/P02_witt2000_gop_citation.md`
- **Verification Status:** `ACCESS_RESTRICTED` (Elsevier paywall; full bibliographic record, abstract, and formulation preserved without unauthorized bypass)
- **Role in RQ1:** The seminal paper introducing the mathematical foundation of Goodness of Pronunciation (GOP), posterior probability estimation over acoustic models, and phone-level likelihood ratio scoring for Computer-Assisted Language Learning (CALL).

#### P06 — Transformer-Based Multi-Aspect Multi-Granularity Non-Native English Speaker Pronunciation Assessment (GOPT)
- **Authors:** Yuan Gong, Ziyan Chen, I-Fan Chen, Patti Price, James Glass (MIT CSAIL & Educational Testing Service)
- **Year / Venue:** ICASSP 2022 / arXiv:2205.03432v1
- **File:** `references/papers/gong2022_gopt.pdf`
- **File Size & Checksum:** 351,260 bytes | SHA-256: `44c5f27ea0977b40396e6fc538a1636074b1484c6b574b968997789b0ceb851c`
- **Verification Status:** `VERIFIED` (Newly acquired Open Access PDF; 5 pages)
- **Role in RQ1:** Establishes the modern Transformer-based multi-aspect scoring paradigm on SpeechOcean762. Demonstrates how to take phone-level GOP features combined with self-attention to predict hierarchical scores (accuracy, completeness, fluency, prosody) at phoneme, word, and utterance granularities.

---

### Pillar 3: Mispronunciation Detection & Diagnosis (MDD)

#### P03 — A Study on Fine-Tuning wav2vec2.0 Model for the Task of Mispronunciation Detection and Diagnosis
- **Authors:** Leda Peng, Kaiqi Fu, Linlin Wang, Lei Xie (Northwestern Polytechnical University & ByteDance)
- **Year / Venue:** Interspeech 2021, pp. 4448–4452
- **DOI:** [10.21437/Interspeech.2021-381](https://doi.org/10.21437/Interspeech.2021-381)
- **File:** `references/papers/peng2021_wav2vec2_mdd.pdf`
- **File Size & Checksum:** 1,442,794 bytes | SHA-256: `8c56fc608b97146b401aa86ddd5390b405e9809ed31ca3f4ac73d267f4aacce9`
- **Verification Status:** `VERIFIED` (Newly acquired Open Access PDF from ISCA Archive; 5 pages)
- **Role in RQ1:** Direct blueprint for the Wav2Vec2 fine-tuning pipeline for MDD. Analyzes CTC fine-tuning on L2-ARCTIC, phoneme error diagnosis (substitution, deletion, insertion), and catastrophic forgetting prevention when adapting native SSL models to non-native speech.

#### P07 — Phonological Level wav2vec2-based Mispronunciation Detection and Diagnosis Method
- **Authors:** Hossamieldeen Shahin, Mostafa Shahin, Beena Ahmed, Julien Epps (UNSW Sydney)
- **Year / Venue:** arXiv:2311.07037v1 (Preprint) / Formal Publication: *Speech Communication*, 2025 (DOI: 10.1016/j.specom.2025.103249)
- **File:** `references/papers/shahin2023_phonological_wav2vec2_mdd_preprint.pdf`
- **File Size & Checksum:** 4,767,893 bytes | SHA-256: `202d722d4b0c15705c1bc0a5f11c924f13cec136705e4b05410ff32a79f4d456`
- **Verification Status:** `VERIFIED` (Newly acquired Preprint PDF; 42 pages comprehensive study)
- **Role in RQ1:** Expands phoneme-level MDD into multi-attribute phonological feature detection (place of articulation, manner, voicing). Critical for generating fine-grained diagnostic feedback explaining *why* an error occurred rather than just binary correct/incorrect.

#### P08 — Mispronunciation Detection and Diagnosis Without Model Training: A Retrieval-Based Approach
- **Authors:** Huu Tuong Tu, Ha Viet Khanh, Tran Tien Dat, Vu Huan, Thien Van Luong, Nguyen Tien Cuong, Nguyen Thi Thu Trang (HUST & VinAI)
- **Year / Venue:** arXiv:2511.20107v1 / IEEE ICASSP 2026 (DOI: 10.1109/ICASSP55912.2026.11462649)
- **File:** `references/papers/tu2025_retrieval_mdd.pdf`
- **File Size & Checksum:** 737,280 bytes | SHA-256: `2eb40576a3c2f4bf59f1471c8bb3c240711d0f13d9af31c19e0454532fac65d1`
- **Verification Status:** `VERIFIED` (Newly acquired Open Access PDF; 5 pages)
- **Role in RQ1:** State-of-the-art training-free baseline on L2-ARCTIC achieving 69.60% F1 using acoustic embeddings and retrieval. Serves as a strong benchmark to evaluate against fine-tuned Wav2Vec2 architectures without compute overhead.

---

### Pillar 4: Non-Native Speech Corpora & Pronunciation Benchmarks

#### P01 — L2-ARCTIC: A Non-native English Speech Corpus
- **Authors:** Guanlong Zhao, Sinem Sonsaat, Alif Silpachai, Inas Lucic, Evgeny Chukharev-Hudilainen, John Levis, Ricardo Gutierrez-Osuna (Texas A&M & Iowa State University)
- **Year / Venue:** Interspeech 2018, pp. 2783–2787
- **DOI:** [10.21437/Interspeech.2018-1110](https://doi.org/10.21437/Interspeech.2018-1110)
- **File:** `references/papers/zhao2018_l2_arctic.pdf`
- **File Size & Checksum:** 721,794 bytes | SHA-256: `03a4b836bca32e6bdf3439bc363f8dbc1610d2186c577f492f4d6c243da5a022`
- **Verification Status:** `VERIFIED` (Newly acquired Open Access PDF from ISCA Archive; 5 pages)
- **Role in RQ1:** Primary corpus documentation. Details recording protocols, 24 L2 speakers across 6 L1s (Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese), acoustic alignment, and phonetic error taxonomy (substitution, deletion, addition) in ARPAbet.

#### P05 — speechocean762: An Open-Source Non-native English Speech Corpus for Pronunciation Assessment
- **Authors:** Junbo Zhang, Zhiwen Zhang, Yongqing Wang, Zhiping Yan, Qiong Song, Yukhai Zheng, Hao Che, Ning Cheng, Longbiao Wang (SpeechOcean & Tianjin University)
- **Year / Venue:** Interspeech 2021, pp. 2781–2785 / arXiv:2104.01378v2
- **File:** `references/papers/speechocean762.pdf` (Alias: `zhang2021_speechocean762.pdf`)
- **File Size & Checksum:** 512,281 bytes | SHA-256: `5edd602a0ee383e3dc20dcc6a462541dd6daf0c2f22c2b3d5d877770391bd657`
- **Verification Status:** `VERIFIED` (Pre-existing on Drive; verified complete 5 pages)
- **Role in RQ1:** Primary benchmark for multi-granularity pronunciation scoring. Documents the 5-expert independent annotation methodology, phoneme/word/utterance scoring rubric, stress ratings, and baseline Kaldi GOP implementation.

---

### Pillar 5: Educational Feedback & CAPT Pedagogy

#### P09 — Computer-assisted pronunciation training: A systematic review
- **Authors:** Moustafa Amrate, Pi-hua Tsai
- **Year / Venue:** 2024 (Published online September 2024), *ReCALL*, Cambridge University Press
- **DOI:** [10.1017/S0958344024000181](https://doi.org/10.1017/S0958344024000181)
- **File:** `references/papers/P09_amrate2024_capt_citation.md`
- **Verification Status:** `ACCESS_RESTRICTED` (Cambridge Core paywall; full bibliographic record, abstract, and pedagogical review findings preserved)
- **Role in RQ1:** Informs educational feedback system design. Synthesizes 30 empirical CAPT studies, identifies pedagogical gaps (over-reliance on mechanical listen-and-repeat drills, neglect of suprasegmentals), and provides actionable guidelines for user-facing corrective feedback.

---

## 3. Additional Literature Candidates (`ADDITIONAL_CANDIDATES`)

*Note: Per acquisition rules, these candidate papers are documented for research roadmap purposes only and have NOT been automatically downloaded.*

1. **Hu et al. (2023)**, *Phoneme-level Mispronunciation Detection and Diagnosis using End-to-End ASR with Multi-task Learning*, ICASSP. (Explores joint CTC-Attention models for simultaneous transcript generation and phone error diagnosis).
2. **Yan et al. (2023)**, *Selective Multimodal Pronunciation Assessment*, Interspeech. (Audio-visual pronunciation assessment on non-native corpora).
3. **Korzekwa et al. (2022)**, *Computer-Assisted Pronunciation Training with Deep Learning: A Survey*, IEEE Access. (Broad technological survey complementing the pedagogical review of P09).
