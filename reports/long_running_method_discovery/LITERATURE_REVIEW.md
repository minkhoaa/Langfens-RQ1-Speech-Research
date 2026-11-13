# Literature review — cycle 1

Evidence entries below distinguish paper-reported results from LANGFENS hypotheses. Search coverage is bounded; this is not a systematic review. Metadata is checked against original venue/publisher pages when available.

## 1. Skip-State HMMs for Forced Alignment of Speech with Phone Deletion

- Authors: Jiahong Yuan, Mark Liberman. “Investigating Consonant Reduction in Mandarin Chinese with Improved Forced Alignment.” Interspeech 2015, pp. 2675–2679. DOI 10.21437/Interspeech.2015-401.
- URL: https://www.isca-archive.org/interspeech_2015/yuan15_interspeech.pdf
- Data/scope: Mandarin HUB4-NE broadcast speech; 7,849 utterances/20 speakers, six speakers/300 utterances held out; manually segmented initials/finals. Speakers are separated by speaker selection, but paper describes training/test speaker subsets; details should be rechecked.
- Method/input: HMM forced alignment with skip-state transitions permitting a phone to have zero duration; acoustic PLP/MFCC-derived features.
- Gold alignment at inference: reference transcript/phone sequence required; no learner-phone gold required. Evaluation uses manually annotated boundaries.
- Error categories: reduction/deletion in ordinary speech, not L2 S/O/I diagnosis.
- Metrics: boundary agreement 10 ms improved 73.3% to 75.6%; 20 ms 93.1% to 93.3%. No AP/F1 for MDD; no insertion event metric or FP operating constraint.
- Reproducibility: HTK identified; code availability not established here.
- Applicability: supports feasibility of optional/deletable phone topology, not proof of improved LANGFENS O AP or exact phone presence classification. Important nuance: deletion may preserve coarticulatory transitions, so a null phone emission still needs neighboring-context evidence.

## 2. Phoneme Mispronunciation Detection by Jointly Learning to Align

- Authors: Binghuai Lin, Liyuan Wang. ICASSP 2022, pp. 6822–6826. DOI 10.1109/ICASSP43922.2022.9746727.
- URL: https://ieeexplore.ieee.org/document/9746727/
- Method: acoustic and canonical-phone encoders, frame/phone attention fusion, local CNN context; joint multi-task optimization for phoneme mispronunciation and alignment.
- Data/task: public L2-ARCTIC, phoneme-level detection; speaker-split details not verified from publisher abstract. Publisher reports F1=63.04%; F1 definition/operating point and insertion matching remain unverified. No AP/low-FP metric reported in reviewed record.
- Applicability: relevant reference-conditioned alignment comparator, but joint learned alignment is not necessarily deletion-aware and F1 is not comparable to LANGFENS Macro-3 Error F1.

## 3. Explore Wav2vec 2.0 for Mispronunciation Detection

- Authors: Xiaoshuo Xu, Yueteng Kang, Songjun Cao, Binghuai Lin, Long Ma. 2021, Interspeech 2021. DOI 10.21437/Interspeech.2021-777.
- URL: https://www.isca-archive.org/interspeech_2021/xu21k_interspeech.pdf
- Data/scope: L2-ARCTIC; paper reports MDD on prompted-text segments. Exact speaker partition not established from reviewed passages; do not claim speaker-disjointness.
- Inputs: waveform, canonical phone and alignment range; Wav2Vec2 representation with convolution/pooling. Alignment segmentations are supplied; this is not exact-gap insertion diagnosis.
- Inference oracle: needs prompted canonical phones/alignment, not realized-phone gold according to method description. Alignment source/provenance needs full-paper audit.
- Task/metrics: binary correct-vs-mispronounced detection; reported F1=0.610. No S/O/I AP or exact insertion matching; not comparable to Macro-3 Error F1.
- Code: not established in reviewed original.
- Applicability: supports testing pretrained contextual representation in low-resource MDD; does not establish improvements at LANGFENS low-FP tail or three-way diagnosis.

## 4. Transformer Based End-to-End Mispronunciation Detection and Diagnosis

- Authors: Minglin Wu, Kun Li, Wai-Kim Leung, Helen Meng. Interspeech 2021.
- URL: https://www.isca-archive.org/interspeech_2021/wu21h_interspeech.html
- Data: CU-CHLOE; speaker split details not exposed in reviewed abstract/page.
- Inputs/models: MFCC Transformer CE and raw-waveform Wav2Vec2 CNN/Transformer with CTC.
- Task: end-to-end phone recognition/MDD. Reported PER/F-measure: T-1 8.69%/77.23%; T-2 5.97%/80.98%; prior systems 11.1%/72.61%, 12.1%/74.65%.
- No exact N+1 insertion matching/AP/FWR in reviewed record. F-measure must not be compared to LANGFENS Macro-3 Error F1.
- Applicability: contextual SSL + CTC is plausible representation baseline, but this paper does not answer deletion-aware O scoring under exact intended phone alignment.

## 5. An Approach to Mispronunciation Detection and Diagnosis with Acoustic, Phonetic and Linguistic (APL) Embeddings

- Authors: Wenxuan Ye, Shaoguang Mao, Frank Soong, Wenshan Wu, Yan Xia, Jonathan Tien, Zhiyong Wu. ICASSP 2022.
- URL: https://www.microsoft.com/en-us/research/publication/an-approach-to-mispronunciation-detection-and-diagnosis-with-acoustic-phonetic-and-linguistic-apl-embeddings/
- Data: L2-ARCTIC; exact split disjointness not stated in reviewed abstract.
- Method: combines acoustic, ASR-derived phonetic and linguistic embeddings; results relative to baseline: +9.93% detection accuracy, +10.13% diagnosis error rate improvement, +6.17% F-measure (paper abstract wording; definitions require PDF).
- Error scope and exact event matching: not established in abstract; no AP/low-FP constraint confirmed.
- Applicability: motivates separating acoustic evidence from lexical/phonetic priors and evaluating prior-only controls; not direct proof lexical prior improves speaker-independent ranking.

## 6. Phonological Level wav2vec2-based Mispronunciation Detection and Diagnosis Method

- Authors: Mostafa Shahin, Julien Epps, Beena Ahmed. 2023 preprint (arXiv:2311.07037); related Interspeech 2024 paper by Mostafa Shahin and Beena Ahmed, “Phonological-Level Mispronunciation Detection and Diagnosis.”
- URL: https://arxiv.org/abs/2311.07037 ; https://www.isca-archive.org/interspeech_2024/shahin24_interspeech.pdf
- Data: L2 English corpora from learners of different native languages (exact corpus/speaker splits need paper audit).
- Method: speech-attribute / phonological features, multi-label CTC over Wav2Vec2; task reports FAR/FRR/DER improvements versus phoneme baseline.
- Not exact S/O/I AP; insertion matching and speaker disjointness unverified in this pass. No direct numeric comparison.
- Applicability: low-resource feature decomposition candidate, but attributes could lose subtle categorical distinctions and require mapping all phone labels carefully.

## Review takeaways

1. Existing literature supports optional deletion paths, contextual SSL representations, and auxiliary phonological attributes as plausible families.
2. None of the reviewed results establishes improved LANGFENS class-wise AP under exact gap semantics and its speaker split.
3. Several candidate papers lack verified split/protocol details; they remain leads rather than comparators until full-paper extraction.
