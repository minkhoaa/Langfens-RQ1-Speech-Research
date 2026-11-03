# Agent E — Original-thesis alignment and feedback audit

**Process:** research-only closure of existing `RQ-RESEARCH-01`; not a new phase.  
**Disposition:** authoritative requirements recovered; conditional family screening completed; **no final architecture selected and no model/product compliance certified**.  
**Authority order:** original KLTN0127 proposal → directly inspected repository artifacts/source → accessed primary literature → explicitly labeled interpretations. Historical research reports are continuity records, not independent scientific evidence.

## 1. Actual inspection, provenance and boundaries

The complete original document was read:

`/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`

- 56,472 bytes, 18 physical lines; SHA-256 `98fa922584b750510d4b0c183c59a84455da9e64ab231b59bdb58862c56133d3`.
- This is an original **“ĐỀ CƯƠNG CHI TIẾT”** proposal, not a completed experimental thesis. The English title is *Developing an English Learning System Integrated with Conversational AI Assistant and Multi-Agent Automated Test Generation* (L6, L10). The listed students are Nguyễn Bá Trọng Khôi and Từ Minh Khoa, with advisor ThS. Trần Thị Hồng Yến (L11–13).
- L14 contains 37,473 characters. To avoid long-line/tool truncation, it was printed in five bounded, wrapped stdout slices covering character intervals `[0,8000)`, `[8000,16000)`, `[16000,24000)`, `[24000,32000)`, `[32000,37473)`. L1–13 and L15–18 were also read. No temporary derivative was written. Citations below use **T:L14, §…** to distinguish sections sharing the same physical line; **T** means the exact absolute path above.
- All of `AGENTS.md` and `docs/process-checkpoints/INDEX.md` were read, including the latter's continuation after the 50-KB output boundary. Relevant overview checkpoints read in full: `manifests/RQ-RESEARCH-01/PROCESS_CHECKPOINT.md`, `manifests/RQ-RESEARCH-01/revalidation-2026-11-03/PROCESS_CHECKPOINT.md`, `manifests/RQ1-DIAG-02B/PROCESS_CHECKPOINT.md`, and `manifests/RQ-MODEL-01/PROCESS_CHECKPOINT.md`.
- Outer HEAD and branch directly verified: `b25a7595360294798d8d06b37592dd65dd4c9c33`, `research-phase`; nested `repo` HEAD: `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`. **Initial current outer status was already dirty:** modified `docs/process-checkpoints/INDEX.md`, dirty nested `repo`, and untracked `manifests/RQ-RESEARCH-01/revalidation-2026-11-03/`. The nested checkout already had modified audit scripts and two deleted reference PDFs. Those are not Agent E changes.
- Source inspection was static, bounded and partial, with exact ranges/locations identified below. No source was executed or imported. No model, checkpoint weights, audio, raw annotation archive, sealed TEST labels/audio, or remote compute was opened. No experiment, calibration fitting, training, inference, deployment, commit, or push occurred. The only authorized write is this isolated report.

### Evidence/read ledger

| ID | Directly inspected underlying source | Supports | Boundary |
|---|---|---|---|
| T | Complete original proposal, provenance above | Authoritative scope, proposed methods, error examples, feedback, evaluation and integration | Proposed expectations, not measured achievements; original-file license not established |
| P | `AGENTS.md`, especially Execution Boundaries, Training Contract & Provenance | Current local operational constraints; SpeechOcean transfer belongs to RQ1; no gold-timestamp inference | Repository policy is binding operationally, but is not original-thesis wording |
| C | `configs/rq1_trn_02a_fold1.yaml:1–29`, complete | Pinned pretrained revision, frozen backbone, two loss weights, seed/batch/schedule/AMP | Configuration intent, not proof of execution or deployment correctness |
| M | `src/langfens_prep/model.py:285–424,475–654`; keyword search elsewhere | CTC, binary and diagnosis head definitions; oracle versus deployable forward branches | Current inspected source, not independent verification of trained diagnosis or model performance |
| Z | `audit_scripts/evaluate_protocol_z.py:324–371`, plus targeted keyword search | Model gets audio + G2P prompt phones; gold retrieved afterward for offline scoring alignment | Static source inspection only; this script was not run |
| A | `src/langfens_prep/alignment.py:1–180` | Canonical-to-canonical DP semantics; boundary pooling gives every interval at least one frame | Neither routine alone demonstrates acoustic omission localization |
| G | `src/langfens_prep/g2p.py:1–190` | Prompt-derived 39-phone frontend; primary dictionary pronunciation selection and OOV handling | Dictionary/model artifacts not loaded; pronunciation variants need separate review |
| S | `src/langfens_prep/speechocean_parser.py`, complete | Repository representation of phone accuracy and higher-level assessment scores | Parser schema, not direct inspection of original SpeechOcean annotations or license |
| E | `manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json`, complete | Archived E0/E1 validation results, denominators and endpoint identifiers | Archived empirical record, not a fresh replay, uncertainty analysis or weight rehash |
| L1 | https://aclanthology.org/2020.bea-1.16/ — title, abstract, bibliographic metadata and site licensing notice | GECToR is a text grammatical-error-correction sequence tagger, not demonstrated speech MDD | Abstract/page accessed; full paper and implementation not read in this lane |
| L2 | https://arxiv.org/abs/2507.16838v3 — abstract, version history, license link | *Segmentation-free Goodness of Pronunciation*, Cao et al.; GOP-SA and GOP-SF; reported pronunciation-assessment evaluation includes SpeechOcean762 | v3, revised 5 February 2026; abstract only, not independent verification of its experiments or omission-specific applicability |
| L3 | https://arxiv.org/abs/1712.05382 and https://arxiv.org/abs/1712.05382v2 — abstract/version/license metadata | *Monotonic Chunkwise Attention*, Chiu and Raffel; original method explicitly targets online/linear-time decoding | v2, revised 23 February 2018; abstract only, not an MDD compliance study |

Verified primary-paper licensing metadata: L1's ACL site states post-2016 materials are CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), subject to its copyright-holder notice; L2's article license link is http://creativecommons.org/licenses/by/4.0/; L3's is http://arxiv.org/licenses/nonexclusive-distrib/1.0/. These observations do **not** establish licenses for model weights, author code, pronunciation dictionaries, teaching materials or datasets. No primary-paper performance/latency number is transferred into this report.

Additional byte checks, not scientific reproduction:

- Frozen split metadata SHA-256 recomputed: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- E summary SHA-256 recomputed: `006611ccf0796938d4114bf62a1d17979c0af9ece362415dc1600670e6b086a3`.
- Current M source SHA-256: `af2a1271c3e4a120f1505aa576bb2d75dc3ed90495b3b7ba675096ebb332f592`; current Z source SHA-256: `aca2406b8b927c01947fb6d42a0fcaa12c11bf1fc624587169f8739fe1fc17c6`. These identify inspected bytes and are not assertions about historical execution-source identity.

## 2. Authoritative requirements, with original Vietnamese evidence

| Requirement | Original evidence | Consequence for closure |
|---|---|---|
| **RQ1 has both acoustic detection and feedback-quality questions.** | “Phương pháp phân tích âm học ở cấp độ âm vị đạt hiệu quả như thế nào trong việc phát hiện các nhóm lỗi phát âm mục tiêu…” and “Các phản hồi hướng dẫn sửa lỗi… có đáp ứng các tiêu chí về tính chính xác, rõ ràng và phù hợp với lỗi được phát hiện hay không?” (T:L14, §2.3, RQ1) | A detection metric alone cannot close RQ1; feedback correctness, clarity and relevance need their own evidence. Neither question presupposes a successful model. |
| **STT and pronunciation analysis are distinct tasks.** | “việc nhận dạng nội dung lời nói không đồng nghĩa với khả năng phát hiện chính xác lỗi phát âm” (§1); “hai nhóm xử lý có mục tiêu khác nhau: nhận dạng lời nói phục vụ hội thoại và phân tích phát âm phục vụ phản hồi sửa lỗi” (§4.c) | Low WER or an intelligible transcript cannot establish accurate phone-level MDD. Conversely, an MDD module need not replace conversational STT. |
| **Whisper is explicitly contemplated for conversational STT.** | “sử dụng Whisper hoặc mô hình Speech-to-Text tương đương… phục vụ phân tích nội dung và tạo phản hồi hội thoại”; “bản phiên âm của mô hình không đủ để kết luận một âm vị đã được phát âm đúng hay sai” (T:L14, §4.c) | Blanket rejection of Whisper/large ASR as outside the thesis is incorrect. Transcript-only pronunciation diagnosis remains insufficient. |
| **Targeted, reference-aware scope, not exhaustive open conversation diagnosis.** | “ưu tiên bài luyện có văn bản tham chiếu” (§3.b, Speaking); “không đặt mục tiêu phát hiện đầy đủ mọi lỗi phát âm trong hội thoại tự do” (§3.b, Giới hạn nghiên cứu) | Read-aloud/prompt-conditioned assessment is aligned. Comprehensive spontaneous-speech diagnosis is not a mandatory closure criterion. |
| **Reference pronunciation comes from ordinary deployment information.** | “Văn bản tham chiếu được chuyển đổi thành chuỗi âm vị kỳ vọng thông qua từ điển phát âm hoặc cơ chế chuyển đổi từ chữ viết sang âm vị” (T:L14, §4.c) | Written prompt and dictionary/G2P are legitimate inference inputs; human error labels/gold TextGrid timestamps are not normal deployment inputs. The latter prohibition is explicit in P. |
| **Alignment plus acoustic scoring/classification is the proposed route.** | “sử dụng Forced Alignment để xác định vị trí tương ứng…”; “dự kiến khảo sát… GOP hoặc phương pháp phân loại lỗi âm vị dựa trên biểu diễn âm học từ mô hình pretrained, chẳng hạn wav2vec 2.0” (T:L14, §4.c) | The proposal is method-specific enough that replacements should explain how they preserve reference/acoustic correspondence. It does not establish that a particular forced aligner, wav2vec variant or architecture has already been selected or validated. |
| **GOP is not standalone subtype diagnosis.** | “Điểm GOP… không được xem là bằng chứng độc lập để kết luận chính xác loại lỗi phát âm” (T:L14, §4.c) | A low score can support “suspect phone,” not automatically “you omitted /t/” or “you said /d/.” |
| **Named target-error examples are provisional and data-dependent.** | “một số nhóm lỗi như thay thế âm vị, bỏ phụ âm cuối và những cặp âm dễ nhầm lẫn; phạm vi cụ thể được xác định dựa trên tính sẵn có và chất lượng của dữ liệu đánh giá” (T:L14, §4.c) | Explicit examples: phone substitutions, final-consonant omissions, confusable pairs. These are not a complete fixed taxonomy. Any final chosen subset needs a transparent label/evidence justification. |
| **Subtypes require verified recognition/classification and reference labels.** | “Việc phân loại nhóm lỗi cụ thể, nếu được triển khai, cần dựa trên phương pháp nhận dạng hoặc phân loại được kiểm chứng bằng dữ liệu có nhãn tham chiếu” (T:L14, §4.c); §2.2.b also conditions specific classification on method/data | A DELETE tag, recognition edit, attention weight or phonological rule is a hypothesis-bearing representation, not automatic validated diagnosis. |
| **Confidence limits the feedback claim.** | “Khi phát hiện lỗi có đủ độ tin cậy…”; “Trong trường hợp dữ liệu không đủ rõ ràng hoặc kết quả dự đoán có độ tin cậy thấp… giới hạn mức độ kết luận hoặc đề nghị người học thực hiện lại bài luyện” (T:L14, §4.c) | Uncertainty/abstention is core alignment, not merely optional product polish. Unsupported subtype certainty is misaligned even if explanations sound fluent. |
| **Feedback progresses beyond an error flag.** | “xác định vị trí âm nghi vấn, đưa ra gợi ý để người học tự điều chỉnh, giải thích cách phát âm và cung cấp mẫu âm thanh bằng Text-to-Speech” (T:L14, §4.c); §2.2.b includes “giải thích nguyên nhân” | Localized suspicion, hints, pronunciation guidance and audio exemplars are expected. “Explain cause” must not invent an articulatory mechanism unsupported by the evidence. |
| **Evaluation separates ASR, MDD and feedback.** | WER, optional CER for STT; “Precision, Recall, F1-score và ma trận nhầm lẫn” for MDD; feedback judged for “tính đúng đắn, sự rõ ràng và mức độ phù hợp” (T:L14, §4.g, Đánh giá nhận dạng giọng nói và phát hiện lỗi phát âm) | PER/AP can be useful repository metrics but do not replace the promised detection/error and feedback evaluations. Detection versus diagnosis denominators must be explicit. |
| **Calibration/testing and comparison must be defensible.** | “Dữ liệu hiệu chỉnh và dữ liệu kiểm thử được tổ chức tách biệt”; prefer speakers absent from calibration where feasible; compare methods “trên cùng tập dữ liệu kiểm thử và cùng đơn vị đánh giá”; baseline choice depends on implemented method/data/resources (T:L14, §4.g) | Do not treat repeated validation selection as an unbiased final evaluation. Do not compare unlike score-correlation and omission-detection tasks. This report does not authorize opening sealed tests. |
| **Grounded explanations and evidence-limited RAG are expected.** | Feedback should be checkable against retrieved sources (§2.2.c); when retrieval is insufficient, “hạn chế việc đưa ra kết luận chắc chắn” (§4.d) | A teaching reference can support the correction advice; it cannot manufacture missing acoustic evidence that the learner made that error. |
| **Integration is part of expected output.** | Backend/API handles authentication, authorization, task management, orchestration and AI integration; long AI tasks may use asynchronous execution (§4.a). Integration evaluation includes interrupted-task recovery and data consistency (§4.g). | A research classifier is not the integrated thesis system. There is no verified end-to-end implementation claim in this audit. |
| **RQ2 is content generation and its coordination trade-offs.** | Compare specialist-agent generation/review/validation/repair against single LLM and sequential self-review (T:L14, §2.3 RQ2, §4.g Multi-Agent) | Acoustic MDD, pronunciation feedback and cross-corpus speaking assessment do not become RQ2 because an LLM explains them. |

All quoted sections above are in T:L14. T:L15 contains the schedule and bibliography; §4.c cites L2-ARCTIC as one candidate corpus, not an exclusive mandatory dataset. The schedule culminates in Speaking/Multi-Agent experiments, controls, integration/error tests, and documented limitations (T:L15). §7 treats broader prosody, stress, intonation, rhythm and fluency as future development: full coverage of these is not an original RQ1 minimum.

## 3. What is and is not a thesis mandate

### Original requirements versus project choices

| Item | Original proposal | Repository/operator constraint or extension | Audit consequence |
|---|---|---|---|
| Pretrained models | Uses pretrained AI; does not aim to train a foundation language model from scratch (T:L14, §3.b) | Exact wav2vec checkpoint and revision in P/C | The exact backbone/revision is a project experimental contract, not original wording. The proposal does not categorically ban adapting a pretrained acoustic model. Current authorization still bans all implementation/training. |
| Frozen backbone | Not specified | `backbone_frozen: true`, C:L10; P Training Contract | Binding current baseline/comparability constraint; not a universal thesis gate against all other acoustic families. |
| Two heads only | Not specified | C:L9 has CTC + binary weights; P expressly preserves two-head training | A candidate with explicit subtype/edit outputs is not inherently outside the thesis, but changing the registered experiment requires separate approval. Merely defining a diagnosis head in M:L360–365 does not establish a trained diagnostic capability. |
| Update budget, optimizer, seed, folds | No exact numerical training contract | C and P define specific protocol; fixed-budget accounting belongs to project | Existing baselines must retain historical semantics. A method requiring another budget is conditionally infeasible under that contract, not scientifically disproven or categorically thesis-incompatible. |
| Colab/Kaggle, T4, FP16 AMP | Not named | P explicitly requires authorized remote GPUs for all training, with platform safeguards | Operationally binding, not evidence that the original thesis required those platforms or a particular precision. No local training fallback is permitted. |
| Latency | Selection considers “độ trễ” (§5); time/cost and feasibility evaluated (§2.2.g, §4.g); asynchronous AI tasks contemplated (§4.a) | Any numeric end-to-end latency threshold must come from another approved requirement | No original numeric SLO or blanket “autoregression too slow” rule was found. No unsupported latency, memory or runtime estimate is used here. |
| SpeechOcean762 transfer | **No occurrence** of `SpeechOcean`/`speechocean` in complete original | P says cross-corpus SpeechOcean762 and pronunciation feedback belong to RQ1 | A coherent project extension to RQ1, not a verbatim original-thesis requirement and not RQ2. |
| All error categories | Examples and data-conditioned scope | Project labels/available annotation constrain experiments | Avoid silently equating every corpus deletion with word-final consonant deletion, or confusable-pair assessment with all substitution cases. |
| Web technology | §5 proposes Next.js/TypeScript, .NET 8, Python/FastAPI, PostgreSQL, possible gRPC/Qdrant/Redis, Docker; choices remain provisional | Actual integration choices need verification | API compatibility must be demonstrated, not inferred from shared programming language or model tensor shapes. |

### SpeechOcean transfer: what it can legitimately test

The transfer question is within **RQ1 speaking/pronunciation assessment**, but its task must be declared. Repository S:L6–26 represents phone accuracy, word stress/accuracy/total, and utterance accuracy/completeness/fluency/prosodic/total. It does **not expose an omission/substitution event taxonomy**. That is an observation about this parser, not a claim that the entire original dataset was independently audited here.

Consequently:

1. Cross-corpus phone-quality association/assessment and omission-event detection are different endpoints. An accuracy score is not automatically a ground-truth DELETE label.
2. Any conversion to binary errors, subtype labels or score targets needs documented annotation meaning, thresholds, mapping and a distinct validation policy. Do not silently reuse L2-ARCTIC calibration or recode historical results.
3. L2's abstract reports pronunciation-assessment studies including SpeechOcean762. This establishes a primary-literature transfer lead, **not** that its features diagnose final-consonant omissions in this project.
4. No transferred model result, native-data license, calibrated cross-corpus operating point or educational validity was verified in this lane. Original SpeechOcean data/primary dataset documentation were not opened; stop short of a definitive label/transfer feasibility claim until that source review is authorized and completed.

## 4. One explicit thesis gate for every family

Apply the same gate to a **fully specified candidate and intended claim**, not to a model-family name. This is a documentary screen derived from T, not a retrospective numerical rubric.

**G1 — Scope:** addresses selected phone-level pronunciation errors in reference-prioritized practice; separates conversational ASR and RQ2 content generation.  
**G2 — Legitimate evidence:** combines actual learner acoustics and prompt/reference information without inference-time gold labels/timestamps. A text prior alone fails this condition.  
**G3 — Honest error semantics/localization:** produces a reference-phone/word-linked suspicion or a separately validated subtype; distinguishes substitutions, final-coda omissions, other edits and uncertainty. A missing phone may have no realized segment.  
**G4 — Empirical evaluation:** tests the claimed target against appropriate labels, separate calibration/evaluation and speaker isolation; fair same-unit comparisons; false alarms and missed errors included.  
**G5 — Feedback/abstention:** conclusions are no stronger than acoustic/label evidence; guidance is correct, clear, error-relevant and source-supported; low-confidence/poor audio triggers limited claims or retry.  
**G6 — Integration/operation:** has a credible path to web task handling, stable reference mapping, result/history storage, access control, failures/recovery and measured time/cost; no invented numeric SLO.  
**G7 — Method traceability:** explains correspondence to the proposal's alignment + GOP/acoustic-classifier route, or records an alternative/deviation for approval. Forced alignment is the proposed mechanism, not a scientific proof that all missing phones occupy valid segments.  
**G8 — Authorization/comparability:** respects current P and frozen historical evaluation semantics; identifies needed contract changes instead of silently implementing them.

**Screen-compatible** below means a candidate can plausibly satisfy these conditions with the specified supporting components. It is *sufficient to retain for research consideration*, not evidence that G2–G6 have been demonstrated. **Conditional** means material representational, scope or contract issues remain. **No family below is certified compliant as a complete RQ1/product solution by this audit.** In particular, documentary compatibility is not a GO for a pilot or a performance PASS.

### Family-to-gate matrix

| Family | Gate-based fit and legitimate contribution | Required correction/evidence before claiming compliance | Screening disposition |
|---|---|---|---|
| **Segmental GOP / forced-aligned posterior scoring** | Directly follows §4.c and supports G1/G2/G7; can rank suspicious reference phones | G3: forcing every canonical phone to a segment can assign neighboring/irrelevant frames to an omission. Low GOP is not subtype proof. Need alignment-failure handling, target-specific validation, G4/G5 calibration and feedback review | **Screen-compatible for suspicion scoring; conditional for omission diagnosis** |
| **Self-aligned / segmentation-free GOP** | Primary L2 defines GOP-SA and GOP-SF; reference-conditioned acoustic compatibility without requiring supplied segment boundaries fits the intent of G2 | G7: document departure from explicit presegmentation. G3/G4: validate final-coda omission and confusable-phone claims, not just overall pronunciation assessment. Marginalizing alignments does not itself prove trustworthy localization or subtype attribution | **Screen-compatible alternative**, subject to recorded method rationale |
| **Acoustic phone recognition + DP/edit alignment** | Realized-phone hypotheses can be compared with expected phones to produce G3 reference-linked mismatch hypotheses; compatible with acoustic classification route | Recognizer deletions/substitutions can be model mistakes. Repeated phones and ties can move the apparent edit. Must distinguish recognition PER from gold pronunciation events and evaluate target localization/diagnosis; do not pass canonical-to-canonical DP off as this acoustic path | **Screen-compatible**, not demonstrated diagnosis |
| **Canonical forced/Viterbi alignment, segment pooling, constrained/regularized attention** | Useful correspondence/pooling mechanisms for G2/G7; can support a downstream scorer | A full-canonical path often cannot express an absent phone without assigning frames. Short duration, low score or improved attention overlap is not a validated omission. G3/G4 require alternative hypotheses and false-positive controls | **Screen-compatible supporting component; insufficient alone** |
| **Skip-capable/edit lattices or explicit null alignment** | Can represent absence without fabricating a realized phone segment, addressing a real G3 limitation | Skip probability must be acoustically grounded and calibrated. Optional skip states may collapse or encode phone-position priors; null state presence alone fails G4/G5. Need permitted inference-only inputs and fair comparisons | **Screen-compatible hypothesis; empirical caution** from E |
| **Explicit subtype head / edit tagging** (KEEP, OMIT/DELETE, replacement labels) | Makes G3 outputs auditable and can distinguish broad detection from subtype diagnosis; aligns with §2.2.b/§4.c if labels permit | DELETE supervision is not acoustic grounding; terminology/direction of edits must be explicit. Text-only GECToR evidence does not validate speech MDD. Need label quality, speaker-disjoint subtype validation, evidence-dependence checks and G8 approval for a changed head/loss contract | **Conditional but viable**; no privileged thesis status |
| **Joint reference-aware acoustic recognition + detection/diagnosis** | Can combine prompt-indexed outputs and realized-phone evidence, potentially meeting G1–G3/G7 | Reference copying or lexical correction may hide learner errors; test dependence on matched acoustics, mismatched controls and targeted confusion/omission labels. More heads/attention do not imply reliability. G8 resource/training changes need approval | **Conditional but viable**; not ruled out by family size alone |
| **Phonological/articulatory feature models** | Potentially useful for confusable pairs and for G5 language-teaching descriptions | Attributes do not establish actual articulator motion or cause. No phone may be realized for an omission. Need a phone/event-level mapping, valid-variant policy, calibrated uncertainty and evaluation of advice against detected evidence | **Screen-compatible complement or conditional primary detector** |
| **Acoustic exemplar retrieval / metric learning** | Can provide acoustically matched exemplars and supplementary evidence; compatible with G2 and explanatory support | No candidate phone segment can make omissions especially challenging. Need a non-oracle query/reference mapping, speaker-independent retrieval, data/license/privacy controls, coverage/abstention and event validation. Retrieval of teaching text is not acoustic diagnosis | **Conditional but viable**; not blanket rejected on an unaudited PER number |
| **Calibrated hybrids / routing / dual-path ensembles** | Can combine acoustic mismatch and reference-aware scoring and decide when to abstain, supporting G2–G5 | Complementarity and calibration are empirical questions. Do not average incomparable scales or count the same prior twice. Evaluate accepted coverage, subtype errors, disagreements, false alarms and end-to-end resource costs | **Screen-compatible hypothesis**, not automatic superiority |
| **Large pretrained ASR / autoregressive encoder-decoder** | Explicitly within conversational STT scope (§4.c). Can also be a component of acoustic/reference-aware pronunciation research if suitable evidence is exposed | Transcript-only MDD fails G2/G3. Language-model normalization/error erasure is a risk to test, not an inevitable property of every model. G4 and G8 feasibility/contract checks remain; do not invent latency or memory disqualifiers | **Screen-compatible for conversation; conditional for MDD** |
| **Transducer / RNN-T and related sequence models** | A recognition/alignment family, not inherently a different thesis scope; a pretrained acoustic realization path could support G2/G3 | Blank/no-emission is not omission ground truth. Need reference-phone mapping and error-preserving validation. New training objectives/budget/backbone must pass G8; no inspected evidence here supports universal OOM/divergence claims | **Conditional research option**, not blanket rejected |
| **MoChA / monotonic chunkwise attention** | Potential acoustic/reference correspondence component. L3 explicitly describes online and linear-time decoding, contradicting a universal “recurrent therefore too slow” dismissal | Monotonic order does not itself encode a missing canonical phone. Need explicit null/edit treatment or a diagnostic component, plus G3–G6 evidence. Generic ASR results are not MDD validation | **Conditional alignment component**, not full diagnosis |

The matrix deliberately separates **ability to represent a possible explanation** from **evidence that the explanation is correct**. No weighted scores, absolute ranking or winner are justified by this gate alone.

## 5. Current artifacts do not close the feedback/diagnosis gap

### What direct repository evidence establishes

M:L348–365 defines CTC recognition, canonical binary detection and a 39-class substitution diagnosis head. C:L9 specifies only CTC and binary loss weights. The inspected trainer region `audit_scripts/train_fold1_ctc_binary.py:714–728` asserts that diagnosis gradients must remain absent. Together these sources support caution: **an output/head definition is not a trained, validated subtype detector**. This lane did not inspect weights or retrain/replay any model.

E records an archived Fold 1 validation population of 300 utterances/9,956 canonical tokens, including 145 omissions. Its stored omission AP is 0.035020 for E0 and 0.032252 for E1; E1 mean skip mass on deletions is 0.0. These are **archived empirical values**, not this audit's experiments. They support retaining the recorded E1 no-rollout caution, not a causal claim that skip models generally cannot work, not a new statistical significance assertion, and not validation of Baseline R as a pedagogically safe production system.

The DIAG-02B checkpoint explicitly preserves archive-parity failure and a causal-training NO-GO. Its checkpoint is used here as a continuity/authorization warning only: the underlying DIAG raw records were not independently reanalyzed by this lane. Historical “production anchor” phrasing does not establish educational correctness or replace G4–G6 evidence.

### Four distinctions that must survive synthesis

1. **Deletion is not CTC blank, silence or a universal zero-duration segment.** A model's non-emission is a latent alignment operation; a gold omission is a reference-conditioned pronunciation event. Weak/unreleased consonants, short transitions, pauses and alignment mistakes can be confounds.
2. **Canonical alignment is not acoustic omission localization.** A:L76–100 aligns gold canonical phones to prompt-G2P canonical phones. Its unmatched entries concern sequence correspondence, not proof that the learner omitted a sound. Z:L344–363 calls the model before retrieving gold and performing this scoring alignment. This separation must remain unchanged.
3. **Forced intervals are not observed segments for missing sounds.** A:L26–30 explicitly guarantees at least one frame for every interval; A:L55–60 imposes a minimum one-frame slice. That can be appropriate for a defined diagnostic pooling routine, but cannot be interpreted as evidence that a deleted phone had an acoustically realized interval.
4. **DELETE supervision / GECToR are not validated acoustic diagnosis.** L1's abstract describes grammatical correction on text with token transformations. The original thesis cites GECToR under Grammar/Writing methodology (§4.f and T:L15 [10]), not as an established speaking-MDD method. Reusing an edit-tag representation for speech is a transfer hypothesis requiring G2–G5 evidence.

### Legitimate feedback ladder

The following is an integration/evaluation specification **for consideration**, not implemented behavior or a new experimental contract:

| Available evidence | Defensible learner-facing response | Unjustified escalation |
|---|---|---|
| Invalid/noisy/truncated input; unsupported prompt or G2P/OOV ambiguity | Explain the input limitation and request another recording/prompt clarification; do not score the phone | “You omitted the final consonant” because no phone was decoded |
| Calibrated low-confidence phone suspicion, no validated subtype | Mark the reference word/phone as possibly unclear; provide a neutral listening/practice hint; allow retry | Assert a substitution or absent consonant, or fabricate a precise missing-phone timestamp |
| Validated target-specific omission hypothesis | “The final /t/ in this word may be missing,” with confidence-appropriate wording and source-supported practice guidance | Present blank posterior, forced duration or null-state mass as direct certainty |
| Validated expected-versus-realized substitution with stable mapping | Identify the expected and probable realized phone; give a checked contrastive cue/example | Assert exact tongue/lip motion observed from the waveform |
| No detected target error on a supported recording | “No targeted issue was detected,” scoped to the supported targets and operating point | “Your pronunciation is perfect,” or a global proficiency/IELTS score |

For an omitted phone the UI can locate the **reference word/phone**, optionally a qualified neighboring acoustic region, without inventing a realized segment. If localization confidence is low, the interface must say so. Audio exemplars/TTS should be pronunciation-checked and not presented as proof that the learner made a particular error. Guidance should accept legitimate pronunciation variants rather than demanding spelling-driven extra segments or an inappropriate audible release.

Confidence needs a defined target: detection probability, subtype probability, mapping certainty and recording adequacy are different quantities. A sigmoid, attention mass or retrieval similarity is not necessarily a calibrated probability. Abstention must not conceal coverage: report how often results are withheld, which groups/errors are withheld, and the false-reassurance risk among accepted “no issue” outputs.

### Required evaluation evidence for eventual feedback compliance

- **Detection:** the original Precision/Recall/F1 and confusion matrix on declared target phones/events; false alarms on correct phones and misses on omissions; subtype- and speaker-aware reporting. AP/PER remain useful supplementary or historically registered measures, not substitutes for the original question.
- **Diagnosis/localization:** correctness of claimed subtype and expected/realized contrast; reference-word/phone indexing accuracy; explicit treatment of repetitions, G2P variants, missing segments and ambiguous alignments. Do not require a fictional exact time interval for absence.
- **Calibration/selective prediction:** calibration of the actual reported confidence target, accepted coverage versus risk, abstention reasons, and failure behavior on insufficient audio. Threshold selection and evaluator changes need separate authorization; none was performed here.
- **Pedagogical feedback:** an independent, domain-informed rubric for correctness, clarity and relevance to the supported detected error. Separate correct acoustic finding + wrong advice, false acoustic finding + fluent advice, and correct limited/abstaining response. Check articulatory guidance, examples/TTS and retrieved-source entailment; RAG grounding alone cannot establish acoustic correctness.
- **Honest claims:** no claim of learning improvement without an appropriate learner-outcome study; the proposal's feedback-quality evaluation is not itself proof of improved learning.

## 6. Integration expectations without implementation

T:L14 §4.a/§4.g/§5 and §6 require more than a notebook score. A candidate may pass the research screen while integration remains unproven. A future reviewable interface should preserve, at minimum:

- task/session and prompt identity; prompt/G2P/reference-version identity; reference word/phone index;
- recording adequacy and assessment status, including unsupported input, abstention, failure and retry;
- detection score versus validated subtype, with separately qualified localization and uncertainty;
- method/model/calibration version so stored histories are reproducible and historical metrics are not silently reinterpreted;
- explanatory text separated from diagnostic evidence, with teaching-source references and checked audio exemplar provenance;
- task progress, recoverable interruption and consistent result storage; access control for learner histories/audio and appropriate data retention.

These are **derived integration recommendations**, not a claim that the thesis prescribed this exact schema. They follow its requirements for structured service integration, history, evidence-limited feedback, authentication/authorization and interrupted-task recovery. Asynchronous processing is explicitly contemplated, so a streaming transducer is not mandatory and offline scoring is not automatically disallowed. Completion/failure rates, latency/time and cost must eventually be measured in the intended environment; no numeric threshold or estimate is invented here.

Boundary of functions:

- **Conversational assistant:** STT → context/history-aware dialogue → optional TTS; validate WER and conversational behavior separately.
- **RQ1 pronunciation:** learner audio + authorized reference → calibrated target-level evidence → bounded pedagogical guidance; SpeechOcean transfer can test a separately specified assessment endpoint.
- **Knowledge/RAG support:** supplies teaching knowledge and citations, not acoustic ground truth.
- **RQ2:** controlled content generation with Coordinator/Generator/Reviewer/Validator roles, comparison against single-model generation and sequential self-review, and explicit time/cost trade-offs. Agent orchestration used to conduct this research is not RQ2 experimental evidence.

## 7. Actionable corrections for the coordinator's closure document

1. Anchor the requirements matrix to T and quote its full RQ1 dual question. Do not close RQ1 with only omission AP or a family ranking.
2. Replace historical blanket family rejection with the common G1–G8 screen. Retain actual project constraints and observed candidate failures; label resource/training-contract incompatibility separately from thesis incompatibility.
3. State that SpeechOcean762 is a repository-defined RQ1 extension, absent from the original proposal. Require verified label/task mapping before treating cross-corpus assessment as omission diagnosis.
4. Preserve the original proposal's intended Forced Alignment + GOP/acoustic-classifier route. For segmentation-free or other alternatives, document the substantive correspondence and proposed method change, rather than pretending the proposal demanded segmentation-free scoring or forbade it.
5. Remove claims that low GOP, blank, canonical Viterbi duration, DELETE supervision or a learned skip token independently prove missing-phone acoustics. Keep acoustic dependence and subtype/localization evaluation distinct.
6. Correct GECToR's evidential role to **text-GEC primary result / speech transfer hypothesis**. Its inclusion in the original bibliography does not grant a speech method priority.
7. Do not call the baseline pedagogically deployment-ready because it is the retained experimental anchor. Distinguish archived validation, historical authorization, source inspection, primary-paper claims and fresh empirical reproduction.
8. Present feedback/abstention and integration as separate unfinished evidence obligations. Do not prescribe a final architecture from documentary screening.
9. Keep the pre-existing source dirt, historical reports and evaluation semantics untouched. Any later pilot, inference, recalibration, implementation or TEST access requires a separate operator decision under P.

## 8. Limitations and exact restart state

- Original thesis was found and fully read, including its bibliography. Its cited papers were **not** all read; only L1–L3 primary landing-page/abstract/metadata evidence was accessed here. No full-paper reproduction, implementation-license review or paper-result verification is claimed.
- This lane did not locate or authenticate a separately supplied “latest omission proposal” or a separate RQ1 document. The earlier checkpoint records this gap; the original thesis is not silently substituted as proof of those documents' contents. Conclusions requiring them stop at this limitation.
- No original SpeechOcean annotation/schema/license source was accessed. S supports a repository-schema observation only; the proposed transfer task remains conditional.
- Source regions were inspected as current bytes, not audited in full. Current dirty nested files are not assumed equivalent to a historical evaluated revision. Byte hashes verify identities, not model correctness.
- No candidate passed a new experiment; no user-study, learner outcome, end-to-end integration test or numerical deployment SLO was verified. All future-benefit statements in the family matrix are hypotheses, not estimates or empirical results.
- **Research state:** no training/inference started; zero successful/skipped optimizer updates in this lane; no new or modified model artifact. The E summary identifies historical checkpoints only; weights were not opened.
- **Safe restart:** read this report and the coordinator's existing-process checkpoint; verify outer/nested HEADs and preserve dirty paths; verify T/split/E hashes; obtain missing proposal and primary SpeechOcean label/license evidence if those claims are needed. Continue documentary clarification only unless separately authorized. Do not execute any pilot, load models/audio, alter calibration/evaluation semantics, stage remote datasets or inspect sealed TEST data under this report.
- Coordinator owns the mandatory checkpoint/index update. Agent E writes no repository checkpoint or index and does not select the final architecture.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Complete original KLTN0127 read with verified hash and nontruncated coverage; Vietnamese requirement/evidence matrix; one G1–G8 gate applied to all requested families; feedback/integration obligations and residual uncertainty explicitly separated from empirical compliance."
    }
  ],
  "changedFiles": [
    "/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/723092bc-f51f-4a2a-8cd9-bf63b7c93d0b/reports/E_THESIS_ALIGNMENT_AUDIT.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "git rev-parse HEAD; git branch --show-current; git status --short; git -C repo rev-parse HEAD; git -C repo status --short",
      "result": "passed",
      "summary": "Verified requested outer/nested HEADs and recorded existing dirty paths without modification."
    },
    {
      "command": "Bounded Python text wrapping and SHA-256 checks of original thesis, frozen split metadata and archived E0/E1 summary",
      "result": "passed",
      "summary": "All 18 thesis lines read; L14 covered exactly [0,37473); thesis/split/summary provenance recorded."
    },
    {
      "command": "Bounded urllib documentation reads of ACL GECToR and arXiv GOP-SF v3/MoChA v2 landing pages",
      "result": "passed",
      "summary": "HTTP 200 primary abstract/version/license metadata inspected; no full-paper or experimental reproduction claimed."
    },
    {
      "command": "Model, unit/regression, training and integration tests",
      "result": "not-run",
      "summary": "Read-only documentary task; no model execution or experiments authorized."
    }
  ],
  "validationOutput": [
    "Original thesis SHA256 98fa922584b750510d4b0c183c59a84455da9e64ab231b59bdb58862c56133d3; 56472 bytes; 18 lines.",
    "No SpeechOcean mention in complete original; original explicitly separates conversational Whisper/STT from phoneme-level acoustic assessment.",
    "All family dispositions are screening judgments, not scientific PASS or implementation authorization.",
    "Report acceptance JSON and required family coverage validated with bounded Python checks; outer and nested staged-file lists empty. Final outer status additionally showed concurrent untracked closure-2026-11-03 artifacts, not created by Agent E."
  ],
  "residualRisks": [
    "Separate latest omission/RQ1 proposal not authenticated in this lane.",
    "Original SpeechOcean label semantics and asset licenses not independently inspected.",
    "Primary literature accessed at abstract/metadata level only; no experiment, model diagnosis, feedback or integration compliance reproduced.",
    "Pre-existing dirty repository/nested source preserved; historical source parity not inferred."
  ],
  "noStagedFiles": true,
  "diffSummary": "Only the isolated authorized report is created by Agent E; no repository source, index, checkpoint, configuration, dataset or historical report edited. Concurrent coordinator/other-lane documentary changes are outside this lane.",
  "reviewFindings": [
    "No final architecture selected.",
    "Substantive correction: SpeechOcean transfer and frozen/two-head/budget/platform constraints are project choices, not original-thesis mandates.",
    "Substantive correction: family compatibility, representational capacity and demonstrated acoustic/feedback compliance must remain separate."
  ],
  "manualNotes": "Full thesis read used bounded stdout wrapping only, with no temporary derivative. Coordinator owns the existing-process checkpoint/index update. Stop after documentary handoff; no inference, training, TEST access or implementation authorized."
}
```
