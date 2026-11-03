# Thesis-alignment audit — authoritative correction addendum

The original **KLTN0127 proposal** was located and fully read, not substituted by a research synthesis. Source: `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`, SHA-256 `98fa922584b750510d4b0c183c59a84455da9e64ab231b59bdb58862c56133d3`. Most content occupies physical L14; bibliography L15. [Independent Agent E report](independent/E_THESIS_ALIGNMENT_AUDIT.md) records nontruncated coverage, original Vietnamese passages, every family mapping and product-scope limitations.

## Authoritative requirements versus invented restrictions

| Original section | Actual requirement / intent | Consequence |
|---|---|---|
| §2.3 RQ1 | Acoustic phone-level detection of selected errors **and** whether generated guidance is accurate, clear and appropriate to detected error | Omission AP or a research ranking alone cannot close RQ1 |
| §§1,4(c) | “việc nhận dạng nội dung lời nói không đồng nghĩa với khả năng phát hiện chính xác lỗi phát âm”; conversational STT differs from pronunciation analysis | Whisper/equivalent is explicitly contemplated for dialogue, but transcript fidelity cannot prove phone correctness |
| §§3(b),4(c) | Reference-prioritized practice, dictionary/G2P expected phones, selected targets conditional on suitable labels | No requirement to diagnose every error in unrestricted conversation; inference cannot depend on gold annotation information |
| §4(c) | Examples: phone substitutions, final-consonant omissions, confusable pairs | Do not replace declared full target scope by a deletion-only surrogate; distinguish coda omissions from all deletions |
| §4(c) | Proposed forced alignment, then GOP **or** pretrained acoustic classification; GOP not independent proof of subtype | Preserve a defensible comparator. Segmentation-free alternatives need explicit rationale for changed alignment treatment, not a fictitious thesis mandate |
| §4(c) | Error classification must be verified against labeled references; low-confidence conclusions limited or retry requested | DELETE labels, skip mass, recognition edits or low GOP alone do not justify confident subtype feedback |
| §4(c) | Reference-phone/word suspicion → hint → explanation/practice → TTS model | Missing phone has no realized segment; do not invent a precise omission timestamp or observed tongue movement |
| §4(g) | Precision/recall/F1/confusion matrix, calibration/testing separated, prefer unseen speakers; compare same units/data | AP/PER useful registered research metrics, not replacements for target-event and feedback evaluation; repeated VAL is exploratory |
| §§2.2(c),4(d) | RAG supplies checkable teaching sources and limits unsupported conclusions | Retrieved advice can ground guidance, not manufacture acoustic evidence |
| §§4(a),4(g),5–6 | Web integration, history, permissions, asynchronous long jobs, error handling/recovery, measured time/cost | No verified universal <300ms CPU inference SLO. Architecture family cannot be rejected with fabricated latency |
| §3(b) limits | Use pretrained models; no foundation **language** model training from scratch | Does not ban every compact pretrained acoustic recognizer/AED/transducer |
| §4(f), bibliography [10] | GECToR cited for Writing/Grammar correction and explanation | Text-edit precedent is not validated speech-MDD evidence or a thesis-preferred acoustic architecture |
| §2.3 RQ2 | Multi-Agent Automated Test Generation and comparison to single-model/sequential self-review | Acoustic diagnosis, phonetic guidance and SpeechOcean assessment remain RQ1, not RQ2 |

**Project choices, not original mandates:** exact backbone revision/freeze, two-head CTC+binary contract, 3,600 updates, folds/seed/scheduler, remote T4/AMP/private storage and SpeechOcean762 transfer. They remain binding current operational/research policies; changing them requires explicit approval for a distinct experimental arm. SpeechOcean762 is absent from the original proposal but is an approved RQ1 extension in AGENTS.md. Its ordinal scores do not supply the same explicit natural-subtype labels as L2-ARCTIC; accuracy 0 may include incorrect/missed phones without separating the same events.

## One gate, every family

The G1–G8 documentary gate is defined once in [Agent E §4](independent/E_THESIS_ALIGNMENT_AUDIT.md) and applied to every bounded candidate in the [unified comparison](UNIFIED_CANDIDATE_COMPARISON.md). It covers targeted scope, acoustic/non-oracle evidence, honest error semantics/localization, fair evaluation, feedback/abstention, integration, methodological traceability and authorization.

- GOP (segmental or segmentation-free), acoustic recognition+DP, skip/edit alignment, compact subtype/edit supervision and reference-aware joint methods are **screen-compatible or conditional**, depending on the completed system and intended claims.
- Phonological attributes and TRAIN-only acoustic exemplars can complement validated detection/feedback, but attribute descriptions or retrieved similarity alone do not prove actual articulation or omission.
- MoChA is a conditional correspondence component; monotonic order alone lacks omission semantics. AED/RNN-T/alternate SSL models remain conditional on appropriate initialization, contract, evidence and burden; none is categorically thesis-invalid by family name.
- Calibrated hybrids need empirical complementarity and risk/coverage evaluation. TTS is aligned as a demonstration source, not automatically a reliable acoustic detector.
- Text-only pronunciation claims, gold-dependent inference, unsupported subtype certainty and comprehensive free-conversation promises fail the gate. A compliant future role is not demonstrated compliance today.

## Feedback and integration evidence still required

Use confidence appropriate to the **actual target**: recording adequacy, mapping confidence, detection and subtype are different. Suspected but unverified event → neutral practice hint/retry, not definitive diagnosis. A model-derived omission region is an uncertain neighborhood between realized neighbors, not an observed missing sound segment. “No targeted issue detected” is not “perfect pronunciation.” Legitimate variants and unreleased/coarticulated phones need safeguards.

Event-level evidence and explanation quality must be assessed separately. Blinded expert review must include false alarms, misses, wrong subtype, correct limited conclusions and abstentions, not only successes. Evaluate correctness, clarity and error-fit; do not claim learner improvement from an explanation-quality study alone.

Future service records need stable prompt/word/phone mapping, model/scorer/calibration provenance, uncertainty/abstention/error status, evidence separate from guidance, teaching-source/TTS provenance, task recovery and access-controlled history. These are derived integration obligations, not an implemented schema. No integration, latency, educational safety or product-readiness claim is verified by this documentation pass.
