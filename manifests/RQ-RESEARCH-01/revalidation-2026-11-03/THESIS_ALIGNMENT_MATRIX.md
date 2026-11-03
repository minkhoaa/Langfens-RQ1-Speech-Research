# Original UIT thesis alignment matrix

**Authority:** `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md` (SHA-256 in evidence matrix). Almost all content occupies physical line 14; cite section headings, not invented line ranges. Reviewed completely, including bibliography. The requested separate RQ1 source and latest omission proposal were not found and are **not represented as reviewed**. Original thesis governs scope; repository AGENTS governs execution. Neither authorizes a new experiment by itself.

## Mandatory gate

| Requirement | Original thesis passage | Acceptance gate for every direction | Current implication |
|---|---|---|---|
| Reference-based phoneme acoustics | §2.2(b): “dựa trên dữ liệu âm học và thông tin phát âm tham chiếu”; §3(b): phoneme target errors, prioritize reference-text practice | Input audio + trustworthy reference/G2P; output tied to canonical phones; demonstrate incremental acoustic evidence | CTC/GOP/verifier/edit/retrieval all conditionally fit; text-only diagnosis fails. |
| Substitutions AND omissions/final consonants | §4(c): “thay thế âm vị, bỏ phụ âm cuối và những cặp âm dễ nhầm lẫn” | Evaluate substitutions and full MDD alongside omissions; do not optimize deletion-only detection at their expense | A deletion auxiliary can be a component, not the entire thesis. |
| Confidence and limited conclusions | §2.2(b), §4(c): consider confidence; low evidence → limit conclusions or ask retry | Calibrated/qualified evidence, ambiguity and abstention; no blank=omission or attention=confidence shortcut | Every family needs confidence validation. A raw score alone does not pass deployment gate. |
| Corrective feedback is part of RQ1 | §2.3 RQ1 explicitly asks whether guidance is accurate, clear and suitable for the detected error; §4(c): locate suspected phone, hints, explanation, practice/TTS | Return target phone/word, evidence limits, possible type, actionable correction, retry option; separately evaluate feedback accuracy/clarity/error fit | Detection AP alone is insufficient. RAG/TTS may explain/demonstrate; cannot fabricate acoustic findings. |
| Pretrained models, labeled data | §3 research limitations; §4(c) pretrained Wav2Vec2/GOP and labeled reference data | Reuse pretrained acoustics and authorized labels; no new foundation-model research | Frozen small heads/scoring best fit existing resources; narrow pretrained recognizer use is not automatically ASR scope expansion. |
| Reproducible comparable evaluation | §4(g): same data/unit for comparators, precision/recall/F1/confusion; separated calibration/test; speaker independence where possible | Frozen splits/denominators; documented versions/checksums; same inference inputs; fixed-FPR plus AP; validation-only selection, no TEST now | Keep E0/E1 contracts unchanged. Proposed pilots are separate, not retroactive redefinitions. |
| Practical web integration | §4(a), §5: frontend/backend/AI service separation; asynchronous tasks for costly AI; processing time/cost | Typed phone results, async status/error handling if needed, measured end-to-end latency/cost | Thesis has **no verified universal <300 ms SLA**. Do not reject methods based on invented timing. |
| STT is separate from diagnosis | §4(c): transcript is insufficient to conclude phone right/wrong; STT evaluated separately | Preserve conversational transcription; acoustic assessment remains a distinct evidence path | No silent replacement of phoneme assessment with Whisper text edits. |
| No general foundation/open-domain expansion | §3 limitations: not every free-conversation error; no foundation language model training from scratch | Narrow targeted read/reference assessment; bounded pretrained model/head adaptation | Full generic ASR/transducer research not shortlisted. Model family alone does not fail gate. |
| RQ2 boundary | §2.3 RQ2 and §4(e): multi-agent generation/review/validation/correction | Research orchestration agents here are tooling, not a new RQ1 multi-agent method | Keep acoustic MDD and generation-harness claims distinct. |

## Family/candidate gate decisions

“Conditional fit” is a **scope** judgment, not evidence of efficacy, operational approval, or deployment readiness.

| Family / candidate | Thesis scope | Missing evidence or condition | Research decision |
|---|---|---|---|
| A: classical GOP/self-aligned CTC GOP | Fits explicit proposed method | Misfit score not standalone diagnosis; forced alignment may squeeze absent phones; confidence and full-task metrics needed | GO as comparator, not omission solution by declaration. |
| A: segmentation-free GOP-SD/SDI | Fits phoneme acoustic matching and alignment research | S1 mixed-error/rating evidence not current omission AP; adapted graph/vocab/numerics/license audit needed | GO to shortlist for design; no implementation authorized. |
| B: existing CTC realized-phone decoding | Fits as acoustic submodule | Decoder deletions ≠ gold omissions; variant and path ambiguity; confidence | GO low-cost control. |
| B/C: supervised reference-phone existence/subtype | Fits if embedded in full pronunciation assessment | Label transport, audio dependence and replacement diagnosis unproven | Conditional shortlist; changed objective separately gated. |
| C: supervised skip loss | Fits targeted learning | Near-zero E1 skip result is not evidence that new loss helps; attention mass not calibrated absence | Reserve minimal ablation, not selected architecture. |
| D: explicit skip/edit path or articulatory monotonic alignment | Fits named alignment research | Need real omitted-reference transitions, calibrated path uncertainty; S8 only aggregate results | Conditional reserve, no generic DP/epsilon readiness claim. |
| D: standard forced CTC path alone | Fits as alignment comparator | Requires each target phone; cannot certify phone absence | NO-GO as standalone deletion mechanism, GO comparator. |
| E: compact acoustic edit tagging / joint recognition+assessment | Fits if learned from speech labels and preserves target breadth | No independent E lane; NLP analogy not validation; extra loss/head changes contract | Provisional reserve pending evidence/operator review. |
| E: full AED/RNN-T replacement | Can fit narrow MDD in principle | Current necessity, cost, compatible weights and benefit not established; high migration burden | NO-GO for this shortlist, on feasibility/evidence, not blanket thesis violation. |
| F: train-only acoustic retrieval / attributes | Fits acoustic evidence and confusable-phone explanation | No direct natural-omission benefit; index isolation, license and feature-map validity | Reserve/substitution-feedback complement. |
| F: TTS/reference-audio comparison | Fits demonstrations/reference evidence | Accent/prosody/template mismatch and absence confidence; measurement absent | Defer as primary detector; demonstration remains in-scope. |
| F: text lexicon/rules/RAG alone | Supports reference generation or pedagogical explanation | No learner acoustic evidence | NO-GO as pronunciation detector; GO only supporting role. |
| Any deletion-only end product or text-only “phoneme diagnosis” | Fails requested breadth/acoustic gate | Drops substitutions or invents acoustic evidence | NO-GO. |

## Execution and research gate are different

- Current action permitted: documentary research and comparison. No pilot execution, training, GPU/Kaggle work, TEST, architecture/source change or integration was performed/authorized.
- A different objective—even with the same two heads—requires explicit approved configuration. Three-way or edit outputs also require a distinct artifact/schema. No edits to E0/E1 or retrospective metrics.
- The historical `RQ-RESEARCH-01` top ranking and D-1/E-1 pilot authorization remain proposals, not approval. This revalidation does not silently implement or select them.
- Open source/proposal provenance and independent lane E must be disclosed before formal decision closure. The original thesis supports a shortlist, not preordained superiority of any approach.
