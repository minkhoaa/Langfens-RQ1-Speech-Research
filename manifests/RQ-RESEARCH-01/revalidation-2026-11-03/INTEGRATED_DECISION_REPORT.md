# LANGFENS RQ1 — comparative research revalidation

**Disposition: PARTIAL delivery; provisional shortlist, NOT an architecture decision.** Five independent fresh-context Luna reports are available. Independent lane E blocked on an unavailable native supervisor-reply route after one recovery; its coordinator supplement is explicitly not independent. The original thesis was verified, but the separately requested RQ1 source and latest omission-detection proposal were not located. Therefore the requested six-independent-report/full-input-review completion gate is **not met**. The comparative synthesis below is usable for operator review, not formal completion or execution authorization.

**Research only:** no implementation, model/audio/label loading, TEST access, GPU/Kaggle/Colab execution, backward or optimizer update. Existing E0/E1 configurations, baseline checkpoints and evaluation contracts remain unchanged. New files are documentary outputs under the existing research process.

## 1. What we know

### Thesis—not a deletion-only surrogate—is the authority

The original UIT proposal §§2.2(b), 2.3 RQ1, 3 and 4(c),(g) requires reference-based phoneme acoustic assessment for target substitutions, final-consonant deletion and confusable pairs; confidence-aware, accurate/clear/suitable corrective feedback; pretrained models, labeled data and reproducible comparisons. It explicitly separates conversational STT from diagnosis and allows abstention/retry. [Thesis matrix](THESIS_ALIGNMENT_MATRIX.md) provides exact passages and gates.

It proposes investigating alignment/GOP or pretrained acoustic classification; it does not prescribe a final architecture. Its GECToR citation is in **Grammar**, not proof of a speech tagging solution. A pretrained recognizer used narrowly for reference-based MDD does not automatically expand RQ1 into general ASR. No verified thesis requirement imposes universal <300 ms inference, fixed 13–15 GB model memory, or a mandatory six-family implementation.

### E0/E1 evidence is negative for E1, not every alignment method

Archived Fold-1 validation: 300 utterances, two speakers NJS/RRBI, 9,956 canonical tokens: 8,799 correct, 1,012 substitutions and 145 omissions. The comparison below is from the pinned **same E0/E1 evaluator**; do not mix older Protocol O/Z/AP figures as if all evaluators were identical. [L2 in evidence matrix](SOURCE_EVIDENCE_MATRIX.md).

| Archived metric | E0 | E1 | E1−E0 |
|---|---:|---:|---:|
| Omission-vs-correct AP | .035020 | .032252 | −.002768 |
| Omission recall at diagnostic correct-token FPR ≤15% | .400000 | .365517 | −.034483 |
| Full binary AP | .317021 | .309293 | −.007728 |
| Greedy CTC PER | .236818 | .232365 | −.004453 |
| Mean skip mass, omitted phones | 0 | rounded .000000 | no demonstrated separation |

**E1 was trained.** Its epoch-6 training log/history records 3,600 successful updates and zero AMP skips; endpoint checkpoint SHA `3b686730…f079d24` matches evaluation metadata. The later benchmark was inference-only; that does not make its model untrained. No weight bytes were loaded/rehashed or remote artifact reverified here. Its zero skip mass is reported at rounded precision, not proof of mathematically exact zero in every token.

**Maintain the existing E1 NO-GO for cross-fold rollout and retain Baseline R.** E1 bundled null attention and a soft diagonal prior, so the comparison does not isolate either mechanism or prove all monotonic/probabilistic alignment fails. Better PER did not yield better MDD/omission detection. Training-history endpoint and later replay differ modestly; that remains a provenance/numerics issue, not grounds to replace baseline numbers.

DIAG-02B supersedes prior causal conclusions: 9/80 archive comparisons exceeded unchanged tolerance; corrected CTC blank AUC was .653 in 66 matched pairs (G=8), not >84% omission confidence. Gold-span summaries are diagnostic, not a deployable segmentation-free detector. Neither causal alignment bottleneck nor acoustic omission readiness is established. [L4].

### A major accessible 2026 result changes the historical shortlist

Cao et al.'s **Segmentation-Free Goodness of Pronunciation** was initially inaccessible to lane A through IEEE. Coordinator retrieved the open **arXiv v3 (5 February 2026)** and inspected primary text and author code. [S1](SOURCE_EVIDENCE_MATRIX.md).

It compares canonical sequence probability with graphs replacing a target phone by:
- **S:** any single phone;
- **SD:** a single phone **or no phone**;
- **SDI:** an arbitrary phone sequence.

It marginalizes paths rather than requiring one forced segment; normalized forward computation and expected activation occupancy address numerical/peakiness issues. CMU Kids mixed-real-error macro phone AUC .914 for SD versus .891 S provides actual deletion-inclusive evidence, but **not deletion-only AP or current L2-ARCTIC omission recall**. SpeechOcean PCC evaluates rating prediction, not omission detection. The author implementation exists, but has research-script assumptions, unverified checkpoint compatibility, no detected root license, and documented ambiguity restrictions.

This makes segmentation-free scoring a stronger concrete research candidate than “standard CTC Viterbi durations must identify deletions” or “GECToR-Speech is already the answer.” It does **not** establish the next Langfens architecture.

## 2. What the local codebase implies

| Actual seam | Consequence for method selection |
|---|---|
| `src/langfens_prep/model.py:434–449` freezes encoder and diagnosis; CTC/canonical attention/binary are trained | Reuse existing acoustic emissions first. No currently validated replacement-phone diagnosis head. A new loss changes training contract even if head count stays two. |
| `dataset.py:97–153`; `textgrid_parser.py:170–239`; `collate.py:25–68` | Natural correct/substitution/deletion labels exist. Deletions are omitted from realized CTC targets and masked from replacement identity; collator currently discards subtype. Training label volume is not proof of per-phone/speaker statistical sufficiency. |
| `alignment.py:16–70,73–205` | Pooling forces at least a frame; sequence DP reconciles gold canonical with prompt-G2P, **not acoustic realization**. It cannot simply be called an existing omission decoder. Audit direction/ties before any reuse. |
| `model.py:51–204,572–677` | E1 null key and diagonal bias are not a normalized monotonic edit path. Localization and acoustic absence are distinct questions. |
| `g2p.py` and Protocol Z path | Prompt-derived phones support deployable assessment; gold phone identities/times must remain isolated as evaluation/supervision. Reference variants/OOV can mimic errors. |
| Existing focal/weighted pilots (Step 18/19) | Generic binary imbalance reweighting already failed to establish benefit. An omission-specific objective is a different hypothesis, but still unproven. |
| Product `ai-service/app/prompts/speaking_grade.py:10` and speech evaluation service | Product transcript prompt explicitly cannot assess phonemes. Word edits/scalar score and an acoustic-failure fallback of 1.0 are not typed phoneme confidence. Integration needs an explicit evidence/uncertainty interface later, not a prompt claim or broad refactor now. |

A deletion has no physical segment or realized-phone identity to retrieve/classify. Canonical→realized omission maps to **restoring/inserting** the missing phone in pedagogical correction. A CTC blank is a frame-path symbol, not a missing-reference-token event. Repeated phones, allowed pronunciation variants, weak releases, silence/noise and 20 ms feature quantization create ambiguity. A canonical-only forced path must realize every target label; an explicit optional/reference-advance transition or alternative sequence is needed to hypothesize omission.

## 3. Common rubric and comparative ranking

**Mandatory thesis gate first.** All in-scope variants below must preserve reference acoustics, substitutions AND omissions, confidence and feedback. Scores rate bounded research candidates, not labels such as “all CTC” or “all retrieval.”

Scale 1–5; weights **academic evidence 25%, expected MDD/omission potential 25%, Langfens compatibility 20%, feasibility/compute 15%, corrective feedback 15%**. Total /100 = `5E + 5M + 4L + 3C + 3F`. Components are coordinator judgments after source review, centrally recomputed, not averages of inconsistent lane scores. No candidate has an observed Langfens win. Ratings are not calibrated probabilities or statistical confidence intervals.

| Priority rank | Candidate | E | M | L | C | F | /100 | Evidence uncertainty / research disposition |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Segmentation-free CTC GOP-SD; normalized variant and S control | 4 | 4 | 5 | 4 | 4 | **84** | Medium/high transfer uncertainty. **Conditional shortlist**: actual primary method + inspectable code; no local omission validation. |
| 2 | Calibrated dual-path acoustic score + frozen binary/sequence evidence | 3 | 4 | 5 | 5 | 4 | **82** | High uncertainty; correlated cues/overfit. **Conditional shortlist**, only if components add held-out value. |
| 3 | Small supervised reference-phone realization/existence + full-MDD subtype assessment | 3 | 4 | 4 | 4 | 5 | **78** | High uncertainty; labels exist, acoustic grounding and objective unproven. **Conditional shortlist direction**, no chosen head design. |
| 4 | Existing greedy free-phone CTC + canonical edit comparison | 4 | 2 | 5 | 5 | 3 | **74** | Medium uncertainty. **Control**, not a posterior deletion detector. |
| 5 | Compact acoustic KEEP/SUB/OMIT tagging/joint recognition | 2 | 4 | 4 | 4 | 5 | **73** | Very high; E independent lane missing and speech tagging superiority unverified. **Reserve**, not “GECToR validated for speech.” |
| 6 | Explicit probabilistic edit lattice / articulatory monotonic skip path | 3 | 4 | 3 | 3 | 4 | **68** | High engineering/identifiability uncertainty. **Reserve**, not needed for first pilot. |
| 7 | Phonological/articulatory attribute stream | 4 | 2 | 3 | 3 | 5 | **66** | Stronger substitution-explanation than omission evidence. **Complement/reserve**. |
| 8 | Classical forced-alignment GOP | 5 | 2 | 3 | 3 | 3 | **65** | Established comparator; omission-specific weakness. **Comparator**, not rejected as scope violation. |
| 9 | Directly supervised E1-like skip mass | 2 | 3 | 4 | 4 | 3 | **62** | High; no verified direct-loss omission result. **Reserve ablation** only. |
| 10= | Train-only acoustic retrieval | 4 | 2 | 3 | 3 | 3 | **60** | High omission uncertainty, index/license costs. **Substitution probe**, not mandatory. |
| 10= | MoChA/monotonic attention without edit state | 4 | 2 | 3 | 3 | 3 | **60** | Order/locality alone not absence. **Alignment control**, not standalone omission solution. |
| 12 | Alternate SSL backbone/phone recognizer | 4 | 3 | 2 | 2 | 3 | **58** | Domain/license/fair-budget uncertainty. **Defer** until scoring/objective deficit clarified. |
| 13 | TTS/reference-audio template as detector | 2 | 2 | 3 | 2 | 4 | **50** | High voice/prosody/variant confounding. **Defer primary detector**; demonstration stays in scope. |

### Ranking uncertainty and sensitivity

A ±1 change in one criterion changes total by 3–5 points; ±1 on both evidence and MDD changes total by up to 10 points. The 84/82/78/74/73 order is consequently **not robust**. These are decision-sensitivity ranges, **not empirical CIs**. Deletion-only performance estimates and directly comparable compute are absent for most papers. A reasonable operator can prefer the smallest supervised pilot over fusion, or scoring over new learning, without contradicting evidence.

Thesis gate failures are not scored into acceptability: text-only pronunciation diagnosis, deletion-only replacement of RQ1, foundation-model/general-ASR research and gold-time inference are NO-GO. Full AED/RNN-T replacement is **not shortlisted** because necessity, compatible initialization and bounded validation cost are unproven—not because every such model is inherently out of scope or known to diverge.

## 4. Tradeoffs, complements and conflicts

### Complementary combinations

1. **Existing CTC + segmentation-free scoring + frozen binary score** shares one encoder/emission computation. Score differences test whether acoustic sequence evidence improves the current ranking before architectural changes. SD supplies omission alternatives; S control checks substitution tradeoff. Small calibrated fusion is justified only after independent component signal.
2. **Supervised target verifier + realized-phone recognition** uses natural operation labels while keeping broad MDD. Recognition can suggest substitutions; explicit realization/omission confidence can support cautious missing-phone feedback. A three-way head, hierarchical subtype or edit tags are alternative realizations—not three systems to implement.
3. **Attributes or train-only exemplars + validated detector** can improve specific confusable-phone explanations. RAG supplies sourced practice guidance; it must not invent observed voicing/place errors. TTS demonstrates the target; synthetic reference similarity does not establish learner error.
4. **Edit graph + acoustic emission scores** can marginalize multiple alignments. This overlaps S1's concrete wildcard graph family; avoid inventing a second generic pair-HMM before establishing why S1 is insufficient.

### Conflicts and risks

- Forced canonical pooling versus true absence: a required-phone window may hide missing realization. Gold intervals are never runtime features.
- Prompt-conditioned recognition versus diagnosis: S5 improves aggregate F1/PER but lowers overall error recall and does not improve correct deletion diagnosis. Strong reference priors can erase the error being assessed.
- Null attention loss versus evidence: forcing a skip label can teach a phone/L1/position prior. Skip separation without acoustic dependence is not success.
- Fusion versus double-counting: CTC scores, greedy edits and binary outputs share a backbone; confidence cannot assume independent evidence.
- Retrieval versus isolation: store only the active fold's TRAIN; never use VAL/TEST to build exemplars. Acoustic embeddings remain subject to dataset licensing/privacy.
- More data versus label sufficiency: 2,464 train omissions is enough to attempt a small supervised hypothesis, not proof that every phone/L1 has adequate coverage. No Vietnamese validation speaker is present in this Fold-1 evidence; no Vietnamese effectiveness claim is justified.
- Published metrics versus actual objective: AUC, PCC, aggregate F1, PER, diagnosis accuracy and omission AP have different denominators. Retrieval's PER 104.08% is a real insertion-sensitive result, not universal impossibility. SpeechOcean lacks the same explicit deletion-label evaluation needed for omission claims.

## 5. Comparable, minimal pilot designs — NOT executed or authorized

Select **at most one initial scoring pilot and, only if justified, one small supervised pilot**. No requirement to implement six families or create new research phases. These are proposed add-on experiments under existing RQ1, not amendments to E0/E1 contracts. All gates and any minimum relevant effect require operator agreement **before** execution; prior historical targets such as AP .12/.15 are not silently adopted.

### Common comparison contract

- Same frozen Fold-1 TRAIN 2,399 / VAL 300 and sealed TEST. Same written-reference/G2P inputs, phone mapping, canonical token denominator, scorer version and candidate eligibility. Preserve existing Protocol O/Z separation: prompt-derived inference is deployable; gold identities/intervals only offline targets/diagnostics.
- Prefer existing verified cached emission outputs if available and whitelist-complete. Otherwise model inference needs a separately approved run; this pass did not execute it. First reproduce E0 **under the selected evaluator** without relaxing DIAG-02B tolerance or replacing historical baselines.
- Overall binary AP; omission AP on omissions-versus-correct; substitution AP on substitutions-versus-correct; recall/precision and achieved correct-token FPR. Report both error-type score semantics: omission ranking may use a dedicated score, whereas joint binary ranking uses total error probability. Exclude insertions from the existing canonical denominator; report insertion/path sensitivity separately rather than changing it.
- Maintain registered CTC greedy PER and blank-collapse guardrails; no decoder change silently substituted. New lattice/score results are auxiliary method outputs, not retrospective baseline PER.
- Threshold/checkpoint selection only as permitted on validation; same-set FPR15 figures are explicitly **diagnostic, selection-optimistic**, not calibrated deployment estimates. Use train-internal speaker-disjoint calibration for learned fusion where feasible; any validation adjustment is disclosed and cannot be called independent evaluation.
- Report per-speaker differences (only G=2 in Fold 1), correct-phone false alarms, word-final/consonant strata, confusable pairs, repeated-phone/variant cases, short-span uncertainty and coverage/abstention. Speaker-resampling intervals with two speakers cannot establish broad significance. No four-fold/Vietnamese/cross-corpus claim from Fold 1.
- Acoustic mismatch/canonical-only controls test incremental audio value. Counterfactual masks are sensitivity probes, not natural omission causal experiments. Do not reintroduce invalid projected-attention bypasses.
- Shared feedback review: blinded examples for true/false/missed substitutions and omissions, plus abstentions; assess correctness, clarity and fit to detected error. A template saying “expected /t/ was not reliably supported; try again” is safer than an unsupported articulatory diagnosis. No product readiness without this second RQ1 endpoint.

### P1 — concrete segmentation-free scoring versus honest controls

**Hypothesis:** optional-target graph scoring improves omission ranking at the same false-alarm budget without degrading substitutions/full MDD, relative to existing binary/greedy evidence.

Use identical frozen CTC emissions. Compare unchanged E0, greedy canonical/realized edits with uncertainty, self-aligned GOP, GOP-SF-S and GOP-SF-SD (normalized/scaled implementation where appropriate). SDI is optional only to test a specific insertion/path explanation. No backbone/head training. Do not transplant the author's training data, model or thresholds.

Before data execution, later approved tests must check repeated labels, empty/edge targets, length feasibility, graph path normalization/terminal states, numerical stability, expected-occupancy floor and the author's next-phone ambiguity restrictions. Source inspection alone is not correctness certification. Obtain code reuse permission/license or independently implement published mathematics after approval. Naive scoring all U targets multiplies the per-target DP cost; published per-target O(T(U+V)) is **not** total utterance cost. Measure encoder and scoring latency separately.

**Falsifier / NO-GO:** no omission gain at matched FPR; gains only from phone/length priors; poor repeated-phone coverage; substitution/full-task regression; or inconsistent DP/provenance. Positive screening keeps the direction open, not automatic architecture promotion.

### P2 — calibrated dual-path combination, only if P1 adds signal

**Hypothesis:** acoustic graph/posterior evidence contains complementary errors to the frozen binary head, not just a monotone transformation of its prior.

Compare component scores alone, a predeclared no-learning combination, and—only if separately authorized—a minimal calibration/fusion fitted on TRAIN-held-out speakers. Keep validation labels out of fusion fitting. Any fitted model/optimizer belongs to explicitly authorized remote training; no local fallback. Same input/model emission budget as P1; no new acoustic encoder.

**Falsifier / NO-GO:** no incremental validation ranking/operating benefit, calibration gains disappear with speaker separation, substitution errors worsen, or abstention hides difficult cases. If cues are redundant, stop at the simpler scorer. Do not stack retrieval/attributes merely to raise family count.

### P3 — one compact supervised acoustic realization/subtype direction

**Hypothesis:** retaining natural deletion identity in supervision improves both useful omission detection and subtype-specific feedback beyond binary BCE, while preserving overall/substitution performance and acoustic dependence.

Choose **one** implementation hypothesis after operator decision: hierarchical correct/realized-error/omitted target, compact KEEP/SUB/OMIT, or a directly supervised existence signal paired with existing binary detection. Direct skip-mass supervision is a reserve minimal ablation, not the default winner. Compare to a parameter-/initialization-matched binary control and existing frozen E0/E1 reports. Use original pretrained initialization, frozen backbone, independent state, matched optimizer/schedule and 3,600 successful-update primary endpoint; validation-selected best is separately reported. Do not warm-start from E1 or another fold. Changed objective/output/schema requires separate reviewed config, not baseline edits.

All training is later authorized remote T4 CUDA FP16 AMP with durable per-epoch checkpoints/readback, exact update/skip accounting and quota verification. Extra head FLOPs should be small relative to encoder, but training time/VRAM remain unmeasured. Archived E1 training logs suggest about 629 seconds job runtime in that environment—not a portable budget estimate for a new model. Historical generic 2.5-hour/13–15 GB claims are not evidence.

**Falsifier / NO-GO:** subtype/skip prediction learns canonical-only priors, no matched-FPR omission gain, full/substitution regression, poor confidence/coverage or failed artifact parity. A new class is not an alignment solution by itself. Broad rollout or product claims require independent later evidence and authorization, not this pilot design.

## 6. Explicit GO / NO-GO and next operator decision

**GO for consideration:** (1) concrete segmentation-free CTC/GOP scoring, (2) minimal calibrated dual-path evidence if complementary, (3) one compact supervised realization/subtype direction. The first is the lowest architectural commitment; the third is the most direct way to test unused natural operation supervision. They are a shortlist, not a chosen architecture.

**NO-GO now:** implementation or pilot execution; E1 Folds 2–4; inference gold-window clipping; treating blank/decoder edits as gold omission; unconstrained full recognizer replacement; deployment/phonetic feedback claims; changing baselines, losses or evaluation denominators without authorization. Synthetic waveform deletion and full attribute/retrieval pipelines are deferred, not universally disproven.

**Decision-relevant gaps:** independent lane E absent; separate RQ1/latest omission proposal absent; no new experiments; DIAG-02B parity unresolved; no same-contract candidate omission results; few validation speakers/no Vietnamese validation evidence; checkpoint/code licenses and total latency/cost not measured. The paper/source matrix distinguishes primary verification from metadata-only leads and unreproduced implementations.

**Recommended next move:** operator reviews this provisional shortlist and supplies the missing source/proposal (or explicitly confirms which existing document was intended). Repair the native lane-E supervision path before claiming six independent reports. Then, if authorized, choose P1 alone or one bounded P3 alternative—do not implement all families or combine speculative changes first. **STOP here; await operator decision.**

## Deliverable navigation

- Independent reports: [A](A_CTC_GOP.md) + [A provenance correction](A_PROVENANCE_CORRECTION.md), [B](B_ACOUSTIC_REALIZATION.md), [C](C_SUPERVISED_OBJECTIVES.md), [D revised](D_EDIT_ALIGNMENT.md), [F](F_RETRIEVAL_HYBRIDS.md). Raw report claims are qualified by the coordinator evidence matrix.
- Missing independent report: [E status + explicitly non-independent coordinator supplement](E_JOINT_MODELING_STATUS_AND_COORDINATOR_SUPPLEMENT.md).
- [Verified source/evidence matrix and corrections](SOURCE_EVIDENCE_MATRIX.md).
- [Original-thesis alignment matrix](THESIS_ALIGNMENT_MATRIX.md).
- [Process checkpoint and safe restart](PROCESS_CHECKPOINT.md).
