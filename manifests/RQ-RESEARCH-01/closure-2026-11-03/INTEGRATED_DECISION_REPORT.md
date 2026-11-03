# LANGFENS RQ1 — integrated research closure and experimental shortlist

**Disposition: documentary closure with explicitly accounted source limitations; no architecture selected.** This continues **RQ-RESEARCH-01**, preserves both historical sessions and appends corrections. Six independently authored fresh-context reports were collected and reviewed. Agent C's report was recovered from its successful write before a usage-limit terminal failure; one native retained recovery confirmed it unchanged. This is not a claim that all six original runs terminated cleanly, nor six independent experiments.

**Boundaries:** research/documentation only. No model implementation, fitting, inference/replay, training/backward/optimizer, GPU/remote execution, Kaggle deployment or sealed TEST audio/labels. No baseline/checkpoint/dataset/evaluator changes. No scientific performance PASS or final thesis/product completion. The [process checkpoint](PROCESS_CHECKPOINT.md) and [verification evidence](VERIFICATION_EVIDENCE.json) record actual execution and safe restart.

## 1. Which methods have the strongest verified evidence for our RQ1 task?

There is **no verified same-contract candidate winner**. Distinguish useful evidence categories:

1. **Concrete acoustic sequence-alternative scoring:** Cao et al. segmentation-free GOP provides the clearest inspected, deletion-capable scoring formulation with author code available for inspection: canonical sequence versus S/SD/SDI wildcard alternatives, marginalizing alignment rather than assuming a correct canonical segment. Table I's mixed-real-error CMU Kids macro phone AUC **SD .914 vs S .891, SDI .859**, with self-aligned control .905, supports studying SD with proper controls. It is **not deletion-only AP, Langfens omission recall or evidence SDI is best**. SpeechOcean PCC is ordinal assessment; supervised downstream models and a different acoustic training setup matter. Raw score/occupancy normalization, code license and repeat/neighbor conformance remain issues.
2. **Speech-native acoustic realization and joint objectives:** CoCA-MDD, Fu's text-dependent CTC model, Zhang's target-conditioned error/realization supervision and Wu's phone-recognition studies supply actual speech evidence. They motivate a compact natural-operation/realization experiment more credibly than text tagging alone. Different corpora/splits/backbones/budgets and aggregate endpoints prevent importing performance. Fu's aggregate F1 improves while correct deletion diagnoses fall; a better global recognizer or classifier does not imply better omissions.
3. **Existing free-phone recognition + reference edit comparison** is the most directly available weight-preserving control, with real speech precedent. Its gaps remain recognition hypotheses, not human omission labels. It should expose both recognized alternatives and ambiguity rather than assert acoustically observed absence.
4. **Phonological attributes** have inspected speech-specific substitution/feature evidence and explanatory potential. They are a complement to validated event detection, not established omission rescue or measured articulator motion. **Retrieval** is scientifically viable but lacks the needed omission breakdown: its high insertion-sensitive PER and canonical F1 are compatible measurements, not proof of invalidity.

**GECToR is verified text GEC only.** Its official Apache-2.0 code and structured edit idea do not validate “GECToR-Speech,” acoustic grounding, CPU/T4 latency or convergence. A compact acoustic KEEP/SUB/OMIT tagger is a new hypothesis, and overlaps ordinary subtype supervision. The thesis cites GECToR under Grammar, not as the chosen pronunciation architecture.

Primary source/revision/table/license details: [evidence matrix](SOURCE_EVIDENCE_MATRIX.md), [independent C](independent/C_PRIMARY_LITERATURE_VERIFICATION.md), [independent Lane E reconstruction by A](independent/A_INDEPENDENT_LANE_E.md).

## 2. Which methods are genuinely compatible with Langfens?

**Structurally low-disruption:** frozen-emission segmentation-free scoring, self-aligned/segmental controls, and free-phone+DP can preserve exact E0 weights. They still need authorized new scorer/decoder code, correct phone/blank mapping, reference variants, numerical/path tests, score orientation and full-coverage evaluation. Source compatibility is not executed load/parity or a license to copy author code.

**Label-compatible but approval-dependent:** compact correct/substitution/deletion, hierarchical realization or edit targets. Natural annotations retain subtype identity, while official collation supplies binary/CTC/masked diagnosis, not subtype targets. CTC already learns chronological **realized** phones (deletions excluded; valid additions retained). Binary already labels deletions as errors. A new objective preserves subtype identity; it does not introduce previously nonexistent deletion supervision.

**Important constraints:** CTC IDs0blank/1–39phones differ from diagnosis IDs0–38. Diagnosis is frozen/untrained. Strict same-base starred substitutions can be errors despite equal normalized phones; unknown `err` has no valid replacement; canonical token tags do not encode arbitrary insertions. E0/E1 binary widths896/897 and skip parameter differ; new heads require explicit partial initialization and fresh training state, not strict resume. Config-only diagonal bias also needs provenance even when shapes load.

**Conditional broader families:** explicit edit lattices, reference-aware joint models, attributes, retrieval banks and alternate recognizers require separate objective/schema/index/initialization/license review. None is categorically excluded by thesis or fabricated resource figures. Current two-loss/frozen-backbone/budget policies remain binding until explicitly amended for a distinct experiment.

B discovered a static blinded-collator uninitialized variable and legacy schema/examples inconsistent with executed CTC/diagnosis semantics. These are documented, **not fixed**; the existing Z evaluator bypasses that collator, so the issue does not invalidate its archive. Full details: [source/data audit](SOURCE_DATA_COMPATIBILITY_AUDIT.md).

## 3. What do E0/E1 failures actually establish?

Archived same-harness Fold1:300 utterances, NJS/Spanish and RRBI/Hindi;9,956 canonical tokens:8,799 correct,1,012 substitutions,145 deletions.

| Registered archived metric | E0 | E1 | E1−E0 |
|---|---:|---:|---:|
| Omission AP, binary-error score on deletion vs correct | .035020 | .032252 | −.002768 |
| Omission recall at diagnostic correct FPR≤15% | .400000 | .365517 | −.034483 |
| Full binary AP | .317021 | .309293 | −.007728 |
| Greedy realized-phone PER | .236818 | .232365 | −.004453 |

**E1 was trained** to3,600 successful updates,0 skips, according to inspected history/endpoint/log; later evaluation was inference-only. Its reported mean skip mass on deletions rounds to.000000, not proof of mathematical zero for every token. The tested skip+soft-diagonal bundle did not achieve its desired Fold1 omission/full-MDD tradeoff despite improved PER. Retain its existing **NO-GO for Folds2–4 rollout** and retain E0 as research reference, not a pedagogically certified product.

What this **does not** establish: a unique causal failure mechanism; insufficiency of all acoustic evidence; failure of every alignment, skip-state, supervised subtype or joint model; impossibility of useful phonological/retrieval methods; or broad speaker/L1 conclusions. Two repeatedly reused validation speakers are not a population study.

**Denominators matter:** omission AP population is145+8,799=**8,944**, prevalence1.6212%; substitutions excluded. Historical PER uses9,881 realized reference phones, not canonical count. At FPR≈.149903, archived rates reconstruct58/145 versus53/145 omission hits with1,319 correct-phone false alarms each: omission-subset precision≈4.21%/3.86%, not a safe feedback guarantee. Dedicated deletion probability, generic error score, recognition gap and skip attention must be named separately.

**DIAG-02B controls causal interpretation:**9/80 archive parity failures persist; matched blank AUC≈.653 in66 selected pairs is modest, gold-interval-conditioned evidence, not a deployable posterior. Original forced-attention intervention bypassed projections. The old1.5×blank gate is impossible on matched correct mean.732156 (requires1.098234>1). No universal blank/duration prerequisite survives. Full corrections: [scientific protocol report](SCIENTIFIC_PROTOCOL_CORRECTION_REPORT.md).

Historical E0 values from other evaluator versions/populations remain unchanged; no new “correct baseline” overwrites .316062/.317012/.317021. Full-precision E1 predictions are absent from inspected archival outputs; AP/paired uncertainty cannot be independently reproduced from rounded aggregates here.

## 4. Which candidates deserve controlled experiments?

**Two independent primary directions, without selecting their eventual architecture:**

- **P-SCORE:** frozen-emission GOP-SF-SD/numerical/occupancy variants versus S, self-aligned/segmental GOP, free-phone DP and unchanged binary evidence. SDI only as a named insertion/flexibility contrast. Test tiny-graph conformance/repeats/edge targets before any approved real-data run. No assumption canonical Viterbi localizes a missing sound.
- **P-SUP:** one compact natural-subtype/existence/realization treatment **or** equivalent acoustic edit tags, compared with a contemporaneous parameter-/routing-/data-/initialization-/successful-update-matched binary control. Isolate objective from routing; retain strict same-base/unknown masks; test canonical-only and matched/mismatched audio. No requirement that P-SCORE pass first.

**Conditional reserve:** normalized edit/skip alignment or compact reference-aware joint hypothesis when a specific unresolved mechanism justifies extra burden; attributes/feedback evaluation as a separate RQ1 endpoint; retrieval/alternate recognizers only with a specific necessary contrast and source/index/privacy controls. **P-FUSION** is warranted only when components actually add complementary held-out evidence.

[Unified comparison](UNIFIED_CANDIDATE_COMPARISON.md) applies one thesis gate and common rubric to **19 bounded candidates/controls** across all families. Balanced readiness scores put SF-SD81, existing decoder control79, compact subtype75, conditional fusion74. Those are **subjective research-readiness judgments**, not performance. Minimal-change weights put decoder control first; feedback-heavy weights put supervised/fusion hypotheses first and raise attributes. One criterion±1 changes3–5points; the leading order is not robust. No winning architecture follows from the table.

[Parallel pilot specifications](PARALLEL_PILOT_SPECIFICATIONS.md) define prompt-only inference, exact registered deletion-vs-correct denominator, legacy versus subtype score semantics, full-coverage handling, speaker-disjoint fitting/selection/calibration, validation reuse, fixed/best checkpoint strata, acoustic controls, falsifiers, feedback review and future resource measurement. Practical-effect/noninferiority margins must be approved/frozen before execution; historical .12/.15 targets were proposals, not measured expectations. All model/optimizer fitting remains separately authorized remote-only; no experiments are authorized here.

## 5. Which combinations are scientifically justified?

**Justified to test, not proved synergistic:**

1. Same frozen CTC emissions + target-alternative scoring + existing binary evidence: shared computation and genuinely different score hypotheses; compare components alone and calibrated/no-learning fusion. Shared encoder means correlated evidence, not independent confidence.
2. Error/subtype supervision + identifiable realized-phone auxiliary targets: targets can provide complementary constraints. Control objective weights/capacity and mask unknown/deleted realization labels; confirm audio dependence rather than prior shortcuts.
3. Explicit edit alternatives + acoustic emissions: normalized paths can express uncertainty/absence. Compare to concrete SD before inventing redundant lattices.
4. Validated event/substitution evidence + phonological/RAG/TTS practice guidance: improves explanatory representation, conditional on independently reviewed correctness/clarity/error-fit and calibrated abstention. Teaching knowledge does not prove the learner's acoustic error.

**Potential conflicts or redundant objectives:** canonical-only forced pooling versus semantic absence; reference copying/strong LM versus error-preserving recognition; forcing attention onto a fictional deletion segment; full-error detection versus subtype-only scores; strict same-base errors versus KEEP-by-phone-identity; arbitrary max/.90 fusion versus calibrated false-alarm control; unnormalized skip/edit priors versus probability claims; retrieval index pooling versus per-fold sealing; too many auxiliary losses/search trials versus fair budget and interpretable attribution. No pair is “100% complementary” merely by diagram.

## 6. What remains unknown?

- **Required source gap:** no distinct latest omission-detection proposal located in the documented project search. Nearest DIAG-01/E1 proposals are not substituted. Operator must identify/provide exact source before any proposal-conformance claim. Original thesis is found and reviewed.
- Joint-Align primary full text blocked IEEE420; detailed mechanism/compute claims excluded. Cao/Fu code licenses unresolved; Najoji author linkage/license unresolved; retrieval official runnable pipeline not established; preprint/publisher and later phonological-version parity not assumed.
- No same-contract candidate omission/full/substitution performance, acoustic grounding, calibration, localization, insertions, ambiguity coverage or resource measurements. DIAG-02B parity/drift and E1 full-prediction reproducibility remain unresolved.
- Sparse subtype/phone support, unknown realizations, pronunciation variants/reduction ambiguity, current fold absence of Vietnamese validation speakers and small/reused speaker sets limit generalization. Counts alone do not prove sufficient supervision.
- No verified error-conditioned feedback quality, learner benefit, end-to-end integration safety or intended-environment latency/memory. A retained baseline is not diagnosis-ready deployment.

These are accounted limitations, not evidence that every candidate will fail. The closure does not invent sources or force consensus on unresolved hypotheses.

## 7. What exact evidence would justify selecting a final architecture?

Selection would require **all** of the following, under separately authorized prospective contracts:

1. **Reproducible artifacts and inference:** immutable source/config/weights/split/G2P/scorer identities; full-precision item predictions and failures; original-tolerance baseline provenance/parity resolved or explicitly scoped; no gold information in runtime. Correct graph/label/strict-error/coverage behavior demonstrated, not only source inspected.
2. **Fair causal comparison of candidate changes:** same allowed data/inputs, fit/selection/calibration opportunities, matched successful updates/seeds/initialization and explicit copied/reinitialized states. Objective-only versus alignment-only ablations where needed; every attempted arm reported. Fixed-budget and selected-best exposure separated.
3. **Meaningful full-task tradeoff:** prospectively agreed omission AP/recall/precision improvement at achieved correct-FPR/coverage, with full/substitution noninferiority margins and insertion/strict-variant checks. Compare to E0 plus relevant S/SA/DP/simple-supervised controls, not only the weakest historical heuristic. No promotion by calibration shift alone.
4. **Independent speaker and robustness evidence:** more than two reused speakers; protected speaker-disjoint confirmation with calibration/search isolated, multiplicity and overlapping-fold limits disclosed. Target-group/coda/confusable-pair and deployment-language evidence adequate to intended claims. TEST remains sealed until separately authorized; reused validation cannot be renamed confirmation.
5. **Acoustic grounding and honest uncertainty:** gains beyond canonical/phone/position priors and mismatched-audio controls; correct semantic distinction of model non-emission/absence; calibrated confidence, abstention and mapping uncertainty with risk/coverage and false reassurance measured.
6. **Second RQ1 endpoint:** blinded domain-informed feedback evaluation showing accurate, clear, appropriate advice for true/false/missed errors and abstentions, with sourced teaching guidance and checked demonstrations. No unsupported articulatory-cause or learner-improvement claims.
7. **Practical and legal readiness:** actual intended-workload encoder+scorer/end-to-end latency, throughput, peak memory/cost/failures; permitted code/data/weights/teaching sources; reliable private artifact durability and web failure/history/access behavior. User approves a tradeoff on the resulting Pareto frontier rather than a fabricated universal score.

A final choice can remain inconclusive or favor a simpler system. The evidence must justify the **specific architecture and intended claims**, not an entire family or report majority.

## Deliverables and stop boundary

- Six substantive reports: [A / independent Lane E](independent/A_INDEPENDENT_LANE_E.md), [B / source-data](independent/B_SOURCE_DATA_COMPATIBILITY.md), [C / primary literature](independent/C_PRIMARY_LITERATURE_VERIFICATION.md), [D / protocol](independent/D_SCIENTIFIC_PROTOCOL_CORRECTIONS.md), [E / thesis](independent/E_THESIS_ALIGNMENT_AUDIT.md), [F / cross-session](independent/F_CROSS_SESSION_CORRECTION_LEDGER.md); [C recovery confirmation](independent/C_RECOVERY_CONFIRMATION.md).
- [Evidence/provenance matrix](SOURCE_EVIDENCE_MATRIX.md), [thesis audit](THESIS_ALIGNMENT_AUDIT.md), [compatibility audit](SOURCE_DATA_COMPATIBILITY_AUDIT.md), [protocol corrections](SCIENTIFIC_PROTOCOL_CORRECTION_REPORT.md), [correction ledger](CROSS_SESSION_CORRECTION_LEDGER.md), [candidate/sensitivity comparison](UNIFIED_CANDIDATE_COMPARISON.md), [parallel pilots](PARALLEL_PILOT_SPECIFICATIONS.md).
- [Checkpoint](PROCESS_CHECKPOINT.md), [verification receipt](VERIFICATION_EVIDENCE.json), `SHA256SUMS`; global index and original phase checkpoint receive append-only closure pointers.

**STOP after documentation.** Research closure is not final architecture selection, pilot authorization, model implementation, GPU deployment, scientific PASS or completion of RQ1 empirical/feedback requirements. No experiments without a separate explicit operator decision.
