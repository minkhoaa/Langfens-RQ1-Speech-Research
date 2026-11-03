# Comparable parallel pilot specifications — NOT executed

These are proposed experiments within existing RQ1, not new phases, implementation authorization or architecture selection. Freeze the operator-approved registration before any future execution. Source/source-license and numerical prerequisites are shared scientific safeguards, **not** a requirement that a scoring pilot succeed before supervised research can begin.

## 1. Common contract

### Inputs and isolation

- Frozen split SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`. Initial screen: Fold 1 TRAIN 2,399 utterances / VAL 300 (NJS, RRBI). TEST audio/labels remain sealed and unstaged. No use of cross-fold checkpoints as pretrained initialization.
- Runtime inputs: waveform + written reference, dictionary/G2P version and any declared pronunciation variants. **No gold TextGrid phones, boundaries, word timings, error identity or realized transcription may enter runtime inference.** A separate offline scorer can read authorized validation labels after predictions are frozen. Training supervision can use authorized TRAIN labels, never VAL/TEST examples.
- Historical Protocol O uses annotation-derived canonical identities; it is a useful oracle-reference diagnostic, not a wholly gold-free deployment contract. Historical E0/E1 O and Z results stay unchanged. Their replay is provenance work, not permission to make a new gold-dependent inference path.
- New candidates and a matched E0 control must share a prompt-only runtime. Report the historical registered numbers as a separate archival table, not as interchangeable with new prompt-only results.
- Pin audio preprocessing, 39-phone normalization, blank ID, prompt/G2P version, variant policy, DP tie order, score orientation, coverage handling and evaluator revision. Missing/invalid inputs produce recorded failures/abstentions, not removal of difficult cases.

### Evaluation populations and score semantics

- Full binary: all 9,956 canonical evaluation tokens; error = substitution OR deletion. Omission: exactly 145 deletions versus 8,799 correct tokens (**N=8,944**, substitutions excluded). Substitution: 1,012 substitutions versus 8,799 correct (**N=9,811**). No candidate-specific omission denominator. Insertions are separate outputs, not silently added to historical canonical-token metrics.
- Every arm declares a full-error score `s_error` and, where available, a separate `s_omit`. Historical E0/E1 omission AP uses the same sigmoid binary-error score as overall AP. For a three-way model, `s_error=1-p_correct`; `s_omit=p_omit` is an explicitly additional endpoint. Report both on the deletion-versus-correct population; never claim a like-for-like score comparison without naming the difference.
- A negative GOP score can be oriented toward error (for example `-GOP`); a likelihood ratio, DP gap, skip attention and subtype softmax are **not automatically calibrated probabilities**. AP is a ranking measure, invariant to strictly monotone transforms; learned fusion and threshold interpretation are not.
- Prompt/G2P reconciliation is offline only. The E0/E1 archive has 9,919 aligned Z tokens, not full coverage. For future full-population comparison, freeze one common full-gold coverage policy before inference: use the existing TRAIN-only prior fallback concept (B1) for full-error scores, with frozen TRAIN-derived subtype priors for omission-specific outputs if required. Publish exact priors, unmatched-token counts, aligned-only results and a conservative abstention sensitivity analysis. These are **new add-on results**, not retrospective changes to the E0/E1 evaluator. If raw scores are not on a probability scale, specify TRAIN-only calibration or a rank-compatible missing-score policy first; otherwise report explicit coverage bounds and do not claim full-population superiority. No VAL-prevalence imputation or favorable case-dropping.
- Report tied-score AP using the existing grouped-tie convention; binary DP decisions have limited ranking resolution. Also report ROC-AUC, precision/recall/F1/confusion counts, achieved correct-token FPR, omission recall at diagnostic FPR ceilings 5/10/15%, coverage/abstention, and substitution/full-task guardrails. Do not equate FPR with false-discovery proportion.
- Keep registered greedy CTC PER versus the realized-phone reference (9,881 phones for historical Fold 1), including recognized insertions. Beam/alternate-lattice PER is an auxiliary result; a recognition deletion is not automatically a learner omission.

### Calibration, checkpoint selection and uncertainty

- Reserve speaker-disjoint TRAIN speakers for calibration/selection, deterministically identified in a later signed registration with label counts. Fit score fusion, temperature/logistic calibration, weights and operating thresholds on that partition only. Models used to produce calibration predictions must not have trained on those held-out speakers; alternatively use speaker-wise out-of-fold TRAIN predictions, explicitly accounting for additional compute. Refitting on all TRAIN invalidates those calibration predictions unless that transfer is separately tested/disclosed.
- Frozen-checkpoint scoring already uses a model trained on all TRAIN; any held-out-TRAIN calibration for it is therefore **in-sample with respect to acoustic training**. Disclose this limitation rather than pretending calibration is fully independent. VAL can still supply a two-speaker exploratory assessment, not confirmatory deployment evidence.
- For a fair supervised comparison, both treatment and newly trained control use the **same** reduced fitting partition, selection/calibration partition and successful-update budget. Compare separately to archived full-TRAIN E0; do not ascribe training-data differences to the objective. A full-TRAIN exact-baseline-budget experiment is another registered stratum, not silently mixed with this one.
- Primary exposure: 3,600 successful updates for each trainable arm, independent original pretrained initialization and independent optimizer/scheduler/scaler/RNG state. Same backbone freeze/eval mode, seed(s), batch 2, accumulation 2, warmup 100 and 25-epoch cosine horizon. If the fitting subset changes updates/epoch, endpoint remains update-defined; report data exposure and elapsed epochs. Early stop must not fabricate a missing fixed endpoint. Validation-selected `best.pt` is separate, selection-optimistic evidence; label-selection use is identical across arms.
- Existing policy freezes two-head CTC+binary baselines. New subtype/edit/attribute/lattice objectives are **approval-dependent amendments for distinct experimental arms**, not changes to E0/E1. Config review must explicitly authorize them before implementation/training. No local fitting/training fallback.
- For the two-speaker Fold-1 screen, publish NJS and RRBI results and paired deltas, counts and disagreements. Cluster intervals at G=2 cannot support population significance. More tokens or seeds do not create more independent speakers. Any bootstrap is labeled exploratory; no token-wise Welch/KS p-value as architectural acceptance proof.
- Predeclare the candidate set and one primary contrast per pilot, report every attempted arm and failure, and use the same selection opportunities. If later confirmatory testing is authorized, freeze one shortlist and use speaker-disjoint untouched evaluation. Repeatedly reused VAL and cross-fold results are exploratory/model-selection evidence; overlapping fold training sets also limit independence.

### Advancement criteria (proposed design, not predicted outcomes)

Before execution, the operator must approve practical-effect and noninferiority margins for omission AP/recall, full/substitution AP and acceptable feedback false alarms. Historical absolute AP .12/.15 or recall .55 targets were hypotheses, not established attainable benchmarks. Do **not** optimize until one of those numbers is reached or redefine success after seeing results.

For initial screening, present a Pareto table of omission AP, recall/precision at matched achieved FPR, full/substitution AP, acoustic dependence, coverage and cost. Advance only with a reproducible favorable tradeoff larger than known numerical drift, not merely a calibration shift or one-speaker win. A result can remain inconclusive. Statistical superiority and final architecture selection require the independent evidence listed in the integrated report, not Fold-1 screening alone.

Acoustic controls for each applicable arm: prompt/phone/position-only comparator; matched audio versus deterministic within-speaker cyclic mismatch with no fixed points; phone/position strata; repeated phones/variants; short/coarticulated but realized controls; silence/noise. Feature/waveform masking, if separately authorized, is a sensitivity probe, not a causal natural-deletion experiment. Reproduce original head/projection semantics; never bypass them as DIAG-02 did.

All new optimizer-based training/fitting is authorized remote T4 CUDA FP16 AMP only, with quota verification, private staging and atomic per-epoch durable checkpoints with SHA-256 readback. No GPU-time/latency/memory budget is asserted as measured. A future approved measurement must separately record encoder/scoring/serialization latency, hardware/precision, duration distribution, p50/p95, throughput, peak memory and failures. Training-free scoring still has inference and DP cost.

## 2. Independent pilots

### P-SCORE — segmentation-free acoustic graph scoring

**Question:** Does explicitly admitting target deletion improve ranking beyond the same emissions without that alternative?

- Reuse identical frozen E0 acoustic emissions; change no weights. Compare E0 binary score, greedy realized-phone + deterministic canonical DP, classical/CTC self-aligned GOP, GOP-SF-S and GOP-SF-SD. Add SDI only as a predeclared insertion/path-flexibility contrast, not a presumed upgrade. Duration/blank-only scores are weak diagnostic controls, never an omission oracle.
- Distinguish normalization of forward computation (numerical stability) from activation-occupancy score normalization. Include the published occupancy-normalized SD variant as a named ablation. Orient scores consistently.
- Before real-data execution, later authorized correctness tests must cover exact tiny-graph path enumeration; repeated identical labels and blank separation; edge/empty contexts; feasible/infeasible target lengths; target/next-phone ambiguity restrictions; epsilon deletion; no accidental deletion of neighbors; normalized mass/terminal states; log-space/scale stability; occupancy floor; and all-canonical-position rather than single-position cost.
- Controls share emissions and mapping, so isolate scoring rather than encoder quality. Published SD mixed-error AUC on CMU Kids is not the expected Langfens AP. Do not use the author's label map/checkpoint without auditing inventory, blank/silence conventions and license.
- **Falsifiers:** no gain versus S/self-aligned controls; gain disappears with speaker/phone matching; catastrophic repeated-phone ambiguity; degraded substitutions/coverage; reliance on unavailable gold contexts; numerical path bugs. Failure restricts this scoring hypothesis, not supervision or every edit lattice.
- **Dependency:** only shared source/numerical/authorization gates. P-SUP can run independently; it need not await P-SCORE results.

### P-SUP — compact natural-operation/realization supervision

**Question:** Does exposing existing deletion identity to a compact acoustic classifier add useful information beyond binary BCE, without learning only canonical priors?

- Choose and freeze **one** primary treatment before implementation: correct/substitution/omission classifier, hierarchical realization+error model, or compact KEEP/SUB/OMIT tagger. These are alternative parameterizations of closely related supervision; do not count all as independent conceptual evidence.
- Primary control: newly initialized parameter-/routing-/budget-matched binary model. Treatment: same acoustic and reference inputs, change target/objective only. If a routing/skip change is also desired, use a predeclared factorial or separate ablation; otherwise the cause of a gain is confounded.
- Optional replacement-phone prediction uses only valid realized-phone targets, excludes deletion/unknown identities with a mask, preserves strict same-base substitutions, and does not identify KEEP solely by phone equality. Insertion detection needs inter-token slots or a realized-sequence decoder; a per-canonical 41-label head does not encode arbitrary insertions.
- Loss: independently weighted CTC realized-sequence loss plus operation CE/hierarchical loss. Weights are selected within the same limited TRAIN-only search for control/treatment. Do not silently import arbitrary deletion weights, focal gamma or a CTC coefficient different from the baseline and attribute results to tags alone.
- Audio grounding: train/evaluate prompt-only and matched/mismatched controls, inspect per-speaker/phone/position error ranks, and assess calibration. High skip/subtype separation without incremental audio evidence is not success.
- **Falsifiers:** gains confined to text priors; only changed sigmoid threshold; no held-out omission benefit at comparable false alarms; subtype gain at full/substitution collapse; invalid source/checkpoint parity. Label supervision creates a learnable target, not proof of acoustic localization.
- **Dependency:** none on P-SCORE. Exact label transport/schema review and distinct-arm authorization are required.

### P-ALIGN — explicit probabilistic edit/alignment comparison (reserve)

**Question:** Is marginalizing deletion/edit paths better than a forced canonical or hard DP alignment under identical emissions?

- Compare ordinary canonical CTC forced path, greedy recognition DP and a declared normalized skip/edit graph. If GOP-SF already supplies the relevant hypothesis, reuse that comparison instead of inventing redundant machinery.
- A canonical advance with no acoustic emission models a missing target; ordinary CTC blank advances time without emitting a phone and is not the same transition. State topology, normalization, repeated-phone behavior and insertion budget must be explicit.
- If trained alignment losses are studied, compare binary-only control, subtype-only, routing-only and combined variants within equal budgets where feasible; gold TRAIN boundaries may supervise but cannot enter inference. Do not force attention onto a physically nonexistent deletion interval.
- **Falsifiers:** apparent gain is only path-prior/phone-position bias; localizes fabricated canonical segments; no benefit over simpler SD; state explosion or unstable marginalization. No claim that canonical Viterbi durations must shrink on true omissions.
- **Dependency:** no scientific need to await P-SCORE or P-SUP; reserve status reflects unverified implementation burden, not experimental disproof.

### P-FEEDBACK — phonological/diagnostic complement (independent secondary endpoint)

**Question:** Do validated replacement/attribute outputs improve feedback accuracy, clarity and fit to detected error?

- Compare a generic correct-target practice template with diagnosis-conditioned phonological guidance. Hold detector predictions fixed to isolate explanation quality. Attribute model training, if needed, is a distinct authorized arm with suitable controls; a static target-phone trait dictionary is not evidence of observed learner articulation.
- Blinded, stratified review includes correct detections, false alarms, misses, substitutions, omissions and low-confidence abstentions; do not curate only successes. Record source provenance for practice guidance and prohibit unsupported physical-cause statements. RAG/TTS provides guidance/demonstration, not independent error evidence.
- SpeechOcean762 is a separately authorized cross-corpus rating/feedback assessment; its score labels cannot be relabeled as the same natural-deletion AP. No corpus audio/TEST access in this documentation pass.
- **Falsifiers:** more specific but less correct explanations; confidently wrong omission claims; failure to abstain; no improvement over simple guidance. This endpoint is required before product readiness, but not a dependency blocking independent detector pilots.

### P-FUSION — complementarity test, not an automatic ensemble

**Question:** Do two already evaluated score streams add held-out value rather than double-count shared-backbone evidence?

- Compare both components alone, a predeclared fixed combination, and a small TRAIN-calibrated fusion with matched fit/search budgets. No hardcoded `DP deletion -> 0.90`, arbitrary max fusion or assumption of independent scores. Include calibration-only E0 to distinguish ranking gains from threshold shifts.
- Freeze features/regularization/calibration splits before VAL. Keep all folds' retrieval/example banks TRAIN-only. Full retriever, alternate backbone/AED/transducer and TTS-template methods remain conditional reserve experiments: first specify compatible initialization, licensed source, a necessary hypothesis and an equal-budget control; do not reject them through invented latency or convergence numbers.
- **Falsifiers:** no incremental AP/operating/coverage benefit; loss on substitutions; gain vanishes with speaker-separated fitting; confidence inflates without improved ranking.
- **Real dependency:** fusion needs component predictions and leak-free fitting data. This logical dependence does **not** imply P-SCORE must succeed before P-SUP can run.

## 3. Required future experiment record

Every authorized run must save: signed protocol/config/source revisions; input/split/label-map/variant hashes; model identity and load compatibility; fit/calibration/evaluation speaker lists; prediction and failure/abstention records; scorer/score orientation/coverage policy; attempted hyperparameters; optimizer successful/skipped counts; fixed and best endpoint identities; atomic durable checkpoint readback; all registered and stratum metrics; per-speaker uncertainty; actual measured resources; blinded feedback sample/rubric; and a process checkpoint. No experiment is authorized by this document.
