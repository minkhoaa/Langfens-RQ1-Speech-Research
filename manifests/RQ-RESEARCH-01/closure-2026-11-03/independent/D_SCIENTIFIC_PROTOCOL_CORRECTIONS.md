# Agent D — Scientific protocol corrections and archival metric audit

**Scope:** research-only closure of existing RQ-RESEARCH-01; no new phase or experimental authorization. **Disposition:** documentary audit delivered; historical E1 Fold-1 screening failure retained; DIAG-02B remains PARTIAL / NO-GO for inherited causal training authorization. No repository files, historical metrics or evaluation contracts changed.

## 1. Authority, actual reads and evidence strength

I read `AGENTS.md`, the global checkpoint index, RQ-RESEARCH-01's historical and 2026-11-03 revalidation checkpoints, RQ-MODEL-01's checkpoint, DIAG-02B's checkpoint/scientific report, historical research pilot specifications, the revalidation integrated pilot specifications, and the original thesis. Repository authority is subordinate to the original thesis, not the reverse.

The original `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`, §2.3 RQ1, §3 limits, §4(c) and §4(g), requires phoneme-level acoustic evaluation, suitable labeled error categories, qualified feedback, comparable evaluation units and separated calibration/evaluation, preferably across speakers. It explicitly distinguishes transcription from pronunciation assessment. It proposes alignment/GOP or pretrained acoustic classification for investigation; it does **not** establish a winning architecture, require GECToR-Speech, or prescribe omission AP .12/.15. GECToR appears in §4(f) Writing/Grammar and reference [10], not as validated speech-MDD evidence. RQ2 remains multi-agent content generation. All of the thesis content is on one unusually long source line 14; its bibliography is line 15, so section citations are more useful than individual lines here.

At inspection, outer HEAD was `b25a7595360294798d8d06b37592dd65dd4c9c33`, branch `research-phase`; nested HEAD was `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`. Outer dirty index documentation/revalidation artifacts and nested modified/deleted source were already present and were not touched. The inspected outer evaluator, AP helper, DIAG-02B diagnostic, model and E1 config were compared byte-for-byte with the **committed nested `baa4989` versions** and all five matched. The dirty nested working tree was not used as historical execution authority.

### Evidence table

| Evidence directly accessed | Location / precise locator | What it supports; limitation |
|---|---|---|
| E0/E1 archived summaries | `manifests/RQ-MODEL-01/evaluation/RQ_MODEL_01_E{0,1}_FOLD1_EVAL/E1_EVALUATION_REPORT.json`, `counts`, `primary_registered_metrics`, `secondary_metrics`, `provenance` | Published rounded values and denominators; not a fresh model replay or token-level AP reproduction. |
| Registered omission and FPR semantics | `audit_scripts/evaluate_e1_mdd.py:108–210`, especially `:125–145` | Deletion-versus-correct subset, substitutions excluded; threshold uses correct scores. |
| Actual score/inference/PER construction | Same evaluator `:308–358`, `:419–434`, `:503–517` | Sigmoid binary score, filtered realized-phone PER reference, offline gold alignment, rounded attention summaries; constructed token records are discarded by `main`. |
| Exact AP / threshold definitions | `audit_scripts/evaluate_fold1_ctc_binary.py:131–150`; `audit_scripts/diagnose_counterfactual_validity.py:139–152` | Tied-score grouped AP and conservative, tie-aware same-set FPR threshold. |
| Trained E1 provenance | `/tmp/e1_train_output/checkpoints/fold1/e1_fold1_alignment_aware/{history.json,training_endpoint_report.json}`; retained `langfens-rq-model-01-e1-fold1-train.log:1826–1827,2360–2405` | Epoch 6, 3,600 successful updates, zero cumulative AMP skips; endpoint hash agrees with evaluation. Logs/metadata, not direct weight verification or remote storage readback. |
| E1 architectural contract | `configs/rq_model_01_e1_fold1.yaml:1–38`; `src/langfens_prep/model.py:137–202` | Frozen pretrained backbone, CTC+binary objective, skip token plus soft normalized-position bias. Not a hard monotone edit model. |
| Canonical DIAG-02B metrics | `manifests/RQ1-DIAG-02B/evaluation-final/REPORT.json:785` (`archive_metrics_reproduction`), `:881` (`parity`), `:1015` (`ctc_matched_discrimination`), `:1801` (10-ms sensitivity), `:3206` (interventions), `:4426` (paired contrasts) | Auditable historical diagnostic results; bounded, selected sample; no new execution in this review. |
| Matched blank means and corrections | `manifests/RQ1-DIAG-02B/reconciliation-final/RECONCILIATION.json:16–31`; `CORRECTIONS.json` | Probability-gate feasibility and archived correction ledger. |
| Counterfactual implementation | `audit_scripts/diagnose_temporal_alignment_validation.py:425–433`; `audit_scripts/diagnose_counterfactual_validity.py:100–129,173–195,432–507` | Invalid historical raw-feature weighted reconstruction versus projection-preserving mask; gold-window diagnostics versus runtime inference. |
| Historical versus revalidated gates | `manifests/RQ-RESEARCH-01/INTEGRATED_DECISION_REPORT.md:181–208`; `revalidation-2026-11-03/INTEGRATED_DECISION_REPORT.md`, §5 | Historical numerical targets and corrected proposed comparisons; proposals are not independent experiment evidence. |
| Earlier validation reuse/statistical errata | `manifests/RQ1-TRN-03A/checkpoints/19-gate-c-scientific-closure.md`, §§2–6; underlying `evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json`, first endpoint | Reused Fold-1 validation, different evaluator-specific baseline numbers. The historical permutation/CI claims were read, **not rerun or adopted as speaker-population inference**. |

No primary paper was newly retrieved in this lane. Literature descriptions in old research reports are not independent evidence. The thesis directly supplies GECToR's reference URL, `https://aclanthology.org/2020.bea-1.16/`; that paper and its implementation/license were **not accessed or verified here**. No new literature performance, latency, memory or compute estimates are offered. The pretrained revision recorded in config is `facebook/wav2vec2-base-960h@22aad52d435eb6dbaf354bdad9b0da84ce7d6156`; no download, loading or license revalidation occurred.

## 2. Denominators and score semantics — preserve the existing contracts

All arithmetic below uses archived JSON counts/metrics, not audio, TextGrids, model outputs newly generated, or raw dataset traversal.

| Population / task | Positive count | Evaluation denominator | Positive prevalence |
|---|---:|---:|---:|
| Fold 1 binary: substitutions + deletions vs correct | 1,157 | 9,956 | 11.621133% |
| **Fold 1 omission: deletion vs correct** | **145** | **145 + 8,799 = 8,944** | **1.621199%** |
| Fold 1 omission census share, not omission-AP prevalence | 145 | 9,956 | 1.456408% |
| Fold 1 substitution vs correct | 1,012 | 9,811 | 10.314953% |
| Four-fold binary | 4,213 | 39,576 | 10.645341% |
| **Four-fold omission vs correct** | **629** | **629 + 35,363 = 35,992** | **1.747611%** |
| Four-fold omission census share, not omission-AP prevalence | 629 | 39,576 | 1.589347% |
| Four-fold substitution vs correct | 3,584 | 38,947 | 9.202249% |

Sources: E0/E1 JSON `counts`; DIAG-02B JSON `scope.census` and `archive_metrics_reproduction`. Census checks and denominator sums were executed. Constant-score AP equals the relevant subset prevalence; the no-skill PR reference is not the all-token deletion share. Exact finite-sample expected random-ranking AP need not equal prevalence. E0/E1 omission AP are approximately 2.1601×/1.9894× **their correct subset prevalence**, not the ratios obtained from .0159.

### What each quantity actually means

- Historical E0 and E1 omission AP both rank **sigmoid binary-head logits**, nominal `P_error`, on the deletion-versus-correct subset (`evaluate_e1_mdd.py:320`). It is not a calibrated `P(delete)`, and a high score can reflect generic error or contextual priors. BCE already supervises natural deletions as positive errors; adding DELETE supervision would retain **subtype identity**, not introduce deletion labels where none existed.
- A future subtype model should report deletion-vs-correct AP using a named deletion score, plus a **legacy-comparable omission AP using total error score**. Joint binary AP should use total error probability, e.g. `1−P(KEEP)` for a coherent categorical model. `P(DELETE|error)` alone is not `P(DELETE)`; a hierarchical joint score requires `P(error)×P(DELETE|error)`. These are model probabilities, not automatically calibrated confidence.
- Substitutions stay excluded from the registered omission-AP denominator. A new deletion-versus-all-other analysis can be useful but must be separately labeled, never substituted retrospectively. Report deletion/substitution confusion to expose subtype false alarms concealed by deletion-versus-correct evaluation.
- A greedy recognizer/Levenshtein **DP deletion** is a hypothesized sequence edit, not gold omission, deletion posterior, or evidence of acoustic absence. Ties, pronunciation variants and recognition misses affect its identity. A 0/1 DP flag creates a coarse two-level ranking; report its ties and attainable operating points rather than inventing a smooth PR curve.
- CTC blank means no label emission at that frame/path; it is neither an omission class nor synonymous with silence. Normal correctly realized speech can have high blank posterior. Annotation deletion does not mean an acoustically observed silent segment.

## 3. Numerically checked E0/E1 result and trained-checkpoint provenance

| Registered Fold-1 metric | E0 | E1 | E1−E0 | Relative change from rounded archive |
|---|---:|---:|---:|---:|
| Omission AP, `P_error`, N=8,944 | .035020 | .032252 | −.002768 | −7.9041% |
| Omission recall at correct FPR ≤15% | .400000 | .365517 | −.034483 | −8.6208% |
| Binary AP, N=9,956 | .317021 | .309293 | −.007728 | −2.4377% |
| Greedy CTC PER | .236818 | .232365 | −.004453 | −1.8803% |
| Substitution AP | .307366 | .300876 | −.006490 | — |
| Protocol-Z aligned binary AP | .313529 | .305736 | −.007793 | — |

JSON precision is six decimals. The omission recall difference is **−3.4483 percentage points**, not −8.62 points. Lower PER is favorable but does not establish useful omission diagnosis. Coverage is `9919/9956 = .996283648`, with **37 excluded unmatched canonical positions**. Protocol-Z alignment uses gold canonical identities offline to join predictions with labels; it does not make its aligned subset an unconditional full-population metric.

### Operating-point reconstruction

The integer counts uniquely compatible with the rounded rates and denominators are:

| Omission-versus-correct diagnostic | TP / 145 | FN | FP / 8,799 | TN | Achieved FPR | Precision on this subset |
|---|---:|---:|---:|---:|---:|---:|
| E0, reported threshold .210404 | 58 | 87 | 1,319 | 7,480 | .149903398 | 4.2121% |
| E1, reported threshold .160890 | 53 | 92 | 1,319 | 7,480 | .149903398 | 3.8630% |

These are **arithmetic reconstructions**, not newly observed confusion matrices. Substitution flags are excluded; these precisions are not overall binary precision or complete subtype diagnostic precision. The unrounded selected threshold was used historically; applying the six-decimal threshold later can change borderline ties. Threshold selection and measurement use the same correct validation tokens, so “≤15%” is an empirical in-sample ceiling, not an independent deployed false-alarm guarantee. Pooled FPR does not guarantee ≤15% for each speaker.

### PER is a different denominator and a different deletion

`evaluate_e1_mdd.py:325–329` builds a **realized-phone** reference from annotation intervals, excluding silence, gold deletions, `has_err_tag` and unmapped realized phones. Greedy collapse precedes edit distance. `PER=(S+D+I)/N_ref`:

| Endpoint | S | D | I | Edits | Realized reference phones | Recomputed PER |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 1,707 | 431 | 202 | 2,340 | 9,881 | .236818136 |
| E1 | 1,665 | 416 | 215 | 2,296 | 9,881 | .232365145 |

Edit components were reconstructed from six-decimal component rates times 9,881 and sum to the archived exact edit totals. Those 431/416 recognizer deletions are **not** the 145 gold canonical omissions. Insertions contribute to PER and can make PER exceed 100%; they do not silently enter the existing canonical binary denominator.

### E1 was trained; only its later benchmark was inference-only

Direct retained evidence establishes the following chain:

1. E1 config specifies original pretrained initialization, frozen backbone, CTC+binary weights 1:1, seed 1337, batch 2/accumulation 2, FP16 training and 25-epoch scheduler horizon. The retained log records verified fresh pretrained initialization and pinned `baa4989` source (`:2360–2378`). These are logged claims, not a fresh audit of parameter tensors.
2. `history.json`, epoch 6, records **7,200 microsteps, 3,600 successful updates, zero cumulative skipped AMP attempts**. The duplicated stdout lines 1826/1827 do not constitute two independent runs.
3. Endpoint JSON records zero-indexed epoch 5 and `fixed_budget_sha256=3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24`, matching E1 evaluation JSON. Thus this is not an untrained skip-token control.
4. E1's validation-best endpoint is distinct: history peaks at epoch 1 / 600 updates, AP .315600; endpoint `best_sha256=31208100a1e48582791f1f8f3efb72b3819514e128d6575f6cf00f416aab44ba`. The benchmark evaluated **fixed-budget**, not this best checkpoint. E0's archived best coincides with its 3,600-update endpoint. Do not compare best versus fixed-budget under one label.
5. E1 epoch-6 training-time AP/PER are .309018/.233377; the later FP32 benchmark gives .309293/.232365: differences **+.000275/−.001012**. Precision/backend/protocol effects are possibilities, not an established explanation. Best AP .315600 is not the benchmark's .309293.

No weights were loaded or rehashed; no live Kaggle/Hugging Face durability was revalidated. The inference-only benchmark's zero training updates describes **that benchmark**, not the model's history.

### Attention and provenance caveats

E1's deletion skip mean is serialized as **.000000**, correct mean .000031; this supports negligible skip usage at report resolution, **not exact mathematical zero on every deletion**. The evaluator explicitly rounds the means to six decimals (`:431`). Under nonnegative valid weights, the rounded mean can conceal positive mass below approximately 5×10⁻⁷. The archived JSON alone cannot identify its distribution or prove total inactivity. In-window conditional mass .035854 versus E0 .032023 and uniform .030883 is descriptive; improved attention mass does not establish omission benefit or causal alignment. The soft bias `−3(p−r)²` (`model.py:152–183`) does not enforce a monotone path or an epsilon deletion transition.

**Reproducibility gap:** `evaluate_e1_mdd.py:503` removes `token_records`; the remaining code writes only aggregate JSON/CSV. Both archived E0/E1 directories contain only those aggregate files. Consequently I could check denominators, deltas and integer compatibility, but **could not independently reproduce E1 AP/AUC, paired error overlap, raw skip distributions, or uncertainty from full-precision predictions**. Preserve the archive; a future separately authorized replay should export immutable full-precision token and CTC records with IDs, score semantics and hashes to a new location.

The same Baseline-R checkpoint has historical AP .316062 in `RQ1-TRN-02A/EVALUATION_REPORT.json`, .317012 in `RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json`, and .317021 in this E0 benchmark. These are evaluator-specific archival anchors, not interchangeable reproductions. Do not “correct” old metrics by replacement or silently choose the most convenient baseline.

## 4. DIAG-02B: what survived and what did not

### Denominator-aware archived results

| DIAG-02B analysis | N / positives | AP | ROC-AUC | Diagnostic recall / achieved FPR |
|---|---:|---:|---:|---:|
| Full archive, all errors vs correct | 39,576 / 4,213 | .272580 | .729022 | Not newly evaluated here |
| Full archive, omission vs correct | 35,992 / 629 | .030506 | .695813 | Not newly evaluated here |
| Matched blank score, 20-ms caliper | 132 / 66 | .643941 | .652663 | 16/66=.242424; 9/66=.136364 |
| Matched binary `P_error` | 132 / 66 | .483501 | .482438 | 7/66=.106061; 9/66=.136364 |
| Matched missing-canonical emission fraction | 132 / 66 | .582365 | .633724 | 0; 0 (ties force threshold above maximum) |
| Matched blank, 10-ms sensitivity | 116 / 58 | .632656 | .641201 | .241379 / .137931 |

Source: canonical JSON fields cited in §1. These are read values rounded for display with integer/rate checks, **not rerun scoring**. The 66 pairs match speaker, phone and word position, without replacement; 41/107 sampled omissions remain unmatched. Matching is duration-caliper rather than exact duration equality. The sample comprises first ten omission-containing utterances per speaker: 80 utterances/2,951 tokens/eight speakers. This is selected and omission-enriched. **Case-control AP at 50% prevalence cannot be compared to census AP .030506 at 1.7476% prevalence as a performance gain.** Annotation windows condition these frame summaries; they are not zero-oracle runtime omission detection.

The blank AUC's archived 500 speaker-cluster bootstrap interval is [.586115,.746434], exploratory at G=8. Multiple scores/calipers/interventions were considered without confirmatory multiplicity correction. Same-control thresholds and fixed-selected-threshold intervals omit independent calibration uncertainty. Neither an AUC above chance nor mean blank .824371 is “82% omission confidence.”

### Archive parity is still failed

`REPORT.json.parity` records **9/80** utterances exceeding unchanged probability tolerance **.000200**, maximum **.001304954**. Reconstructed head versus forward and all-open mask logit differences are zero on the bounded sample; fresh-process saved-logit parity is also zero. Those are different comparisons from archived CUDA-score parity. Sample binary AP .292173109 archived versus .292111817 CPU (−.000061292) does not waive pointwise failures. Full 1,199-utterance forward replay is explicitly **NOT_EXECUTED**. The continue-on-drift flag authorized continuation of diagnostic collection, not PASS. Cause of drift remains unresolved.

### Causal and localization limits

- Historical forced attention used uniform averaged weights multiplied by **raw hidden states** (`diagnose_temporal_alignment_validation.py:425–433`), bypassing learned per-head value/output projections and original attention computation. Its approximately +.26 score drift cannot prove downstream-MLP OOD failure or identify natural omission causation.
- Corrected `head_forward` preserves original MHA and masks allowed frames. Gold windows enter **offline interventions only**, not a deployable predictor. Local-minus-random omission probability change .014506 is an artificial sensitivity contrast; it is not the causal effect of naturally omitting a phone.
- Local omission AP .07262 versus baseline .06899 has archived paired change CI [−.00448,+.00990]; recall increases at τ=.20 alongside correct FPR increasing .1633→.2342 (`SCIENTIFIC_REPORT.md`, §4; underlying `REPORT.json.intervention_metrics`). This does not demonstrate a matched-FPR improvement.
- Post-encoder feature zeroing does not remove waveform information retained in contextual features elsewhere. Waveform zeroing reruns the encoder but introduces artificial boundary effects; it is neither natural deletion nor time-removal. The waveform sample is only 16 omission/control pairs, with case-control prevalence. Neither probe establishes acoustically grounded omission diagnosis.
- `RECONCILIATION.json` confirms 500/629 annotation spans coincide with neighbor gaps. This is an annotation/span fact, not observation of the acoustics of an absent phone. Duration-boundary counts 293 raw versus 284 rounded under 40 ms reflect conventions, not different datasets.
- The historical omission enrichment .00× was corrected to ~1.3966×; speaker-macro masking .001845 to .001720. These historical corrections are recorded, not silently substituted into original artifacts. Interval attention's uniform comparison is K/T, not pointwise 1/T.

**Conclusion:** descriptive omission weakness and modest matched CTC discrimination survive. The broad alignment hypothesis remains open. “Temporal attention is the proven primary cause,” “CTC already has sufficient omission evidence,” and automatic training GO do not survive. Nor does this establish that alignment-aware training cannot work.

## 5. Pilot gate feasibility and required corrections

### 5.1 Historical blank-ratio gate can be impossible

Historical research §6.1 requires mean omission blank ≥1.5× correct mean and p<10⁻⁵. Canonical reconciliation JSON supplies:

| Arithmetic from matched means | Value |
|---|---:|
| Omission mean blank | .824370932850 |
| Correct-control mean blank | .732156167202 |
| Observed ratio | 1.125949585319 |
| 1.5× correct mean | **1.098234250803** |
| Maximum possible omission/correct ratio, omission mean ≤1 | **1.365828828325** |

Thus **on this matched-control distribution the gate is mathematically unattainable**: any correct mean >2/3 rules it out. A future predicted-window distribution might have different means; the present calculation is not proof every future score distribution fails. Nevertheless a probability-ratio gate with this ceiling is not a defensible universal prerequisite. Failure cannot disprove all CTC guidance, optional-target graph scoring, learned existence or supervised subtype models. Replace with a predeclared discrimination/operating comparison to suitable controls, not an arbitrary mean ratio.

The 40/80-ms duration rule is likewise not a universal acoustic criterion. A full-canonical CTC forced alignment normally **must emit every canonical label**; an absent target gets an assigned frame/spurious segment rather than an honest zero-duration state. Consecutive repeated targets also impose blank-separation feasibility. Deletion-capable graph transitions must be explicitly modeled and validated. A canonical forced alignment alone does not localize omissions.

Welch/KS tests on thousands of correlated phones cannot justify p<10⁻⁵ as speaker-independent proof. Whole-speaker dependence, selection and multiplicity matter; increasing the bootstrap replicate count does not create independent speakers.

### 5.2 Recall, precision and AP are not interchangeable gates

For prevalence π, recall r and FPR f, precision is `rπ / [rπ+f(1−π)]`. With P=145, N=8,799, meeting recall ≥.55 requires **at least 80 true positives** (80/145=.551724). At the largest allowable integer FPR count, floor(.15×8,799)=1,319 false positives, precision is only **80/1,399=.057184**. This is a warning about false alarms at low omission prevalence, not proof that recall is unattainable. Include lower-FPR and precision/coverage endpoints before pedagogical use.

For that exact TP/FP operating point:

- A hard two-score classifier's tied-group AP is `r×precision + (1−r)×π = .038817`.
- For distinct-score rankings consistent with that operating point, putting all permitted false positives first or all true positives first gives attainable AP extrema approximately **.021964 to .587027**. This calculation orders the remaining positives/negatives adversarially/favorably; it is not a model result.
- Therefore omission AP ≥.15 and recall ≥.55 at FPR≤.15 are **not intrinsically contradictory** for a rich ranking, but neither follows from the other. A hard DP flag with the illustrated operating counts does not deliver AP .15 merely by attaining recall .55.

AP .12/.15 and minimum recall are aspirational historical targets, not evidence-based expected results. They need a registered effect rationale, score definition, precision/coverage trade-off and separate discovery/confirmation. Preserve their historical text; append current qualifications. The revalidation §5 correctly declines automatic adoption of those targets.

### 5.3 No universal unsupervised P0 prerequisite

A common **artifact-integrity/parity check** is necessary; an unsupervised blank/duration scorer's scientific success is **not** a necessary precursor to every supervised pilot. A weak frozen-head blank cue cannot establish that a separately trained deletion-conditioned model lacks useful acoustic information. Conversely DELETE-label success cannot establish alignment or acoustic grounding without controls. E1 failure rules against cross-fold promotion of that tested implementation; it does not prove that every skip mechanism, edit tagger or monotonic model fails. Historical focal/weighted-binary screening failures concern those arms/settings, not all subtype objectives or universal futility of reweighting.

## 6. Independent, comparable pilot designs — design only

These are alternative bounded tests within existing RQ1, not new phases, training launches or changes to archived E0/E1 contracts.

### Shared selection and isolation plan

1. **Freeze provenance and estimands first:** manifest SHA, original pretrained revision, phone mapping, Protocol O versus Z, allowed runtime inputs, denominator, score orientation, tie/coverage policy, evaluator version and checkpoint SHA. Resolve archival parity at the original tolerance before claiming replication; do not overwrite old reports. Export full-precision predictions and update/skip ledgers for any future approved execution.
2. **Speaker-disjoint development:** within the fold's authorized TRAIN speakers, prospectively reserve disjoint model-fit, hyperparameter-selection and calibration speakers with deletion support. Do not use TEST speakers/audio/labels. Fit all trainable scoring/fusion parameters only in permitted fit partitions; freeze model and threshold before the final validation screen. If an additional held-out speaker set is unavailable, admit that limitation rather than splitting one speaker's utterances and calling it speaker-independent.
3. The existing Fold-1 NJS/RRBI validation set has already selected checkpoints, loss variants, thresholds, diagnostics and research directions. A new frozen screen there remains **adaptively reused exploratory validation**, even if its labels are not directly fitted. Confirmation requires later separately authorized speaker-disjoint evaluation protected from this search; no TEST use is authorized now. Report both speakers individually; L1 and speaker are confounded (NJS Spanish; RRBI Hindi), so this is not an accent-level sample.
4. Equal candidate eligibility, inference inputs and scoring coverage; exact same scored positions, plus all exclusions. Report total binary AP, legacy-score omission AP, named deletion-score omission AP where applicable, substitution AP, actual FPR, precision, recall/FNR, per-speaker counts, uncertainty/abstention, and greedy realized-reference PER. Preserve insertion accounting separately.
5. Predeclare one primary contrast/effect threshold and secondary guardrails; limit candidate/hyperparameter/seed search and record every attempt. Use a hierarchical decision rule or multiplicity-adjusted confirmatory analysis if confirmation is later possible. Avoid cherry-picked best-checkpoint comparisons: exactly 3,600 **successful** updates for trained primary endpoints; report validation-selected best separately. Skipped attempts are not successful updates.
6. Use paired predictions on identical items. Speaker is the highest relevant sampling unit; G=2 cannot support general-population bootstrap claims. Resampling utterances within each of two fixed speakers estimates conditional within-speaker variation only. Resampling two speakers has almost no independent information. Even a speaker-level exact two-sided sign/permutation test has only four sign assignments; tiny p-values are impossible. Eight-speaker/four-fold summaries remain small and folds share training data; neither four folds nor repeated seeds create independent new speaker populations. Bootstrap positive fractions are not p-values or probabilities of cross-fold success.

### Pilot A: fixed-emission acoustic/edit scoring, independently eligible

**Question:** does an explicitly defined acoustic score improve omission ranking/precision at fixed false-alarm budgets beyond the frozen binary score and canonical priors?

Use identical, integrity-verified cached emissions if available; obtaining new emissions requires separate inference authorization. Compare unchanged E0 scoring, hard greedy-DP flag, a declared continuous posterior/graph score, and a same-input prior-only control. Specify full-canonical versus optional-deletion graphs, pronunciation variants, repeated labels, terminal states, normalization and impossible-path handling. No gold timing at runtime. Any fitted calibration is training and needs separate authorization, not a “training-free CPU replay.”

Use matching of phone/position/speaker/duration only for **secondary offline sensitivity diagnostics**; primary AP retains the full registered deletion-versus-correct denominator. Predeclare mismatch controls within speaker, coverage and abstention so gains cannot come from deleting difficult cases. A failed frozen score only rejects that scorer/graph/normalization under these conditions, not every acoustic or supervised direction.

### Pilot B: compact supervised subtype/existence hypothesis, independently eligible

**Question:** does preserving natural deletion identity in a compact objective improve useful omission detection beyond a matched binary control, and is the benefit dependent on matched audio?

Choose one reviewed compact KEEP/SUB/DELETE or hierarchical existence formulation. Train it **and a contemporaneous parameter-/initialization-matched binary control** from the original pretrained initialization with the same frozen backbone, data subsets, successful-update budget, optimizer/scheduler horizon, seed pairing and checkpoint-selection opportunities. Account for head dimensionality/parameter changes; do not ascribe a bundled head+loss+alignment change to DELETE labels alone. Use a minimal ablation isolating the objective while holding alignment fixed. Do not warm-start from E1 or another fold. Historical E0/E1 remain context, not a substitute for the matched causal control.

Control canonical-only scoring, matched versus within-speaker mismatched audio, and subtype confusion; assess whether gains survive speaker-disjoint calibration. These are necessary checks against a model learning phone/position priors. Direct skip supervision, if chosen, tests a different registered hypothesis and must not be retrofitted into E1's historical objective. Better subtype predictions do not by themselves prove temporal localization, articulatory causes or feedback readiness.

All future training/fitting is separately authorized remote-GPU work under the project contract (CUDA FP16 AMP, exact update accounting, independent optimizer/scheduler/scaler/RNG states, private durable per-epoch mirroring and SHA readback). No local fit or fallback. Compare total training/search and inference cost honestly; no numerical runtime/VRAM claim is supported here. Pilot B does not depend on Pilot A passing an arbitrary unsupervised threshold.

### Pilot C: fusion only after complementary evidence, not mandatory

If separately authorized, compare each component alone, a predeclared nonlearned combination and minimal fusion fitted on TRAIN-held-out speakers. A dual path must beat the strongest component under the same speaker split and selection budget; monotone score rescaling cannot improve AP absent tie changes. Keep calibration fitting outside the validation score measurement. No extra encoder/search budget hidden in the fusion arm. Failure means no demonstrated complementarity for those components, not universal rejection of joint modeling.

**Any positive pilot:** supports a bounded screening signal only. It is not automatic cross-fold rollout, general Vietnamese performance, deployment calibration, acoustic causal diagnosis or fulfillment of RQ1's feedback endpoint. Feedback needs a separate blinded correctness/clarity/error-fit review and safe abstention, not an explanation template treated as evidence.

## 7. Verification record, limitations and safe restart

Executed only read-only Git inspection, bounded text/source inspection, SHA-256 of lightweight JSON artifacts, source-byte comparison against committed nested `baa4989`, and standard-library Python JSON arithmetic. No evaluator import/execution, torch/model loading, inference, calibration fitting, bootstrap, tests, GPU/remote command, dataset archive traversal, sealed TEST access, repository edits, commit or push occurred. Current research successful updates **0**, skipped updates **0 / no attempt**; no model artifact changed.

Recomputed hashes:

| Artifact | SHA-256 |
|---|---|
| Frozen split JSON | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` |
| E0/E1 summary JSON | `006611ccf0796938d4114bf62a1d17979c0af9ece362415dc1600670e6b086a3` |
| E0 evaluation JSON | `b2315fef480daa37df2e69f60d6a3594d3bfafbb5589bb68bf00b00476685e3e` |
| E1 evaluation JSON | `0d9c6e6a3d63d26f8f3bcc4780389e96494f9ba33a6dccc05c425136c449b0d4` |
| Canonical DIAG-02B JSON | `38aa543a46d5c2da856e0d73f1887a12285576a205c66fc41c0faa194f8647b6` |
| Retained E1 training history JSON | `52ab6f31302f0c572d3abbd9b09fc338ba98c221a37f690223966ae973178a5e` |

Arithmetic assertions passed for census sums, rounded-rate integer compatibility, edit totals and nine parity failures. These are **documentation/arithmetic checks**, not independent scientific reproduction or a model PASS. Full E1 prediction records were not found in the two archived evaluation directories; stop at that limitation rather than fabricate AP recomputation. Remote checkpoint presence, licenses, exact weight content and drift cause remain unverified here. The separately requested latest omission proposal is not identified in the supplied existing-session material; no content is inferred for it.

**Actionable append-only corrections:** distinguish trained E1 from inference-only benchmark; correct omission prevalence/score labels; qualify rounded skip zero; retain evaluator-specific baseline numbers; withdraw infeasible universal blank gate and causal overclaims; register independent speaker-disjoint alternatives and held-out calibration; preserve all historical artifacts and TEST sealing.

**Restart state:** coordinator may cite this report and write its required checkpoint/index entry. No separate repository checkpoint was written by this lane. Before any future execution, verify both HEADs and pre-existing dirty state, this report's evidence, immutable lightweight hashes, missing prediction/proposal availability and explicit new authorization. Do not repair the dirty nested repository, relax parity, overwrite historical JSON, or initiate a model/data job as a continuation of this research-only closure. STOP after documentary delivery.

```acceptance-report
{
  "criteriaSatisfied": [
    {"id": "criterion-1", "status": "satisfied", "evidence": "Self-contained audit with source locators, independently checked archival JSON arithmetic, protocol corrections, proposed independent pilots and explicit limitations."}
  ],
  "changedFiles": ["/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/723092bc-f51f-4a2a-8cd9-bf63b7c93d0b/reports/D_SCIENTIFIC_PROTOCOL_CORRECTIONS.md"],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "git rev-parse HEAD; git status --short; nested HEAD/status inspection", "result": "passed", "summary": "Expected HEADs confirmed; pre-existing dirty repository content preserved."},
    {"command": "Standard-library Python JSON arithmetic and SHA-256 checks", "result": "passed", "summary": "Verified counts, prevalence, deltas, integer-compatible operating points/PER, probability-gate feasibility and lightweight artifact hashes."},
    {"command": "Read-only byte comparison with git -C repo show baa4989:<five inspected source/config paths>", "result": "passed", "summary": "All five inspected outer files match committed nested historical source."},
    {"command": "Model replay / training / test suite", "result": "not-run", "summary": "Outside this research-only authorization; no model execution or backward passes."}
  ],
  "validationOutput": ["Blank-ratio target 1.098234 exceeds probability ceiling 1 on the archived matched controls.", "Fold1 omission denominator 8944; reconstructed E0/E1 TP 58/53, FP 1319 each.", "Historical DIAG-02B remains 9/80 parity failures; no new scientific PASS."],
  "residualRisks": ["Full-precision E1 prediction archive unavailable in inspected archived evaluation directories.", "DIAG-02B archive drift unresolved; E1 training/evaluation metric differences not explained.", "Only two repeatedly reused Fold1 validation speakers; no independent generalization claim.", "No direct checkpoint-weight verification, remote durability/license revalidation or missing-proposal reconstruction."],
  "noStagedFiles": true,
  "diffSummary": "Only the authorized isolated report was written; repository unchanged by this lane.",
  "reviewFindings": ["Scientific blockers to causal/deployment claims remain; documentary audit complete within scope."],
  "manualNotes": "Historical report values are not counted as independent evidence; underlying JSON/source/logs were directly accessed. Coordinator owns repository checkpoint/index documentation. Stop before any experimental execution."
}
```
