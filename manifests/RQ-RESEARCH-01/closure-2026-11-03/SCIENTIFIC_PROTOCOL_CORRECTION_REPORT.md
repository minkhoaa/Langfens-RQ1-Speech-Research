# Scientific protocol correction addendum

**Research/documentation only.** Archival arithmetic and source interpretation, not fresh model evaluation. Historical E0/E1 metrics and evaluators remain unchanged. Read this with [independent Agent D audit](independent/D_SCIENTIFIC_PROTOCOL_CORRECTIONS.md) and the final integration acceptance notes.

## 1. Same-contract E0/E1 comparison

Source: `manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json` and the two `RQ_MODEL_01_E{0,1}_FOLD1_EVAL/E1_EVALUATION_REPORT.json` files. Executed-source pin: `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`. Coordinator byte-compared ten critical outer source/config files against this pin: all equal, including evaluator, trainer, model, parser, dataset, collator, losses and label mapping. This is source parity, not weight/forward parity.

| Metric | E0 | E1 | E1−E0 |
|---|---:|---:|---:|
| Omission-vs-correct AP | .035020 | .032252 | −.002768 |
| Omission recall at diagnostic FPR≤15% | .400000 | .365517 | −.034483 |
| Actual correct-token FPR | .149903 | .149903 | 0 |
| Full binary AP | .317021 | .309293 | −.007728 |
| Substitution-vs-correct AP | .307366 | .300876 | −.006490 |
| Greedy realized-phone PER | .236818 | .232365 | −.004453 |
| Protocol Z aligned binary AP | .313529 | .305736 | −.007793 |

Population: 300 utterances; NJS (Spanish), RRBI (Hindi); 9,956 canonical tokens = 8,799 correct + 1,012 substitutions + 145 deletions. Some older prose reverses these speakers' L1s; use evaluator demographics and source dataset documentation. Archived by-L1 metrics aggregate exactly those two individuals, not independent L1 population estimates.

- Binary prevalence = 1,157/9,956 = **11.6211%**.
- All-token omission prevalence = 145/9,956 = **1.4564%**.
- Registered omission-AP prevalence = 145/(145+8,799) = **1.6212%**, N=8,944. Substitutions are excluded, not negative examples. The AP .035020 is about 2.16× this prevalence, not a calibrated probability or strong useful precision.
- Substitution denominator = 1,012+8,799 = 9,811. PER denominator = **9,881 realized phones**, not 9,956 canonical tokens. E0 edits 2,340; E1 edits 2,296. Realized references retain additions and exclude annotated deletions/silence/invalid targets. CTC recognition deletions and gold learner omissions are different quantities.
- Integer counts compatible with rounded archival rates reconstruct E0 at FPR15 as 58/145 detected deletions and E1 as 53/145. Both reconstruct to 1,319/8,799 correct-phone false alarms under this diagnostic threshold; these are arithmetic reconstructions, not newly observed prediction records. Omission-versus-correct precision is approximately **4.21%** and **3.86%** respectively. This omission-only precision does not count substitutions as negative false alarms; full-MDD precision has its own denominator. FPR15 is not “15% of alerts are wrong.”
- Hypothetical recall .55 at FPR .15 and this prevalence implies only about **5.7%** omission precision. It is an experimental target, not sufficient justification for confident corrective feedback.

### Model training versus benchmark execution

The saved benchmark was inference-only. **E1 itself was trained:** archived `/tmp/e1_train_output/checkpoints/fold1/e1_fold1_alignment_aware/{history.json,training_endpoint_report.json}` records 3,600 successful updates, zero AMP skips in history, and endpoint hash `3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24`, matching the evaluation JSON. Coordinator inspected/rehashed these metadata files, not model weights or remote durability. History hash `52ab6f31302f0c572d3abbd9b09fc338ba98c221a37f690223966ae973178a5e`; endpoint metadata hash `8ded5f32ca2ef57cf18529f32ca46676ee0bd1e0f5f480b1d806c2dbc58777fa`.

Six-decimal skip mass .000000 on deletions means **no demonstrated useful skip separation at stored precision**, not a proof of exact zero for every token. E1 bundles skip-token attention and diagonal bias; this is not a factorial causal test of each. Retain the archived NO-GO for E1 rollout; do not extend it to every alignment, null-state or supervised realization method.

Historical E0 .316062 (older Protocol O replay), .313530 (older Z), .317021 (E0/E1 evaluator), and four-fold macro/micro APs belong to separately identified evaluators/populations. Do not overwrite or average them into a new “correct baseline.” Endpoint history and later benchmark also have modest numerical differences that this documentation pass does not resolve.

## 2. DIAG-02B, not DIAG-02 causal prose, governs interpretation

Authoritative correction: `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md` §§1–8 and its machine-readable report. Current disposition remains **PARTIAL; 9/80 archive parity failures at unchanged .000200 tolerance; NO-GO for causal training authorization**. A full 1,199-utterance model replay was not executed there or here.

| Evidence | Proper reading | Not established |
|---|---|---|
| Four-fold omission AP .030506 on 629 deletions vs 35,363 correct | Descriptive ranking deficit; denominator 35,992 and prevalence **1.7476%** | Prevalence .015893 as omission-AP denominator; reliable detector |
| Mean blank AUC .653 in 66 matched pairs (G=8) | Modest interval-conditioned discrimination in selected case-control data | Calibrated omission posterior, segmentation-free readiness or population AP |
| Matched-case AP .644 at 50% prevalence | Selected case-control ranking | Improvement from corpus omission AP .030506 |
| Valid projection-preserving local/random attention restrictions | Exploratory location-specific sensitivity | Natural omission causal mechanism or training benefit |
| Historical forced attention bypassed learned projections | Prior +.26 drift is not valid causal evidence | Proven MLP OOD breakdown |
| Small feature-erasure response | One post-encoder sensitivity probe | Proof original waveform contains no acoustic evidence or attention is irrelevant |
| Zeroing waveform interval | A distinct artificial intervention | A naturally omitted phone, physical silence, or correct omission localization |

Annotation deletion intervals often represent a neighboring transition/gap; their positive annotation duration is not the duration of a physically present omitted phone. Quantization, coarticulation and annotation uncertainty are explicit limitations. Repeated target labels, short realized phones and allowed pronunciation variants are mandatory controls.

## 3. Retire the 1.5× blank-posterior gate

For bounded posterior means, `mean_blank_deletion ≥ 1.5 × mean_blank_correct` requires `mean_blank_correct ≤ 2/3`. DIAG-02B's matched control mean is **.7322**, so the old gate would require **1.0983 > 1**, mathematically impossible in that sample. The observed means .8244/.7322 imply a ratio about **1.126**, not 1.5. These rounded gold-interval diagnostic means are not the new pilot's unknown self-aligned statistics; the impossibility does not automatically transfer numerically to every future population. Agent D's full-precision archival calculation gives a required 1.098234 and observed ratio 1.125950. Original Lane D also proposed a conflicting **2×** gate, even less feasible on these controls.

Even where a ratio is feasible, it does not assess useful ranking, matched-FPR recall, calibration, acoustic specificity or feedback correctness. Replace it with the common endpoint/coverage/control contract in [parallel pilot specifications](PARALLEL_PILOT_SPECIFICATIONS.md). Do not optimize annotation duration thresholds, claim minimum 20-ms canonical forced paths are true omission spans, or treat a small p-value in thousands of correlated tokens as a deployment gate.

Ordinary canonical CTC paths must emit every reference phone; blank does not delete one. Forced alignment can fabricate a tiny segment or allocate a neighboring sound, but neither outcome is guaranteed. An explicit optional-target graph or realized-sequence comparison is a distinct hypothesis.

## 4. Statistical and selection boundaries

Fold 1 has **two validation speakers**, not 9,956 independent subjects. Any speaker bootstrap is highly discrete; an exact two-sided sign-flip test over two independent speaker effects has a minimum attainable p-value of .5. Thus a conventional .05 population-significance gate is not attainable with this simple two-speaker design. Token-level uncertainty and extra seeds do not cure missing speaker replication.

Report both speakers, not only a pooled result; acknowledge two L1s and no Vietnamese validation speaker in this fold. Pooled AP is not the mean of speaker APs. Validation-selected thresholds, checkpoints, loss weights or method shortlist create selection optimism. Diagnostic achieved-FPR thresholds computed on the same correct tokens are not independent calibrated deployment operating points. Four folds/eight validation speakers improve descriptive coverage but have overlapping training sets and repeated research-driven selection; do not relabel them pristine confirmation.

No universal AP/recall/latency superiority claim follows from generic focal/weighted BCE failures. A calibration shift is not a ranking improvement. No unique causal explanation follows from E1's failure, nor does direct DELETE CE guarantee acoustic grounding. Proposed absolute AP targets, iteration counts and resource budgets remain design choices/estimates unless actually measured in the relevant arm.

## 5. Additional cross-session corrections

- The original synthesis's **59.90% omission FNR at 10% FPR** mixes estimands: Agent F traced 59.90% to Fold-1 all-error binary FNR, while DIAG-01's pooled deletion FNR at 10% FPR is .8251 (519/629), recall .1749. Do not silently substitute either historical metric for the other.
- Original focal-pilot “p=.056” was a bootstrap positive fraction, not the paired permutation p-value .479 documented by Step 19. Neither is a speaker-population guarantee. The weighted arm ended at 3,598 successful updates and is not a 3,600-update fixed endpoint.
- The E0/E1 evaluator removes full token records before writing aggregate JSON/CSV. The inspected archived directories do not supply full-precision E1 predictions: AP/AUC, paired disagreement and raw skip distributions cannot be independently recomputed from those aggregate files. Future authorized replay should export immutable records to a **new** directory. No aggregate arithmetic check is a full AP reproduction.
- Historical source ID drift is material: original A.3 described an explicit deletion-arc trellis, and F.3 a constrained-Viterbi probe; the synthesis relabeled A.3/F.3 as external MFA. Keep candidate definitions fixed before scoring.
- “No-skill AP equals prevalence” is the constant-score PR reference. Exact finite-sample expected random-ranking AP need not equal prevalence; do not turn the reference line into an exact finite-sample theorem.
- A hierarchical deletion score requires `P(error) × P(DELETE|error)`, not just the conditional subtype score. Deletion-versus-correct AP alone does not reveal DELETE false alarms on substitutions; report the subtype confusion table as an auxiliary endpoint.

## 6. Safe decision boundary

Proceed only to documentation review. Independent scoring, compact supervision and explicit probabilistic alignment pilots can be designed in parallel, each with controls; one need not pass before another is scientifically meaningful. Fusion genuinely depends on component predictions and leak-free calibration. All implementation, fitting, inference/replay, remote staging, training, TEST evaluation and product integration require separate operator authorization. No historical evaluator semantics were changed to produce this report.
