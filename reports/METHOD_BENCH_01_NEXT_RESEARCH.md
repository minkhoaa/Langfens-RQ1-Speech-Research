# LANGFENS RQ1-METHOD-BENCH-01 — Next research cycle

## Current decision state

Research-first gate is closed for the first experimental cycle. The external method agenda remains open. Three distinct mechanisms now have local CPU screening evidence: phone-conditioned correct-only centroids (H02), similarity-soft realized-phone target-window recognition (H01), and discrete-unit phone-conditioned surprisal (H03). All current new results are one-seed exploratory screens; none supports a new method hypothesis yet. No full new S/O/I model has exact-gap insertion capability.

## Ranked next work

| Rank | Experiment or research task | Why now | Success evidence | Failure/stop signal |
|---:|---|---|---|---|
| 1 | Recover cached CTC/GOP logits or event score artifacts and implement TRAIN-only constrained-alternative scoring (H04) | It isolates the phonological-prior contribution at very low CPU cost and directly follows Parikh et al. | Matched S AP improvement over unrestricted GOP and prior-only control, with no leakage | No competitor identity/posterior information is available; do not synthesize it from aggregate margins |
| 2 | Audit sequence-level CTC artifacts and implement edit-path scoring (H05) | Best current route to an explicit omission and insertion mechanism; CTC paper motivates alignment-free scoring | Event S/O ranking plus verified mapping of every extra-phone span to an exact N+1 gap | Only target-window summaries exist and no event timing or acoustic span can be recovered |
| 3 | Revisit H03 with sequence/text conditioning | Unigram discrete surprisal scored 0.14318 AP on SEARCH; needs to test whether source paper's sequence LM and Text2DUnit-DTW explain the gap to stronger acoustic models | Matched local surprisal/DTW components improve target anomaly AP and produce auditable spans | Artifact/code/checkpoint unavailable, or local gains track phone/speaker identity only |
| 4 | Improve H01 as an actual sequence-recognition adaptation | Current phone head has no free sequence decoder; soft labels decreased S/O AP versus hard targets | Multi-seed sequence mapping improves O/S AP and complete I gap ranking; hard target remains a matched control | Soft target effect remains class-negative or edit mapping is unstable |
| 5 | Revisit H02 with speaker-normalized or contrastive normal models | Simple centroid did not generalize well; this is a representation/objective test, not a repeat | Speaker-held-out S/O AP rises over centroid and speaker-only baseline | Distances remain driven by speaker or syllable identity |
| 6 | Inspect original paper artifacts for Zhou, SylPh, discrete-unit surprisal, OpenPA, and speech-text MDD | Determine whether faithful reproduction is feasible and whether cited code/checkpoints are usable | Record commit, equations, training labels, license, checkpoint size and inference contract | Keep candidate research-only when sources cannot be verified |
| 7 | GPU lease review at next work boundary | CUDA is preferred for suitable neural training, but previous shared reservation must be safely released | Verified shared-wrapper lease with campaign slot and memory receipt | Continue CPU work; never infer lease from idle GPU or empty lock files |

## Immediate execution plan

1. Inspect output bundles from the prior H001 alignment workstream without changing or overwriting them. Confirm whether raw CTC phone posteriors or only scalar event scores remain.
2. If logits exist, run H04 on frozen scores with a confusion set/graph built from TRAIN only. Evaluate S domain only. Keep unmatched insertion/omission blank.
3. If sequence posteriors or aligned phone spans exist, run H05 and audit exact-gap topology with a fixed event mapper. Otherwise record this method as not locally executable from available artifacts.
4. Try a fair second seed for H01 hard-vs-soft only if it does not displace H04/H05 information-gain screens. Use the same CPU/GPU budget per arm.
5. Continue external search for recent methods outside the current taxonomy, especially structured phone recognition, low-resource event localization, phone-level uncertainty and non-L2 acoustic anomaly methods. Add only verified primary sources and explain transfer limits.

## Current benchmark status

- Reproducible CPU training receipts: two H01 MLP arms; H03 codebook/token model.
- Existing H02 artifact: correct-only centroid prototype, no neural training.
- Numerical event rankings: H01 target S/O AP; H02/H03 binary target-error AP.
- Unsupported so far: exact-gap insertion for all new screens; full Macro-3 AP; multi-seed support.
- All DEV_SEARCH results remain exploratory and have not been used for parameter or threshold fitting.
