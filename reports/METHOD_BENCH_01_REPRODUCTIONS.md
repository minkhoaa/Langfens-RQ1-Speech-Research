# RQ1-METHOD-BENCH-01 — Method reproduction log

## Current status

The campaign recovered prior A/B/C research artifacts. Two inspired CPU methods have now been benchmarked locally; neither is a faithful paper reproduction. CUDA was not used because a safe campaign lease was not verified.

## CPU screen: MB01-H02-SYLPH-PROTOTYPE

Agent A identified Jin et al.'s Interspeech 2025 SylPh paper (DOI 10.21437/Interspeech.2025-1285). The paper's public abstract describes contrastive clustering of normal stimulated-syllable representations and pseudo-negative examples; it does not provide a LANGFENS-compatible four-way target/exact-gap output. No implementation link was located during the handoff. I therefore ran an `INSPIRED_IMPLEMENTATION`, not a reproduction: uniform mean pooling of the existing frozen 17-frame H-002 target window, cosine distance to TRAIN correct-only intended-phone centroids, no fitted thresholds, no DEV_SEARCH fitting.

| Split | Target binary error AP | Prevalence | AP lift |
|---|---:|---:|---:|
| DEV_CAL | 0.3602 | 0.2233 | 1.6128 |
| DEV_SEARCH (exploratory) | 0.1564 | 0.1001 | 1.5632 |

The target-only DEV_SEARCH score is below the H-002 source receipt's binary MDD AP 0.3457 on the same target population. The prototype screen does not estimate S/O diagnosis AP, exact-gap insertion AP, Macro-3 Error AP/F1, or Macro-4 F1. Its weak held-out ranking argues against spending CUDA time on this specific centroid adaptation without a better acoustic representation or a stronger paper-faithful objective. Outputs: `experiments/rq1_method_bench_01/agent_b/MB01-H02-SYLPH-PROTOTYPE.{json,csv}`; rerun with `./.venv/bin/python experiments/rq1_method_bench_01/agent_b/sylph_phone_prototype_screen.py`.

## Reproduction categories

Use `FAITHFUL_REPRODUCTION`, `ADAPTED_REPRODUCTION`, `INSPIRED_IMPLEMENTATION`, or `NEW_RESEARCH_VARIANT`. Record paper/repository commit, adaptation, split, seed, training and inference costs, prediction bundle, EVAL-V3.1 receipt, and limitations for each completed method.

## Protocol

Fit learned parameters on TRAIN only. DEV_CAL may support documented calibration/model selection. DEV_SEARCH is exploratory and cannot fit thresholds, calibrators, weights, or parameters. Event predictions must preserve target and exact N+1 gap topology. Use `research.evaluation_v3.evaluator.UnifiedEvaluatorV3` for all comparable results.

## CPU screen: MB01-H01-SOFT-PHONE-CPU-S101

**Status:** INSPIRED_IMPLEMENTATION, trained and benchmarked for target S/O only; not a faithful reproduction of Zhou et al. (Interspeech 2025). The run trains a 51→128→64→phone-class MLP on cached CTC-derived target-window features and TRAIN realized-phone supervision. The soft arm assigns 0.75 target mass to the realized phone and distributes 0.25 over other phones using exp(-phonological_distance / 0.09) from the existing project feature table. Omission remains a hard blank class. Matched hard-target and similarity-soft arms used seed 101, 18 epochs, batch 512, AdamW at 0.001, and CPU training. Runtime was 8.44 s and 8.03 s respectively. Checkpoint hashes and source-data hashes are in experiments/rq1_method_bench_01/agent_b/soft_phone_recognition/results.json.

On DEV_SEARCH, hard-target AP was S 0.36356, O 0.15360, binary target MD 0.38365. Similarity-soft AP was S 0.36214, O 0.14672, binary target MD 0.38832. The small binary AP increase does not compensate for lower S/O AP in this one-seed screen; the similarity-soft ranking hypothesis is **not supported at this configuration**. Results are exploratory because DEV_SEARCH has prior campaign exposure and no seed replication has been done. The recognizer predicts one target-aligned phone or blank per canonical target; it has no free-running phone sequence and no insertion-gap predictions. EVAL-V3.1 therefore reports target AP, while exact-gap I AP and Macro-3 AP are not estimable. Argmax decisions are uncalibrated and their warning rates are operating-point evidence only.

Reproduction command: ./.venv/bin/python experiments/rq1_method_bench_01/agent_b/soft_phone_recognition_screen.py. Artifacts: script, hard_seed101.pt, similarity_soft_seed101.pt, target prediction JSONL files, results.json, comparison.csv, and run.log under experiments/rq1_method_bench_01/agent_b/soft_phone_recognition/.

## Current status summary

- **RESEARCHED:** 22 unique source records / 15 method candidates or controls in the database; nine method families.
- **SELECTED:** H01–H05 in the experiment portfolio.
- **IMPLEMENTED:** H01 target-window recognizer; H02 correct-only centroid; H03 discrete-unit surprisal. H01/H03 are inspired adaptations.
- **TRAINED:** H01 hard/soft MLPs on CPU seed 101; H03 codebook and token counts on CPU seed 101. H02 correct-only centroids are not neural training.
- **BENCHMARKED:** H01 S/O target AP; H02/H03 binary target-anomaly AP. No full new S/O/I Macro-3 benchmark is complete.
- **SUPPORTED:** None of the new hypotheses yet.
- **NOT SUPPORTED AT CURRENT CONFIGURATION:** H01 soft targets did not improve S/O AP; H02/H03 binary target AP were below the H-002 SEARCH reference. These screens do not reject their full method families.

## CPU screen: MB01-H03-DISCRETE-UNIT-SURPRISAL-CPU-S101

**Status:** INSPIRED_IMPLEMENTATION, trained and benchmarked as binary target-anomaly ranking only; not a faithful reproduction of Sara and Chowdhury (arXiv:2606.19910). A 64-centroid MiniBatchKMeans codebook was fit on a random 60,000-frame sample from TRAIN H-002 target-window embeddings. Smoothed per-phone unigram unit probabilities were estimated only from TRAIN gold-correct target windows. At inference, each 17-frame acoustic target window and intended phone identity yield mean negative log token probability. No native speech-trained SSL codebook/token LM or Text2DUnit-DTW is used.

CPU training took 4.074 seconds including token-count estimation; K-means fit was 2.430 seconds. Inference took 0.466 seconds for DEV_CAL and 0.892 seconds for DEV_SEARCH. Binary target-error AP was 0.34781 on DEV_CAL and 0.14318 on DEV_SEARCH (target error prevalence on SEARCH is 0.10008; AP lift 1.430). H-002 binary target AP reference is 0.34570 on SEARCH and the earlier simple H02 cosine centroid scored 0.15644. These numbers are same-domain exploratory ranking context, not evidence of an improvement over H-002. No S/O diagnosis, class decisions, or exact-gap insertion scores are produced. EVAL-V3.1 reports target binary AP from raw score and marks target decision coverage as zero because predictions abstain; this is intentional for a ranking-only specialist.

Reproduction command: ./.venv/bin/python experiments/rq1_method_bench_01/agent_b/discrete_unit_surprisal_screen.py. Outputs are under experiments/rq1_method_bench_01/agent_b/discrete_unit_surprisal/, including codebook, full evaluator receipt, prediction JSONL and run log.
