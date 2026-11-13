# METHOD-BENCH-02 Agent B — Alternative acoustic representations

**Date:** 2026-11-08  
**Evaluation:** official `UnifiedEvaluatorV3` / EVAL-V3.1  
**Data boundary:** every fitted parameter used TRAIN only; DEV_CAL and DEV_SEARCH were evaluation-only; TEST was not opened.

## Summary

Two CPU-first screens tested mechanisms distinct from repeating H01 soft targets:

1. **Discrete acoustic-unit context:** extend the recovered MB01 H03 K=64 TRAIN codebook and phone-conditioned unigram with phone-conditioned within-window unit bigram likelihoods. The transition model reduced target-error AP from 0.14318 to 0.11736 on exploratory DEV_SEARCH. Local transition context is not supported at this configuration.
2. **Ordered subphonetic states with transport targets:** an inspired OTTC adaptation trained a framewise MLP on 17-frame Wav2Vec2 layer-6 windows. A balanced one-dimensional monotonic transport plan maps uniform frame mass to three ordered states for each realized phone. Target binary anomaly AP was 0.19146 on DEV_SEARCH; substitution AP 0.15842; omission AP 0.01058. Exact-gap insertion AP and Macro-3 AP remain unestimable.

Both runs are one-seed screens. DEV_SEARCH is descriptive only; no threshold, model selection, calibration, or fitting used it. Neither result establishes a full method-family verdict.

## MB02-B-DISCRETE-UNIT-TRANSITION-S101

The run recovers the K=64 codebook created in MB01 from TRAIN H-002 hidden frames, then fits unigram and adjacent-token transition tables by intended phone using only gold-CORRECT TRAIN windows. It compares the original unigram score with mean within-window `-log P(unit_t | unit_(t-1), phone)`; no new model selection or tuning used evaluation splits.

| Split | Score | Binary target-error AP | ROC AUC | Target/gap rows |
|---|---|---:|---:|---:|
| DEV_CAL | Phone-conditioned unigram | 0.34781 | 0.64644 | 10,043 / 10,343 |
| DEV_CAL | Phone-conditioned bigram | 0.25997 | 0.56495 | 10,043 / 10,343 |
| DEV_SEARCH | Phone-conditioned unigram | 0.14318 | 0.58149 | 19,755 / 20,354 |
| DEV_SEARCH | Phone-conditioned bigram | 0.11736 | 0.54700 | 19,755 / 20,354 |

The unigram values reproduce MB01 H03, providing a matched baseline. Bigram transition likelihoods were worse on both splits. All canonical target rows were present. This cache contains no gap acoustic windows, so all N+1 gap events are represented to the evaluator as abstentions; insertion AP is not estimable. Binary anomaly scores do not diagnose S versus O.

**Receipt:** CPU, seed 101, 6.17 s for transition fitting plus evaluation; 59,562 TRAIN targets, of which 50,356 were gold-correct. Codebook SHA256: `a62711e947ee24037ab408087e75dff21da730a560e6a7213af94d85fbf91be6`. Training feature and manifest SHA256s: `eb8b23a22c2b497e5ac41a07c2ab20ab4f4baac1228605f9481a5e949489392d` and `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff`.

**Artifacts:** [screen script](../experiments/rq1_method_bench_02/agent_b/discrete_unit_context_screen.py), [results and EVAL-V3.1 summaries](../experiments/rq1_method_bench_02/agent_b/results.json), and per-split unigram/bigram target prediction JSONL files are under `experiments/rq1_method_bench_02/agent_b/`.

## MB02-B-OTTC-ORDERED-STATES-S101

Geng et al. propose ordered phone-internal states and optimal temporal transport classification to produce dense monotonic frame evidence. The paper describes learned frame marginals and phone-sequence decoding; the associated IF-MDD repository provides CROTTC model code and public L2-ARCTIC inference bundles. This experiment is an **INSPIRED_IMPLEMENTATION**, not a faithful reproduction: it uses cached Wav2Vec2 layer-6 features (51-dimensional context at 17 frames), independent target windows, a static balanced transport coupling, and no sequence decoder. It does instantiate transport-based ordered-state supervision; it is not a target-vector-only proxy.

The framewise MLP has a shared hidden layer and 3 output states per realized phone. Its transport target is the exact overlap coupling between 17 equal-mass frame intervals and 3 equal-mass state intervals. Fit uses TRAIN manifest realized-phone supervision and gold-CORRECT TRAIN windows only. At evaluation, the summed three-state posterior for the intended phone is recorded as `prob_correct`; `1 - prob_correct` is the ranking-only binary anomaly score. The model has no blank class, so omission AP is weak and cannot be interpreted as an omission classifier.

| Split | Binary target-error AP | Substitution AP | Omission AP | Target/gap rows |
|---|---:|---:|---:|---:|
| DEV_CAL | 0.43744 | 0.29291 | 0.04600 | 10,043 / 10,343 |
| DEV_SEARCH | 0.19146 | 0.15842 | 0.01058 | 19,755 / 20,354 |

All target events have prediction rows. The N+1 gap manifest was passed to EVAL-V3.1 but has no acoustic scores (10,343 / 20,354 gaps are missing predictions); insertion AP and Macro-3 AP are unestimable. Target predictions abstain, so decision metrics/FWR are intentionally not claimed. DEV_SEARCH scores are exploratory and below the previously recovered H-002 binary target AP of 0.34570; the DEV_CAL AP does not justify a performance claim.

**Receipt:** CPU, seed 101, six epochs, batch 512, 48.73 s training; 59,562 TRAIN targets and 50,356 gold-correct windows. Checkpoint SHA256: `6c830bd653fc078b738d370c1b0e7a59a20e2529ea8f699f628d15cbffb41da8`. Feature SHA256: `eb8b23a22c2b497e5ac41a07c2ab20ab4f4baac1228605f9481a5e949489392d`; TRAIN manifest SHA256: `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff`.

**Artifacts:** [screen script](../experiments/rq1_method_bench_02/agent_b/ottc_subphonetic_screen.py), checkpoint `MB02-B-OTTC-ORDERED-STATES-S101.pt`, [receipt and metrics](../experiments/rq1_method_bench_02/agent_b/ottc_results.json), and DEV_CAL/DEV_SEARCH target prediction JSONLs are under `experiments/rq1_method_bench_02/agent_b/`.

## External source and feasibility notes

- Sara and Chowdhury, [*Light-weight Pronunciation Assessment via Discrete Speech Token Surprisal*, arXiv:2606.19910](https://arxiv.org/abs/2606.19910): primary source specifies a token language model over native acoustic-unit sequences plus transcript-guided Text2DUnit-DTW. The locally recovered H-002 cache and MB01 codebook support only a much smaller target-window adaptation; native audio-unit / text-to-unit assets were not recovered.
- Geng et al., [*Subphonetic Acoustic Modeling via Optimal Transport for Pronunciation Assessment*, arXiv:2609.13694](https://arxiv.org/abs/2609.13694): primary source defines topology-aware framewise modeling with OTTC and ordered phone states. The author's [IF-MDD repository](https://github.com/Secondtonumb/IF-MDD) contains CROTTC code; its [CROTTC-IF L2-ARCTIC bundle](https://huggingface.co/Haopeng/CROTTC-IF-l2-arctic) advertises model and perceived-SSL checkpoints. Those corpus-specific weights were not used because held-out split provenance is not adequate for this TRAIN-only experiment. The local cache permitted the explicit static-plan adaptation above.
- Geng et al., [*Beyond Acoustic Sparsity and Linguistic Bias: A Prompt-Free Paradigm for MDD*, arXiv:2604.22133](https://arxiv.org/abs/2604.22133): the same public [IF-MDD repository](https://github.com/Secondtonumb/IF-MDD) provides CROTTC, IF-MDD, and CROTTC-IF code, plus L2-ARCTIC release bundles. The full method requires an acoustic CROTTC model, an implicit-fusion sequence model and training data/compute beyond the cached target windows. Its bundled corpus-specific checkpoints were not a safe substitute for fresh TRAIN-only fitting. A distinct prompt-free adaptation was therefore not run: the available cache does not include utterance-level acoustic sequences needed to implement its acoustic/sequence decoupling faithfully, and a phone-window classifier would reproduce H01-like behavior instead.

The public checkpoint/code availability is evidence of potential reproducibility, not proof that those checkpoints are uncontaminated for this split or that their licensing covers data/model artifacts. No external code or weights were downloaded for this CPU experiment.

## Reproduction

```bash
./.venv/bin/python experiments/rq1_method_bench_02/agent_b/discrete_unit_context_screen.py
./.venv/bin/python experiments/rq1_method_bench_02/agent_b/ottc_subphonetic_screen.py
```
