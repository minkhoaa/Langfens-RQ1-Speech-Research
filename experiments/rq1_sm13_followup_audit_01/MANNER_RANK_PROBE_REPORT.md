# SM13 manner-ranking pilot: completed single-candidate screen

**Date:** 2026-11-13  
**Decision:** Do not promote this candidate as the vowel/stop fix. It modestly improves stop ranking but degrades vowel ranking and does not meet the illustrative 15% recall / 50% precision gate.  
**Evaluation boundary:** TRAIN speaker OOF only; no DEV or TEST label or score was opened by this pilot. All threshold searches are retrospective on exposed TRAIN data.

## Controlled setup

The candidate re-fit the actual **633,361-parameter SM13 `ScalarMixingTargetHead`** on the same 13 frozen Wav2Vec2 layers, 2,304D acoustic features, 39D target-phone reference, 12 epochs, batch 512, AdamW 0.001 / 0.01, original `[1, 2.5, 4]` weighted CE, seed 101, and the original four outer speaker folds. The only training change was a fixed `0.1 ×` pairwise softplus term that ranks each vowel or stop substitution above the 32 highest-scoring clean phones of the same manner in its minibatch. All 59,562 TRAIN target positions were scored out of fold. Gap-I and Fixed-O scores were copied from the frozen baseline during policy comparisons. The baseline was the archived SM13 seed-101 OOF predictions, not a retrained control. The candidate configuration was fixed before its outcomes were inspected; this is one exploratory screen, not a hyperparameter search.

The run used the cooperative GPU lease and finished in **204.5 seconds**. Four target-head checkpoints and exact-order OOF scores are saved beside this report. No full-TRAIN production model was replaced.

## Ranking results

| Target manner | Gold substitutions | Baseline AP | Candidate AP | Baseline AUROC | Candidate AUROC |
|---|---:|---:|---:|---:|---:|
| Vowel | 2,280 | 0.2348 | **0.2275** | 0.7280 | 0.7247 |
| Stop | 862 | 0.1968 | **0.2190** | 0.7044 | 0.7165 |
| Other | 4,239 | 0.5246 | **0.5188** | 0.8513 | 0.8481 |

Stop AP improved in all four outer held-speaker folds; vowel AP improved in two of four. Thus the pooled stop improvement has some speaker-group consistency, while vowel regression is material to the requested joint objective. Twelve speakers and one seed do not support a narrow uncertainty interval.

| Outer fold | Held speakers | Vowel AP baseline → candidate | Stop AP baseline → candidate |
|---:|---|---:|---:|
| 0 | ABA, ASI, BWC | 0.2264 → 0.1940 | 0.2167 → 0.2210 |
| 1 | EBVS, HJK, HKK | 0.2596 → 0.2617 | 0.1077 → 0.1227 |
| 2 | HQTV, LXC, NJS | 0.2526 → 0.2689 | 0.1492 → 0.1629 |
| 3 | PNV, SKA, SVBI | 0.2434 → 0.2192 | 0.2681 → 0.3171 |

## Operating-region results

Each row below changes one manner's threshold at a time while the other alert streams retain their respective model scores and the frozen O/I scores. Vowel and stop rows spend the warning budget separately and cannot be combined. Thresholds were chosen on the **same** OOF scores being summarized, so these are optimistic diagnostic frontiers.

| Manner and search rule | Baseline TP / FP | Baseline precision / recall / joint FWR | Candidate TP / FP | Candidate precision / recall / joint FWR |
|---|---:|---:|---:|---:|
| Vowel, FWR ≤0.20 | 81 / 143 | 36.16% / 3.55% / 0.1994 | 71 / 122 | 36.79% / 3.11% / 0.1956 |
| Vowel, FWR ≤0.20 and precision ≥50% | 9 / 9 | 50.00% / 0.39% / 0.1250 | 0 / 0 | undefined / 0% / 0.1278 |
| Stop, FWR ≤0.20 | 78 / 143 | 35.29% / 9.05% / 0.1994 | 80 / 126 | 38.83% / 9.28% / 0.1983 |
| Stop, FWR ≤0.20 and precision ≥50% | 22 / 19 | 53.66% / 2.55% / 0.1306 | 38 / 38 | 50.00% / 4.41% / 0.1494 |

A further descriptive threshold-transfer check picked each manner threshold on nine TRAIN speakers' OOF scores and applied it to three other TRAIN speakers. For stop with train precision ≥50%, the candidate pooled at **26 TP / 26 FP, 3.02% recall, 50.0% precision, FWR 0.1428**; the baseline yielded **19 TP / 26 FP, 2.20% recall, 42.22% precision, FWR 0.1344**. For vowel under a risk-only rule, candidate yielded **73 TP / 138 FP, 3.20% recall, 34.60% precision, FWR 0.2044**, versus baseline **94 TP / 151 FP, 4.12% recall, 38.37% precision, FWR 0.2039**. Threshold transfer here is **not fully nested independent calibration**: the models that produced the training speakers' OOF scores may themselves have been fit on some target held-speaker labels. This check is descriptive and cannot certify deployment thresholds.

## Decision and interpretation

The pilot does **not** establish that a loss-only change fixes the detection problem. It shifts the stop ranking slightly, but the useful-recall gain is small and vowels deteriorate. Its frozen-policy threshold still detects zero vowel substitutions and only one stop substitution in TRAIN OOF. The candidate should remain an exploratory artifact, with no replacement of the frozen policy or production checkpoint.

The next model hypothesis should add **measurable acoustic evidence** of vowel trajectories and stop closure/release transitions or test a phone/context-specific scorer, while retaining speaker-held-out validation and a fixed warning budget. Human review of the prepared audio packet should determine whether the intended feedback target includes the corpus's `*` deviation labels before such a model is promoted. Any favorable new design needs genuinely new speakers for final confirmation; the historical TEST and DEV_REPLICATION populations are exposed.

## Reproduce

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job SM13-MANNER-RANK-PROBE-S101 -- .venv/bin/python experiments/rq1_sm13_followup_audit_01/train_manner_rank_probe.py
.venv/bin/python experiments/rq1_sm13_followup_audit_01/evaluate_manner_rank_probe.py
```

Inputs: immutable TRAIN manifest and 13-layer acoustic cache; archived baseline `predictions_sm13_seed101.npz`; frozen O/I score arrays. Outputs: `MANNER_RANK_PROBE_OOF.npz`, `MANNER_RANK_PROBE_RESULTS.json`, four checkpoints, and this report.
