# METHOD-BENCH-06 Agent A — Matched E4/neural evidence integration

## Verdict and scope

**Descriptive matched OOF analysis only; nested learned integration not run.** Recovered MB03 `RQ1-MB03-C-SPEAKER-GRL-GROUPCV-S101` PLAIN target probabilities and MB05 canonical E4 mean seed101 event scores, then matched every target on exact `(utterance_id, slot_idx)`. The 59,562 target events contain 7,381 S and 1,825 O positives across 12 TRAIN speakers and the four authoritative held-speaker groups. MB03's receipt identifies the architecture/training protocol (five epochs, batch 1,024; nine speakers train, three held out). MB05 folds are ABA/ASI/BWC; EBVS/HJK/HKK; HQTV/LXC/NJS; PNV/SKA/SVBI. E4 scores were checked against per-utterance target-array lengths. No DEV_SEARCH or protected data, fitting, threshold selection, or GPU/checkpoint was used.

The earlier fixed pooled percentile-rank mean had attractive descriptive pooled AP (.44547 S/.35708 O versus neural .42321/.27671), but pooled score normalization uses all OOF events, so it is not an untouched outer-fold result. This does **not** justify a learned-fusion claim. In this deeper analysis E4 alone is below neural AP in both classes, top-K results vary by fold, and intended-phone identity is a material S control. An expensive nested four-fold design is deferred until fold-valid evidence demonstrates conditional E4 value. Never interpret ordinary pooled OOF stacking as cross-fitted.

## Exact matched pooled results and diagnostics

| Class | Support / prevalence | Neural AP | E4 AP | Spearman | Neural max-probability lowest quartile: n / prevalence; neural AP / E4 AP | Within-phone E4 percentile AP | Phone-prevalence-only AP* |
|---|---:|---:|---:|---:|---:|---:|---:|
| S | 7,381 / .12392 | .42321 | .31972 | .41049 | 14,891 / .21577; .33016 / .33927 | .22168 | .33608 |
| O | 1,825 / .03064 | .27671 | .25936 | .22936 | 14,891 / .05923; .19626 / .28315 | .20351 | .10240 |

*Phone prevalence is a retrospective identity diagnostic computed from all labels, not a valid trained/deployable baseline. Within-phone E4 percentile AP ranks E4 only among events with the same intended phone; it removes between-phone rank offsets descriptively but is not a cross-fitted conditional model. S's phone-only AP is substantial (.33608), higher than E4 AP, and within-phone E4 AP drops to .22168. This warns that E4 S ranking may inherit intended-phone prevalence. O's phone-only AP is lower (.10240), while within-phone E4 remains .20351; this is suggestive of within-phone omission ranking, but not a trained conditional-information test.

The low-neural-confidence subset is defined as the bottom quartile of neural `max(P(C),P(S),P(O))`, not the bottom quartile of class score. In this subset E4 AP exceeds neural AP for both S (.33927 vs .33016) and O (.28315 vs .19626); prevalence is .21577 and .05923 respectively. This supports a targeted follow-up, not independent conditional evidence: confidence selection and phone composition are retrospective pooled OOF diagnostics.

### Per-fold AP and precision/recall at top 5%

P@5%/R@5% selects the highest 5% of each fold's events **separately for each method** (K and supports differ slightly by fold). `Both TP` is intersection of true-positive events in the two selected sets. It should not be confused with model union benefit.

| Fold (speakers) | Class | n / positives | Neural AP | E4 AP | Neural P@5 / R@5 | E4 P@5 / R@5 | Both TP; neural-only TP; E4-only TP |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 (ABA/ASI/BWC) | S | 14,816 / 1,859 | .4440 | .2858 | .5884 / .2345 | .4062 / .1619 | 131; 305; 170 |
| 1 (EBVS/HJK/HKK) | S | 14,848 / 1,524 | .3610 | .3041 | .4980 / .2428 | .3943 / .1923 | 111; 259; 182 |
| 2 (HQTV/LXC/NJS) | S | 14,989 / 2,079 | .4595 | .3598 | .6520 / .2352 | .4840 / .1746 | 100; 389; 263 |
| 3 (PNV/SKA/SVBI) | S | 14,909 / 1,919 | .4169 | .3279 | .5429 / .2110 | .4584 / .1782 | 120; 285; 222 |
| 0 (ABA/ASI/BWC) | O | 14,816 / 357 | .2656 | .2337 | .2254 / .4678 | .2375 / .4930 | 88; 79; 88 |
| 1 (EBVS/HJK/HKK) | O | 14,848 / 301 | .1903 | .1755 | .1736 / .4286 | .1682 / .4153 | 54; 75; 71 |
| 2 (HQTV/LXC/NJS) | O | 14,989 / 773 | .3555 | .3688 | .4027 / .3907 | .4107 / .3984 | 113; 189; 195 |
| 3 (PNV/SKA/SVBI) | O | 14,909 / 394 | .2997 | .2148 | .2668 / .5051 | .2024 / .3832 | 69; 130; 82 |

At pooled top 5% (K=2,979 per method), S: neural P/R=.0930/.0375 and E4=.0681/.0275 (277 vs 203 TP; 194 E4 TP outside neural top set). O: neural=.0201/.0329 and E4=.0188/.0307 (60 vs 56 TP; 56 E4 TP outside neural top set). (Fold diagnostics in `expanded_error_diagnostics.json` also include top 1% and 10%, speaker-level results, and overlap.) E4's non-overlapping true positives demonstrate different rankings, not utility at a common operating policy. For S the neural top set is consistently stronger; O E4 has an AP and top-K advantage only in fold 2, not a stable per-fold win.

## Complementarity assessment and nested design gate

The original descriptive pooled percentile rank-mean AP gains (+.02227 S, +.08037 O) and the confidence-subset O result motivate investigation. But those gains are not fold-local calibrated evidence; score rankings diverge moderately (Spearman .41/.23), E4-only AP trails neural overall, and phone identity explains considerable S ranking. Unions and low correlation alone do not establish complementarity beyond intended-phone identity or confidence. No nested cross-fit score is claimed.

If follow-up evidence justifies it, the required design is strict: for each outer 3-speaker fold, use only the other nine speakers; make inner grouped OOF base predictions among those nine; fit any meta-model, normalization, calibration, or threshold solely there; fit base models for that outer fold using those same nine speakers and score the untouched three. Outer labels may not enter any choice. Existing MB03/MB05 OOF predictions were generated under four fixed three-speaker holdouts and cannot simply be pooled and stacked to satisfy that contract. A nested result would require retraining/re-scoring base models to create the inner predictions within each outer training set.

## Artifacts and reproduction

Owned artifacts under `experiments/rq1_method_bench_06/agent_a/`:
- `diagnose_matched_oof.py` and original `matched_score_diagnostics.csv`, `complementarity_diagnostics.json` (historical initial analysis preserved).
- `expanded_error_diagnostics.py` and `expanded_error_diagnostics.json` (this report's exact-key expanded diagnostics, including all fold/speaker AP, top 1/5/10%, overlap, low-confidence and phone controls).

Commands from repository root:

```bash
.venv/bin/python experiments/rq1_method_bench_06/agent_a/diagnose_matched_oof.py
.venv/bin/python experiments/rq1_method_bench_06/agent_a/expanded_error_diagnostics.py
```

Inputs/provenance: MB03 `speaker_grouped_oof_predictions.npz`, `train_event_ids.jsonl`, and `speaker_grouped_oof_receipt.json`; MB05 fold-specific seed101 E4 mean score files and grouped-fold receipts; TRAIN manifests. The script uses NumPy/SciPy/scikit-learn on CPU, no training and no checkpoint; runtime under two seconds. No CUDA lease/device. The speaker-level values are retained in JSON rather than compressed into an unstable prose ranking. No claims extend beyond this TRAIN OOF population; speaker/fold and phone composition are entangled, no external population was tested, and no threshold/policy was evaluated.
