# METHOD-BENCH-06 — Phân tích complementarity (E4 vs neural)

**Status:** matched TRAIN OOF diagnostic; no learned integration, calibration, or nested validation.

## Matching and populations

MB03 PLAIN grouped neural outputs (`RQ1-MB03-C-SPEAKER-GRL-GROUPCV-S101`) joined to MB05 E4 seed101 canonical mean at exact `(utterance_id, slot_idx)`. All 59,562 TRAIN target events align; supports S=7,381, O=1,825; four folds are ABA/ASI/BWC; EBVS/HJK/HKK; HQTV/LXC/NJS; PNV/SKA/SVBI. No DEV_SEARCH fitting.

## Main evidence

| Class | Neural AP | E4 AP | Spearman | Phone-only AP* | Within-phone E4 AP* | E4 over neural in low max-prob quartile |
|---|---:|---:|---:|---:|---:|---:|
| S | .42321 | .31972 | .41049 | .33608 | .22168 | .33927 vs .33016 |
| O | .27671 | .25936 | .22936 | .10240 | .20351 | .28315 vs .19626 |

*Phone controls use all labels and are retrospective descriptive diagnostics, not held-fold-fitted models. Low-confidence group is the bottom quartile of neural max(C,S,O), n=14,891; descriptive selection uses pooled OOF. It is not proof E4 contributes conditional signal.

| Fold | Speakers | AP-S neural/E4 | AP-O neural/E4 |
|---|---|---:|---:|
| 0 | ABA/ASI/BWC | .4440/.2858 | .2656/.2337 |
| 1 | EBVS/HJK/HKK | .3610/.3041 | .1903/.1755 |
| 2 | HQTV/LXC/NJS | .4595/.3598 | .3555/.3688 |
| 3 | PNV/SKA/SVBI | .4169/.3279 | .2997/.2148 |

At each method's pooled top 5%, neural selected 277 S / 60 O true positives; E4 selected 203 S / 56 O. E4's unique top-5% TPs outside neural set: 194 S / 56 O. Supports ranking diversity only; top-K supports are method-specific and FP costs are not policy-matched.

A pooled percentile-rank mean yielded AP `.44547 S/.35708 O`, but normalizing over pooled OOF sees the whole evaluation score distribution. This is **exploratory only**, not a fixed fold-valid gain and not a trained model. E4 alone trails neural pooled AP in both classes; E4 exceeds neural O AP in only fold 2. Rank correlations are moderate and do not establish conditional value.

## Verdict

E4 rankings contain distinct cases; useful independent evidence beyond neural confidence and intended-phone priors remains **INCONCLUSIVE**. Phone identity is a strong S control, and within-phone ranking reduces E4 S AP. O within-phone E4 AP remains suggestive, but needs fold-safe modeling. Do not spend on an expensive nested ensemble until an explicit conditional signal survives cross-fitted tests. Any future confirmatory test must outer-hold three speakers, make inner grouped OOF base scores on the other nine, train meta policy only there, train outer bases only on those nine, then apply to untouched outer speakers. Existing pooled OOF cannot meet that requirement.

## Reproduction

```bash
.venv/bin/python experiments/rq1_method_bench_06/agent_a/diagnose_matched_oof.py
.venv/bin/python experiments/rq1_method_bench_06/agent_a/expanded_error_diagnostics.py
```

See `experiments/rq1_method_bench_06/agent_a/expanded_error_diagnostics.json` for full speaker/fold/top-1/5/10% supports, overlaps and PR diagnostics. CPU-only, under two seconds, no checkpoint.
