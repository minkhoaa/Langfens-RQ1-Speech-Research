# METHOD-BENCH-07 Agent A — Nested complementarity, seeds 101 and 202

## Results

Completed two full four-outer-fold nested evaluations. For every outer fold, the nine training speakers were divided into three inner held groups of three; each inner direct model and CTC head trained on the other six. The normalizers/meta-model were fit only on inner OOF rows. Outer neural heads trained on all nine outer-training speakers. Outer E4 scores came from the exact matching seed/fold MB05 CTC/E4 artifacts trained on those same nine outer-training speakers. Outer held-out labels were used only for final scoring. No DEV_SEARCH, global OOF normalization, or outer-label tuning was used.

This is **nested direct acoustic neural + phone-prior + E4 fusion** (the neural part is a direct event MLP, not neural CTC). Both seeds use the same architecture, objectives, epoch budgets, folds, and meta regularization. CTC seeds are deterministic `seed + outer_fold*10 + inner_fold`. E4-only meta feature is a coherent three-class simplex from softmax([0,SLLR,OLLR]); the learned output is multinomial LogisticRegression C=.1 with feature standardization fit on inner OOF only. The fixed 50:50 controls are prespecified simplex averages with no test-population fitting. The primary nested fusion result is AP-S/O; a separate supervisor-derived target-only warning-policy evaluation is reported below, without insertion/Macro-3 claims.

### Matched outer-fold AP-S/O

| Seed | Fold / held speakers | Neural | E4 | Phone prior | Fixed 50:50 neural/prior | Fixed 50:50 neural/E4 | Learned neural+prior+E4 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 101 | 0 ABA/ASI/BWC | .4030/.2322 | .2508/.2085 | .3196/.0761 | .4132/.2120 | .4020/.3102 | **.4411/.3223** |
| 101 | 1 EBVS/HJK/HKK | .3351/.1826 | .2566/.1517 | .3019/.0714 | .3382/.1765 | .3656/.1965 | **.3848/.2139** |
| 101 | 2 HQTV/LXC/NJS | .4110/.3709 | .3091/.3285 | .3877/.1657 | .4473/.3657 | .4370/.4612 | **.5039/.4662** |
| 101 | 3 PNV/SKA/SVBI | .3964/.2965 | .2737/.1916 | .2859/.0904 | .3732/.2878 | .4101/.3519 | **.4116/.3679** |
| 202 | 0 ABA/ASI/BWC | .4066/.2721 | .2649/.2175 | .3196/.0761 | .4287/.2522 | .3956/.3462 | **.4342/.3426** |
| 202 | 1 EBVS/HJK/HKK | .3453/.1437 | .2703/.1546 | .3019/.0714 | .3460/.1457 | .3712/.1911 | **.3930/.2193** |
| 202 | 2 HQTV/LXC/NJS | .4052/.3451 | .3330/.3416 | .3877/.1657 | .4453/.3451 | .4461/.4493 | **.5013/.4626** |
| 202 | 3 PNV/SKA/SVBI | .3801/.2704 | .2844/.1999 | .2859/.0904 | .3651/.2644 | .4040/.3181 | **.4063/.3531** |

Learned fusion exceeds standalone neural, standalone E4, phone prior, and fixed neural/prior blend for both classes in every fold/seed. Against fixed neural/E4 blend it is higher in 15/16 fold×class comparisons; seed202 F0 omission slightly favors fixed (`.3462` vs learned `.3426`). Pooled fixed neural/E4 O AP also slightly exceeds learned in both seeds. Thus learned meta-model is not universally best among fixed compositions. The pooled/fold aggregation differences and two-seed/four-group scope limit conclusions.

### Pooled concatenated outer predictions (secondary)

| Seed | Neural AP-S/O | E4 AP-S/O | Prior AP-S/O | Fixed neural/prior AP-S/O | Fixed neural/E4 AP-S/O | Learned fusion AP-S/O |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | .3841/.2657 | .2722/.2290 | .3070/.0883 | .3825/.2400 | .4027/.3308 | **.4233/.3251** |
| 202 | .3802/.2561 | .2882/.2368 | .3070/.0883 | .3825/.2366 | .4021/.3340 | **.4230/.3286** |

Foldwise AP is primary because fold-specific meta outputs may have calibration differences; pooled numbers are secondary aggregation only. Against the historical MB03 neural PLAIN controls on the same outer folds, learned fusion improves O AP in 4/4 folds for both seeds; S improves in only 2/4 folds and pooled S is essentially tied. Thus the clearest complementarity beyond the established neural reference is omission.

## Method, joins, and leakage controls

- Outer groups: F0 ABA/ASI/BWC; F1 EBVS/HJK/HKK; F2 HQTV/LXC/NJS; F3 PNV/SKA/SVBI.
- Direct predictor: frozen H-002 11-frame crop (`train_frame_context[:,3:14]`), crop mean/std/end-minus-start plus canonical reference-phone one-hot; MLP 256/128, C/S/O weighted CE [1,2.5,4], 12 epochs, AdamW lr .001/wd .01. Per-fit standardization only. This is a direct event MLP, not CTC.
- Phone prior: add-one smoothed C/S/O frequency by canonical/intended phone. Each inner OOF prior uses only that inner six-speaker training set; outer prior uses only the nine outer-training speakers. A discovered initial inner-prior leakage (all nine pool labels were used) was corrected, rerun, and the invalid prior outputs overwritten; only corrected outputs feed final receipts.
- CTC: cached frozen Wav2Vec2 Base 960h hidden state 6, 768 dimensions, 20-ms clock; linear 768→43, CTC blank 42, AdamW 1e-3/wd 1e-4, 8 epochs, batch 16, clip norm 5. Six-speaker inner fits each contain 900 utterances, held groups 450. Each seed has 12 new inner CTC heads; seed202 uses derived seeds 202+10×outer+inner.
- E4: MB05 mean aggregation of CTC sequence likelihood differences. Each inner model's substitution/insertion alternatives were estimated solely from its six-speaker training predictions and canonical references; the held inner speakers only receive scores. Outer E4 uses matching MB05 seed101/202 outputs and their matching outer-fold CTC models trained on nine outer-training speakers.
- Authoritative H-002 TRAIN event IDs are `{utterance_id, slot_idx, speaker_id}`. The direct script asserts 59,562 unique `(utterance_id,slot_idx)` events and exact label equality with `data/derived/l2arctic/manifests/train.jsonl`. E4 validation checks held-speaker assignments, per-event target-slot counts, 178,686 inner `(outer fold,event)` scores per seed, 59,562 complete unique outer event joins, and exact matching outer speakers. Saved predictions retain utterance ID, slot index, combined event ID, speaker, labels, and scores. No outer-test labels were used to fit any model/transform/meta-model.

## Compute, artifacts, and reproduction

CPU was used (no GPU lease needed); frozen MB03 W2V2-H6 cached train representations were reused. Seed101 direct nested refit time 138.06 s (receipt) and CTC/E4 nested 614.95 s; seed202 direct 145.45 s and CTC/E4 nested 613.05 s. Each CTC receipt records epoch losses and SHA256 per checkpoint. Existing MB03/05 artifacts were read only; no historical output overwritten.

Executable scripts, logs, output receipts/predictions, and 24 inner checkpoints are under `experiments/rq1_method_bench_07/agent_a/`. Main files include `nested_direct_seed{101,202}.py`, `ctc_nested_e4_seed{101,202}.py`, `evaluate_nested_e4_fusion{,_seed202}.py`, `nested_e4_fusion_seed{101,202}_predictions.npz`, receipts and CTC model checkpoints. `nested_ap_comparison.csv` contains all eight fold/seed rows including the fixed neural/E4 ablation. Prediction/code/manifest hashes are in receipts. Full implementation summary also available in output artifact for this run.

## Supervisor-derived target-only warning-budget evaluation

A separate nested operating analysis applies the same procedure to all target-only competitors: choose a max(P(S),P(O)) threshold from inner OOF scores at budgets 0.2/0.5/1/2 warnings per utterance, freeze it on each outer test fold, and use a fixed SHA-256 event-ID tie-break (no outer score-percentile ranking). Learned meta-model policy-selection scores are generated by a second inner-group cross-fit, not in-sample meta predictions. These are target-only warnings: insertion scores are not included, and no Macro-3/FWR-complete-system claim is made. This is a score-threshold policy analysis, not probability reliability calibration.

| Seed | Budget | Neural FWR / false warnings / target Macro-2 F1 | Fixed neural+E4 | Learned neural+prior+E4 |
|---:|---:|---:|---:|---:|
| 101 | 0.2 | .105 / 57 / .030 | .223 / 122 / .093 | .099 / 50 / .027 |
| 101 | 0.5 | .294 / 168 / .076 | .495 / 307 / .158 | .344 / 193 / .078 |
| 101 | 1.0 | .686 / 474 / .142 | .985 / 713 / .226 | .902 / 568 / .171 |
| 101 | 2.0 | 1.602 / 1,315 / .248 | 1.966 / 1,623 / .319 | 1.841 / 1,346 / .290 |
| 202 | 0.2 | .160 / 85 / .044 | .231 / 133 / .092 | .112 / 55 / .031 |
| 202 | 0.5 | .398 / 244 / .089 | .515 / 358 / .150 | .377 / 219 / .084 |
| 202 | 1.0 | .802 / 577 / .154 | 1.009 / 756 / .227 | .942 / 604 / .175 |
| 202 | 2.0 | 1.792 / 1,535 / .252 | 2.021 / 1,678 / .325 | 1.845 / 1,382 / .282 |

The simple fixed neural/E4 blend has higher target Macro-2 F1 than learned fusion at all four budgets in both seeds; learned fusion exceeds neural-only at 1/2 budgets but not consistently at 0.2/0.5. Thus the AP gain does **not** establish a superior operating policy. Realized FWR is target-only and may differ from requested budget. Full per-class precision/recall/F1 and thresholds are in `reports/METHOD_BENCH_07_TARGET_WARNING_BUDGETS.csv`; pooled count aggregation is in `_SUMMARY.csv`.

## Residual risks

1. Two seeds/four outer speaker groups are still small for uncertainty estimation; replicate or report speaker-level intervals before strong claims.
2. Inner six-speaker vs outer nine-speaker fit calibration may shift.
3. Direct model uses frozen H-002 embeddings and intended-phone input; phone identity is intentionally strong prior information.
4. Target-only thresholds/F1 were evaluated, but no validated inner insertion scorer was fused; no complete S/O/I warning policy, insertion F1, Macro-3 F1 or full-position FWR is claimed.
