# METHOD-BENCH-05 B — Xác nhận E4 theo bốn nhóm người nói

## Kết luận

Trên bốn fold TRAIN giữ speaker, E4 mean cải thiện xếp hạng substitution và omission so với E3 ở mọi fold, qua cả seed 101 và 202. AP insertion không cải thiện ổn định. Đây là OOF theo speaker thật: mỗi CTC recognizer và thống kê đối thủ E4 chỉ fit trên chín speaker, rồi chấm ba speaker bị giữ lại.

Pooled raw AP seed 101: E3 S/O/I=.23888/.09205/.03462; E4 mean=.31972/.25936/.03159. Seed 202: E3=.24859/.10568/.03717; E4=.32829/.27157/.03743. E4 tăng S và O ở cả 8 fold-seed comparisons; I tăng ở 3/8. Macro-3 pooled AP tăng .12185→.20356 (seed101) và .13048→.21243 (seed202). Insertion remains weak and near E3, so the cross-speaker experiment does not establish an E4 insertion advantage.

## Giao thức và thực thi

Fold 0 giữ ABA/ASI/BWC; fold 1 EBVS/HJK/HKK; fold 2 HQTV/LXC/NJS; fold 3 PNV/SKA/SVBI. Mỗi fold fit chín TRAIN speakers, đánh giá ba speaker khác. OOF gồm 1,800 utterances, 59,562 target events, 61,362 gaps; mỗi utterance được dự đoán đúng một lần khi speaker của nó bị giữ lại.

Recognizer là Wav2Vec2 Base 960h frozen hidden state layer 6, 768 chiều, frame clock 20 ms, realized-phone targets, tokenizer và blank ID 42 như MB03; trainable head Linear(768,43). AdamW lr=1e-3, weight decay=1e-4, batch 16, 8 epochs, gradient clip 5, 85 batches/epoch, bốn Torch threads. Dùng cached acoustic features. Không dùng gold timing hoặc realized phones làm feature. Trong từng fold, E1/E3/E4 dùng cùng posterior arrays; thống kê E4 chỉ dùng fit speakers.

GPU driver không khả dụng, nên chạy CPU. Seed101 training wall seconds theo fold: 62.7, 59.9, 66.3, 61.4; seed202: 70.8, 70.7, 63.9, 61.7. E4 mean scoring mất 42–47 giây/fold, E1/E3 mất 80–100 giây/fold. Không có batch topology nào bị loại; peak memory không được instrument.

E4 mean là aggregation chuẩn METHOD-BENCH-04 và là kết quả chính. Max giữ riêng làm sensitivity analysis. Một số output seed101 ban đầu chỉ có max, nên mean được tính lại từ cùng posterior mà không retrain CTC. Seed202 max chỉ có fold 0–1; không dùng để suy luận pooled.

## PER recognizer

Beam-10 realized-phone PER trên ba held-out speakers:

| Seed | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
|---:|---:|---:|---:|---:|---:|
| 101 | .40754 | .44347 | .43632 | .40579 | .42328 |
| 202 | .38016 | .40848 | .41539 | .39325 | .39932 |

Trên 8 điểm fold-seed, Pearson r giữa beam PER và E4 AP là .31 (S), .19 (O), .29 (I). Đây là mô tả trên cùng bốn nhóm lặp lại, không phải kiểm định độc lập; PER tốt hơn không dự đoán MDD AP tốt hơn theo cách đơn điệu.

## Raw AP theo fold

Bảng dùng raw per-event score cho mỗi class để giữ cùng score semantics với pooled raw-score AP. Macro-3 là trung bình AP-S, AP-O, AP-I.

### Seed 101

| Fold | Method | AP-S | AP-O | AP-I | Macro-3 AP |
|---:|---|---:|---:|---:|---:|
| 0 | E1 | .15535 | .04991 | .02195 | .07574 |
| 0 | E3 | .22287 | .09824 | .03878 | .11996 |
| 0 | E4 mean | .28580 | .23373 | .02545 | .18166 |
| 1 | E1 | .15493 | .02799 | .00984 | .06426 |
| 1 | E3 | .22654 | .04856 | .02513 | .10008 |
| 1 | E4 mean | .30410 | .17550 | .03463 | .17141 |
| 2 | E1 | .18701 | .07954 | .01267 | .09307 |
| 2 | E3 | .26696 | .13553 | .04575 | .14941 |
| 2 | E4 mean | .35983 | .36881 | .04471 | .25778 |
| 3 | E1 | .17117 | .04799 | .01045 | .07654 |
| 3 | E3 | .23900 | .08640 | .03328 | .11956 |
| 3 | E4 mean | .32791 | .21475 | .02938 | .19068 |

### Seed 202

| Fold | Method | AP-S | AP-O | AP-I | Macro-3 AP |
|---:|---|---:|---:|---:|---:|
| 0 | E1 | .16401 | .04946 | .02161 | .07836 |
| 0 | E3 | .23490 | .10315 | .03744 | .12516 |
| 0 | E4 mean | .29301 | .24529 | .03338 | .19056 |
| 1 | E1 | .15593 | .02954 | .01086 | .06545 |
| 1 | E3 | .23133 | .06202 | .02699 | .10678 |
| 1 | E4 mean | .31073 | .18620 | .05329 | .18341 |
| 2 | E1 | .19633 | .08222 | .01807 | .09887 |
| 2 | E3 | .27949 | .15948 | .05769 | .16556 |
| 2 | E4 mean | .37261 | .38316 | .04724 | .26767 |
| 3 | E1 | .17716 | .04845 | .01415 | .07992 |
| 3 | E3 | .24660 | .08898 | .02961 | .12173 |
| 3 | E4 mean | .33320 | .22391 | .03126 | .19612 |

## Concatenated TRAIN OOF

Pooled AP is evaluated over all four held-out ledgers, not averaged per-fold.

| Seed | Method | AP-S | AP-O | Exact-gap AP-I | Macro-3 AP |
|---:|---|---:|---:|---:|---:|
| 101 | E1 | .16686 | .05084 | .01271 | .07680 |
| 101 | E3 | .23888 | .09205 | .03462 | .12185 |
| 101 | E4 mean | **.31972** | **.25936** | .03159 | **.20356** |
| 202 | E1 | .17347 | .05217 | .01576 | .08047 |
| 202 | E3 | .24859 | .10568 | .03717 | .13048 |
| 202 | E4 mean | **.32829** | **.27157** | .03743 | **.21243** |

Supports S=7,381, O=1,825, I=560. Prevalence and prior-only AP are .123921, .030640, .009126. E4 is above prior-only ranking for each class, but insertion advantage is small: pooled I is below E3 for seed101 and effectively tied for seed202.

An AP discrepancy is due to score semantics, not label alignment: the saved matched E1/E3 EVAL-V3.1 summary computes target AP after mapping S/O marginals through a normalized three-way probability vector; the pooled raw ledger ranks raw class-specific scores. The fold AP tables here use raw scores for E1, E3 and E4, matching pooled calculation. EVAL-transformed scores remain reported separately under operating metrics.

## Held-out speaker AP, seed 101

Raw AP-S/O/I; supports are gold-positive S/O/I counts.

| Speaker | Supports S/O/I | E3 AP-S/O/I | E4 mean AP-S/O/I |
|---|---:|---:|---:|
| ABA | 402/31/76 | .15621/.01869/.05491 | .20615/.11951/.03256 |
| ASI | 761/120/25 | .23941/.10541/.02243 | .28612/.20909/.01466 |
| BWC | 696/206/33 | .28024/.12698/.04535 | .36615/.29833/.04899 |
| EBVS | 876/194/81 | .29393/.08206/.03436 | .37437/.25590/.05252 |
| HJK | 330/4/7 | .17357/.00253/.00218 | .25859/.01708/.00345 |
| HKK | 318/103/39 | .15291/.06428/.03518 | .26196/.14118/.03307 |
| HQTV | 809/477/64 | .30886/.23779/.05539 | .37235/.50944/.05195 |
| LXC | 709/216/52 | .25545/.11359/.05065 | .38155/.25658/.06488 |
| NJS | 561/80/34 | .22070/.05304/.02651 | .32136/.16824/.02190 |
| PNV | 631/246/16 | .28205/.16467/.02382 | .33509/.34559/.07056 |
| SKA | 571/59/105 | .21360/.03849/.07381 | .33351/.15233/.04638 |
| SVBI | 717/89/28 | .23194/.04942/.00997 | .32328/.13333/.01687 |

S AP improves for all twelve speakers; O improves for all twelve, though HJK has only four omission positives. I AP improves for six speakers (BWC, EBVS, HJK, LXC, PNV, SVBI) and decreases for six (ABA, ASI, HKK, HQTV, NJS, SKA). Low support makes some per-speaker AP estimates unstable.

## EVAL-V3.1 operating summaries

Raw AP ranks events; the default operating policy has different behavior. Four-fold mean Macro-3 Error F1 / Macro-4 mixed-domain F1:

| Seed | Method | Macro-3 F1 | Macro-4 F1 | Binary MD AP | Joint FWR / utterance | False alerts / 1000 positions |
|---:|---|---:|---:|---:|---:|---:|
| 101 | E3 | .1998 | .3474 | .3099 | 11.19 | 166.5 |
| 101 | E4 mean | .1561 | .3139 | .4216 | 13.61 | 202.5 |
| 202 | E3 | .2081 | .3570 | .3175 | 10.53 | 156.8 |
| 202 | E4 mean | .1654 | .3246 | .4288 | 12.35 | 183.8 |

E4’s better S/O ranking AP did not improve default point F1. Joint FWR remains high; no low-warning operating claim is made. Per-fold EVAL-V3.1 summaries hold class supports, coverage, binary MD metrics, and warning counts.

The pooled EVAL-V3.1 point-policy metrics below are computed by concatenating each event's prediction from its own held-out fold. Target decisions use each method's fixed saved class mapping and argmax; gap decisions use the saved E1/E3 probability >= 0.5 or E4 sigmoid(log-likelihood-ratio) >= 0.5. No threshold was selected from held-out labels. Supports are constant across methods/seeds: Correct=50,356, S=7,381, O=1,825, I=560.

| Seed | Method | Class | Precision | Recall | F1 |
|---:|---|---|---:|---:|---:|
| 101 | E1 | S | .2381 | .3761 | .2916 |
| 101 | E1 | O | .0639 | .6066 | .1157 |
| 101 | E1 | I | .0485 | .0911 | .0633 |
| 101 | E1 | Correct | .9416 | .5720 | .7117 |
| 101 | E3 | S | .2812 | .4636 | .3501 |
| 101 | E3 | O | .0908 | .5397 | .1555 |
| 101 | E3 | I | .0654 | .1911 | .0974 |
| 101 | E3 | Correct | .9394 | .6817 | .7901 |
| 101 | E4 mean | S | .3076 | .2409 | .2702 |
| 101 | E4 mean | O | .0840 | .7940 | .1519 |
| 101 | E4 mean | I | .0280 | .2411 | .0502 |
| 101 | E4 mean | Correct | .9363 | .6792 | .7873 |
| 202 | E1 | S | .2502 | .3924 | .3056 |
| 202 | E1 | O | .0671 | .5907 | .1205 |
| 202 | E1 | I | .0614 | .1268 | .0828 |
| 202 | E1 | Correct | .9427 | .5976 | .7315 |
| 202 | E3 | S | .2947 | .4715 | .3627 |
| 202 | E3 | O | .0986 | .5436 | .1669 |
| 202 | E3 | I | .0669 | .2000 | .1002 |
| 202 | E3 | Correct | .9386 | .7026 | .8036 |
| 202 | E4 mean | S | .3407 | .2333 | .2769 |
| 202 | E4 mean | O | .0886 | .8099 | .1597 |
| 202 | E4 mean | I | .0363 | .2482 | .0633 |
| 202 | E4 mean | Correct | .9355 | .7025 | .8024 |

Pooled binary MD precision/recall/F1/AP are E1=.2561/.8061/.3887/.2364, E3=.3037/.7593/.4339/.3139, E4=.2986/.7471/.4267/.4278 at seed101; seed202 E1=.2669/.8014/.4005/.2446, E3=.3151/.7485/.4435/.3231, E4=.3111/.7348/.4371/.4356. Pooled Macro-3 Error F1 / Macro-4 F1 are seed101 E1=.1568/.2956, E3=.2010/.3482, E4=.1574/.3149; seed202 E1=.1696/.3101, E3=.2100/.3584, E4=.1666/.3256. These pooled point metrics differ slightly from the fold-average summaries above because their aggregation populations differ.

The complete fold and pooled class P/R/F1/support and binary MD metrics are in [grouped_oof_operating_metrics.csv](../experiments/rq1_method_bench_05/agent_b/grouped_oof_operating_metrics.csv) and [grouped_oof_operating_metrics.json](../experiments/rq1_method_bench_05/agent_b/grouped_oof_operating_metrics.json). The compact JSON (184 KB) retains EVAL-V3.1 confusion matrices, per-fold summaries, exact policies, coverage, and false-warning counts; it omits joined per-event rows.

## Max sensitivity, verdict and limitations

Seed101 max E4 AP-S/O/I by folds 0–3: .29623/.23373/.03154; .31190/.17550/.04138; .37187/.36881/.04881; .33644/.21475/.03454. S/O remain strong; I beats E3 in one fold. Seed202 max was run only for folds 0–1: .30059/.24529/.04027 and .31541/.18620/.05396; no pooled seed202 max claim.

**Supported:** E4 mean S/O ranking gains transfer across four held-out TRAIN speaker groups and keep the same direction across seeds 101 and 202.

**Not supported:** E4 insertion ranking is better than E3 across speakers. Five of eight fold-seed comparisons favor E3; three favor E4, with pooled AP-I lower/tied.

**Limitations:** The folds exhaust 12 TRAIN speakers, not an external cohort. Seeds repeat fits over the same speakers, not independent evaluation populations. Speaker and fold error/phone composition remain entangled. A frozen-encoder linear head does not establish a ceiling for CTC or direct-event methods. No protected holdout or DEV_SEARCH selection was used.

## Reproduction and artifacts

From repository root, substitute fold 0–3 and seed 101/202:

    .venv/bin/python experiments/rq1_method_bench_05/agent_b/train_fold_ctc.py --fold 0 --seed 101
    .venv/bin/python experiments/rq1_method_bench_05/agent_b/matched_baselines_fold.py --fold 0 --prefix mb05_f0_s101_ --split train_speaker_holdout
    .venv/bin/python experiments/rq1_method_bench_05/agent_b/e4_fold.py --fold 0 --seed 101 --agg mean
    .venv/bin/python experiments/rq1_method_bench_05/agent_b/summarize_grouped_oof.py
    .venv/bin/python experiments/rq1_method_bench_05/agent_b/summarize_grouped_oof_operating.py

Fold contract, scripts, checkpoints, posterior arrays, decodes, score ledgers and receipts are under experiments/rq1_method_bench_05/agent_b/. Derived files: grouped_oof_raw_ap.csv, grouped_oof_summary.json, grouped_oof_pr_curves.csv, grouped_oof_precision_recall_at_k.csv, grouped_oof_speaker_ap_seed101.csv. Recognizer receipts record fit/held-out speakers, metrics, durations and posterior hashes. Preserve the difference between raw score AP and EVAL-V3.1 normalized-target AP when comparing outputs.

