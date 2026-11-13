# METHOD-BENCH-05 — Tái lập thí nghiệm

## Môi trường và dữ liệu

- Repository: `/home/khoa/Projects/New-Method-Research`; Python executable `.venv/bin/python`.
- Evaluator: EVAL-V3.1. Dataset split: TRAIN 1,800 utterances / 12 speakers; four authoritative speaker groups. DEV_CAL chỉ là development; DEV_SEARCH đã exposed nên không dùng chọn model/policy. Không truy cập protected holdout.
- NVIDIA driver không khả dụng trong lần chạy này; các thí nghiệm MB05 ghi nhận CPU. Không có peak memory instrumentation. CTC heads dùng cached frozen Wav2Vec2 Base 960h hidden state 6, 768 chiều, 20 ms, phone tokenizer 42 nonblank symbols, blank ID 42.
- Không ghi đè checkpoints/posteriors MB03/04. Mọi MB05 output nằm dưới `experiments/rq1_method_bench_05/{agent_a,agent_b,agent_c}/`.

## Fold-valid CTC recognizers và E1/E3/E4

Fold definitions:

| Fold | Speakers held out | Speakers used for fit |
|---:|---|---|
| 0 | ABA / ASI / BWC | Remaining nine TRAIN speakers |
| 1 | EBVS / HJK / HKK | Remaining nine TRAIN speakers |
| 2 | HQTV / LXC / NJS | Remaining nine TRAIN speakers |
| 3 | PNV / SKA / SVBI | Remaining nine TRAIN speakers |

Train fold recognizers (the script loads cached frame features; no feature encoder retraining):

```bash
.venv/bin/python experiments/rq1_method_bench_05/agent_b/train_fold_ctc.py --fold 0 --seed 101
.venv/bin/python experiments/rq1_method_bench_05/agent_b/train_fold_ctc.py --fold 0 --seed 202
```

Repeat with `--fold 1`, `--fold 2`, `--fold 3`. Receipts identify `RQ1-MB05-B-CTC-F{fold}-S{seed}`, speaker provenance, posterior hashes, PER, training history, and wall time. Every fold had 85 valid batches/epoch and zero CTC topology skips.

Matched E1/E3 baselines per fold:

```bash
.venv/bin/python experiments/rq1_method_bench_05/agent_b/matched_baselines_fold.py --fold 0 --prefix mb05_f0_s101_ --split train_speaker_holdout
```

Use the corresponding posterior prefix for each fold/seed (e.g. `mb05_f2_s202_`). Canonical E4 **mean** scoring:

```bash
.venv/bin/python experiments/rq1_method_bench_05/agent_b/e4_fold.py --fold 0 --seed 101 --agg mean
```

`e4_fold.py` defaults to max, so `--agg mean` is required to reproduce the canonical MB04 E4 configuration. `--agg max` is retained as a separately labeled sensitivity. The primary four-fold results use `sub-k=4`, `ins-k=8`; candidate statistics are fold-train-only. The pooled OOF score ledgers, raw AP, PR points, P@K/R@K, and per-speaker seed101 AP can be replayed with:

```bash
.venv/bin/python experiments/rq1_method_bench_05/agent_b/summarize_grouped_oof.py
.venv/bin/python experiments/rq1_method_bench_05/agent_b/summarize_grouped_oof_operating.py
```

Main receipts include `RQ1-MB05-B-CTC-F{0,1,2,3}-S{101,202}_RECEIPT.json`. Score files and posterior exports are `mb05_f{fold}_s{seed}_...`; the detailed inventory is in the Agent B report. Derived summary tables include `grouped_oof_raw_ap.csv`, `grouped_oof_summary.json`, `grouped_oof_pr_curves.csv`, `grouped_oof_precision_recall_at_k.csv`, `grouped_oof_speaker_ap_seed101.csv`, and `grouped_oof_operating_metrics.csv`.

## Exact-gap collision and temporal scores

Collision audit and matched PR curves:

```bash
.venv/bin/python experiments/rq1_method_bench_05/agent_a/collision_audit.py
.venv/bin/python experiments/rq1_method_bench_05/agent_a/save_pr_curves.py
.venv/bin/python experiments/rq1_method_bench_05/agent_a/temporal_gap_score.py
.venv/bin/python experiments/rq1_method_bench_05/agent_a/candidate_inventory_ablation.py
```

These reuse fixed MB03/04 CTC posteriors on the 300-utterance HJK/HKK TRAIN speaker holdout, with 10,139 gaps and 46 insertion positives. T2 uses Viterbi predicted boundaries and a ±2-frame candidate log-posterior/blank score; no gold timing is used. The candidate-set script compares the fixed TRAIN-fit top-8 against all 42 nonblank tokenizer symbols, preserving max aggregation. It reports full-candidate coverage/collision as post-hoc diagnostics only. The forward-backward T3 attempt did not yield a valid score; do not attempt to reproduce an unrecorded result as if it were measured.

## Independent direct-event model

Primary model, acoustic/reference controls, and PR diagnostics:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_05/agent_c/direct_event_mlp.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_05/agent_c/direct_event_mlp_controls.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/rq1_method_bench_05/agent_c/evaluate_direct_event.py
```

The direct MLP uses frozen H-002 layer-6 target crops, TRAIN-only fold standardization and supervision, canonical intended-phone one-hot, and predicts C/S/O. Receipts, OOF arrays, controls, PR curves and class supports are under `experiments/rq1_method_bench_05/agent_c/`. It has no exact-gap insertion output.

## Main artifact map

- A report and scores: `reports/METHOD_BENCH_05_A_TEMPORAL_LOCALIZATION.md`, `experiments/rq1_method_bench_05/agent_a/`.
- B report and fold checkpoints/posteriors: `reports/METHOD_BENCH_05_B_GROUPED_E4_VALIDATION.md`, `experiments/rq1_method_bench_05/agent_b/`.
- C report and OOF direct-event controls: `reports/METHOD_BENCH_05_C_DIRECT_EVENT_RESEARCH.md`, `experiments/rq1_method_bench_05/agent_c/`.
- Supervisor outputs: campaign report, hypothesis verdicts, leaderboard, ablation table, and next-research report in `reports/`.

## Reproduction cautions

Raw per-event AP is the primary ranking metric. Do not substitute thresholded F1 or warning rate. EVAL-V3.1 normalized-target AP in saved E1/E3 summaries may differ slightly from raw class-score AP used in pooled OOF tables; both are valid but use different score transforms. Seed 101 and 202 are repeated fits on the same 12-speaker dataset, not independent cohorts. `DEV_SEARCH` remains development-exposed and was not used for model selection here.
