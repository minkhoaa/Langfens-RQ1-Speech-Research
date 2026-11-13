# METHOD-BENCH-06 — Reproduction receipts

## Provenance and splits

Repository `/home/khoa/Projects/New-Method-Research`; Python `.venv/bin/python`; Torch 2.6.0+cu124. Reuse the official TRAIN 1,800 utterances / 12 speakers and four exact groups:

| Fold | held-out |
|---|---|
| 0 | ABA / ASI / BWC |
| 1 | EBVS / HJK / HKK |
| 2 | HQTV / LXC / NJS |
| 3 | PNV / SKA / SVBI |

No DEV_SEARCH fitting or protected data. CUDA was directly rechecked: `nvidia-smi` showed idle NVIDIA GeForce GTX 1650 4GB; PyTorch reported `cuda_available=True`. MB06 analytical scripts/GMM ran CPU; no CUDA training in this campaign; no peak-memory measurements.

## A: matched event diagnostics

Sources: MB03 `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_predictions.npz`, `train_event_ids.jsonl`, receipt; MB05 fold score ledgers `experiments/rq1_method_bench_05/agent_b/mb05_f{0..3}_s101_e4_mean_scores.jsonl`, receipts, labels/manifests. Agent A exact joined pair `(utterance_id,slot_idx)`; check counts and identity, not merely row totals. Artifact `experiments/rq1_method_bench_06/agent_a/expanded_error_diagnostics.json`.

```bash
.venv/bin/python experiments/rq1_method_bench_06/agent_a/diagnose_matched_oof.py
.venv/bin/python experiments/rq1_method_bench_06/agent_a/expanded_error_diagnostics.py
```

CPU, <2 s, no checkpoint. MB03 experiment ID `RQ1-MB03-C-SPEAKER-GRL-GROUPCV-S101`. E4 scores are recovered historical results, not rerun. Pooled rank-mean is exploratory only; do not use it as nested fusion.

## B: insertion localization

Verified MB06 experiment `RQ1-MB06-B-GROUPED-TEMPORAL-LOGREG-S101`; no checkpoint; frame-logit probe/source/receipt/OOF scores/PR coordinates under `experiments/rq1_method_bench_06/agent_b/`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_06/agent_b/grouped_temporal_probe.py
```

CPU wall time 15.13s; peak memory not measured. Uses frozen CTC logits from cached evaluation waveform replay, predicted canonical alignment, ±5-frame window and six pooled summaries; fits balanced logistic regression in each speaker-held-out fold. No gold timing at inference. All 61,362 N+1 gap IDs are unique; 560 positives; four folds complete. AP-I pooled `.02220`; fold held AP `.02004/.03068/.01982/.02621`; matched same-classifier static features `.03557`; historical MB03 neural `.162125`. No extra seed.

The MB03 event-level gap predictions are `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_predictions.npz`, keys `gap_PLAIN_0..3`; arrays 15,266×2, 15,298×2, 15,439×2, 15,359×2. Receipt AP-I `.16212489699`, supports 134/127/150/149 and fold AP `.188792/.102708/.109449/.251541`. Agent B corrected its prior missing-baseline claim.

The H-002 cache `research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/train_frame_context.npy` is target-indexed `(59562,17,768)`, not a frame timeline. Cached waveform replay archives under `experiments/evaluation_v3/closure/waveform_full/utterances/` provide frame CTC logits; this experiment used those, not raw WAV re-inference. The local pooling result is negative against its static control and neural reference, but does not falsify all temporal models. MB05 T2 can be separately reproduced with `.venv/bin/python experiments/rq1_method_bench_05/agent_a/temporal_gap_score.py`.

## C: MixGoP-inspired adaptation

ID `RQ1-MB06-C-MIXGOP-ADAPT-GROUPED-S101`; output/receipt under `experiments/rq1_method_bench_06/agent_c/`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_06/agent_c/mixgop_adaptation.py
```

CPU, verified rerun 4.344 s. No checkpoint, peak memory not recorded. Fold PCA is fit on outer-train features; per-phone diagonal GMM fit on canonical correct TRAIN events only; models score held-out targets. K=1/K=4, target S/O only. Receipt carries folds and input paths. This is a MixGoP-inspired adaptation, not paper reproduction.

## New MB06 artifacts

- Agent A report: `reports/METHOD_BENCH_06_A_EVIDENCE_INTEGRATION.md`; scripts/JSON in `experiments/rq1_method_bench_06/agent_a/`.
- Agent B report: `reports/METHOD_BENCH_06_B_INSERTION_LOCALIZATION.md` (reports no new model; its initial baseline-inventory conclusion is corrected by NPZ audit above).
- Agent C report: `reports/METHOD_BENCH_06_C_EXTERNAL_RESEARCH.md`.
- No nested integration, probability calibration, warning-budget analysis, or additional-seed MB06 run.
