# Controlled middle-layer LoRA adaptation

## Question

Does low-rank attention adaptation in Wav2Vec2 middle blocks improve held-speaker target error detection over a frozen encoder control with the same five-state mixture, fit-only warm-start, head, and optimizer schedule?

## Result

The eight paired held fold/seed comparisons show essentially no average change from LoRA: mean paired delta in macro S/O/I AP is **+0.00014** (median **−0.00012**, range **−0.00233 to +0.00325**); mean delta in macro S/O/I F1 is **+0.00007** (median **+0.00162**, range **−0.00985 to +0.00611**). The sign changes by fold and seed. These small differences do not establish a reliable gain.

The per-seed pooled OOF F1 deltas are **−0.00145** (seed 101) and **−0.00073** (seed 202); these differ from the mean fold-level paired delta because F1 after pooling events is not the average of fold F1 values. The canonical all-13-layer SM13 baseline remains stronger on pooled out-of-fold macro error AP for both seeds. Relative to that baseline, macro error F1 is lower for LoRA on seed 101 and slightly higher on seed 202. All three arms have pooled false-warning rates above the 0.20 reference in both seeds.

| Seed | Model | Macro S/O/I AP | Macro S/O/I F1 | Joint FWR |
|---:|---|---:|---:|---:|
| 101 | Canonical all-13 SM13 | 0.24687 | 0.12495 | 0.21889 |
| 101 | Matched frozen five-state control | 0.23251 | 0.12254 | 0.21889 |
| 101 | Five-state LoRA | 0.23325 | 0.12110 | 0.21444 |
| 202 | Canonical all-13 SM13 | 0.25071 | 0.12987 | 0.23500 |
| 202 | Matched frozen five-state control | 0.23516 | 0.13234 | 0.21056 |
| 202 | Five-state LoRA | 0.23463 | 0.13161 | 0.21167 |

These are pooled held predictions from four disjoint outer folds within each seed, evaluated with EVAL-V3.1 after each fold's calibration threshold was frozen. Per-seed OOF predictions contain all 59,562 target events and 61,362 gap events with no duplicate keys.

### Pooled classwise results

Each cell is precision / recall / F1 / average precision. `S`, `O`, and `I` denote substitution, omission, and insertion.

| Seed | Model | Substitution | Omission | Insertion |
|---:|---|---|---|---|
| 101 | Canonical all-13 SM13 | 0.6715 / 0.0504 / 0.0938 / 0.3827 | 0.5252 / 0.0800 / 0.1388 / 0.2525 | 0.3798 / 0.0875 / 0.1422 / 0.1054 |
| 101 | Matched frozen control | 0.6250 / 0.0393 / 0.0739 / 0.3465 | 0.5349 / 0.0882 / 0.1515 / 0.2456 | 0.3798 / 0.0875 / 0.1422 / 0.1054 |
| 101 | LoRA | 0.6250 / 0.0386 / 0.0727 / 0.3478 | 0.5377 / 0.0860 / 0.1483 / 0.2465 | 0.3798 / 0.0875 / 0.1422 / 0.1054 |
| 202 | Canonical all-13 SM13 | 0.6707 / 0.0599 / 0.1100 / 0.3744 | 0.5187 / 0.0685 / 0.1210 / 0.2647 | 0.3836 / 0.1000 / 0.1586 / 0.1130 |
| 202 | Matched frozen control | 0.6986 / 0.0396 / 0.0749 / 0.3419 | 0.5206 / 0.0970 / 0.1635 / 0.2505 | 0.3836 / 0.1000 / 0.1586 / 0.1130 |
| 202 | LoRA | 0.7000 / 0.0398 / 0.0754 / 0.3429 | 0.5133 / 0.0953 / 0.1608 / 0.2480 | 0.3836 / 0.1000 / 0.1586 / 0.1130 |

Insertion scores are intentionally identical across the three models: both target arms use the canonical all-13-layer baseline's fold/seed gap probabilities. This isolates target-side encoder adaptation; the insertion values are shared baseline results, not evidence about adapter insertion performance.

## Design and protocol

- Model: `facebook/wav2vec2-base-960h`, cached snapshot `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.
- Four grouped outer folds, seeds 101 and 202. Each fold uses six speakers for fit, three for calibration, and three held speakers. Training uses fit speakers only. Calibration freezes the B=.2 warning threshold. Held predictions are evaluated once per arm; held labels never select epochs or settings.
- The five-state mixture combines pooled hidden states 5–9 (outputs after zero-based transformer blocks 4–8). Candidate adds rank-8, alpha-16 LoRA residuals on attention q/v projections in those five blocks. All original encoder weights remain frozen.
- Frozen control and LoRA candidate share the same fit-only two-epoch head/mix warm-start, standardizer, target head, per-utterance shuffle and dropout stream, then two paired training epochs. The control trains only its target head and scalar mixture. LoRA trains the same parameters plus its adapter weights.
- Both arms use target weighted cross-entropy weights `[1, 2.5, 4]`, AdamW target-head/mix LR `1e-3`, LoRA LR `1e-5`, weight decay `0.01`, gradient clipping at `1.0`, and 113 fit optimizer updates per epoch. The compared five-state mixture is distinct from the canonical all-13-layer baseline.
- Gap probabilities for both arms come unchanged from the fold/seed canonical all-13-layer SM13 baseline. The all-13 baseline trains its target head for 12 epochs and gap head for five epochs on fit speakers.
- Only `train.jsonl` raw audio and its precomputed label-independent alignment centers were used. Raw Wav2Vec2 extraction used `AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")` at 16 kHz with the extractor's default normalization enabled. No dev/test audio or labels were used.

## Numerical parity correction

The initial frozen controls used historical float32 pooled features while the LoRA candidate consumed an fp16 cached prefix. The measured cache-prefix error versus full forward was at most `9.67e-4`; suffix output error was at most `6.46e-4`. To remove this small input precision mismatch, all eight primary frozen controls were retrained on layer 5–9 features generated from the same fp16 prefix through frozen blocks 4–8. The original float32-cache control results are preserved under [`folds/superseded_quantization_mismatch`](folds/superseded_quantization_mismatch/README.md) and excluded from the primary comparison. Candidate checkpoints and predictions were retained unchanged.

Matched feature-cache hashes and exact sizes are in [`matched_control_features/receipt.json`](matched_control_features/receipt.json). A sampled comparison against historical float32 pooled caches found target max/mean absolute differences no larger than `0.00664` / `1.82e-5`, and gap differences no larger than `0.00439` / `1.45e-5` across fold/seed probes. Both arms use the same matched prefix path in the primary paired comparison.

## Gradient and hardware checks

- Preflight with the actual fp16 prefix found max prefix error `0.000967`, max no-adapter suffix error `0.000646`, and max pooled-cache error `4.77e-7`.
- Across all eight candidate runs, LoRA A, LoRA B, and target mixture gradient norms were finite and nonzero after the first adapter update. LoRA A's initial gradient is zero at zero-initialized B, as expected; the recorded post-update A gradient is nonzero in every fold/seed.
- Candidate trainable parameter count: 756,232 (633,347 target-head weights, five mixture logits, and 122,880 LoRA parameters). Original encoder weights are frozen. Candidate train time was 57.7–60.7 seconds per fold/seed on a GTX 1650 4 GB; measured peak CUDA allocation was 593–609 MB. CPU threads were capped at two.
- Environment: Python 3.11.17, PyTorch 2.6.0+cu124, Transformers 5.22.1, CUDA runtime 12.4.

## Limits

This tests one target-only adaptation schedule over five middle states. The two paired adaptation epochs are fixed, not selected from held results; the fit-only head warm-start uses two additional epochs. The schedule is intentionally modest and does not establish that longer or different encoder adaptation cannot help. Prior fine-tuning history showed validation AP still rising from epoch one to epoch two, so that limitation is material. The study covers this corpus and these held speakers, not external data. Fixed-precision held PR diagnostics in per-fold JSONs are oracle diagnostics only and were not used for selection.

## Reproduction and artifacts

```bash
flock -x experiments/sm13_three_directions_20261113/gpu.lock \
  .venv/bin/python experiments/sm13_three_directions_20261113/encoder_adapter/run_encoder_adapter.py \
  --folds 0 --seeds 101
flock -x experiments/sm13_three_directions_20261113/gpu.lock \
  .venv/bin/python experiments/sm13_three_directions_20261113/encoder_adapter/extract_matched_control_features.py
flock -x experiments/sm13_three_directions_20261113/gpu.lock \
  .venv/bin/python experiments/sm13_three_directions_20261113/encoder_adapter/run_encoder_adapter.py \
  --matched-control-only --folds 0 --seeds 101 --skip-preflight
.venv/bin/python experiments/sm13_three_directions_20261113/encoder_adapter/aggregate_results.py
```

The fixed shared protocol is [`../shared/protocol.py`](../shared/protocol.py). The preregistration and technical amendment are [`PREREGISTRATION.md`](PREREGISTRATION.md) and [`AMENDMENT_01_MATCHED_CONTROL.md`](AMENDMENT_01_MATCHED_CONTROL.md). Matched per-fold/seed summaries are `folds/matched_pair_seed*_fold*.json`; fold-keyed predictions are `encoder_adapter_lora_*` and `encoder_adapter_control_prefix_*` NPZs. Per-seed pooled metrics and predictions are `folds/oof_seed*_...`.
