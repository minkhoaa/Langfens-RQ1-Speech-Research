# Temporal local-context direction

## Claim

Test whether a small order-aware temporal CNN over frozen Wav2Vec2 layer-6
windows improves over the shared 13-layer SM13 neural target/gap heads when the
speaker split, calibration rule, target and gap labels, optimizer schedule, and
EVAL-V3.1 scorer are held constant.

## Split and decision protocol

For fold `f`, `GROUPS[f]` is held once, `GROUPS[(f+1)%4]` calibrates thresholds,
and the remaining six speakers are fit-only. Seeds are 101 and 202. This is
exploratory grouped CV over L2-ARCTIC TRAIN and not a fresh confirmatory test.
Only fit labels train heads; calibration labels select deterministic B=.2
warning cutoffs; held labels enter the evaluator after decisions are fixed.
Target warning score ranks the normalized S/O-vs-C margin, with S/O warning type
given by the larger class probability. SHA-256 event-key tie-breaking matches
the shared deterministic policy. Held fixed-precision summaries are explicitly
oracle diagnostics.

The common evaluator is `research.evaluation_v3.UnifiedEvaluatorV3` (EVAL-V3.1)
and checks exact target/gap event identity and N/N+1 topology. Prediction NPZs
retain event IDs, golds, decisions, and probabilities; JSONs retain fold
provenance, cal-frozen thresholds, EVAL class metrics, model training settings,
and the diagnostic held fixed-precision points.

## Models

The shared baseline uses the original SM13 13-layer scalar-mixing head classes
over the frozen pooled Wav2Vec2-960h layer caches and intended-phone references.
The target schedule is 12 epochs, batch 512, AdamW `lr=1e-3`, `weight_decay=.01`;
the gap schedule is 5 epochs, batch 1024, AdamW `lr=8e-4`, `weight_decay=.0001`.
Both heads fit only six speakers per fold. This is a neural-only component
baseline; it is not the deployed asymmetric E4 system.

The matched pooled-L6 control uses the same target head and training schedule,
with layer-6 features repeated across the scalar-mix input slots. The temporal
head preserves the order of each aligned 17-frame target window (and 10-frame
left/right gap window), runs two width-128 1D convolutions, then learned
attention and mean pooling before the event classifier. It uses the same
target/gap fit schedules as SM13. The primary architecture comparison is
temporal target + the shared SM13 gap; the full temporal target/gap model is a
secondary end-to-end candidate.

## Inputs and exclusions

The temporal windows are extracted by the encoder-adapter owner from the frozen
local Wav2Vec2-960h model and TRAIN audio only. Hidden frames are alignment
centered and edge clipped; the receipt and array hashes are in
`../encoder_adapter/raw_cache_receipt.json`. No dev, test-sealed, or
SpeechOcean data are read by this direction.

## Reproduction

Use the project environment. First run the one-fold, one-epoch smoke (separate
outputs, not benchmark results), then run the all-fold fit-only comparator, then
the complete temporal run after shared GPU work is scheduled:

```bash
.venv/bin/python experiments/sm13_three_directions_20261113/temporal/run_temporal.py --smoke
.venv/bin/python experiments/sm13_three_directions_20261113/temporal/run_temporal.py --baseline-only
.venv/bin/python experiments/sm13_three_directions_20261113/temporal/run_temporal.py
```

The runner acquires `../gpu.lock` and releases it after each fold. Baseline
training receipts, fold-level keyed scores, and pooled OOF aggregation are
stored under the sibling `shared/` and `temporal/` output directories.

## Current status

The shared all13 SM13 and pooled-L6 baselines are complete for both seeds and
all four folds. The one-epoch smoke path completed. The first registered full
temporal fit (fold 0, seed 101) completed: macro3 AP was 0.23370 for temporal
target + shared SM13 gap and 0.22829 for full temporal target/gap, versus 0.29714
for all13 SM13 and 0.26059 for pooled-L6 target + shared SM13 gap. This one held
group is a negative exploratory result, not pooled evidence; the other seven
full temporal fits remain scheduled around the encoder-adapter and
data-expansion GPU jobs.

