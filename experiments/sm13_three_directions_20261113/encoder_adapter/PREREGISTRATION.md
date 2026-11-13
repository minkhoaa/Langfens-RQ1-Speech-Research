# Encoder adapter experiment preregistration

## Question

Does task-specific attention LoRA adaptation in Wav2Vec2 layers 4–8 improve exact SM13 event prediction over an identically trained frozen encoder control using the same trainable layer mixture and heads?

## Data and split

- Use only `train.jsonl` raw audio and its precomputed, label-independent alignment centers.
- Four outer folds from `DatasetManager.GROUPS`: fold `f` held speakers are `GROUPS[f]`; calibration speakers are `GROUPS[(f+1)%4]`; fit speakers are the remaining six.
- Seeds: 101 and 202. Fit only on the six-speaker fit partition. Calibration may determine the predeclared B=.2 operating policy and choose among the fixed epoch-1/epoch-2 snapshots if needed. Held speakers are read once for the final per-fold/seed evaluation and never select architecture, hyperparameters, or epochs.
- Candidate and control use identical partitions, data ordering seed, loss, optimizer schedule, target/gap heads, middle-layer mixture, and number of head updates.

## Frozen/adapted comparison

- Encoder: `facebook/wav2vec2-base-960h`, with the same model revision and feature extraction as the frozen SM13 reference.
- Both arms use a softmax scalar mixture over the five pooled hidden-state outputs after zero-based transformer blocks 4–8 (HF `hidden_states[5:10]`), initialized uniformly, followed by the exact target and gap acoustic pooling used in `features/extract_all_features.py`; reference vectors are concatenated to those features. This is explicitly a middle-layer mixture, distinct from the canonical all-13-layer SM13 baseline.
- Both arms train the same target head on fit speakers. Control freezes all encoder weights. Candidate adds rank-8, alpha-16 LoRA residuals on attention `q_proj` and `v_proj` in zero-based transformer blocks 4–8; all original encoder weights stay frozen. LoRA B matrices start at zero, so adapter-disabled outputs must match the frozen control before training.
- Gap probabilities are fixed to the matched fold/seed canonical all-13-layer frozen SM13 baseline for both arms. The adapter objective is target-only, so the shared evaluator's insertion metrics are identical by construction and are reported as shared-baseline values.
- The LoRA and scalar-mixture parameters must receive finite, nonzero gradients on a fit-only backward check. No posthoc head-only model is counted as encoder adaptation.
- If caching the frozen prefix through layer 3, compare cached-prefix layer 4–8 outputs with a full no-adapter forward pass within the stated numerical tolerance before using the cache. The cache contains fit audio only; calibration and held audio are evaluated with the full encoder.

## Training bound

- Warm-start target/gap heads and mixture logits for two epochs on cached frozen middle-layer features from fit speakers only. Copy this exact fit-only initialization into both arms. Then train each arm for a fixed two additional epochs, with identical shuffle, per-utterance updates and head schedule. No adaptive extension after inspecting held results.
- AdamW, target-head/mix LR `1e-3` with weight decay `0.01`, LoRA LR `1e-5` with weight decay `0.01`; gradient clipping at 1.0; deterministic seed per fold/seed pair and identical shuffle/dropout streams between paired arms. Weighted cross-entropy follows the existing RQ1 encoder benchmark (`[1, 2.5, 4]` target weights). The gap warm-start artifact is retained only for reproducibility and does not contribute to the paired adaptation objective or reported predictions.
- Training uses fit raw audio only; no dev/test raw audio, labels, features, or outcomes.

## Evaluation and reporting

- Use shared `UnifiedEvaluatorV3` evaluation and calibration policy from `shared/`. Preserve keyed predictions for exact target/gap topology. Report class-wise precision/recall/F1/AP, macro error AP/F1, false-warning rate/count at the calibration-derived B=.2 policy, plus diagnostic held PR operating points at fixed precision .75/.80 clearly labeled as oracle diagnostics.
- Report every fold/seed against its matched frozen control and the shared canonical frozen baseline. Include trainable parameter counts, gradient checks, checkpoints, runtime/peak memory if measured, environment, and exact reproduction command. Any missing fold/seed is labeled incomplete; no aggregate claim is made from partial runs.

## Prior evidence and limits

Earlier upper-layer unfreezing on one inner split improved validation AP from epoch 1 to epoch 2, so this experiment uses two fixed epochs and does not interpret an early negative run as evidence that all fine-tuning fails. Results cover this architecture, this corpus, and this schedule only.
