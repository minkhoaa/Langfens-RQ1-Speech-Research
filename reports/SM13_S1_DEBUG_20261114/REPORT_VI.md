# Debug S1 SM13 — completed CPU-only audit, no retraining

## Verdict

**Strong evidence of head overfitting/overconfidence; no evidence found of a label,
event-order, checkpoint-resume or detection-count bug explaining the regression.**
There is a concrete **conditioning mismatch** in transplanting unstandardized scalar
mixing from S0 to native WavLM Large. Its contribution to quality loss remains a
hypothesis until a matched controlled refit; it is NOT a proven causal explanation
or proof that WavLM is intrinsically worse.

All work local CPU-only/two threads, inference and tests only. No production fitting,
GPU/remote calls, test/DEV_CAL, CROTTC changes or historical artifact mutation.

## Evidence and limits

### 1. Labels, references, split, optimizer and evaluator

- Reconstructed TRAIN labels from the1800 manifest rows: **0 target/gap label
  mismatches**, **0 gap-reference adjacency mismatches**; keyed target/gap order
  reproduced. Fit9speakers/heldABA+ASI+BWC disjoint.
- Four inherited all-zero intended-phone vectors: ER), D_, SPN, Z_. Canonical
  mapping otherwise consistent. These affect both S0/S1 and are not a new S1
  regression. Do not silently relabel them or open gold timing for inference.
- Exact comparisons of **all four head migrations** (target/gap at50 and150)
  preserve model/Adam/RNG/history/cursors/scaler/scheduler except provenance tags.
  Adam counters/LR/weight decay at50/150/350 are continuous/unchanged.
- Fresh CPU head inference for **all fit/held events** at50/150/350 reproduces stored
  GPU predictions: maximum probability delta about **5.94e-6**, **zero argmax
  disagreements** in all six held replays. No dropout/eval-mode or stale head found.
- Independent sklearn replay reproduces typed AP and binary P/R/F1 at fixed
  epochs50/100/150/200/250/300/350 (tolerance1e-12). No threshold search performed.
-18 inherited nonmonotonic predicted alignments,6in held; event order preserved,
  no sorting/drop. Shared historical alignment is still a limitation; this audit
  does not prove alignment quality or compute gold-timestamp agreement.

Evidence: `cpu_audit_v2/AUDIT.json`, `INDEPENDENT_METRIC_REPLAY.json`, prior404-state
verification at `../SM13_S1_PLUS200_20261114/VERIFICATION.json`.

### 2. Overfitting is visible in fresh eval-mode inference, not merely loss logging

Target head binary S/O detection at0.5, SAME completed checkpoints and populations:

| Epoch | Fit F1 | Held F1 | Fit weighted CE | Held weighted CE | Wrong held predictions with confidence>=99% |
|---|---:|---:|---:|---:|---:|
|50 |90.61%|39.43%|0.1024|1.4602|626|
|150|97.71%|41.42%|0.0295|2.3781|1093|
|350|98.79%|37.74%|0.0209|3.9133|1330|

These are **target-only**, not pooled target+gap F1. Target accuracy rises while
error detection deteriorates; accuracy alone is misleading for this imbalance.
Wrong predictions become more confident, so held CE rises even as total mistakes
fall. Top-label ECE10 increases about0.104→0.125. No temperature/threshold tuning.

Gap fit F1~99.53/99.42/98.95%, held F1~25.62/30.50/31.28% at50/150/350: memorization
vs generalization is also substantial. Held insertion positives134/15266; reporting
98–99% gap accuracy alone would hide this limitation.

Training history `train_loss` is a mean of weighted batch CE with dropout enabled
while held loss is pooled weighted CE in eval mode. These are not identical
estimators; the audit recomputed **both populations in eval mode with the same
pooled denominator**. The overfitting gap persists, so it is not just a plotting
artifact. Never rewrite the historical loss series as if they used the same estimator.

Audit v1 initially reconstructed CE from stored softmax probabilities; a few late
FP32 probabilities underflow to zero. Audit v2 uses fresh logits/log-softmax for CE,
which reproduces saved held CE3.9133 instead of a probability-floor artifact3.9940.
Production evaluator already used logits correctly. Regression fixture added.

### 3. Identified adapter weakness: raw-layer scale disparity

Per-layer RMS on **all fit-only events** (44746targets /46096gaps):
- S1 target max/min RMS **89.48x**, gap **100.19x**.
- On the same deterministic256fit-event sample, S0 ratios are only **3.10x/6.51x**.
- S1 intermediate layer22 RMS is about20(target), final layer24 about0.224;
  gap about17.8 vs0.178. Uniform softmax scalar weights do NOT mean equal acoustic
  contribution when amplitudes differ this much.
- Local installed `WavLMEncoderStableLayerNorm.forward` appends intermediate states
  before its final `self.layer_norm(hidden_states)`, then appends the normalized
  final state (`transformers/models/wavlm/modeling_wavlm.py`, approx490–516).
  This agrees with the observed scale discontinuity. It is NOT a layer-index or
  geometry mismatch: the registered0,2,...24 indices and13x1024width are correct.
- At350, target scalar `scale` shrinks to0.0437 while first acoustic weight
  Frobenius norm grows26.7→56.7 from50→350. A rough weight*RMS proxy assigns~72%
  to layers20/22 and~0.35% to24. Correlated layers mean this proxy is **not causal
  attribution**; no claim that a particular layer is best.

Old adapter intentionally preserved S0 raw-feature recipe, so this is a
**transfer/conditioning design weakness**, not a hidden data change. Merely adding
more epochs cannot be assumed to fix it.

### 4. Native feature/cache replay and waveform normalization

Three preselected TRAIN utterances were rerun from pinned native315M weights on CPU,
with fresh feature pooling and the350head:
- Cache relative L2 differences~1.5e-6…3.7e-6; final-head probability deltas<=3.32e-6,
  zero argmax disagreements on these samples.
- **Not all pooled-feature CPU/GPU allclose gates passed**: sparse near-zero values
  exceed the prior rtol1e-3/atol3e-4 envelope on LXC/PNV (4/1 and14/22 target/gap
  coordinates). No tolerance was silently widened or cache overwritten. Auditv1
  stopped at this gate and is retained. Auditv2 records the failures and measures
  downstream impact instead of claiming full native parity. This is consistent
  with small numerical backend differences, NOT proof of a corrupt cache or a
  universal CPU/GPU equivalence guarantee.
- HF normalization followed by SB layer_norm changes input values by~1.7e-5…6.7e-5
  on these samples. One HF-only counterfactual gives pooled relative L2~1.8e-6/1.4e-6.
  Thus no evidence here that double waveform normalization explains the large
  quality gap. Only one counterfactual: cannot rule out effects on other audio.

### 5. Diagnostic ablations do not establish a replacement method

Fixed2048held-event **inference-only** ablations show target performance depends
on both acoustic and canonical-reference inputs. At350, S AP~38.81% unmodified,
31.36% with zero acoustic,24.68% with zero reference on that sample. This is
out-of-distribution input suppression, NOT refitted reference-only/acoustic-only
baselines and NOT proof of causal shortcut leakage. Gap strongly depends on acoustic
features. No threshold or layer search was run.

## Implemented opt-in conditioning candidate (not a production fix/gain claim)

New `research/sm13_backbone_side/conditioning.py`:
- Streaming one positive RMS scalar per layer, computed **only from explicit fit
  speakers**; reject overlap, unknown/missing speakers, nonfinite/zero fit layers.
- Frozen13-element normalizer buffer persisted in the NEW head state; canonical
  references untouched. It does not change the old cache or trained checkpoints.
- Separate wrapper namespace rejects direct loading of legacy trained heads.
  You must refit fresh heads under a new registered, matched experiment.
- `CONDITIONING_CANDIDATE_STATS.json` records full-fit statistics/cache SHA/fit-index
  SHA/source SHA. On the fixed fit sample, conditioned layer RMS is roughly
  **0.987–1.009 target /0.989–1.010 gap**, proving scale balancing numerically,
  **not proving better detection**. Held features/labels were not used to fit scales.

Tests157passed for SM13 CPU suites, including new leakage/serialization/gradient/
legacy-checkpoint incompatibility tests and exact-logit CE underflow fixture.
No whole-repository PASS claim; no new real-data head training.

## Next bounded experiment, still pending registration/admission

1. Fresh raw vs fit-only RMS-conditioned S1 heads on the identical1350/450protocol,
   same seed/recipe/finite budget/evaluator, so scale conditioning is the only change.
   Do not combine LR/architecture/normalization/layer changes into an uninterpretable
   gain claim. Preserve existing350 as history, not a new starting point for the
   conditioned head.
2. For the method-selection/final comparison phase, lock a TRAIN-derived
   early-stopping/checkpoint policy and threshold calibration before fresh fitting;
   do not retroactively replace the fixed350 result with whichever epoch looks best.
   Epoch/selection controls must match S0/S1. Held reuse remains exploratory.
3. FullTRAIN1800→DEV_SEARCH599/common CROTTC adapter gates still open. S0 encodes
   an ASR-fine-tuned backbone while S1 is native pretrained; capacity/normalization/
   representation differences still prevent a pure size/backbone-effect claim.

No sweep/full-backbone fine-tuning/new corpus/test access or automatic launch is
authorized by this debug report. Disk currently near the14GiB CROTTC export floor;
keep read-only caches and avoid new feature materialization.
