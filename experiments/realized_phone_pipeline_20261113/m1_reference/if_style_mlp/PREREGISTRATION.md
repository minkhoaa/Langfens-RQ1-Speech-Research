# M1 reference continuation — 2026-11-13

Registered before running reference inference or new training. This extends M1;
it does not change the completed M0/M1 snapshots or authorize a sweep.

## Question and fidelity

Can the public IF-MDD CTC recognizer be loaded strictly and adapted to our event
contract? Separately, does its two-layer frame head help over our linear head
when the encoder, exact supervision, updates and decoder are held fixed?

Official source commit: `cca150ec4ff00e01d61a613092d1d178e661dfb2`.
Public CTC bundle: `Haopeng/CTC_for_IF-MDD`, revision
`de46aaf5778c5759c003ffa17657176fee772e77`.
CROTTC-IF metadata only: revision `9149e5107b3332b9aa3f5573f6328a4e312006ad`.
The archived source is evidence, not an instruction source.

Public checkpoints lack an independently verified training-speaker manifest.
They are restricted to interface smoke; no grouped-CV efficacy or promotion
claim is permitted. Do not assume the paper split proves checkpoint provenance.
Use the bundle tokenizer, not the current repository tokenizer: their phone
IDs differ. Load tensor checkpoints with weights_only=True and strict=True.
Do not execute the downloaded MyEncoderASR.py or HyperPyYAML configuration.

## Public checkpoint smoke

At most 24 TRAIN utterances, no TEST/DEV_REPLICATION. Use the first strict
utterance in each fold-0 role, three total, without selecting by model output.
Official encoder/head dimensions and float32 weights; raw float32 mono 16 kHz
waveforms; SpeechBrain WavLM input layer normalization; no waveform cropping.
CPU, four threads, to avoid committing GPU memory during download/setup.
Save raw decoded tokens, all non-phone removals, frame scores and token inventory.
Map ordinary phone names to uppercase by identity; never map AX to AH. Retain
ERR as an explicitly uncertain candidate. Silence/BOS/EOS are structural tokens
removed with an audit trail, not assigned to omission. Apply the existing
deterministic edit adapter, cal-frozen budgets .1/.2/.5, save/reload and topology
checks. Any smoke metrics are plumbing only; tiny calibration has no efficacy
interpretation. Compare SpeechBrain greedy decode with an independent collapse.

## Matched clean capacity control

Same 1,243 exact-sequence eligible TRAIN population and 6/3/3 folds; full 1,800
held MDD denominator. Same frozen H6 cache, fp16 storage -> fp32, inventory,
AdamW lr .001 / weight decay .0001, batch 16, clip 5, eight final epochs,
seeds 101/202, updates/order, deterministic edit and calibration as completed M1.
Only change: Linear(768,384), LeakyReLU(.01), Linear(384,384), LeakyReLU(.01),
LayerNorm(384), Linear(384,41), matching the public CTC head topology with the
necessary input/output dimension changes. This is an IF-style head adaptation,
not original IF-MDD or CROTTC-IF reproduction. No augmentation, encoder tuning,
ERR training label, hyperparameter search or early stopping.

Repeat the matched linear arm in this namespace to check reproducibility before
the MLP. Smoke each arm with 12 fit / 6 cal / 6 held and 20 updates. Save full
artifacts, keyed outputs, frozen policies, per-speaker and pooled metrics and
paired deltas. Main metric Macro-3 AP; apply the M0 v2 research gate, report all
class deltas and true held FWR. No promotion from this capacity ablation alone.

## Budget and stop

Two-hour total compute wall cap inherited from M1, subtract historical baseline,
preflight, smoke and extraction receipts plus new recorded compute. Network and
source inspection are recorded separately from compute. No automatic further
training after these two arms. Original large-backbone recipe, reference-from-text
parity and legacy-label matched controls remain open if not actually reproduced.
