# M1 baseline training — registered before any new training

Date: 2026-11-13. User explicitly requested training after M0.

This run trains the first executable baseline, not a novel or promoted model.
It follows the local MB03 continuous frozen Wav2Vec2-H6 linear CTC mechanism,
with audited exact realized-phone supervision and the four registered M0 folds.
This is a partial reproduction with changed labels/splits, not a full reproduction
of Shahin 2024 or CROTTC-IF. Those reference fidelity gates remain open.

- Population: only 1,800 TRAIN utterances / 12 speakers. Strict supervision uses
  the 1,243 M0-eligible utterances, further restricted to each fold's six fit speakers.
- Calibration: next group of three speakers. Held: disjoint group of three speakers.
  All held utterances remain in MDD evaluation. Exact realized PER uses eligible
  held sequences only, with explicit support and coverage.
- Encoder: cached `facebook/wav2vec2-base-960h`, revision
  `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, frozen hidden state 6, 20 ms frames.
  [Publisher model card](https://huggingface.co/facebook/wav2vec2-base-960h)
  describes LibriSpeech pretraining/fine-tuning and Apache-2.0 licence (checked
  2026-11-13). No L2-trained public checkpoint is used. Cache cannot be reused
  until manifest/order/audio hashes, dtype, shape, finite values and two waveform
  probes pass. Probe tolerance registered here: max absolute error .003 and RMSE .0005.
- Inventory: sorted 39 ARPAbet phones + AX, blank at index 40. No ERR/SPN,
  malformed symbols or deviation-marked sequences become exact training labels.
  Reference symbols remain unchanged for historical event topology.
- Model: one Linear(768,41) head; encoder stays frozen. CPU / four torch threads.
- Optimizer: AdamW, lr .001, weight decay .0001, batch 16, clip gradient norm 5,
  eight fixed epochs. No checkpoint selection on calibration or held speakers.
  Nonfinite loss or infeasible repeated-label CTC alignment stops the run.
- Seeds 101 and 202; four folds each. No architecture/hyperparameter sweep.
- Decoder: audio-only greedy CTC collapse, then deterministic unit-cost edit
  alignment to the supplied reference; tie order diagonal, omission, insertion.
  Keep all inserted phone candidates. CTC emission spans are not gold phone boundaries.
- Event scores: binary scores from the deterministic edit path. They have coarse
  ranking and are uncalibrated; 0/1 is not a claim of certainty about an actual error.
  Frozen warning policies may suppress edit events and must retain that distinction.
- Policy budgets .1/.2/.5 warnings per utterance, .2 primary; target/gap allocation
  .8/.2 and event-hash tie handling inherited from M0. Fit thresholds on cal only.
- Smoke: 12 fit, 6 cal, 6 held utterances; 20 updates with batch four; complete
  train/predict/calibrate/evaluate/save/reload before full training. Smoke metrics
  are not performance evidence. CPU cap: 2 hours total run wall time including
  preflight and smoke; no GPU launch while CUDA unavailable.
- Save every fold/seed checkpoint, learning curve, calibration policies, keyed
  held predictions, realized decodes and frame posteriors, source/artifact hashes,
  classwise counts, per-speaker results, full-population pooled OOF metrics per seed.
- A strict-vs-legacy supervision causal claim is not possible from this run.
  The training subset/inventory differs from SM13 and legacy CTC; historical
  scores are context only. No replacement of historical checkpoint/policy.
- Stop/reassess after this baseline. No gain claim based only on PER or a different
  warning budget; original-reference reproduction, attribute ablation, new-speaker
  confirmation and product acceptance remain outstanding.
