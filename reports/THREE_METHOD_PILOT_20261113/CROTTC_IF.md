# CROTTC-IF local pilot

**Status:** compact mechanism pilot executed; not a reproduction of the paper.

## Method implemented

The method follows the paper's three-part idea: frame-level acoustic phone prediction with monotone optimal transport (OTTC), consistency regularization (CR) across perturbed acoustic views, and indirect fusion (IF) through a privileged canonical/error teacher that transfers information into a prompt-free phone sequence model. The reference is [Beyond Acoustic Sparsity and Linguistic Bias: A Prompt-Free Paradigm for Mispronunciation Detection and Diagnosis, arXiv:2604.22133v1](https://arxiv.org/abs/2604.22133), especially Sections 3–4.

The implementation is in [model.py](/home/khoa/Projects/New-Method-Research/research/three_method_pilot/crottc_if/model.py) and [run.py](/home/khoa/Projects/New-Method-Research/research/three_method_pilot/crottc_if/run.py). It has:

- A learned frame-mass head and a full-utterance, monotone 1-D quantile coupling. The coupling's marginals are checked by a synthetic test and its mass parameters receive gradients. Cross-entropy over the coupling supervises the frame phone head.
- Two length-preserving perturbed acoustic views with symmetric frame-posterior KL consistency loss.
- An acoustic-attention autoregressive phone decoder trained on the FIT realized phone sequence.
- A FIT-only privileged teacher conditioned on canonical phones, OT-pooled acoustics, and pooled autoregressive decoder states. Its error-type labels train an error head; its perceived-phone distributions also train a teacher phone head. The teacher phone distribution is explicitly KL-distilled into the autoregressive student's phone logits.
- A FIT-supervised log sequence-length predictor used to stop autoregressive decoding at an acoustic-only predicted length.
- Acoustic-only autoregressive decoding at inference. The inference method accepts acoustic features only and does not call the canonical embedding, teacher, or teacher heads.

No annotation timestamps or gap boundaries enter model inputs or alignment. The separate canonical teacher alignment is also estimated from full-sequence transport.

## Bounded execution

The corrected run used the registered manifest `experiments/three_method_pilot_20261113/pilot.json`, frozen H6 features with SHA-256 verification, CPU only, two threads, 24 FIT utterances, and two epochs. It completed in 1.570 seconds, within the 300-second limit (`.venv/bin/python`, PyTorch 2.6.0+cu124, NumPy 2.4.6). The runner read FIT supervision and FIT annotations, exact realized-sequence CAL supervision for checkpoint selection, feature metadata, and the manifest role IDs. All 6 CAL records were exact-sequence eligible. It did not open held request records or held labels.

Checkpoint selection used CAL phone error rate (PER) only:

| Epoch | FIT total loss | OTTC | CR | AR phone LM | Length | Teacher | Distillation | CAL PER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.4917 | 3.7112 | 0.0038 | 3.6102 | 0.0664 | 4.4527 | 0.1467 | 1.0592 |
| 2 | 7.9342 | 3.4841 | 0.0036 | 3.4718 | 0.0566 | 3.6896 | 0.1527 | 1.0855 |

The selected checkpoint is epoch 1 by exact realized-sequence CAL PER (1.0592). Both `last.pt` and `best.pt` were atomically saved, loaded back strictly, and verified to contain model, optimizer, scheduler, scaler, RNG states, and metadata. The selected prediction artifact has 36 records (24 FIT, 6 CAL, 6 HELD) with the required `utterance_id`, `role`, and `recognized_phones` fields at `experiments/three_method_pilot_20261113/crottc_if/recognized_phones.jsonl`. Held predictions are emitted for the root-owned evaluator; this report does not score held data.

The initial auxiliary-only pilot was preserved at `experiments/three_method_pilot_20261113/crottc_if_auxkd_v0/` and marked superseded. The first autoregressive run was preserved at `experiments/three_method_pilot_20261113/crottc_if_ar_v1/`; it was superseded because its length predictor had not been supervised. The next run is preserved at `experiments/three_method_pilot_20261113/crottc_if_canonicalcal_v2/`; it was superseded because it selected checkpoints against canonical phones instead of exact realized CAL sequences. None is counted as the final run.

## Verification

Five synthetic mechanism checks passed when invoked directly under Python: OT marginal preservation and non-crossing support; nonzero gradients to the learned OT mass head; CR behavior; canonical/error teacher and teacher-phone distillation gradients into autoregressive phone logits; nonzero FIT length-head gradient; teacher-free inference; and fail-closed handling of nonfinite inputs. The runner also verified each used feature file's SHA-256 and shape, saved and reloaded each epoch checkpoint, and emitted all role outputs.

## Deviations and limits

This is a resource-bounded adaptation, not the paper's full architecture or result reproduction:

- Uses frozen Wav2Vec2-base hidden-state-6 feature files and small trainable heads, rather than end-to-end WavLM-Large training.
- Uses equal mass per target phone for the compact target transport; it does not implement the paper's richer ordered subphonetic-state construction.
- Uses a compact GRU-cell decoder with acoustic attention, FIT-supervised length regression, and greedy autoregressive decoding, not the paper's two-layer Transformer decoder and joint AM/LM search.
- Uses a small canonical/error teacher MLP with direct phone-distribution distillation, not the paper's dual fusion networks and CNN teacher.
- CR uses feature-space noise and short time masking, not the paper's waveform/spectrogram time-warp and masking setup.
- The pilot is one seed, one fold, 24 FIT examples and two epochs. Best exact-realized CAL PER is 1.0592 phone edits per reference phone; this is weak pilot performance and is not an MDD detection metric. The paper's reported detection F1 cannot be compared to this PER.
- The target-phone teacher labels include correct, substitution, and omission classes. Gap-level insertion annotations are not separately represented in this compact token-level teacher.

The implementation and checks establish the intended mechanisms at small scale; they do not establish the paper's reported performance, generalization, or product readiness.
