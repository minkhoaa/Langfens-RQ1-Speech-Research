# M1 completion execution — 2026-11-13

User requested completion of M1 after the capacity continuation. Registered
before the new smoke or training. No previous results are overwritten.

## Acceptance scope from GOAL section 8

Reproduce one suitable sequence-MDD implementation under the local speaker
protocol, establish SM13 under that protocol and preprocessing parity, and
produce runnable N-target/N+1-gap outputs with every paper deviation explicit.
Paper-result replication and product acceptance are separate: M1 can establish
an executable adapted reference without claiming the original paper's metric.
The earlier continuation's `full reference recipe` wording is not permission
to relabel its frozen head as a full reproduction. This execution adds an
actual end-to-end speech-encoder/phone-CTC training baseline.

## Sequence reference

Reference: official IF-MDD PhnMonoSSL CTC mechanism, source pinned in
../m1_reference/upstream (commit cca150ec4ff00e01d61a613092d1d178e661dfb2).
The source supports wav2vec2_base as an encoder backend. Resource adaptation:
facebook/wav2vec2-base-960h, revision 22aad52d435eb6dbaf354bdad9b0da84ce7d6156,
instead of default WavLM Large. The complete base encoder is trainable except
CNN feature extractor; two-layer 384 LeakyReLU/LayerNorm head is unchanged in
topology; output inventory is our 40 phones plus blank instead of public 44.
Only non-L2 pretrained weights initialize the recognizer; no public L2 bundle.

TRAIN 1800 / 12 speakers, same 4 folds of 6 fit / 3 calibration / 3 held.
Fit complete realized sequences only (1243 eligible globally); full held
population for MDD. Seeds 101/202, final fixed TWO epochs per fold/seed.
This short recipe is an implementation/protocol reproduction, not a claim
to reproduce 100-epoch WavLM/paper effectiveness. No best-checkpoint selection.

Adam (not AdamW), head lr .0003 and encoder lr .00001; effective utterance
batch 16 through microbatch 1/accumulation 16; last partial batch divides by
actual size. Freeze CNN, enable gradient checkpointing, fp32 parameters and
forward/loss. Clip norm 5, stop on nonfinite gradients or loss, no skipping.
Waveform float32/mono/16k; feature-extractor normalization then SpeechBrain
layer_norm; speed perturb 95/100/105 during fit only, sampled by seeded RNG.
SpecAugment follows the pretrained model training config. No gold boundaries.
Two epochs instead of 100, effective batch 16 instead of 32, precision fp32
and correct accumulation are declared deviations. Upstream auto-mixed-precision
fit_batch clears gradients on every microbatch, so its accumulation flag does
not implement true accumulation in that branch; we do not copy that behavior.

Audio-only greedy CTC then unit-cost deterministic edit. Existing cal-frozen
B=.1/.2/.5 and evaluator. Save encoder/head checkpoint, fit IDs, policies,
keyed outputs, full log posteriors, PER S/D/I, per-speaker/class/phone analyses.
Verify actual encoder changes; waveform save/reload; acoustic independence
from supplied-reference perturbations. Smoke: 12 fit/6 cal/6 held, 20 optimizer
updates on microbatch 1, then complete predict/calibrate/evaluate/save/reload.
CUDA lease required. Record peak allocated/reserved VRAM and compute timings.

## Controls and diagnostic inference

1. Legacy MB03 CTC: same strict utterance IDs, epochs=8, update/order/optimizer
   as sealed linear M1; original 42-phone inventory including ERR/SPN, blank=42.
   Audit old timestamp-derived realized sequence against event-interleaved strict
   sequence first. If identical, label-policy parity is established on that
   subset; do not invent a supervision ablation. Compare inventory change only.
2. SM13 all-13: same 6/3/3 population, existing 12 target/5 gap epoch recipe.
   Replay full-fit historical baseline on all budgets and rerun a strict-fit
   subset control with the same event-head schedule. This matches fit utterances
   but not architecture/updates/label-task; no single-factor causal claim.
3. E4 finite-likelihood control on the completed frozen-MLP recognizers, same
   frame posteriors as E1, no new recognizer. Competing phones fixed by the
   predefined inventory ordering (four substitution/eight insertion alternatives),
   not chosen using cal/held gold. Unknown reference tokens remain in topology
   and are explicitly unsupported by the CTC inventory (neutral scores + mask).
   Ratios are uncalibrated scores; no event-posterior claim.
4. Candidate-coverage diagnostic: first six held utterances in each fold at
   each seed, beam widths 1/4/8 with at most 12 token expansions per frame;
   rescore unique candidate sequences with exact CTC forward after frame
   renormalization. Report retained mass and changes across widths, include
   greedy/canonical separately, no coverage guarantee from n-best normalization.

## Reference and inference contract

Provide a gold-free WAV + independently supplied phones path (validated inventory)
and a deterministic CMUDict text path. OOV/ambiguous unsupported input must be
explicit, never silently shortened. Text-reference outputs belong to their own
reference profile and cannot be joined to corpus gold whose phones differ.
Recheck exact request/reference/event IDs for all 1800 corpus requests. Measure
fit-only unanimous canonical prompt catalogs on held speakers separately; never
select a lexical variant by held labels or repair gold to match predictions.
Text/corpus mismatch remains a disclosed limitation, not hidden normalization.

## Budget and closure

Inherited cap: two CPU hours and two GPU hours; subtract existing measured M1
compute (including historical extraction) from the relevant device account.
No sweep or extension of the two-epoch reference after seeing results.
If the smoke resource projection exceeds remaining cap, stop and preserve
incomplete status; do not silently reduce the population/seed count.
M1 closure requires all the above registered artifacts/replays, documented
fidelity/unsupported inputs and a runnable supplied-reference inference path.
If they pass, close M1 as LOCAL_PROTOCOL_REFERENCE_IMPLEMENTATION_REPRODUCED;
leave original-paper metric replication, M2–M6 and promotion unachieved.
