# METHOD-BENCH-03 — next research decisions

## Recommendation

Do **not** integrate current CTC E3 or GRL as a production/scoring candidate. Keep the implementation and posterior artifacts as a verified baseline and a useful failure case. The strongest next investment is recognition quality and event localization together, under speaker-grouped validation—not broader architecture sweeps.

## Ranked follow-ups

1. **Raise sequence posterior coverage and calibration.** The frozen Wav2Vec2 head is a credible recognizer baseline, but top-10 candidates retain tiny CTC mass. Export a tractable prefix lattice or calibrated n-best set; show mass and S/O/I convergence over search depth, then rerun exact-path marginalization. Keep candidate-mass diagnostics as a gate on posterior language.
2. **Connect sequence scores to speaker-grouped OOF.** Current CTC/edit events are DEV-only, while Agent C OOF streams are from another small head; complementarity and fusion are unknown. Produce fold-excluded CTC event predictions on the same speaker-grouped TRAIN identities and compare E1/E3, canonical/prior controls and a matched acoustic classifier without using held-fold labels for alignment or tuning.
3. **Improve the phone model only against grouped folds.** Compare a lightweight trainable adapter/temporal head over frozen Wav2Vec2 to the three-seed linear head, using realized and canonical PER, blank statistics, speaker-wise error supports and MDD event AP. Require both acoustic and MDD evidence; the pretrained recognition baseline still trails canonical-sequence diagnostic PER.
4. **Validate external eSpeak-to-LANGFENS mapping.** Build a mapping grounded in the inventory, define composite-token and diacritic handling, quantify coverage, and lock examples before evaluating the 392-token model. Its license/provenance are receipted; no LANGFENS metric exists yet.
5. **Improve exact-gap scoring after candidate coverage.** Compare probability of any insertion with expected inserted-phone count, interval-aware evidence and boundary/repeated-phone cases. Preserve N targets and N+1 gaps, EVAL-V3.1 identities, and matched deterministic controls; never use evaluation labels to resolve alignments.
6. **Test distinct speaker invariance only with a stronger rationale.** GRL did not help repeatably. If grouped-OOF analysis supports acoustic identity shift, a Spin-style speaker-perturbation/content-consistency method is a distinct candidate; run TRAIN-only paired augmentation against same-architecture plain controls and repeated seeds.

## Success and stop rules

Advance a candidate only if a matched speaker-grouped OOF comparison improves a supported event class without degrading others materially, and direction repeats across seeds/folds. Require exact-gap I coverage and verified event identities. DEV_CAL may select a frozen policy; DEV_SEARCH remains descriptive/exposed. Stop model integration if PER improves but MDD AP does not, if posterior coverage remains unquantified, or if gains depend on random-utterance validation only.

## Unresolved hypotheses

U2 insertion gap failure reflects severe in-sample overfit, but speaker identity, label/prevalence shift, acoustic condition, alignment geometry and training limits have not been causally separated. The CTC comparison has three projection-head seeds but one frozen encoder family and one corpus; the GRL study has one training seed. Low CTC performance may reflect capacity, optimization, supervision noise, or acoustic input limitations, which current evidence does not separate.
