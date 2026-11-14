# H-001 — preflight and execution checkpoint

Status: `EXECUTED` for A/B frozen frame-classifier max-sum scoring; `NOT_RUN` for CTC sequence deletion likelihood; overall result is exploratory, not independently confirmed.

## Source topology correction

`tools/extract_acoustic_features.py::viterbi_align` builds `[phone_0, SIL, phone_1, SIL, …]`, permitting stay, +1 and a +2 jump when the intervening state is SIL. No arbitrary deletion arc exists for phone slots 1 onward. The initializer permits both states 0 and 1 on frame zero, so phone 0 can be skipped already. No distinct leading-SIL state is defined: that SIL state is separator/gap 1. N=0 is not handled by this routine. The omission report documents these exact quirks.

## Cache discovery and class semantics

`experiments/evaluation_v3/closure/waveform_full/utterances/*.npz` contains one `ctc_logits` tensor per authorized utterance plus `canonical_phones`, exact target/gap feature summaries and `STU-01-GOLD-ONLY_seed101` outputs. The index lists 5,398 artifacts and the A/B run verified all hashes and sizes. Full population includes 1,800 TRAIN, 300 DEV_CAL, and 599 DEV_SEARCH utterances; aligned logits total 977,062 frames.

The logits are emitted by `charsiu/en_w2v2_fc_10ms`; config says `Wav2Vec2ForFrameClassification`, and CharSiU describes this W2V2-FC variant as frame classification trained on forced-aligned labels. The local artifact name `ctc_logits` does not make them CTC posteriors. They validly support replaying the existing max-sum aligner and a matched optional-phone max-sum path. They do not support a scientifically identified CTC sequence marginal. H001-C awaits a verified phone-CTC model and blank/token contract.

## Executed method

`run_h001_viterbi_optional_cpu.py` uses stored logits only. A is negative mean target-phone log posterior on the reproduced current forced path. B removes one target phone state at a time, keeps all remaining phone and SIL states, and scores deletion-path minus full-path Viterbi score divided by utterance frame count; skip penalty is fixed at zero. Each arm has a fixed-recipe one-dimensional TRAIN-fitted O-vs-C logistic readout; unchanged Gold-Only S and gap-I heads are retained. All outputs use full N/N+1 populations and EVAL-V3.1.

Canonical outputs use full event IDs and are independently replayed through `UnifiedEvaluatorV3`; recorded AP/F1 values match with zero delta. Search values remain exploratory, and no model/threshold was selected on SEARCH. Full evidence and exact hashes are in [RC2_H001_ALIGNMENT.md](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H001_ALIGNMENT.md) and the run receipt under `run_20261107_viterbi_optional_cpu_v3/`.

## Limits and resume

Optional B retains adjacent SIL states around the deleted phone, which can be too permissive; no SIL-collapse ablation was run. O AP improves in the exploratory comparisons, but O F1 remains zero on SEARCH under the common argmax policy. Four SEARCH speakers cannot support strong generalization inference. Intended-reference independence remains unverified.

Next bounded studies: compare B with an explicitly specified single-gap/SIL-collapsed null topology; fit calibration/readout with speaker-grouped TRAIN OOF; then run C only with a verified phoneme-CTC posterior source. Do not access TEST_SEALED.
