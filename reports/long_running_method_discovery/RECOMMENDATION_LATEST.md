# Recommendation — cycle 1 (research proposal only)

## Problem

Omission scoring is vulnerable to forced alignment that allocates frames to every intended phone. In Phase 3 diagnostics, 207/209 SEARCH omissions still received positive aligned duration. Existing ranking at the low-false-warning tail remains limited, even under a SEARCH-label oracle upper bound. The duration is therefore not proof of acoustic realization.

## Evidence

The source topology has no arbitrary intended-phone deletion transition. A skip-state HMM paper demonstrates zero-duration phone paths for speech reduction/deletion, but reports boundary alignment—not L2 MDD AP. LANGFENS EXP-24 O ranking is promising but not causal proof that attention fixes this issue. DEV_SEARCH is development-exposed, and reference provenance is unresolved.

## Proposed method

First specify an **offline, frozen-evidence optional-phone alignment ablation**: permit a null path for each intended phone with a fixed TRAIN-selected penalty; retain neighboring coarticulatory acoustic evidence; expose null-vs-emitting path margin and alignment uncertainty. Keep lexical transition priors separable from acoustic scores. Do not train or deploy a new model in this proposal.

## Scientific hypothesis (HYPOTHESIS)

Compared with forced emitting alignment, a deletion-aware null path may improve O ranking in low-FP regions while avoiding false presence evidence from forced duration. Improvement is not assumed; short legitimate phones may also be skipped.

## Minimal experiment

Same cached frame scores, same phone sequence, forced-vs-optional DP; penalty fixed on TRAIN; matched short-phone/correct controls; audio-only, lexical-only and current EXP-24 controls. Metrics: AP O, precision at fixed warning counts, false-null rate on short correct phones, AP S/I, speaker-paired uncertainty. DEV_SEARCH remains exploratory.

## Resources / baseline

If frame posteriors already exist: CPU dynamic programming and metric analysis; no GPU, no neural training. Strongest baseline: current frozen forced aligner + EXP-24 scorer, with Phase 3 specialist ranking separately reported. If requisite per-frame outputs are absent, implementation/training requires separate approval.

## Acceptance and failure

Accept for a later controlled model experiment only if O-tail gains reproduce across independent speaker groups and do not arise from lexical prior alone or increased false nulls on short legitimate phones; require no material S/I regression. Fail if gains are SEARCH-only, one-speaker-only, or driven by prior. No current result validates this method.
