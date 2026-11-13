# Phase 4 handoff — next defensible research steps

## Current gate decision

Do not call EXP-24+34 a validated winner; retain it as a post-hoc/exploratory SEARCH candidate. The official frozen EXP-36 recipe reproduces exactly and leads canonical DEV_SEARCH macro-3 AP among listed existing controls, but SEARCH is development-exposed and speaker count is four. No acoustic causal conclusion or UPED improvement has been established. No Phase 4 neural training was run.

## Priority 1 — close provenance before more modeling

1. Retrieve primary EXP-24 and EXP-34 training/preprocessing implementations, feature schemas, labels-used proof, normalization/checkpoint lineage, and exact recipe history for the two-model candidate.
2. Reconcile intended-reference derivation against gold realized annotations. Define approved orthographic-text-to-phone/G2P and variant resolution, enumerate OOV **tokens** and affected utterances with exact denominators.
3. Preserve EVAL-V3 event-ID mapping, all N target and N+1 gap slots, speaker-disjoint memberships, and source hashes in an immutable experiment manifest.
4. Establish a clean, exclusive CUDA lease and capacity check before any neural workload. Preserve shared active artifacts and do not treat stale lock markers as current leases.

## Priority 2 — paired acoustic falsification study

Pre-register B0-B6 from `PHASE4_B_ACOUSTIC_DEPENDENCE.md`, define each intervention from verified feature ancestry, and freeze CAL-only selection. Run full audio/reference first; then matched prior/prosody masking and prior-only controls. Mismatched audio/reference results remain sensitivity diagnostics, never valid label-scored outcomes. Verify event key conservation and official evaluator parity. Report CTC omission topology and gold-derived diagnostics separately from deployable inference.

## Priority 3 — unified neural development

1. Freeze D0/D1/D2 architecture and full EXP24 training contract; choose equalized updates/objectives and matched seeds 101/202/303 before training.
2. Use TRAIN for fitting and DEV_CAL for checkpoint/threshold/calibration decisions according to one frozen protocol. Do not optimize on SEARCH.
3. D3/D4 require B's lexical/prosodic provenance and intervention gates; D5 only after both pass. D6 distillation only if validated complementarity justifies the added complexity.
4. Train/infer only under authorized CUDA lease. Publish atomic, immutable artifacts and evaluate with EVAL-V3 including full target/gap coverage.
5. Compare ranking, classification, calibration and operating policies separately. Do not impose FWR<=0.20 as a capability gate.

## Confirmation and uncertainty

DEV_SEARCH has already been exposed and must not serve as independent confirmation. Use a new authorized speaker-disjoint evaluation population if available; otherwise state that confirmation is blocked rather than claiming population generalization. With four SEARCH speakers, avoid event-wise significance tests and avoid false-precision speaker confidence intervals. Use speaker-cluster analysis only as descriptive evidence until cluster count is adequate.
