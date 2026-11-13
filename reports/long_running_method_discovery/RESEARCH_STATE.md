# Research state — cycle 1

Last updated: 2026-11-07. Scope: research-only review of authorized repository evidence and public papers. No training, inference rewrite, registry edit, checkpoint selection, or sealed-data access was performed.

## Verified context

- Phase 3: Gold-Only and KD-All comparable overall; three paired seeds do not establish KD superiority. Low-FWR ranking remains weak.
- Phase 3 diagnosis: shared S/O threshold suppresses rare substitution recall. Even class-wise oracle use of the full 119-FP SEARCH budget yields only 10.69% S and 13.88% O recall for Gold (upper-bound diagnostics, not deployable results).
- Alignment code/report audit: forced Viterbi topology emits every target phone and has no arbitrary-phone epsilon deletion transition. 207/209 SEARCH gold omissions still receive positive model-derived duration. That duration is not evidence of acoustic realization.
- TMCI: EXP-24 has useful O ranking; EXP-34 provides lexical transition priors; EXP-28 provides gap energy/prosody. Compositions are exploratory and DEV_SEARCH is development-exposed (4 speakers).
- Exact insertion evaluation requires all N+1 gaps, including leading and trailing positions.
- Intended reference is annotation-derived; independent text-to-phone provenance remains unverified. CMUdict lookup alone misses OOV words.

## Cycle 1 priority

Investigate deletion-aware reference-conditioned scoring without treating an aligner-generated nonzero span as evidence of presence. First experiment should be an offline, matched scoring ablation using cached acoustic frame posteriors/features if authorized existing artifacts suffice; otherwise specify a future training task only. No claims of improvement before controlled evaluation.

## New campaign evidence checked

- EXP-81 (Subframe Temporal Contrastive Aligner) review reports failure against its own Macro F1/AP and false-warning guardrails on DEV_REPLICATION and attributes errors to rigid contrastive boundary sharpening. Its benchmark uses a different protocol/metrics from EVAL-V3 Macro-3 Error AP/F1, so values are not cross-comparable. It strengthens caution against hard boundary-separation losses but does not directly test optional deletion paths.

## Current ranking (hypotheses, not results)

1. H-001: optional-phone / deletion-aware local alignment with explicit null path and uncertainty; targets O false presence under forced alignment.
2. H-002: low-FP focused acoustic representation using frozen intermediate SSL layers plus local reference-conditioned pooling; targets weak S/O tail ranking.
3. H-003: exact N+1 gap-conditioned acoustic detector with boundary and energy trajectory; preserves insertion semantics.
4. H-004: specialist heads / disagreement-aware teacher transfer; exploratory after a strongest acoustic baseline.
5. H-005: speaker-grouped calibration and selective risk/coverage reporting; calibration is distinct from ranking.

## Validity cautions

DEV_SEARCH is exploratory, not independent final validation. Do not compare literature binary F1 with LANGFENS Macro-3 Error F1. EXP-24 mechanisms are not causal merely because ablations rank components. Avoid lexical priors as substitutes for acoustic evidence; report acoustic-only and prior-only controls. Tiny speaker count makes speaker-stratum uncertainty large.

## Restart

Read NEXT_CYCLE.md, refresh newly published Phase 4 findings if present, then pursue one bounded question: can a zero-phone/null alignment path improve O ranking on matched known cases without hurting short legitimate phones? Consult original skip-state / jointly learned alignment papers and design controls before recommending implementation.
