# Research log

## Cycle 1 — 2026-11-07

- Read PHASE3_SO_FAILURE_DIAGNOSIS, PHASE3 final/paired reports, TMCI validity/final/handoff, EVAL-V3 comparability/specification and EXP-34 review.
- Reconciled findings: shared threshold is a demonstrated policy bottleneck but not the only limitation; ranking at low-FP remains limited. Positive alignment duration is model/reference-derived, not genuine phone-presence evidence. DEV_SEARCH has four speakers and is development-exposed. Reference provenance remains unresolved.
- Audited narrative strength: TMCI reports describe EXP-24 attention as explaining boundary jitter and EXP-34 as causal source of substitution benefit. Treat as mechanistic interpretations until matched causal ablations; some language in historical reports is stronger than its evidence. Speaker/family slices are descriptive and sample-limited.
- Read EXP-81 review/spec as the latest relevant campaign evidence. Its own criteria failed and review rejects rigid contrastive boundary sharpening; because its metrics/protocol differ, this is directional evidence only, not a numeric comparison to EVAL-V3. It does not falsify softer/null-path alignment.
- Literature search consulted original ISCA/IEEE/Microsoft records for skip-state forced alignment, SSL MDD, Transformer MDD and APL embeddings. Metrics/protocols are not directly comparable to LANGFENS.
- Search limitation: exact author metadata for skip-state and detailed protocol for jointly learned alignment still require primary-PDF inspection. Logged as unresolved rather than guessed.
- Artifacts created: RESEARCH_STATE, LITERATURE_REVIEW, METHOD_COMPARISON, HYPOTHESIS_REGISTRY, EXPERIMENT_PROPOSALS, NEXT_CYCLE, RECOMMENDATION_LATEST.
- Next: inspect full original jointly learned alignment paper and audit available frozen frame-level artifacts/feature paths read-only; decide whether E-001 can be evaluated without producing new experimental results or whether only a future task specification is feasible.
