# LANGFENS RQ1-METHOD-BENCH-01 — Research gaps and verification queue

## Evidence gaps before stronger claims

1. **Exact-gap mapping is a major unsolved transfer step.** Several papers address sequence insertions, but not LANGFENS ranking across all N+1 gaps. Define a mapping from decoded phone spans to gaps using only audio and intended reference, then test edge gaps, repeated phones, and ambiguous edit paths. Do not count sequence insertion as exact-gap capability until this is validated.
2. **Omission evidence is under-isolated.** Alignment-free CTC and sequence recognition make O plausible, but their gains may come from overall recognition quality, duration, or reference priors. Compare matched target-level O scores, duration-only controls, and acoustic sequence evidence.
3. **Rare insertion prevalence can hide ranking failure.** Report exact-gap AP and PR curves with all gaps as negatives; do not tune on DEV_SEARCH. A positive bag or word insertion alert is not event-level evidence.
4. **Speaker and phone identity may confound anomaly methods.** Prototype/OCSVM distances can encode speaker, recording or phone-frequency information. Use per-speaker reporting, speaker-only controls, normalization ablations and held-out speaker results.
5. **Acoustic versus linguistic prior contribution is not resolved.** For constrained confusion graphs, alternate references, and text-conditioned models, compare acoustic-only, prior-only and combined scores. Build graph counts from TRAIN only.
6. **Published benchmarks are not directly comparable.** Corpora include SpeechOcean762, L2-ARCTIC, Mandarin LATIC, child speech, speech audiometry and synthetic accents; reported outcomes include PER, WPER, AER, F1, FAR/FRR, PCC and word/utterance ratings. None is LANGFENS Macro-3 Error AP on matched event/gap populations.
7. **Full implementation reproducibility varies.** OpenPA links a repo; the linked code, commit, checkpoint, license and local size remain to audit. Several recent records have no verified code/checkpoints. IEEE speech-text MDD full-text method and model artifacts are not confirmed. Mark such work gated rather than approximate it and report a false reproduction.
8. **Recent 2026 evidence remains heterogeneous.** Some items are arXiv preprints or proceedings; verify final version, supplements, code and reported split before drawing method conclusions. Challenge-level gains can reflect additional authentic data as well as architecture.
9. **Construct validity and annotator uncertainty are open.** Current event gold is treated as the benchmark target, but perceptual acceptability may depend on listener and norm. Inter-rater data are needed before changing labels or interpreting irreducible disagreement.
10. **Discrimination and thresholded utility differ.** AP and classwise PR measure ranking; F1 depends on thresholds and prevalence. Freeze any policy on DEV_CAL, then report DEV_SEARCH descriptively; never frame threshold tuning as improved discrimination.
11. **Original and adapted methods must be separated.** Similarity-soft targets, SylPh-inspired centroids, attention-to-phone localization and discrete-unit mapping can be useful adaptations, but must not be presented as faithful reproductions unless original equations, data, training and inference details are matched.
12. **Resource evidence is incomplete for large encoders.** GTX 1650 VRAM and concurrent RQ1-UNIFIED-02 reservations make model size decisive. Checkpoint availability, actual peak memory, batch shape and CPU runtime are required before committing.

## Next verification work

| Order | Work item | Evidence needed | Decision it informs |
|---:|---|---|---|
| 1 | Audit local cached feature/logit inventories and event topology for CTC/GOP screens | File provenance, label coverage, real audio/reference contract, device-independent feature hashes | Whether H04/H05 can be run without new training |
| 2 | Inspect Zhou 2025 paper supplements/data/code | Similarity matrix definition, synthetic-error generation, target unit, source implementation, license | Faithful reproduction versus compact inspired adaptation |
| 3 | Inspect SylPh full text and original implementation availability | Positive/pseudo-negative bag construction, contrastive loss, adaptive text embeddings, localization method | Whether to invest in contrastive normality after centroid screen |
| 4 | Resolve discrete-unit surprisal artifacts | Encoder/codebook, native speech source, token LM form, Text2DUnit-DTW, code/license | Whether H03 has a feasible cached/CPU path |
| 5 | Confirm current GPU lease at a safe boundary | Shared wrapper lease receipt and free memory | Which neural workloads move to CUDA |
| 6 | Validate insertion mapping on synthetic examples, then genuine predictions | exact target-gap key coverage, no gold timestamps at inference, ambiguity audit | Whether sequence-family methods can claim I |
| 7 | Verify full text and artifacts for joint speech-text MDD / OpenPA | Paper method, checkpoint sizes, inference inputs, source commits/license | Keep gated or promote into follow-up candidate |

## Explicit non-claims

- The centroid screen does not show that SylPh was reproduced or that anomaly detection as a family fails.
- Published insertion/deletion metrics do not establish LANGFENS exact-gap AP.
- A higher external F1/PCC is not a LANGFENS benchmark result.
- The existence of a code repository does not establish a runnable, licensed, checkpoint-complete reproduction.
- Weak omission, rare insertion, alignment quality, representation quality and speaker generalization are research questions, not established causes from this review.
