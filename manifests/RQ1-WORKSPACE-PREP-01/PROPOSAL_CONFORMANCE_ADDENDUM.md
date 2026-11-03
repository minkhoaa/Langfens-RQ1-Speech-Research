# Shared proposal-conformance addendum — P-SCORE / P-SUP preparation

Disposition: **original-UIT-scope design alignment established; latest distinct omission-proposal conformance UNVERIFIABLE (source missing); empirical/product conformance NOT established.** This completes the missing documentary addendum with an explicit source gap, not a fictional approval. No final architecture selected.

## Sources actually reviewed

1. Original UIT KLTN0127 proposal: `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`, SHA-256 `98fa922584b750510d4b0c183c59a84455da9e64ab231b59bdb58862c56133d3`. Original Vietnamese §§2.2(b), 2.3 RQ1, 3(b), 4(c), 4(g) and bibliography reviewed; long single-line source reformatted only in `/tmp` for nontruncated reading.
2. Latest local research checkpoint/report: `manifests/RQ-RESEARCH-01/closure-2026-11-03/{PROCESS_CHECKPOINT,INTEGRATED_DECISION_REPORT}.md`; thesis, source/data and parallel-pilot closure reports. Closure and revalidation SHA256SUMS rechecked successfully before preparation. These were uncommitted on local HEAD `b25a7595360294798d8d06b37592dd65dd4c9c33`; preparation commits their existing bytes rather than reverting to an older remote.
3. Current outer source: phone normalizer, TextGrid parser, dataset, collator, model head/freezing sections, E1 evaluator O/Z/metric logic, alignment/G2P source seams and baseline config; archival E0/E1 summary/checkpoint. Nested `repo@baa4989243d5ff46b8b3bf5a3c280cdf2857382b` has pre-existing mode/symlink dirt and is not the working authority for new edits.
4. No distinct latest acoustic-grounded omission proposal located. Rechecked local documentary names/content in outer docs/manifests, closure's enumerated nested-search evidence, and KLTN top-level files. No raw dataset/archive or credential search. Nearest DIAG-01/E1 proposals and closure pilot designs are **not** substituted for the requested missing source. Search scope is not proof that no such document exists elsewhere. Supply exact path, version/date and digest, then append a clause-by-clause reconciliation before claiming full conformance.

## Reconciliation and explicit deviations

| UIT requirement / closure constraint | P-SCORE disposition | P-SUP disposition | Shared evidence still needed |
|---|---|---|---|
| RQ1 acoustic phone-level selected-error detection, not STT transcript correctness | Same E0 acoustic emissions, S vs SD hypotheses, acoustic/DP controls | Compact natural-operation/existence target, contemporaneous matched binary control | Prospective same-contract performance and incremental audio dependence |
| Target substitutions, final consonant omissions, confusable pairs subject to labels | Omission endpoint plus full/substitution guardrails; coda stratum separate from all deletions | Strict natural correct/substitution/deletion labels, including same-base errors | Phone/coda/confusable support and error-specific confusion/precision |
| Written-reference practice, dictionary/G2P; proposed forced alignment + GOP OR pretrained acoustic classifier | Segmentation-free alternatives explicitly relax a forced target segment that can hallucinate existence; retain self-aligned/segmental GOP comparator | Existing acoustic-reference routing retained initially; no gold boundary runtime | Alignment uncertainty, variants/repeats, no fabricated deletion interval |
| GOP is compatibility evidence, not independent subtype proof | Raw ratios/ranks named; graph deletion is a model hypothesis | Class labels are supervised hypotheses, not observed articulation | Calibrated confidence and abstention, expert checking |
| Same units/data; precision/recall/F1, false alarms; calibration distinct from evaluation, prefer unseen speakers | Shared gold denominators, prompt-only replay, raw ranking; E0's CAL acoustic-training overlap explicitly disclosed | Disjoint FIT/SELECT/CAL/VAL speakers; same conditions and successful updates | More independent speakers; G=2/reused VAL screen cannot establish population claims |
| RQ1 feedback accurate, clear, appropriate; hint → explanation/practice → TTS | No automatic definitive omission message from low GOP/DP gap | No automatic definitive subtype/cause message from softmax | Blinded stratified expert rubric, source-backed teaching guidance, retry/abstention |
| Web integration/history/permissions/recovery; measured feasibility | Future score service schema, no latency claim | Future task/checkpoint/deployment design only | Actual end-to-end measurements and integration tests |
| RQ2 is Multi-Agent Automated Test Generation | Not reassigned to RQ2 | Not reassigned to RQ2 | SpeechOcean762 transfer/feedback remains RQ1 extension, not natural-deletion label equivalence |
| Current project: frozen pretrained backbone, two-head CTC+binary, remote-only training | Exact-weight sidecar, no training | New operation/existence objective requires explicit distinct-arm policy amendment; not currently permitted by two-head rule | Operator authorization of objective/head/parameter changes before implementation/training |

The thesis considers GOP or acoustic classification; it does not mandate skip tokens, a universal scoring-first prerequisite, a complex edit architecture, a new foundation model, fixed 300 ms CPU latency, or that any candidate wins. Both lanes remain independent. P-SUP may proceed under its own future authorization without P-SCORE passing.

## Research closure carried forward without historical rewrites

- E1 trained to 3,600 updates but did not improve registered omission/full AP; retain E1 Folds-2–4 NO-GO. Rounded zero skip mass is not universal exact zero. E0 remains a research anchor, not certified diagnosis-ready product.
- DIAG-02B 9/80 parity failures remain PARTIAL and causal-training NO-GO; no invalid forced-attention or impossible 1.5×blank criterion is reused.
- O annotation identities differ from prompt-only Z. No gold TextGrid inference inputs in new arms. Historical evaluator semantics/numbers/tolerances remain byte-unchanged.
- Primary-source code-license gaps (including Cao/Fu reuse), Joint-Align full-text block and absent full-precision E1 records remain accounted limitations, not invented evidence. Do not copy unresolved-license code; an independently specified implementation still needs provenance review.
- Independent-pilot readiness is not RQ1 empirical closure. Shared V1 contract freezes populations/labels/masks/coverage/uncertainty and feedback obligations; arm registrations must later approve actual graphs/objectives, effect margins and execution authority.

## Conformance gate / source-gap resolution

**GO for requested setup/documentation only. NO-GO for implementation or experiments.** To close missing-source conformance, operator supplies the distinct latest proposal or explicitly accepts its unavailability for the next bounded step. Append source digest, exact clauses, differences from UIT/closure/V1, and resolution/authorization; never replace this historical limitation silently. If its requirements change V1, create a separately reviewed shared V2 before either arm executes. Feedback/integration/uncertainty requirements cannot be discarded merely because the omission detector improves.
