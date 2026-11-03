# GOP-SF primary formulation, graph and license audit

**Documentary/source inspection only; no scorer/model executed.** Current public sources re-fetched to memory, not persisted/copied as implementation. No author-code copying without a verified license. No Langfens omission-performance or resource PASS.

## Authenticated sources actually inspected

| Public locator | Identity / current retrieval evidence |
|---|---|
| `https://arxiv.org/abs/2507.16838v3` | Cao, Fan, Svendsen, Salvi; v3 revised 2026-02-05; related DOI `10.1109/TASLPRO.2026.3654852`; HTTP200, 43,284 bytes, SHA256 `1038ff68023653e76d3d26e8cd457ad3d5eb8617bd697150b6c1175234401ef6` |
| `https://arxiv.org/html/2507.16838v3` | HTTP200, 292,492 bytes, SHA256 `01cf33796950f3f58fda2e214b781435ec8bc0dd79829eae0d8afe25f70725cf`; primary §§II–III including §§III-B–G read; declares CC BY4.0. HTML page wrappers can change; versioned scholarly identity, not future wrapper-byte stability, is the source pin. |
| `https://api.github.com/repos/frank613/CTC-based-GOP/git/trees/ffc4823330efe2c836691931a456ea9e3360871e?recursive=1` | HTTP200; 15,349 bytes, SHA256 `00c18ec338724f168aa025cbdbad7c50bb7e16f0fe6cd6c3ce002fb14bb83ceb`; `truncated=false`, no LICENSE/COPYING-named entry |
| Pinned raw `README.md` at that commit | HTTP200; 1,014 bytes, SHA256 `fa61652c01131622096bb65ab193849f02d9bf7b2a48abe40b599a9ee62e6e60`; differentiates `is24` precursor from `taslpro26` journal extension; author list matches primary |
| Pinned raw `taslpro26/gop_sf_sd_norm.py` | HTTP200; 14,999 bytes, SHA256 `2fb86f3120874f155e4b32d69ff4b0427c5d5a35ba0aaa6c11081d9cff670f66`; relevant initialization/neighbor/output lines 155–194,248–287 inspected as text, never imported/executed |

Raw prefix: `https://raw.githubusercontent.com/frank613/CTC-based-GOP/ffc4823330efe2c836691931a456ea9e3360871e/`. The paper explicitly links this repository; README links the journal extension. Publisher PDF byte equivalence was not checked. Full primary-literature and result qualifications remain in closure independent C; that report was read in full, not treated as a fresh experimental replication.

## Exact hypothesis and mathematical assumptions

Let canonical sequence be `C=(L,c_i,R)` and model emissions be normalized CTC phone/blank probabilities. Canonical probability sums **CTC paths**, not a best segment. Relaxed alternatives at one position are:

- S: `L . R`, exactly one arbitrary nonblank phone;
- SD: `L .? R`, one phone **or empty**;
- SDI: `L .* R`, arbitrary phone sequence including empty (separately registered reserve).

Eq.12 raw goodness is `L_CTC(relaxed)-L_CTC(C) = log P(C)-log P(relaxed)`. Error orientation is its negative. A single target-removed sequence is a different deletion-only contrast, **not** the whole SD denominator. Substitution/deletion alternatives need correct marginal mass and terminal acceptance. The uniform alignment-prior assumption in Eq.9–10 cancels a prior; it is a modeling assumption, not a demonstrated acoustic fact. Left/right contexts are fixed canonical strings and may themselves contain pronunciation errors.

Eq.6 divides raw goodness by estimated activation occupancy; Eq.13 uses normalized forward occupancy at central wildcard nodes, floored at 1 for near-zero occupancy. This is **not gold duration**, and not automatically forward-backward posterior duration. Scaling forward probabilities for underflow prevention (GOP-SF-numerical) differs from occupancy-normalizing the score (GOP-SF-Norm). Author lines 285–287 write raw GOP **and** occupancy as separate columns; no division at that output site.

Neither a score ratio nor permitting empty realizations is a calibrated human-omission posterior. Blank consumes frame time/non-emission; target deletion advances canonical position without an emitted target. Neither means observed physical silence or an exact absent-phone segment.

## Repeat, graph and neighbor qualification — do not claim exact reproduction

Author lines 176–193 and253–256 exclude the next canonical phone from wildcard choices for disambiguation. This is material for target==next-phone and adjacent repeats: canonical-path inclusion and a nonpositive goodness sign cannot simply be assumed for that restricted implementation. A clean independently specified unrestricted sequence-language union can differ from this author graph. Name and hash the actual topology/restriction; do not silently call it exact author reproduction or clip unexpected ratios to hide disagreement.

Before any experimental use require: explicit start/terminal/transition and epsilon-closure definitions; whether paths or distinct emitted strings are summed; no double counting when substitutions yield duplicate strings or deletion yields repeated neighbors; correct CTC repeat/blank handling; zero-frame/empty/single/first/last targets; no deletion of neighbors; impossible canonical paths typed separately; all-canonical-position accounting. Prove expected mass inclusion relations only for graphs that actually satisfy them. A graph-language baseline is a legitimate named hypothesis, not conformance certification for author code.

Log-sum-exp reference and/or scaled forward recursion with recovered scales are required for long sequences/zeros. Keep exact impossible-path infinities internal, emit typed abstention/null diagnostics rather than JSON infinities or favorable error scores. Compare to exhaustive tiny-path enumeration; occupancy floor/normalization and precision tolerances prospectively fixed. Author code inspection is not a numerical proof. Per-target complexity in §III-E is `O(T*(|C|+|V|))`; all-position scoring repeats denominators. Encoder reuse does not make scoring free; no milliseconds/VRAM forecast is attested.

## Prospective implementation registration (not author-graph certification)

The bounded implementation lane is registered to the explicitly selected `ctc-union-language-v1` contrast: S and SD alternatives, with distinct emitted strings deduplicated after collapsing CTC paths. A deletion permits neighboring equal phones to become an ordinary repeated-phone CTC sequence; standard CTC blank/repeat separation still applies. This is an independently specified unrestricted sequence-language union, **not** a reproduction of the author's neighbor-restricted topology or Eq.13 occupancy approximation. SDI is not in the current implementation scope. Preserve raw contrast as the primary score; no occupancy-normalized score, threshold, calibrated probability, subtype posterior, or scientific benefit is assumed. Score orientation is higher-is-error. Synthetic exhaustive tiny-path tests can verify implementation mechanics only, not acoustic validity or author parity.

This registration does not choose an experiment winner, authorize real-data scoring, or resolve owner acceptance margins. Any later empirical use still requires the existing experiment/data/asset gates and a separately frozen control and criteria.

## Evidence and reuse disposition

Closure primary Table I reports **mixed-real-error** CMU Kids macro-phone AUC SD .914, S .891, SDI .859; not isolated deletion AP, Langfens performance or proof more flexible graphs win. SpeechOcean PCC includes a different assessment task and sometimes supervised downstream models. Different encoder/fine-tuning/corpora prevent importing those results.

Paper CC BY4.0 permits reading/citing subject to its terms; it does **not** establish a license for the linked source code, weights, dependencies, datasets or downstream product. Pinned complete tree still lacks a verified code reuse license. **COPYING/PORTING AUTHOR CODE: BLOCKED pending permission/license review.** Independently authored published-math implementation, clearly citing assumptions/deviations and not presented as licensed author reproduction, is distinct and allowed in the present bounded synthetic tranche. It does not clear future model/data deployment rights. Fu code license, Joint-Align full primary text and other optional family licenses remain unresolved only where those assets/claims are pursued; they are not universal prerequisites for independent JSON/schema work.
