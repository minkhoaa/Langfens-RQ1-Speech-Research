# Unified comparison and ranking sensitivity

**This is a research-readiness ranking, NOT measured performance, a probability of success, an architecture decision or experiment authorization.** Every candidate below lacks a same-contract new Langfens result. Scores are explicit coordinator judgments grounded in the [evidence matrix](SOURCE_EVIDENCE_MATRIX.md), not a vote or average of lane rankings. Historical scores are retained in their original files and corrected separately.

## 1. One thesis gate before the rubric

Use the same G1–G8 gate from [Agent E §4](independent/E_THESIS_ALIGNMENT_AUDIT.md): (1) targeted reference-aware pronunciation scope, (2) legitimate learner acoustics/no inference gold, (3) honest reference-phone/error semantics and uncertainty, (4) fair labeled evaluation, (5) accurate/clear/error-fit feedback with abstention, (6) credible web/failure/history integration, (7) traceability to proposed alignment+GOP/acoustic classification or documented method deviation, and (8) current authorization/comparability constraints.

**Documentary screen-compatible ≠ demonstrated thesis compliance.** A component may pass only as part of an explicitly completed system; no low rubric score changes a false acoustic claim into an acceptable one. Hard exclusions are text-only phone diagnosis, gold-informed inference, unsupported subtype certainty, changing baseline semantics, foundation-language-model training from scratch, or presenting a deletion-only detector as fulfillment of all declared RQ1 targets. These are claim/use exclusions, not blanket bans on model-family names.

All 19 bounded variants below are screen-compatible or conditional **for their stated role**, not certified complete RQ1 systems. Conversational Whisper is separately in scope; transcript-only pronunciation diagnosis fails the gate. Large pretrained recognizers/AED/RNN-T are conditional rather than proven too slow/divergent. Frozen-backbone/two-loss/budget constraints are operationally binding current contracts, not original-thesis dictates.

## 2. Common rubric

Components 1–5 (larger is better):

- **E, verified task-relevant evidence:** 1 conjecture; 2 adjacent-task/representation motivation; 3 speech-native or concrete method but limited target-specific evidence; 4 directly inspected substantial speech-task precedent; 5 replicated same-contract target evidence. **No new candidate receives E=5.** Venue prestige alone is not evidence.
- **M, expected full-MDD/omission utility:** 1 no meaningful mechanism; 2 indirect/limited absence representation; 3 plausible with consequential failure modes; 4 explicit useful alternatives/supervision; 5 demonstrated useful target tradeoff here. These are **hypotheses**, not predicted AP.
- **L, Langfens compatibility:** 1 unavailable incompatible contract; 2 broad changes/alternate initialization; 3 significant adapters/objectives; 4 existing labels/features but changed approved arm; 5 sidecar reuse of exact weights/emissions. Strict resume and partial initialization remain distinct.
- **C, bounded implementation/validation feasibility:** 1 large unresolved prerequisites; 2 broad new pipeline; 3 material graph/data validation; 4 compact bounded adapter/new objective; 5 existing lightweight decoder/control. This is seam analysis, **not measured time, memory or cost**.
- **F, corrective-feedback usefulness:** 1 opaque unsupported; 2 broad suspicion; 3 reference-linked suspicion requiring more diagnosis; 4 explicit event/uncertainty interface; 5 attribute-rich explanation support. High F is representational usefulness, **not validated advice quality**.

Base weights: **E25/M25/L20/C15/F15**, inherited as a transparent common preference scheme, not retrospectively called preregistered. `total = (25E+25M+20L+15C+15F)/5`, on a /100 display scale. A tie remains a tie. No decimal precision is manufactured.

## 3. All-family comparison

| ID / base position | Bounded candidate | E | M | L | C | F | /100 | Gate role / key uncertainty |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Q01 / 1 | Segmentation-free **GOP-SF-SD**, numerical/occupancy variants | 4 | 4 | 5 | 4 | 3 | **81** | Shortlist scoring hypothesis; deletion-capable alternatives, not calibrated subtype. Primary mixed-error evidence; high domain/emission/code-conformance uncertainty |
| Q02 / 2 | Existing free-phone greedy CTC + canonical DP | 4 | 3 | 5 | 5 | 3 | **79** | Mandatory meaningful control, not final winner. Recognition gap ≠ true omission; ties/variants/localization uncertain |
| Q03 / 3 | Compact natural correct/substitution/omission or hierarchical realization supervision | 3 | 4 | 4 | 4 | 4 | **75** | Independently eligible supervised shortlist; speech-native auxiliary precedent, existing labels. New objective/schema/grounding unverified |
| Q04 / 4 | Calibrated acoustic/binary/sequence fusion | 2 | 4 | 5 | 4 | 4 | **74** | Conditional complementarity experiment, not standalone source of new evidence. Correlated cues and fitting optimism; high uncertainty |
| Q05 / 5 | Segmentation-free **GOP-SF-S** | 4 | 2 | 5 | 4 | 3 | **71** | Essential denominator/substitution control. No empty target alternative; not presumed sufficient for omissions |
| Q06 / 6 | Compact canonical acoustic KEEP/SUB/OMIT / replacement tagging | 2 | 4 | 4 | 4 | 4 | **70** | Viable **alternative parameterization of Q03**, not independent validated family. GECToR speech transfer very uncertain; strict same-base/unknown/insertion semantics need design |
| Q07 / 7= | Explicit normalized probabilistic edit/skip lattice | 3 | 4 | 3 | 3 | 4 | **68** | Reserve independent hypothesis; explicit absence transition. Priors, path multiplicity/repeats/epsilon correctness and identifiability unverified |
| Q08 / 7= | Segmentation-free **GOP-SF-SDI** | 3 | 3 | 5 | 3 | 3 | **68** | Optional graph-flexibility control; more alternatives not automatically better; canonical metric undercounts insertions |
| Q09 / 9 | Phonological/articulatory attribute stream | 4 | 2 | 3 | 3 | 5 | **66** | Feedback/substitution complement or conditional detector, not demonstrated omission solution. Trait labels ≠ observed articulator motion |
| Q10 / 10 | Compact reference-aware joint recognition + phone/error objectives (CoCA/Fu-inspired) | 4 | 3 | 3 | 2 | 4 | **65** | Real speech precedents, broader adapter/training burden; prompt copying, shared errors and missed-deletion evidence. No transplant of published private/alternate setup |
| Q11 / 11 | Canonical CTC/self-aligned segment pooling and duration/blank features | 3 | 2 | 4 | 4 | 3 | **62** | Alignment/scoring control, **not omission oracle**. Every required canonical phone gets model occupancy; physical timing uncertain |
| Q12 / 12 | Classical external forced-alignment GOP | 4 | 2 | 3 | 3 | 3 | **60** | Thesis-proposed comparator; legitimate deployed inference alignment but weak absent-phone semantics. No universal runtime rejection |
| Q13 / 13 | Direct skip/attention-routing supervision | 2 | 3 | 4 | 3 | 3 | **59** | Separate E1-like ablation only, not historical E1 objective. Explicit target may teach priors; additional loss needs approval |
| Q14 / 14= | Pretrained phonetic AED/encoder-decoder | 4 | 3 | 2 | 2 | 3 | **58** | Conditional broader recognition experiment; speech precedent, different corpus/freeze/budget; necessity unproven, not universal divergence |
| Q15 / 14= | Alternate SSL encoder / acoustic recognizer | 4 | 3 | 2 | 2 | 3 | **58** | Conditional model-quality contrast; confounds scoring/objective unless matched; source/model license/init and domain uncertainty |
| Q16 / 16= | TRAIN-only acoustic retrieval + sequence comparison | 3 | 2 | 3 | 3 | 3 | **55** | Viable reserve/substitution probe. Aggregate primary result but no omission breakdown; insertion-heavy recognition, index isolation/license requirements |
| Q17 / 16= | MoChA / monotonic attention without explicit edit state | 3 | 2 | 3 | 3 | 3 | **55** | Conditional alignment component/control, not standalone absence model; no inherent too-slow claim |
| Q18 / 18 | TTS/reference-audio template matching as detector | 2 | 2 | 3 | 2 | 4 | **50** | Conditional reserve; voice/prosody/variant confounds and no verified target evidence here. TTS **demonstration** is separately in scope, not this detector hypothesis |
| Q19 / 19 | Pretrained phonetic RNN-T/transducer | 2 | 3 | 2 | 2 | 3 | **48** | Conditional reserve; no verified exact target-specific source/initialization here. Blank not semantic omission; resource claims unmeasured |

**Why controls can rank above a new model:** readiness includes actual available interfaces/known decoder behavior. Q02 is readily researchable but has no demonstrated omission advantage. Its rank is not an architecture endorsement. Likewise Q09 is not an omission-first replacement just because it is useful for explanations. Candidate roles and gate conditions survive all weight changes.

No family is rejected through invented CPU milliseconds, GPU hours, memory or universal convergence claims. Q03/Q06 overlap; Q01/Q05/Q08 are graph variants, not three independent replications. Q07 may overlap Q01's concrete graph mechanism; justify extra machinery before duplicating it.

## 4. Numerical ranking sensitivity

Fixed component judgments above; alternative declared weight profiles:

| Profile | E | M | L | C | F |
|---|---:|---:|---:|---:|---:|
| Balanced | 25 | 25 | 20 | 15 | 15 |
| Evidence-heavy | 40 | 20 | 15 | 10 | 15 |
| Minimal change | 15 | 20 | 30 | 25 | 10 |
| Feedback-heavy | 15 | 15 | 15 | 15 | 40 |
| Omission-heavy | 20 | 40 | 15 | 10 | 15 |

| Candidate | Balanced | Evidence | Minimal change | Feedback | Omission |
|---|---:|---:|---:|---:|---:|
| Q01 SF-SD | 81 | 80 | 84 | 75 | 80 |
| Q02 greedy CTC/DP control | 79 | 78 | 85 | 75 | 74 |
| Q03 supervised subtype | 75 | 72 | 77 | 77 | 76 |
| Q04 calibrated fusion | 74 | 67 | 80 | 77 | 75 |
| Q05 SF-S control | 71 | 72 | 76 | 69 | 64 |
| Q06 edit tags | 70 | 64 | 74 | 74 | 72 |
| Q07 edit lattice | 68 | 67 | 66 | 71 | 71 |
| Q09 phonological complement | 66 | 70 | 63 | 76 | 62 |
| Q10 reference-joint | 65 | 69 | 60 | 68 | 65 |

Consequences, not a forced consensus:

- Minimal-change preference ranks the existing decoder control above SF-SD; feedback-heavy preference gives Q03/Q04 77 and Q09 76, ahead of SF-SD/DP at 75. No single first place persists across profiles.
- Omission-heavy preference retains the explicit-alternative and supervised hypotheses while lowering attribute-only/segmental methods. This reflects the chosen utility, not new measurements.
- At balanced weights, one component changed by ±1 moves a total **3–5 points**; ±1 on E and M moves it by up to 10. The top four's separations (2/4/1 points) are not robust to reasonable judgment uncertainty. Those are sensitivity bounds, **not statistical confidence intervals**.
- More aggressive evidence discounts for unverified code/license/emission transfer can demote Q01; more value on available natural labels can promote Q03. The same gate still forbids inference gold and unsupported feedback claims. Licenses may block copying an implementation, not necessarily independent study of published mathematics.
- No useful numeric ranking distinguishes Q03 from Q06 until their exact targets/capacity/objectives are fixed. The difference here is evidence maturity/label-design burden, not known performance.

## 5. Experimental shortlist without a winning architecture

**Independent initial contrasts:** P-SCORE (Q01 against Q02/Q05/Q08/Q11/Q12 controls) and one bounded P-SUP (Q03 **or** Q06). Neither needs the other to pass. Q07/Q10 are reserve mechanism tests if a specific unresolved question justifies their burden; Q09 supports the separate feedback/substitution endpoint. P-FUSION is conditional on component predictions and leak-free calibration, not mandatory or automatically beneficial.

Final choice requires same-input/same-denominator outputs, successful-update matching, acoustic controls, correct/omission/substitution tradeoffs, independent speaker evidence, resource measurements and validated feedback/abstention. Until that evidence exists, retain multiple plausible explanations and stop at [pilot specifications](PARALLEL_PILOT_SPECIFICATIONS.md).
