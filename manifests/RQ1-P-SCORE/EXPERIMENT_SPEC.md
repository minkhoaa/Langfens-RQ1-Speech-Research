# P-SCORE — frozen E0 emission scoring pilot (design only)

**DOCUMENTATION_READY; IMPLEMENTATION_AND_EXECUTION_NO_GO.** Workspace `/home/khoa/Projects/KLTN/langfens-rq1-p-score`, branch `prep/rq1-p-score`. Frozen common base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`. Binding contract: `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`, SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`. No dependency on P-SUP or on a successful scoring outcome before supervised research.

## Question and primary contrast

Under exactly the same frozen E0 CTC emissions and prompt, does admitting a target-deletion alternative improve omission ranking versus allowing substitutions only, without destroying full/substitution discrimination or coverage? Primary contrast: **GOP-CTC-SF-SD minus GOP-CTC-SF-S**, raw error-oriented score on deletion-vs-correct AP. E0 binary evidence, self-aligned/segmental GOP and greedy CTC+canonical DP are required controls. Dedicated deletion likelihood contrasts, occupancy normalization and SDI are separately named auxiliary analyses, not additional primary claims.

No numerical expected improvement, resource estimate or winner is asserted. Practical AP/recall gains, full/substitution noninferiority, coverage and false-alarm margins require prospective operator approval before execution. All attempted settings/failures reported; no post-hoc graph/normalization sweep on VAL.

## Immutable inputs and controls

- E0 `best.pt`, 3,600 updates, expected SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`; backbone pretrained revision/freeze as shared contract. No parameter change, head replacement, optimizer or warm start. E1 is archival context only, never substituted for E0.
- Runtime waveform/prompt only (Z-PROMPT-V1). Prompt/dictionary/normalizer/source hashes are in shared provenance. Corpus package identity is archival until actual authorized readback. No data staged here.
- Future emissions cache must be private external storage, keyed by checkpoint/source/audio/preprocessing hashes and valid frame length. Store FP32 log probabilities, vocabulary/blank map, shape/dtype/device/precision/source; no padded-frame scoring. Same cache bytes feed S, SD, self-aligned and DP controls; existing binary scores separately tied to same runtime inputs. Never reuse a cache with gold-derived cropping or canonical-specific emission conditioning.
- Report historical O E0/E1 separately; new candidates and new prompt-only E0 replay share allowed IDs and offline mapping. No O label queries or gold boundaries in new runtime.

## Scorer specification to implement only after separate authorization

Canonical sequence C and observations O: canonical log likelihood is `log P_CTC(C|O)`. At each **position i**, fixed left/right canonical context surrounds target-relaxed alternatives. S permits one phone; SD permits one phone OR empty; SDI permits an arbitrary phone sequence including empty (reserve only). Blanks are CTC path mechanics, not a DELETE class. Raw GOP is canonical log likelihood minus relaxed-graph log likelihood (equivalently relaxed CTC loss minus canonical CTC loss); error orientation is its negative. A single target-deleted path is **not** the full SD denominator. Canonical term is shared; every position needs its own denominator.

Before implementation, approve an explicit state/transition/terminal and epsilon-closure table, path-set de-duplication/weight semantics, next-phone wildcard restriction and repeat rules against the inspected primary source. Mathematical graph semantics and author-code disambiguation can differ: name any independently specified variant, do not quietly call it exact reproduction. Record whether canonical paths are a subset, duplicate alignments counted once, and mass relation expected. Do not invent a normalized posterior from an overlapping-path sum.

Separate three issues:

1. Numerical forward stability: log-sum-exp reference; scaled/probability recursion equivalence and final scale recovery. No clipping/underflow that changes rankings unnoticed.
2. Activation-occupancy normalization: named `raw` versus `norm`; paper floors estimated central wildcard occupancy at 1, not gold duration. Verify occupancy definition and divide explicitly when that named variant is enabled. Author code emits raw GOP and occupancy separately; file existence is not normalized-score conformance.
3. Topology semantics: epsilon target deletion must consume no acoustic frame and must not delete neighbors; adjacent/repeated target phones require proper CTC blank separation and position identity. Impossible contexts yield typed failures, not favorable deletion evidence.

Controls:

- Existing frozen E0 sigmoid binary score, legacy-comparable `s_error`.
- Greedy CTC phone sequence + fixed edit DP against prompt. Save hard gap indicator plus any separately defined continuous margin; hard DP ties limit AP resolution. Recognition deletion ≠ true omission.
- Self-aligned/segmental GOP: derive acoustic activation alignment from model/prompt only, declare pooling/denominator/duration conventions; cannot use TextGrid intervals. Forced canonical segment for a missing phone is not physical localization evidence.
- S/SD primary pair, identical normalization regime; occupancy-normalized pair secondary. SDI requires an explicit additional registration; do not assume more path freedom wins.
- Acoustic dependence: matched vs within-speaker shift-1 mismatch, phone/position prior and blank/duration diagnostic controls under shared policy. No causal forced-attention shortcuts.

## Outputs and assessment

Use shared per-token/runtime/offline schema plus `OUTPUT_SCHEMA_AND_TASK_PLAN.md`. `s_error` is the declared negative raw GOP for primary S/SD; it is a ranking score, **not calibrated probability**. Dedicated `s_omit` only for an explicitly registered additional deletion contrast, otherwise null/not supported. All scores retain algorithm/normalization identity. Report full binary, omission-vs-correct and substitution-vs-correct AP with same score plus dedicated scores separately; recall/precision/achieved FPR and PER/coverage as V1. No complete-population superiority at incomplete mapping coverage. Confidence/feedback absent until evaluated; no automatic definitive omission message.

E0 trained on all TRAIN: CAL/SELECT are not independent of its acoustic fitting. Primary ranking is frozen-weight exploratory validation, no learned fusion/calibrator. Calibration work requires separate authorization and explicit historical-overlap qualification. Two repeatedly reused VAL speakers are not confirmatory significance evidence.

## GO / NO-GO

GO now: review documents and provenance. NO-GO: any implementation/scoring/inference/model tests/fitting/GPU/deployment. Future implementation gate: missing proposal supplied or limitation accepted, primary source/license resolution for code reuse, graph/occupancy/DP registration approved, shared hash intact, synthetic no-training tests authorized. Future real-data gate: prospective effect/noninferiority margins; source/weight/package/cache hash verification, baseline parity qualification, failure accounting and operator authorization. NO-GO on graph mismatch, repeats/terminal bugs, hidden gold input, TEST membership, unsupported score probability claims or changed historical evaluators. Failure of P-SCORE does not veto P-SUP.
