# METHOD-BENCH-01 Candidate recommendations

## Transparent ranking (0–2 each)

Score dimensions: potential on weak event classes, mechanism distinction, source/reproducibility, 4GB feasibility, information value. Scores are triage judgments, not measured quality.

| Rank | Candidate | Weak-class potential | Novel mechanism | Source/reproducibility | Feasibility | Information value | Total / 10 | Recommendation |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Realized-phone recognition + soft similarity | 2 | 2 | 1 | 1 | 2 | 8 | First training experiment if target/realized-phone labels and code/data are obtainable; compact head adaptation is feasible. |
| 2 | Correct-only prototypes / SylPh-inspired | 2 | 2 | 1 | 2 | 2 | 9 | Start with frozen features and prototype scoring; explicitly test S/O and exact-gap I separately. |
| 3 | Restricted confusion-graph GOP | 1 | 1 | 2 | 2 | 1 | 7 | Cheap scoring test on cached logits; call it inspired unless exact code/formula is reproduced. |
| 4 | Discrete token surprisal | 1 | 2 | 1 | 2 | 2 | 8 | Inspect 2026 original paper and test representation availability; promising particularly for gap occupancy. |
| 5 | OpenPA cross-attention | 1 | 2 | 2 (code linked) | 0 | 2 | 7 | CPU/frozen localization probe only; does not natively provide phoneme error scores. |
| 6 | Joint speech-text MDD encoder | 2 | 2 | 0 | 0 | 1 | 5 | Request/inspect full paper and weights before committing compute; otherwise blocked. |
| 7 | Phonological features (H-006) | 1 | 0 | 2 | 1 | 0 | 4 | Existing project family; do not count as discovery novelty. |

## Dispatch-ready suggestions

1. **MB01-H02:** Use frozen acoustic features and correct-only target-phone prototypes. Fit no parameters using DEV_SEARCH. Return event scores for S/O and each exact target gap; compare against the existing gold-only and specialist controls with official EVAL-V3.1. Distinguish alignment/localization heuristic choices in manifests.
2. **MB01-H01:** Inspect paper supplementary/GitHub and VCTK-accent availability. If direct adaptation is too costly, train a lightweight phone-sequence head on existing permitted training representation cache. Label as `INSPIRED_IMPLEMENTATION` unless full formulation is matched.
3. **MB01-H04:** Reuse frozen CTC logits, derive confusion alternatives only from TRAIN, then calculate restricted versus unrestricted logit margins; no GPU training needed. This is a fast test of whether a new phonological prior contributes beyond current GOP.
4. **MB01-H03:** Continue discrete-unit surprisal beyond the current phone-conditioned unigram screen. Inspect the original code and test sequence surprisal plus text-guided alignment; treat exact-gap insertion as unsupported until unmatched token spans map to every N+1 gap without gold timestamps.
5. Keep OpenPA / speech-text encoder as gated investigation pending VRAM, checkpoint/license and event compatibility checks.

No candidate is currently supported as a likely outperformer of the approximately 0.3018 exploratory DEV_SEARCH specialist-fusion reference. That comparison requires matched event predictions and careful use of exposed DEV_SEARCH. The scientific case for a sequence/edit formulation is strong enough to test; outcome is unknown.
