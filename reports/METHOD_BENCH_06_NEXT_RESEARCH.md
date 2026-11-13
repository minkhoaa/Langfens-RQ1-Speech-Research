# METHOD-BENCH-06 — Ưu tiên nghiên cứu tiếp theo

## 1. Exact-gap acoustic localization — ưu tiên cao nhất

Current gap baseline is meaningful (`AP-I=.162125`, four fold AP `.18879/.10271/.10945/.25154`), while E4 and historical T2 are much weaker. MB06 did not train a new method. Recover the frame-level signal, not just target-indexed H-002 crops or event-level gap scores. Raw WAVs are available; test whether frozen acoustic frame embeddings and predicted reference-boundary distributions can be generated within CPU/GPU budget. Proposed falsifiable method: weakly supervised framewise extra-phone eventness, aggregated into all exact N+1 canonical gaps using predicted alignment uncertainty. Use gold timing only for TRAIN supervision, never inference. Controls: acoustic-eventness only, phone/gap prior only, and predicted-boundary aggregation; full outer 3-speaker holdout and inner selection. Preserve leading/trailing gaps. First reconcile MB03 NPZ IDs to labels, then run baseline paired on exact identical gaps.

Do not substitute neighbor-phone pooled MLP, retest T2 local max, or report AP from unsupported boundary gaps. If frame representations cannot be made, file an explicit feasibility stop rather than a simulated temporal model.

## 2. E4 conditional omission evidence before nesting

Existing evidence suggests O may retain E4 within phone and in low neural-confidence regions; S appears strongly confounded by phone prevalence. Make a cheap fold-local diagnostic on allowed TRAIN data: fit phone priors and confidence conditioning only within outer-train speakers, apply to held-out fold, report paired per-fold AP differences and calibration-free ranking. If no stable conditional O/S signal, stop. If justified, implement the full nested outer/inner grouped protocol (not pooled OOF rank stacking); learn one constrained C/S/O simplex model and compare neural-only/fixed/pre-prior controls.

## 3. Calibrated operating policy only after ranking signal

Use DEV_CAL only under declared policy; do not fit on DEV_SEARCH. Report F1, precision/recall, warning budgets `.2/.5/1/2` per utterance, unrestricted diagnostic point, and per-class curves. Keep raw AP separate. Do not call sigmoid of CTC likelihood ratio calibrated without calibration evidence.

## 4. External method follow-up

MixGoP K4 adaptation is negative against direct neural targets and has no insertion prediction; do not increase mixture count. A worthwhile reproduction would need the source MixGoP feature extractor/frame treatment/data recipe, while preserving speaker-grouped evaluation. Alternative source leads in Agent C's paper review and Agent B joint-alignment papers should be screened for actual available inputs before implementation.

## Stop rules

- No nested E4 fusion unless fold-local conditional gains survive intended-phone and confidence controls.
- No extra GMM seed solely to search for a positive result; one seed screen is strongly below neural.
- No claim about temporal localization from sequence AP or pooled neighbor crops.
- No promotion of a unified checkpoint until exact-gap method is tested and calibration decisions are separated from raw ranking.
