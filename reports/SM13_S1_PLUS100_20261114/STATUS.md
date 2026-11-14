# S1 local +100: terminal150 verified after CPU-only report recovery

Latest evidence: `VERIFICATION.json` (`S1_PLUS100_STATES_AND_FULL_CURVES_VERIFIED`),
`RECOVERY_VERIFIER_STATUS.json`. Both heads completed150 total epochs, with target13200
and gap6900 optimizer updates. Training PID1579721 and observerPID1583121 are terminal;
no training was restarted during reporting repair. Additional100 window took855.09s
including preflight and the initial failed report step. Original50 parents remain intact.

## Fixed endpoints on the same450 reused TRAIN-held utterances

| Total epoch/head | Detection micro F1@0.5 diagnostic | F1 using frozen epoch50 calibration thresholds | Macro S/O/I AP |
|---|---:|---:|---:|
| 50 |38.6507%|42.6627%|25.7442%|
|100 |41.7724%|44.0037%|27.3496%|
|150 |40.6829%|43.7263%|28.9543%|

No new threshold tuning or best-epoch checkpoint replacement. Every1..150 curve point
and fixed50/100/150 endpoints reproduce from keyed saved predictions. Verified204 states:
200 new epoch states,2 metadata-only migrated epoch50 states and2 final last states;
model/Adam/RNG/cursors/history/finite checks and retained cache hashes passed.

Mean diagnostic F1 over41–50 /91–100 /141–150 is39.7819 /39.9470 /39.7981%:
late F1 is approximately flat/noisy, despite higher AP. Macro AP improves+3.2101pp
from50 to150, but remains below historical S0@50=33.4066% (not an epoch-matched
backbone-effect estimate). Target fit CE drops0.2011→0.0958 while held CE rises
1.4602→2.3781, consistent with overfitting/increasing confidence on held errors.
No claim of large-backbone superiority, full fusion or final common-model evaluation.

Curve: `VERIFIED_LEARNING_CURVES.png`; data: `CURVE_REPORT.json`.
This is single-seed exploratory reused TRAIN calibration1350fit/450held, not
fullTRAIN1800→DEV_SEARCH599, unseen confirmation, test or paper hierarchical F1.
No automatic extension after150. All compute was local; no CROTTC/runtime change.

## Reporting incident and transparent recovery

The training driver saved both complete150 states, then failed inside `curve_report`
with `invalid detection score`; the first finite observer correctly refused to certify it.
Original `STATUS.json`, `VERIFIER_STATUS.json`, error logs and original evaluator source
are preserved under the runtime / `POSTPROCESS_INCIDENT/` (not rewritten into successes).

Seven valid FP32 target probability rows at epochs83/109/111/134/135/146/147 produced
p(S)+p(O)=1.0000001192092896 (one FP32 ulp above1). The narrow derived-score guard now
bounds only dtype-roundoff within2eps; material errors, invalid direct exports, NaNs,
missing keys and duplicate keys still fail. Saved class probabilities are NOT altered or
renormalized. Typed AP/training remain unchanged. All441-grid/fixed/frozen threshold
binary decisions were independently checked unchanged; saturated binary-AP score ties
use the now-valid bounded score definition.

`POSTPROCESS_RECOVERY.json` records the sole reporting source change (`detection.py`),
seven adjustments, before/after SHA of protected training artifacts and no GPU/training/
remote activity. CPU126 SM13 regression tests passed, including actual FP32 reproducer
and rejection of material invalid scores. No full-repository PASS claim.

The verifier was explicitly extended for this report-only recovery and subsequently
verified all204 states/full curves/cache. `VERIFICATION.json` records the preserved
original failed driver status and recovery receipt SHA. Original training registration
hashes remain immutable; do NOT rerun training with changed evaluator sources as if it
were the old registered run. Future training needs its own explicit registration.
