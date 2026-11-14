# S1 raw vs fit-only layer-RMS; conditional total300

Decision `SM13-RAW-NORMALIZED-CONDITIONAL300-20261114-v1`.
Latest user: **đối chứng raw vs normalized trước, nếu ổn thì train lại với300epoch**.
Local only; preserve CROTTC. This protocol is frozen before pilot outcomes.

## Claim and controlled setup

Question: On the same reused TRAIN1350fit /450heldABA+ASI+BWC protocol, does
fit-only13layerRMS preprocessing improve S1 native frozen WavLM Large head detection?

Both arms are freshly initialized, same seed101(target)/1101(gap), same trainable
initial weights and random draws, full existing native FP32 cache, canonical refs,
registered layerindices0,2,...24, heads/loss/LR/weightdecay/batches/dropout. The only
arm difference is13 frozen acoustic-layer scale divisors fitted on fit speakers.
No backbone forward/backward, new feature extraction, new corpus, layer/LR/seed sweep,
E4/fusion change, threshold tuning, final DEV_SEARCH or sealed test/DEV_CAL access.

Both heads/arms run50 total epochs, order rawtarget→normalizedtarget→rawgap→normalizedgap.
Same environment/shared GPU lock/two CPU threads. One paired seed: no stable-seed or
unseen-confirmation claim. Hardware/environment and profile recorded. Runtime is not
claimed as a throughput benchmark; no statistical performance claim from single timings.

Statistics are recomputed streaming from44746target/46096gap fit events, no held inputs;
frozen buffers saved in every normalized checkpoint. Canonical reference vectors unchanged.
Trainable initialization equality checked; optimizer/RNG/shuffle/cursor/full state atomic
per epoch and safe optimizer-boundary pauses. Synthetic resume tests include normalized
buffers and matched weights/Adam. Raw50 must reproduce historical same-recipe model
within declared weight tolerance1e-5relative/1e-6absolute before admitting300.

## Locked gate — operational definition of “ổn”

Correctness must pass in a **separate CPU subprocess verifier**, including every pilot
state/prediction, recomputed fit-only scales, matching event IDs/populations, fresh full
held head inference, optimizer/RNG/cursors and exact recomputation of the gate.

All quality conditions below must pass, using fixed epoch50 endpoints and identical
score/decision policy (target=pS+pO, gap=pI, threshold0.5, score>=threshold):

1. Pooled held micro detection F1 normalized minus raw **>=+1.00 percentage point**.
2. Mean paired F1 delta over epochs41–50 **>=0**.
3. Macro S/O/I AP delta **>=−0.50 percentage point**.
4. Each S/O/I AP delta **>=−2.00 percentage points**.
5. At least2of3 held speakers have nonnegative pooled F1 delta.
6. False warnings increase **<=0.50 per utterance**.

No best epoch, class dropping, threshold retuning, gate loosening after outcomes or
choosing a better seed. This is an exploratory development admission rule, not a
scientifically established success/thesis threshold or unbiased final evaluation.
Report full curves, deltas, per-speaker counts/P/R/F1, S/O/I AP and FWR regardless of
pass/fail. Exact27 ordered speaker bootstrap delta outcomes give a **descriptive**
interval only;3 reused speakers/one paired seed are weak uncertainty evidence.

If correctness or quality fails: **STOP after pilot**, preserve failures/results,
no300 fitting or automatic retry. If all pass: record verified pilot SHA and fresh
resource admission before conditional continuation.

## Conditional300 semantics

Fresh normalized head trained0→50 in the pilot may continue with its exact complete
model/Adam/RNG/history/cursor state50→300: **300 total/head, not300 additional**.
No warm start from old raw350 or converting a trained raw head. Pilot50 artifacts,
comparison/gate and verification remain immutable. Same recipe/scales/threshold0.5;
no early stopping or hyperparameter modification introduced here.

Save full checkpoint every epoch. Final endpoints50/100/150/200/250/300 plus complete
curves. Separate CPU final verifier checks602 normalized states, predictions/metrics,
fit-only normalization buffers and unchanged pilot artifacts. Comparing final normalized300
against raw50 is **not epoch-matched normalization-gain evidence**; normalization effect
is supported only by the paired50vs50 pilot. Historical raw300 can be an exploratory
reference, not a freshly paired replicate. Stop at300, no autoextension.

## Storage/resource admission

Existing cache remains read-only under `/var/tmp/.../sm13_s1_train_cache_v1/`.
That filesystem currently has too little space for another several-GiB checkpoint
campaign plus CROTTC export reserve. Do NOT delete or compress historical artifacts.

New weights/state go **outside Git** to
`/home/khoa/.local/share/langfens/sm13_raw_normalized_20261114_v1/` on the separate
/home filesystem. Admit only with **4GiB /home floor +5.5GiB projected new outputs**;
monitor4GiB floor during run. Independently retain the unchanged **14GiB /var/tmp
CROTTC export floor**. RAM16GiB initial/4GiB runtime, VRAM512MiB; GPUsharedlock,
2CPUthreads, one bounded<=4h window. Stop/pause on floors/nonfinite/mismatch/OOM;
never skip failed batches or stop CROTTC. No remote use/fallback/API work.

## Provenance/reproduction

`research/sm13_backbone_side/raw_normalized.py --prepare` creates immutable
`REGISTRATION.json` before running. Driver invoked with `--registration ...`.
`tools/verify_sm13_raw_normalized.py --stage pilot|final` is CPU-only/no launcher,
invoked as a blocking subprocess by the bounded driver before admission and at completion.
Original `epoch50.py`/conditioning wrapper sources are copied into `PROVENANCE/` before
adding the optional factory/mixture-property interfaces; existing default raw behavior
is unchanged and regression-tested. No Git commit/push or publication authorized.
