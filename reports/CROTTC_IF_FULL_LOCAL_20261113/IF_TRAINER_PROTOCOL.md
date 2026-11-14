# Full IF trainer contract — 2026-11-14

Implementation: `research/crottc_if_official/if_train.py`. This prepares complete
native IF stages while AM CROTTC trains. **Native IF resource profiling/training
has not run**: the required completed parents do not yet exist. Unit tests and
compilation do not substitute for that evidence.

## Stages and fidelity choices

| Stage | Parents | Trainable modules | Epoch target |
| --- | --- | --- | --- |
| IF CTC initialization | completed AM CTC300, validation-selected weights | native SSL transformer + complete Conformer AM heads + actual LM/teacher; native CNN stays frozen | 200 |
| IF CROTTC transfer | completed AM CROTTC300 and completed IF CTC200 | transfer the selected IF LM/teacher; replace **entire** SSL/projection/Conformer/heads from CROTTC AM and freeze/eval that AM | 200 |

200 **per** IF stage is the existing provisional local allocation, not an author-
verified allocation. Omega1=.5 follows paper implementation §6.2; its conflicting
value .3 remains recorded. Omega2=1 and omega3=10. Use actual two-layer/384-hidden
source LM and independent fusion branches; preserve full C/S/D/I supervision.
No ConPCO objective is added to Eq18. Source regular-MHA/RelPosMHAXL and CNN layout
differences stay disclosed in IF_COMPONENT_REGISTRATION.

Joint IF CTC uses clean audio through the full AM, native SSL SpecAugment during
training, a CTC acoustic term and source teacher/LM losses. IF CROTTC freezes the
entire AM in eval. It still measures the CROTTC paper objective with two registered
source waveform views as a diagnostic constant in Eq18, rather than reporting a
fabricated zero acoustic loss. Those diagnostics cannot update frozen AM weights.
The LM/teacher gets a fresh Adam at stage transfer; each stage's last checkpoint
then resumes its own Adam/scheduler/scaler/RNG/partial gradients exactly.

SSL learning rate during joint IF: 1e-5. Head/LM/teacher: 2e-4. Constant LR is an
explicit paper interpretation; source NewBob and extra freezing hooks are not used.
Seed3047, FP32, microbatch1/accumulation32, CUDA full-model compute, CPU Adam/master/
gradient accumulation and RAM activation offload follow the tested AM backend.
All trained native parameter sizes are preserved. Effective batch accumulation
still differs from true batch32 BatchNorm statistics.

## Admission and resource bounds

Before any IF optimizer update, verify actual parent checkpoint SHA, provenance,
full epoch/update/scheduler counts, full epoch validation history and selected
weights/F1. A status string, profile or live parent is insufficient. Child strict
loads enforce actual architecture state. Shared GPU flock prevents overlapping
training. All TRAIN/DEV_SEARCH audio/source/native hashes are reverified.

`--profile` is a **scratch** representative 32-row TRAIN resource test (median/p95/
longest), at most900 seconds and one update. It never writes a campaign checkpoint,
cannot be resumed/promoted, and cannot count as a completed stage. No smaller model
or shortened target schedule is used for production.

Production windows are at most14,400 seconds, checked before each microbatch and
validation sentence, plus in-flight computation/metric/checkpoint finalization.
Stop on nonfinite loss/gradient, mismatch or resource exhaustion without skipping
a failed batch. Require12GiB available RAM initially and stop below2GiB between
microbatches. Disk preflight: IF CTC12GiB fresh/6.5GiB resume; IF CROTTC6GiB fresh/
3GiB resume. Every atomic save independently checks actual payload size +1GiB.
Actual native IF VRAM/RAM profile remains required before a sustained campaign.

Save every5 updates, before/after validation, and at clean window end. Preserve
pending unscaled FP32 gradients in RAM by module/parameter name. Save full SSL,
AM heads and LM/teacher weights together with optimizer/scheduler/scaler/RNG and
cursors. Reload/compare atomic weights and partial gradients. Save best weights
separately; selected-parent provenance is immutable in REGISTRATION.json.

## Validation contract

Decode all599 frozen DEV_SEARCH WAV using only audio-derived acoustic memory,
beam10, temperature1.1, fixed preregistered AM weight .99. Do not construct teacher
supervision or use canonical/error/timestamp inputs during validation. Only after
the entire population is decoded may the pinned metric see reference annotations.
Interrupted validation selects no checkpoint. Keep unique per-attempt prediction/
metric reports and progress logs. An undefined upstream metric stops selection;
it is not replaced by zero or another evaluator.

The offline scoring helper preserves upstream metric limitations and explicit
denominators. Strict typed S/O/I, exact realized-phone diagnosis and feedback are
separate remaining evaluations. Local DEV_SEARCH F1 is not a paper-split score.
Official test and DEV_CAL remain sealed/excluded under existing boundaries.

## Commands (parents must first complete)

```sh
# Read-only eligibility checks; no GPU allocation or optimizer update.
CUDA_VISIBLE_DEVICES= .venv-crottc-official/bin/python -m research.crottc_if_official.if_train --stage if_ctc --check-parents
CUDA_VISIBLE_DEVICES= .venv-crottc-official/bin/python -m research.crottc_if_official.if_train --stage if_crottc --check-parents

# Run with offline/thread env from GPU_RAM_EXECUTION.md and a free GPU lock.
.venv-crottc-official/bin/python -m research.crottc_if_official.if_train --stage if_ctc --profile --wall-seconds 900
.venv-crottc-official/bin/python -m research.crottc_if_official.if_train --stage if_ctc --wall-seconds 14400
.venv-crottc-official/bin/python -m research.crottc_if_official.if_train --stage if_ctc --resume --wall-seconds 14400
```

The analogous IF CROTTC stage admits both completed parents. `--run-root` and
`--parent-root` permit explicitly chosen local checkpoint storage; no remote runtime,
Drive streaming, automatic data extraction, or test opening is implemented.

CPU-only trainer unit checks use2 threads, <=180 seconds and no optimizer updates.
They test source-sized IF objective gradients using synthetic AM memory, loader/
freeze invariants using tensor fixtures, and post-decode gold/coverage boundaries.
These are scoped unit checks, not evidence of completed native IF training.
