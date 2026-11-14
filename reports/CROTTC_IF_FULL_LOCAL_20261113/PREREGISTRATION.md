# CROTTC-IF — local reproduction registration, 2026-11-13

Decision: `CROTTC-IF-FULL-LOCAL-20261113-v1`, recorded in root GOAL before implementation.
Training is explicitly authorized. No official test was unsealed by this decision.

## Claims and source

The objective is a complete, auditable local reproduction, followed by development
of the thesis method. **Current status is implementation and measured training
profile, not a reproduced paper result.** Exact paper benchmarking remains
unavailable under the current sealed split. A development run is labeled a local
variant even when its full architecture and epoch targets match the paper.

Paper: https://arxiv.org/html/2604.22133v1 .
Pinned author source: https://github.com/Secondtonumb/IF-MDD/tree/cca150ec4ff00e01d61a613092d1d178e661dfb2 .
Unmodified source hashes: `experiments/crottc_if_full_local_20261113/SOURCE_RECEIPT.json`.
The historical source and previously verified release bundle are not edited.

## Stage contract

| Stage | Required architecture/objective | Target epochs | Status at registration |
| --- | --- | --- | --- |
| AM CTC | original WavLM Large, 2-layer Conformer, 384 hidden, CTC | 300 | pending |
| AM CROTTC | same AM, official bucketized OTTC, two views, symmetric detached KL | 300 | implementing/profile first |
| IF CTC initialization | joint CTC/2-layer Transformer decoder + teacher | 200 | pending implementation |
| IF CROTTC transfer | replace pretrained CTC AM with trained CROTTC AM, fine-tune LM/teacher | 200 | pending prerequisites |
| Final decoding | audio only; beam 10, temperature 1.1; report AM weight sweep | — | official release smoke already verified separately |

The paper specifies 200 epochs for LM/teacher but does not separately specify
allocation between its two stages. 200 per stage above is a provisional bounded
target, **not a verified author schedule**. Resolve before claiming schedule fidelity.
AM learns SSL transformer layers and following modules; only the SSL CNN feature
extractor is frozen. Initialization is pinned `microsoft/wavlm-large` revision
`c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, never the released L2 fine-tuned model.
AM Adam learning rates: SSL 1e-5, following modules 3e-4. LM/teacher: 2e-4.
Seed 3047. No performance target is declared passed without measurements.

## Data and known fidelity gaps

Freeze and SHA-256 verify 1,800 TRAIN and 599 DEV_SEARCH WAV and annotation
records. Keep all 1,800 training utterances. The former 557 incomplete realized
sequences remain in training with explicit `err` targets; **this does not recover
their exact realized phones**. Raw interval order and silence are retained for
TRAIN. Starred phones become `err`; AX maps to AH; four known malformed spellings
(`er)`, `d_`, `z_`, `AH)`) become `err`, not guessed phones. Other unknowns fail.
Deletion placeholders are removed from perceived training sequences. The author
did not supply the preparation script or frozen manifests in the pinned source;
this normalization must not be called author-equivalent preprocessing.

DEV_SEARCH currently uses existing authorized derived labels. These lack raw
silence intervals and star markers; scores are developmental and annotation-policy
dependent. No DEV_CAL fitting, test loading, archive-wide extraction, or official
test access. All accessed ZIP members belong to explicitly authorized TRAIN IDs.

Paper population: 2,429 train / 268 validation / 900 test; paper test speakers
NJS, TLV, TNI, TXHC, YKWK, ZHAA. Local TRAIN includes NJS; local sealed speakers
are ZHAA, TNI, MBMPS, TLV. Consequently even the training population differs.
Generic authorization to run a full paper does not override the explicit test seal.
An exact paper-split result needs a separately registered, authorized data protocol.

## Code/paper differences that need tracking

| Item | Paper | Public pinned recipe | Local treatment |
| --- | --- | --- | --- |
| AM config | Conformer 2 layers | README default YAML names conformer but omits its module; separate `_confEnc` YAML contains it | use `_confEnc` modules |
| Conformer kernel | 3 | `_confEnc` default 31 | explicit override 3 |
| CR objective | Eq. 5: class sum / (2 frames), Eq. 6 eta=1 | `0.5*OT(two views) + 0.1*mean(B,T,C) symmetric KL` | profile/main default paper equation; source option separately labeled |
| Perturbation | time warp 80, time/frequency masks ratio (.1,.3] | waveform DropFreq/DropChunk, speed 95/100/105; defined warp unused | currently source augmentation, fidelity unresolved |
| IF acoustic encoder | pretrained CROTTC including Conformer | released IF YAML does not register its separate Conformer | explicit integration required; pending |
| IF fusion branches | two parallel branches | `modules.fuse_net_dec` aliases encoder fusion; a separate top-level object exists | verify/fix in separate effective config; pending |
| Teacher labels | C/S/D/I four classes | DataIO generates only `int(canonical != perceived)` | local staged labels preserve C/S/D/I; training integration pending |
| IF loss | paper Eq.18, omega1 .3 versus implementation section .5 | extra ConPCO losses enabled | register chosen interpretation before training IF |
| Checkpoint selection | validation F1 | AM YAML defaults PER; IF validation teacher-forced | local AM audio-only greedy F1; IF audio-only AR required |

None of these pending rows may silently become a PASS.

## Resources, checkpoint and stopping contract

Registered machine-readable limits: `experiments/crottc_if_full_local_20261113/COMPUTE_BUDGET.json`.
Local profile: one real update, up to 32 microbatches, at most 900 seconds plus
checkpoint finalization. Check RAM/disk/GPU, acquire shared GPU flock before use.
Full SSL weights, gradients and FP32 Adam states reside on CPU; small heads and
OT operator use CUDA. Autograd remains connected across transfers. Activation
checkpointing enabled. Four CPU threads. This is not a smaller model or LoRA.

Effective batch count 32 by accumulation with microbatch 1. This does not reproduce
the original batch statistics of BatchNorm modules; execution difference recorded.
No optimizer update is called successful unless loss/gradients are finite and an
SSL transformer weight actually changes. No batch is skipped on failure.

First sustained execution window <=4 hours, resumable without changing 300-epoch
targets. Save atomic `last.pt` at every epoch and clean window end with model,
optimizer, scheduler (constant LR), scaler (disabled FP32), Python/NumPy/Torch/CUDA
RNG, epoch and data cursor, source/data hashes. Reload and compare all weights.
Keep best weights separately; do not retain hundreds of full Adam checkpoints on
the 21 GiB free disk. Resource failures preserve logs and status. No historical
checkpoint is deleted to make room.

## Evidence still required for “full paper”

Main experiment alone does not reproduce every table. Track CTC, CRCTC, OTTC,
CROTTC, CTC-IF, CROTTC-IF, removal of each teacher fusion branch/error head,
CTC-LM, decoding-weight analysis, LLaMA/Qwen canonical and potential-pronunciation
conditions (oracle labeled separately), SO762/ERJ transfer, and Iqra'Eval2 results.
These are pending, with unavailable datasets/labels recorded as NOT_REPRODUCED.
Do not claim a leaderboard reproduction from a released checkpoint smoke.

Report hierarchical TA/TR/FA/FR, CD/ED, F1/P/R/FRR/FAR/EDR/PER/COR with denominators.
Keep strict typed S/O/I and thesis feedback precision/coverage separate. No direct
rank comparison between local development scores and the published 71.77% F1.
