# IF component implementation contract — 2026-11-13

This prepares the required full-size IF stages while AM CROTTC is running.
It does not change the running AM implementation or authorize use of sealed test.
No IF optimizer training or claim of a trained IF checkpoint is made here.

## Source and decisions

Use actual modules from pinned `CROTTC_IF.yaml` and `ConPCO_TransASR.py`, with
384 hidden units, two LM decoder layers, and two layers in each fusion branch.
Construct without SSL/model/data downloads or checkpointers. The AM supplies its
complete projection + Conformer memory, shared by acoustic scoring and LM/teacher.
Remove the unused LM 1024-to-384 prenet to avoid duplicate acoustic processing.

The YAML aliases `modules.fuse_net_dec` to `fuse_net` while separately declaring
an independent top-level `fuse_net_dec`. Use that independent object. Preserve the
source CNN/downsampler and other heads; ensure each fusion branch receives gradients.
Fix decoder-memory masking to use perceived decoder length, not canonical length.
Retain four true teacher classes (correct=0, substitution=1, deletion=2, insertion=3)
from frozen TRAIN, including its declared silence placeholders and unknown targets.
Canonical/error labels enter the TRAIN teacher only. Joint AR search receives
audio-derived AM memory and AM CTC head only, beam10, temperature1.1 by default.

Loss interpretation for the forthcoming separately registered IF stages:
paper Eq.18 with omega1=.5 (implementation §6.2), omega2=1, omega3=10; source
label-smoothed LM NLL and source batchmean binary/class losses, binary positive
weight2, guided-attention sigma .4. Exclude ConPCO losses absent from Eq.18.
The paper also states omega1=.3 in §4.3; .5 is an explicit interpretation, not
evidence that the ambiguity was resolved by the authors.

## Fidelity limits remain explicit

- SpeechBrain's actual LM decoder in the pinned runtime is regular MHA plus
  positional embedding, even though the encompassing TransformerASR is configured
  with RoPEMHA. Teacher fusion uses source RelPosMHAXL. Do not claim full RoPE
  fidelity from the YAML name alone.
- Source shared CNN path is 384→128→64 before binary/type projections. Paper
  description puts a shared 128 trunk before a 64 binary branch. Preserve source
  in this implementation and disclose the difference.
- Source factor4 acoustic teacher projector also includes a 384→2048→384 MLP.
- Two-stage epoch allocation and source/paper AM perturbation are unresolved as
  recorded in PREREGISTRATION and RESUME_REGISTRATION. No sustained IF stage
  starts until its parent checkpoints, freeze policy, schedule and budget are
  registered. Full IF trainer/transition integration is still required.
- Synthetic AM memory tests are component tests, not full WavLM audio training,
  acoustic validation or benchmark scores.

## Verification budget

CPU component tests only while AM owns GPU lock. Run with CUDA_VISIBLE_DEVICES
empty, 2 CPU threads, local/offline mode. At most 180 seconds; no optimizer steps,
downloads, dataset extraction, checkpoint copies, or parameter-size reductions.
Check resource availability before running. Test real 384-dimensional two-layer
modules, ragged masks, both teacher gradient paths into AM memory and LM, causal
decoding, four-class targets, and teacher exclusion during beam10 search.
Supervision validation reads only the frozen authorized TRAIN JSONL and verifies
its receipt hash. Do not construct or read test or DEV_CAL manifests.

## Observed verification

Seven CPU tests passed in 4.84 seconds. All 1,800 frozen TRAIN records retain
1,110 unknown `err` tokens and teacher classes 0/1/2/3. Both fusion branches
backpropagate into acoustic memory and the actual LM decoder; changing a
substitution to deletion changes type loss while keeping binary loss equal.
Beam10 search bypasses the teacher and rejects canonical input. Initial failed
gradient test and successful rerun logs are retained. Detailed code/log hashes:
`IF_COMPONENT_VERIFICATION.json`. These are component checks; full IF training
and stage transitions remain pending.

## Trainer progress — 2026-11-14

Full GPU/RAM trainer and stage initialization/transfer logic are now implemented
in `if_train.py`, with thirteen trainer/stage unit checks passed. Native IF
resource profiling and production training remain pending completed parents.
See `IF_TRAINER_PROTOCOL.md` and `IF_TRAINER_IMPLEMENTATION_VERIFICATION.json`.
