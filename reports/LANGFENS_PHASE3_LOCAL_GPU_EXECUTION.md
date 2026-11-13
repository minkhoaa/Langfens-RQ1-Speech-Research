# LANGFENS — PHASE 3: LOCAL-FIRST UNIFIED STUDENT + MULTI-TEACHER KNOWLEDGE DISTILLATION

## DIRECT EXECUTION ORDER FOR Pi/OMP AUTONOMOUS RESEARCH AGENTS

**Execute real experiments now. This is an implementation-and-training directive, not a request to produce only a plan.**

You are the LANGFENS autonomous speech ML research director. Use native Pi/OMP subagents and the operator's **local NVIDIA GTX 1650 CUDA GPU**. Prioritize local, cached, low-overhead training because the approved L2-ARCTIC subset is small. **Do not use Colab, remote GPU, cloud training, SSH, or outside data.** Do not create a new project.

**Fixed root:** `/home/khoa/Projects/New-Method-Research`.

**Scientific objective:** establish whether multi-teacher KD gives a unified target-phone / exact-gap insertion student a measurable advantage over a matched gold-only student, especially error recall/F1 at the same **joint FWR <= 0.20**. Never promise that the student will match teacher high recall at this risk limit; this is the hypothesis under test.

## 0 — Immediate local preflight (bounded, then proceed to training)

1. `cd /home/khoa/Projects/New-Method-Research && pwd -P && git status --short --branch && git rev-parse HEAD`.
2. Inspect Phase 1 frozen checkpoints (S, O, I-F1, I-Precision), Phase 2 CPL-01/02/03 outputs, source code, evaluator, manifest/feature contracts and `.venv`. Record actual hashes; the reported Phase 1 commit is `41f02ad9ad4ea2df4734ebc38bc2052503586fb6`, **not an instruction to reset Git**.
3. CUDA check via `nvidia-smi` and a real PyTorch CUDA tensor operation. Record GPU, available VRAM, dtype support and existing resource leases. If available, require `device=cuda:0` for neural forward/backward training and record GPU runtime. CPU is appropriate for I/O, metrics and simple statistical calculations. Do not silently fall back to CPU heavy training.
4. Respect other sessions' active resource leases. Operator permits up to **three simultaneous GPU-heavy jobs**, provided measured VRAM and safety permit it. Attempt concurrency up to 3; on CUDA OOM lower microbatch/sequence length, apply accumulation/checkpointing or reduce concurrency based on telemetry. Do not assert 3-way GPU compute overlap without evidence.
5. Verify canonical manifest fingerprints, speaker partition and exact event accounting. Reported: TRAIN 1,800 utterances / 12 speakers / 59,562 target phones / 61,362 gaps; DEV_CAL 300 utterances / 2 speakers / 10,043 targets / 10,343 gaps; DEV_SEARCH 599 utterances / 4 speakers / 19,755 targets / 20,354 gaps. Treat these as **reported** until checked on disk.
6. Apply only targeted Phase 2 corrections, **without launching another general audit campaign**: CPL-01 omission coverage union is **85/209** from its table (not 94/209); distinguish student baseline from F2-O-PRIO and F1-UNION; correct I-Precision actual precision = 21.57% at the reported threshold; rederive FWR counts and Pareto membership directly from ledgers. Record corrections and continue.
7. Never access TEST_SEALED, GPG encrypted archives or intentionally isolated holdouts. Never inject gold TextGrid timing, realized-phone labels or evaluation labels into inference features. Gold labels are permitted for TRAIN supervision and evaluation only.

## 1 — Local-first acceleration strategy (mandatory)

**Use the small subset to our advantage rather than repeatedly re-encoding audio.**

- Reuse existing verified frozen Wav2Vec2/CTC representations where source provenance, speaker split, event IDs and no-oracle inference contract are valid. If a required frame-level representation is missing, extract it **once locally using CUDA**, save in a versioned, content-addressed feature cache with extractor+checkpoint+manifest SHA-256. Do not assume 51D target_X and 6D gap_X arrays contain enough information for frame-level cross-attention.
- Prefer an initial lightweight **shared acoustic/reference-conditioned sequence student** over end-to-end acoustic encoder fine-tuning. Keep Wav2Vec2 frozen first. No heavyweight download or end-to-end retraining is required for the first valid gold-only vs KD-All experiment.
- Use efficient utterance/length bucketing, pinned data loading when useful, local SSD feature cache, mixed precision **only if supported and stable**, gradient accumulation, small microbatches and deterministic padding masks. On GTX 1650, validate FP16; do not assume BF16. Record real throughput/peak VRAM.
- Avoid recomputing teacher forwards inside every student training epoch: cache teacher soft probabilities/logits once, per canonical event ID, with explicit logits/probability and temperature semantics.
- Build the smallest working architecture first; benchmark memory, latency and actual model parameter count from running code. Claims such as '850k parameters' or '<2ms inference' require measurement, not estimates.
- Hyperparameter search must be bounded and justified; favor a few interpretable candidates, paired ablations and checkpoints over an uncontrolled sweep on DEV_SEARCH.
- **Run STU-01-GOLD-ONLY immediately** once the sequence features and label contracts pass. Parallel agents may prepare teacher OOF scores and KD variants concurrently. Do not wait for extensive writeups before training.

## 2 — Student input/output and architecture contract

Input at inference: **learner speech audio plus intended text/reference phoneme sequence**. Alignment must be derived only from model outputs and intended reference, never gold timing.

- Shared acoustic encoder/projection. Start with a lightweight temporal model and optional local cross-attention to reference phone embeddings; then test a compact Conformer if it fits measured resources. Use the **same chosen architecture** for all main KD ablations.
- Target head: `N x 3` normalized probabilities for `CORRECT / SUBSTITUTION / OMISSION`, with strict phone-index identity.
- Insertion head: `(N+1) x 2` normalized probabilities for `NO_INSERTION / INSERTION`, including gap 0 and gap N; score **exact gap and utterance**, not count-only insertion.
- Correct target/gap masks and sequence lengths. Support variable length including empty/adversarial cases without losing slots.
- No classifier that 'drops' difficult examples from metric denominators. Abstentions require explicit coverage and correct scoring.
- Do not assume a trainable cross-attention Conformer can be built faithfully from utterance-aggregated arrays alone; recover or generate legitimate frame-level context, otherwise train a legitimate lightweight common-feature student and document limitation.

## 3 — Leakage-safe teacher KD targets

The frozen Phase 1 teacher checkpoints were fitted on TRAIN. Their in-sample TRAIN predictions may be overfit and cannot by themselves justify a strong KD claim.

- Preferred: generate **speaker-grouped out-of-fold teacher scores on TRAIN**. Train economical teacher replicas per fold with identical feature/inference contracts and no access to the held-out speaker fold; use frozen/cached acoustic features and CUDA-trained heads. Keep fold membership and checkpoints. Original full-TRAIN frozen teachers remain eligible for inference on DEV_CAL and DEV_SEARCH, but cannot substitute for OOF teacher predictions on TRAIN.
- If budget limits OOF, document the in-sample teacher-score variant explicitly as **exploratory only**, compare to gold-only, and do not claim leakage-free KD effectiveness. Continue valid Gold-Only training rather than blocking the phase.
- Every score must be joined by `dataset_id:speaker_id:utterance_id:target|gap:index` and checked for duplicate/missing/cross-speaker IDs.
- S and O are **binary specialists with different labels**, not a pre-normalized joint 3-class teacher distribution. Preserve label semantics. Build KD losses against appropriate student class marginals / binary logits; do not naively concatenate S/O scores into a simplex.
- Compare I-F1 and I-Precision as different distillation sources; include a version that uses both with explicit reconciliation or reliability gating. Do not relabel I-Precision as higher-precision than historical B102 across all cohorts.
- Prefer proper teacher probability calibration **on DEV_CAL only** and compare raw logits vs calibrated soft scores. Report Brier/ECE/AUPRC before/after. Do not assume uncertainty correlations justify `1-u` weighting without ablation.
- Gold supervision remains the primary learning signal. Teacher noise cannot override trusted gold labels indiscriminately.

## 4 — Training arms (run, do not merely specify)

Use the **same architecture, splits, input features, budget, seed and evaluation code** for matched main comparisons. Main seed = `101`; after identifying a promising result, repeat the key matched arms with at least two additional seeds, if feasible. Record all seeds rather than cherry-picking.

1. `STU-00-LIGHT-GOLD`: small shared-encoder dual-head baseline without KD (useful for speed/architecture comparison).
2. `STU-01-GOLD-ONLY`: chosen unified architecture trained only against gold target/gap labels. **Run this first.**
3. `STU-02-KD-S-ONLY`: same architecture + Teacher S KD.
4. `STU-03-KD-O-ONLY`: same architecture + Teacher O KD.
5. `STU-04-KD-I-ONLY`: same architecture + one insertion teacher. Test I-F1 and I-Precision separately if inexpensive.
6. `STU-05-KD-ALL`: same architecture + S/O/I KD (primary comparator to STU-01).
7. `STU-06-CONF-GATED`: same architecture + calibrated/reliability-gated KD; do not hard-code unvalidated `u < 0.30` as a scientific truth.
8. `STU-07-KD-I-ALTERNATE` (optional if helpful): compare I-F1, I-Precision or dual teacher KD at matched gold loss/training budget.
9. **Frozen non-student controls:** `UPR03_SYNTHESIS_CHAMPION`, Phase 2 best safe FWR policies, and standalone teachers. These are evaluation controls, not student training arms.

**Train in two waves:** Wave A = STU-00, STU-01, STU-05 with immediate real local GPU updates. Wave B = all single-teacher and uncertainty ablations, plus any justified compact Conformer variant. Do not skip the matched gold-only control. Use early stopping from appropriate development metrics; choose checkpoints without looking at a sealed evaluation set.

Loss starting point (weights are tunable hypotheses, not validated constants):

`L = L_gold_target + alpha*L_gold_gap + beta_S*L_KD_S + beta_O*L_KD_O + beta_I*L_KD_I`.

- Target: correctly masked three-class cross-entropy or focal loss with TRAIN-derived class weighting; guard against class-collapse.
- Gaps: correctly masked binary BCE/focal loss, with TRAIN-derived weighting and event support; evaluate probability ranking as well as thresholded F1.
- KD: temperature-aware, probability-contract-correct teacher/student objectives. For a binary S teacher, match `p_student(SUB)` versus `p_teacher(SUB)`; similarly O and I; ensure KL direction and `T^2` factor are implemented consistently.
- Optional consistency loss `p_sub*p_omis` must be an **ablation**, not assumed physically valid or mandatory. Multiclass softmax already enforces mutually exclusive hard labels.
- Default sampling/weights should derive from TRAIN; tune only within specified calibration/development process. No leakage from DEV_SEARCH into training or early stopping.

## 5 — Local GPU execution orchestration

Launch three parallel agent workstreams:

- **GPU Agent A — Core student:** build/validate architecture, local feature cache and Gold-Only baseline; then main KD-All matched run.
- **GPU Agent B — KD targets:** create speaker-grouped OOF teacher soft scores on TRAIN, validated DEV_CAL scores, S/O/I calibration; run teacher forwards on CUDA when necessary.
- **GPU Agent C — Evaluation and alternatives:** implement/verify target/gap event evaluator, FWR accounting and pairwise policy comparison; run compact-student or single-teacher variants on GPU after prerequisites.

Attempt up to 3 concurrent GPU-heavy operations if leases and observed memory permit. Coordinate via local job queue and shared GPU lock/leases; do not bypass other active research sessions. Prefer tasks that keep the GPU productive without redundant re-extraction. Preserve all crashes/oom logs and retry intelligently. Record device, CUDA steps, duration, VRAM and whether overlap was real.

## 6 — Fair, class-specific evaluation

Use TRAIN for learning, DEV_CAL for score calibration and policy/threshold selection. DEV_SEARCH is **repeatedly development-exposed**; use for matched exploratory results and do not call it fresh blind verification. Never access encrypted TEST_SEALED/isolated datasets.

Evaluate full target and gap populations with exact event identities and **consistent evaluator versions**. Report, at a minimum:

- Per-class TP, FP, FN, TN, precision, recall, F1, AUPRC for S, O and exact-gap I; correct P/R/F1; macro-3 and macro-4; false warnings per utterance; counts and calibration (ECE/Brier) with stated semantics.
- Student-vs-Gold-Only, Student-vs-UPR03, Student-vs-best Phase 2 safe fusion at **the same FWR budget**, including `<=0.05, <=0.10, <=0.15, <=0.20, <=0.30, <=0.50, <=1.00`.
- On 599 utterances, `FWR <= 0.20` permits **at most 119 false warnings** under the documented joint warning count; distinguish this from the sum of per-class FP counts. Guard against warnings on incorrect error types and double counting.
- Provide correct/incorrect confusion and selectivity, speaker/family breakdowns, thresholds selected on DEV_CAL, uncertainty and honest limitations of 4-speaker DEV_SEARCH. Separate oracle ceilings from implementable results.
- Actual inference wall-time, model parameter count, peak VRAM, checkpoint size and end-to-end vs cached-feature latency (do not confuse them).
- Explicit class-collapse, label/no-oracle controls and calibration sanity tests.
- Student superiority requires improvements on matched metrics **without falsely using accuracy or CORRECT F1 as the primary success signal**. An always-CORRECT baseline has high correct-F1 on this population and detects no errors.

The central scientific questions:

1. Does `STU-05-KD-ALL` beat `STU-01-GOLD-ONLY` under matched conditions?
2. Can student improve one or more error recalls/F1 at `FWR <= 0.20` against UPR03 and strongest Phase 2 safe policy, while reporting regressions in others?
3. Which teacher contributes actual improvement, and does confidence gating help?
4. Are improvements stable across seeds and speaker groups?

A negative KD result is legitimate; do not declare breakthrough without evidence. Do not invent test-set generalization.

## 7 — Deliverables inside the existing repository

Create/reuse:

```text
research/student/
research/student/kd/
experiments/student_training/
experiments/student_evaluation/
models/student/
coordination/PHASE3_JOB_QUEUE.jsonl
coordination/PHASE3_GPU_LEDGER.jsonl
reports/PHASE3_PROGRESS_LATEST.md
reports/PHASE3_EXPERIMENT_RESULTS.csv
reports/PHASE3_GOLD_VS_KD_PAIRED.md
reports/PHASE3_FWR_RECALL_PARETO.csv
reports/PHASE3_TEACHER_OOF_PROVENANCE.md
reports/PHASE3_FINAL_SCIENTIFIC_REPORT.md
reports/PHASE3_REPRODUCTION.md
```

Every experiment record includes code/data/model/feature hashes, exact command, seed, epochs/updates, GPU utilization/peak VRAM, training time, performance, score threshold, split exposure, raw predictions, checkpoint hash and relevant failure cases.

Commit **only newly created or deliberately edited Phase 3 files** after tests; do not overwrite historic baselines or unrelated uncommitted changes. Publish a Vietnamese operator report with actual numbers, URLs/paths where available, explicit VERIFIED vs REPORTED status, limitations and executable reproduction commands.

## 8 — Acceptance gates and continuation

Minimum technical completion:

- A legitimately trained local CUDA unified dual-head student, checkpoint and usable end-to-end inference adapter.
- Two **matched** runnable student arms: Gold-Only and KD-All; all remaining requested ablations run or explicitly accounted for with evidence.
- Trustworthy phone/gap population accounting and genuine per-class evaluation under no-oracle inference.
- DEV_CAL threshold selection, DEV_SEARCH exploratory matched comparisons and FWR–recall Pareto outputs.
- Evidence about whether teacher KD adds value, including failures and uncertainty; not just 'training completed'.

If OOF scoring or a Conformer component proves impractical on the local GTX 1650, preserve the valid lightweight student experiments and make a scientifically justified bounded alternative; do not silently replace invalid supervision, nor block all progress pending unrelated audits. Avoid unsupported claims about 80% performance or a theoretically necessary Conformer.

**START:** inspect the working tree and authorized manifests; immediately prepare the cached local features and launch the first real CUDA `STU-01-GOLD-ONLY` training, while parallel agents prepare OOF teacher scores and evaluator. Then train `STU-05-KD-ALL` under an identical contract, run all Phase 3 ablations, perform local calibration/evaluation, and write the final evidence-backed report.
