# RQ1-TRN-02B Acoustic Conditioning Improvement Design

**Revision:** DESIGN-R1. **Status:** design-only; no implementation, training, optimizer update, GPU allocation, or official-test access is authorized. Any execution requires separate explicit authorization.

## Provenance and frozen context

This revision is on `docs/rq1-trn-02b-acoustic-conditioning-design`, based on design commit `68363673f04719eb976a5e690e0845c0164adc3c`, whose parent is frozen baseline commit `c4a658ffbd5a2afd2ebdce172e47af43ed9bb63e`. The baseline facts and historical results are preserved in `BASELINE_FREEZE_LEDGER.md`; no 02A artifacts or results are changed. Fold 1 contract is the frozen split (2,399 train, 300 validation speakers NJS/RRBI, 900 sealed test), and the split SHA-256, checkpoint SHA-256, training execution SHA, Protocol O/Z results, canonical-only baseline, acoustic swaps, frozen Wav2Vec2 backbone, and disabled diagnosis training remain exactly as recorded in that ledger.

02A is the immutable CTC + Binary historical baseline. The experiment asks whether dropout of the Binary head's direct canonical-phone feature encourages acoustic discrimination, not whether the model is diagnosis-ready. Existing validation evidence and its provenance caveats are retained: Protocol O AP .317021 in the matched/mismatched control (a separate selection replay reported .316062); train-only canonical prior AP .303063; constant prevalence .116211; same-speaker shift-1 AP .294722; Protocol Z aligned-only AP .313530 on 9,919/9,956 positions (99.63% coverage). Do not reconcile distinct historical executions by silently substituting values.

## Hypothesis and minimal intervention

The canonical phone embedding queries cross-attention and is also concatenated as a direct lexical feature to the Binary MLP. The hypothesis is that modestly disrupting only this latter path can reduce reliance on phone-specific priors and increase acoustic discrimination. This remains untested.

Mask the **full-vector** phone embedding at probability `p` only on valid non-padding token positions, using zero masking with no inverted-dropout rescaling. Apply after the original canonical embeddings have been consumed by cross-attention, only on the direct lexical embedding path entering the Binary MLP. Preserve embeddings used for cross-attention queries and leave aligned acoustic features unchanged. Enable only in training; disable for evaluation. At `p=0`, preserve exact original behavior. Do not add an acoustic gate, contrastive loss, head, or architecture redesign. Keep CTC + Binary BCE objectives and their 1:1 weights unchanged. Keep Wav2Vec2 frozen and in eval; diagnosis head remains disabled/untrained.

## Controlled experiment matrix and estimands

| Arm | Initialization and training | Binary embedding mask | Role |
|---|---|---:|---|
| **Reference R** | Original RQ1-TRN-02A `best.pt`; no optimizer updates or model modification | n/a (`p=0` historical behavior) | Immutable historical anchor; preserve original Protocol O/Z results |
| **Control C0** | Initialize weights from exactly the same `best.pt`; reset optimizer, scheduler, and AMP scaler per shared policy; fine-tune on frozen Fold 1 TRAIN | `p=0.00` | Measures additional fine-tuning without intervention |
| **Treatment C1** | Identical initialization and reset policy; fine-tune on same TRAIN | `p=0.25` | Binary-only prompt-dropout intervention |

Primary causal comparison is **C1 versus C0**, with all non-intervention variables and training exposure matched. R is not a substitute control. Report separately **C0 − R** (additional fine-tuning effect), **C1 − R** (combined fine-tuning + dropout effect; not a causal dropout estimate), and **C1 − C0** (incremental dropout effect within this controlled design). No p=.50 arm, extra seed, or additional fold in this initial design.

## Shared training contract and budget

C0 and C1 share identical initial checkpoint weights, seed `1337`, frozen split and train/validation assignments, oracle-canonical training input policy, AdamW settings, LR/weight decay, batch size 2, accumulation 2, loss weights/objectives, warm-up and scheduler policy, gradient clipping, AMP configuration, validation/checkpoint policy, metrics, and evaluator implementation. Preserve the production optimizer hyperparameters from the ledger (AdamW lr 1e-3, weight decay .01; cosine schedule/100 warm-up), subject to preflight confirmation of exact baseline implementation. Reset optimizer/scheduler/scaler identically; do not resume 02A optimizer state. Preserve frozen backbone and diagnosis exclusion.

Use separate reproducible RNG streams for intervention masks where feasible, so drawing C1 masks cannot perturb the shared dataloader order or unrelated stochastic components. Use identical deterministic sample order and seed policy in both arms. Record Python/NumPy/Torch/CUDA RNG state, per-loader and mask RNG state, deterministic-kernel settings, software/driver versions, and any nondeterministic kernels or unavoidable arm differences. Exact bitwise identity is not promised where GPU kernels are nondeterministic.

Frozen TRAIN has 2,399 utterances, batch size 2, approximately 1,200 microbatches/epoch, and accumulation 2: approximately 600 successful optimizer updates/epoch if no AMP steps are skipped. Thus 6 epochs ≈3,600 successful updates; 11 epochs ≈6,600, not 3,600.

| Budget limit | C0 | C1 | Combined |
|---|---:|---:|---:|
| Maximum epochs | 6 | 6 | 12 arm-epochs |
| Hard cap on successful optimizer updates | 3,600 | 3,600 | 7,200 |

Count optimizer-step **attempts** separately from successful parameter updates. Advance successful-update count only when the optimizer actually updates parameters; account for accumulation boundaries, short final batches, and AMP-skipped steps. Log attempted/successful/skipped steps and examples/tokens seen. Never exceed either per-arm cap or aggregate cap.

### Matched budget, validation, and early stopping

Both arms use the same six-epoch data exposure and planned schedule. Epoch 6 (or the last common completed epoch if a documented safety stop prevents completion) is the **predeclared common comparison checkpoint**, distinct from exploratory `best-validation-AP` selection. Save/evaluate both arms at each epoch; report the matched epoch-6 pair as primary, and report best-validation-AP checkpoints separately as exploratory secondary results. Do not select different checkpoints and present them as the primary causal pair.

Retain patience=5 as a monitored validation/checkpoint policy, but it does **not** terminate either arm before the common six-epoch endpoint. With a six-epoch maximum, patience=5 cannot ordinarily trigger before the cap; any stop for numerical/safety failure invalidates the matched primary comparison unless both arms have a comparable checkpoint and exposure, and must be disclosed. No unequal early stopping for validation performance: do not allow different training exposure to masquerade as a treatment effect. Validation AP may select exploratory `best.pt` under identical tie handling, but not alter the common checkpoint, budget, intervention, or threshold. Thresholds remain fixed/historical; no test tuning.

### Colab resource estimate and execution boundary

Plan two separate official Colab T4 runs: `RQ1-TRN-02B-C0` and `RQ1-TRN-02B-C1`. Planning estimate is approximately **3–5 T4 GPU-hours per arm / 6–10 T4 GPU-hours total**, including startup, validation, evaluation, and transfer; this is a rough estimate, not measured evidence. Confirm per-arm runtime and remaining Colab quota with an authorized preflight. If estimated time or available quota cannot support **both** arms, stop before allocation and report the requirement. Do not run C1 alone, locally train as fallback, or claim a controlled experiment without both arms. No GPU is allocated by this design revision.

Before any separately authorized run, stage only authorized Fold 1 train/validation WAV and TextGrid package from Drive, verify its SHA-256 on VM-local disk, and train from local VM disk (never stream audio from Drive). Keep official test sealed and unstaged. Each arm has isolated config, seed/RNG metadata, optimizer/scheduler/scaler state, checkpoint directory, logs, predictions, training history, and Drive destination under a new 02B location; never write into the 02A checkpoint directory. Store source/config/split/package/initial-checkpoint SHA-256 ledger and environment inventory. Atomically persist `last.pt` and `best.pt` after every epoch to the arm's Drive destination, verify remote presence and hash before continuing, and include model/optimizer/scheduler/scaler/RNG/metadata for resume continuity. Stop if sync/hash verification fails; release idle runtime promptly. Use official Colab CLI and check quota/auth before allocation; never use interactive CLI automation.

## Evaluation contract

**Primary question:** Does Binary-only prompt embedding dropout improve acoustic discrimination relative to an otherwise identical no-dropout fine-tuning control? Evaluate C0 and C1 at matched training budgets, on the same utterances and token positions. Preserve established deterministic audio permutation policies and frozen evaluators.

For both arms, report:
1. Binary AP under matched audio.
2. Binary AP under deterministic mismatched audio (existing within-speaker shift-1 policy).
3. Matched-minus-mismatched AP gap.
4. AP against the train-only canonical prior, computed on the same scoring population.
5. Paired C1-minus-C0 AP differences (and separately C0−R, C1−R; do not call C1−R causal dropout effect).
6. Class-conditional score/decision changes and TP/FP/FN/TN at the fixed historical threshold, plus per-speaker results.
7. Greedy CTC PER regression guardrail, with baseline replay provenance; guardrail remains no more than 1.0 absolute PER point degradation from a reproducibly measured R/B0 anchor.

A higher matched AP alone does not establish increased acoustic reliance. Interpret alongside mismatched AP, matched-minus-mismatched gap, prior, and class-conditional changes. Use paired procedures on the same utterances/token positions; for uncertainty, use the established utterance-cluster bootstrap with identical resampled utterances across arms/conditions and resampling within each of the two speakers. Report paired AP intervals and per-speaker values; do not use token-i.i.d. intervals or claim population generalization.

**Protocol Z secondary only:** use the frozen CMUdict frontend and no gold timestamps at inference. Report aligned-only Binary AP, gold alignment coverage, and C0 versus C1 on the exact same matched scoring population. Oracle-assisted imputation policies are secondary sensitivity analyses only. No Protocol Z G2P fine-tuning and no tuning intervention, checkpoint, or threshold using Z. Protocol Z inference is zero-oracle, but offline scoring still requires gold annotations.

Preserve the historical Protocol O matched and deterministic mismatched audio policies, canonical-only controls, evaluator provenance, and sealed test rule. Protocol O is explicitly oracle-canonical evaluation; never imply it represents deployable inference. Do not modify historical evaluator scripts or records.

## Limitations and interpretation

Both C0 and C1 are fine-tuned from a validation-selected production checkpoint. The initial comparison uses one seed. Validation contains only two speakers. Paired bootstrap intervals do not establish cross-speaker population generalization. Checkpoint selection on validation creates optimism bias. Protocol Z inference is zero-oracle, while its offline scoring still requires gold annotations. Training uses oracle-canonical inputs whereas Protocol Z is a secondary G2P input-shift evaluation; this experiment does not resolve that mismatch. Results are exploratory and do not establish diagnosis readiness.

## Stop conditions and future implementation tasks

Stop on split/hash/speaker mismatch, test data presence/access, label/timestamp leakage into model inputs, changed evaluator, baseline anchor failure, backbone unfreezing, diagnosis gradients/training, objective mismatch, NaN/Inf, CTC guardrail violation, checkpoint persistence failure, quota/auth insufficiency, or either update cap. Do not continue one arm alone after the other becomes infeasible and report the pair as successful.

Only after separate implementation authorization: (1) add/configure the mask with exact p=0 and evaluation parity tests; (2) add tests for padding, cross-attention preservation, mask RNG isolation, gradients, frozen backbone and diagnosis exclusion; (3) prepare dedicated isolated 02B configs/entrypoint enforcing both caps, atomic Drive persistence and full metadata; (4) prepare paired O/Z evaluation and cluster-bootstrap harness without editing historical scripts/artifacts; (5) perform local code/config/manifest audits. **Implementation, preflight, training, GPU allocation, official-test evaluation, diagnosis training, and multi-fold execution each remain unauthorized by this design document and require separate authorization.**
