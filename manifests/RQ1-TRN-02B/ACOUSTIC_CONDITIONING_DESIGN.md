# RQ1-TRN-02B Acoustic Conditioning Improvement Design

**Status:** proposed bounded design; GPU execution requires separate explicit authorization. **No training, optimizer update, or official-test access is authorized here.**

## Purpose and frozen context

RQ1-TRN-02A is the immutable Fold 1 CTC + Binary baseline documented in `BASELINE_FREEZE_LEDGER.md`. This design tests whether modestly reducing the binary head's direct dependence on canonical-phone identity improves *acoustic* discrimination, rather than merely shifting prompt priors. The backbone remains frozen. The diagnosis head remains disabled/untrained; no gold timestamps may enter model inference.

### Evidence vs. hypotheses

**Observed in existing validation experiments:** (i) Baseline binary AP is .316062 on Protocol O selection replay and .317021 on an independent matched/mismatched control; numbers are not interchangeable across executions. (ii) Train-only canonical phone prior AP is .303063; constant prevalence is .116211. (iii) Same-speaker shift-1 audio swap lowers AP from .317021 to .294722, but retains substantial performance; overall decisions are 96.76% identical at .50 while error-class agreement is 86.60%, with 201 vs 172 true positives and 208 vs 250 false positives. (iv) Protocol Z aligned-only G2P evaluation AP is .313530 on 9,919/9,956 aligned tokens (99.63% coverage). (v) Training used oracle-canonical phone labels; Protocol Z inference conditions on CMUdict G2P phones. That is a real input-distribution mismatch, not proof that either protocol is deployable.

**Architectural hypotheses (not yet established):** The canonical phone embedding both queries cross-attention and is concatenated unregularized into the binary MLP. It can encode phone-specific error priors and possibly dominate token decisions. Prompt dropout may force the binary MLP to exploit aligned acoustic features more; alternatively it may harm calibration/CTC sharing, or merely alter prevalence-sensitive rankings. Audio swaps establish sensitivity but not causal or generalizable superiority; two validation speakers severely limit population inference.

## Alternatives considered and selection

1. **Selected: binary-path prompt embedding dropout.** During training only, independently zero a complete phone embedding vector with probability `p=0.25` *after* cross-attention has consumed the undropped phone embedding, and before concatenation into the Binary MLP. No rescaling (mask-to-zero) so the counterfactual is explicit. Apply only at non-padding token positions. Eval is unchanged. The acoustic aligned vector, CTC head, attention queries and targets are untouched. This is a small isolated intervention and directly tests the unregularized prompt shortcut.
2. **Acoustic-conditioned residual/gate.** A learnable gate over acoustic and phone features is plausible, but requires more new parameters and initialization/regularization choices; pre-existing `gated_residual` mode affects diagnosis path, not Binary MDD, so reusing it would not test the target.
3. **Matched-vs-mismatched contrastive supervision.** Could explicitly teach correspondence but requires robust definition of positive/negative phone/audio pairs, additional loss weighting and sampling; high confounding risk and not minimal.

## Controlled experiment matrix

| Cell | Initialization | Binary prompt mask | Purpose |
|---|---|---:|---|
| B0 baseline replay | Frozen production `best.pt`; no update | p=0 | Fixed-weight anchor; reproduce outputs and controls; no added training |
| B1 primary | Same production `best.pt` weights; optimizer/scheduler newly initialized | p=.25 in training only | Causal intervention comparison; one seed initially |
| B2 sensitivity (not in initial budget) | Same initialization | p=.50 | Only request later if B1 completes and a separately authorized follow-up is warranted |

B1 adds no parameters. Parameter groups remain only `ctc_head`, `cross_attention_module`, `binary_head`; Wav2Vec2 stays `requires_grad=False` and in eval. Diagnosis head excluded; objective/loss remains CTC + Binary BCE, weights 1:1. Keep optimizer and all other baseline hyperparameters fixed unless train-only policy below applies. Baseline `best.pt` is the common initialization; this is a fine-tuning-from-baseline ablation, not a from-scratch rerun. Record optimizer reset explicitly. Seed 1337 for B1; deterministic loader/shuffle and RNG state as supported by existing code. Any future multi-seed replication requires separate authorization and paired seeds for baseline and intervention.

### Input-policy limitation

For a controlled, attributable first experiment, preserve baseline Fold 1 training supervision/input pipeline (oracle canonical phones) and compare Protocol O matched/mismatched evaluation with identical prompts, labels and mappings. Also report Protocol Z G2P as an explicitly secondary input-shift stress test, using frozen CMUdict/G2P and the established aligned-only scoring / 99.63% coverage. Do not tune on Protocol Z. This experiment does **not** solve oracle-canonical training versus G2P inference mismatch; resolving that requires a distinct, pre-registered G2P-canonical training intervention and must not be bundled into B1.

## Experimental contract

- **Data/split:** frozen manifest SHA in ledger; 2,399 train and 300 validation utterances (NJS/RRBI); test 900 utterances remain sealed and unstaged. Existing authorized fold package only, locally copied to Colab VM and SHA-256 checked before any training. Never read TextGrid timestamps as model inputs.
- **Train-only decisions:** any choice of p, regularization, threshold, architecture, epoch limit, or checkpoint must not use validation/test. p=.25 and maximum epochs 11 are fixed now as design choices, not validation-selected. No new threshold search. The historic .50 threshold is reported as fixed operating point; .20 is reported only as historical baseline reference, not re-optimized. AP is threshold-free primary.
- **Validation/checkpoint policy:** historical baseline selected by validation AP; B1 will use the same predeclared patience=5 and maximum 11 epochs (approximate same horizon as production run) with validation AP as selection metric, exact tie handling copied from baseline implementation; candidly report selection bias and do not claim unbiased generalization. Evaluate matched and mismatched audio at best-checkpoint only. Report every epoch and both `last.pt` and `best.pt`.
- **Primary endpoint:** Protocol O matched-audio binary AP versus train-only canonical prior AP=.303063 on the same token population, and paired B1-minus-B0 AP. Report absolute AP, absolute delta, validation prevalence and matched population. Require both improved matched AP over canonical-only prior and increased acoustic dependence (matched-minus-mismatched AP gap vs B0) to call the outcome supportive; this is a directional design criterion, not a significance claim.
- **Prompt-prior discriminator:** Evaluate B0 and B1 on exactly the same validation token positions under matched audio, deterministic within-speaker deranged audio, and canonical-only prior. Report each AP and paired `matched AP - mismatched AP`; additionally report paired per-token score shift, class-conditional prediction agreement, TP/FP/FN/TN and per-class decisions at fixed τ=.50. A change only in matched AP without a larger matched-vs-mismatched contrast is insufficient to attribute gains to acoustic discrimination; gains replicated under audio mismatch or matched by canonical-only prior indicate prompt-prior/calibration effects. Also report per-speaker AP with caution.
- **G2P/Protocol Z secondary stress test:** frozen CMUdict SHA `81917843...`, G2P tokens, no TextGrid at inference; aligned-only AP and coverage (and Policy B1 imputation only as labeled audit sensitivity). No tuning/checkpoint selection from Z. Keep O and Z results separate; never conflate denominators.
- **CTC guardrail:** greedy CTC PER on fixed Protocol O validation; compare B1 to the production report value (23.64%; disclose replay source if a different estimate such as 23.68% is reproduced). Guardrail: no >1.0 absolute PER point degradation from reproducibly measured B0 replay. No CTC decoding changes.
- **Statistics:** paired utterance-cluster bootstrap over the 300 validation utterances (same resampled utterance indices for B0/B1 and matched/mismatched arms; preserve speaker strata, resample within each of the two speakers). Report paired AP deltas and percentile 95% intervals plus per-speaker values. With only two speakers and one seed, inference is exploratory, not cross-speaker significance. Do not use token-i.i.d. confidence intervals or claim generalization.

## Budget, execution, persistence

Request **one Fold 1 B1 run, at most 11 epochs / 3,600 successful updates**, T4 GPU only via official Colab CLI after explicit authorization. This is capped at the production baseline's observed 3,600 updates (the epoch count/update schedule must be checked at preflight); estimated duration up to one comparable baseline run, roughly 3–5 T4 GPU-hours including startup, validation, control inference and transfer. Estimate is planning-only; use Colab usage before allocation, stop on quota/auth blockers, never fallback to local CPU/GPU training. No multi-fold or official-test evaluation.

Use a uniquely named Colab runtime and RQ1-TRN-02B Drive folder. Stage only authorized train/validation WAV + TextGrid package, verify SHA on VM local disk, train from VM disk (not Drive mount), and sync `last.pt` and `best.pt` atomically after every epoch; verify remote file presence and SHA before continuing. Checkpoints must include model, optimizer, scheduler, scaler, Python/NumPy/Torch/CUDA RNG and metadata (git SHA, config, split/package/checkpoint hashes, epoch/update, seed, mask rate, trainable parameter names, metric protocol). Persist logs, prediction records, environment/version inventory and per-epoch metrics. Stop/release idle runtime promptly. If synchronization/verification fails, stop training until resolved.

## Risks and stop conditions

- Prompt masking can create train/eval mismatch; mask only Binary MLP's direct embedding path, not attention queries; test p=0 equivalence and eval identity before GPU.
- Encoder frozen but shared heads change CTC; abort if backbone unfreezes, diagnosis receives gradients, CTC contract violations occur, NaN/Inf occurs, or PER guardrail exceeded at a checkpoint.
- Stop on manifest/hash/speaker mismatch, test data presence/access, labels/timestamps leaking to forward, non-deterministic or altered evaluator, baseline anchor failure, checkpoint persistence failure, quota/auth absence, or budget exceeding 3,600 updates/11 epochs.
- Baseline is chosen on validation AP, and B1 also selects on same two-speaker validation; paired bootstrap does not remove selection bias. Results are exploratory; test remains sealed pending a separate protocol/authorization.

## Implementation tasks after approval

1. Add config flag and Binary-only embedding mask in `src/langfens_prep/model.py`; preserve baseline default (`p=0`). Assert eval and p=0 parity; padding unaffected; cross-attention query unaffected.
2. Add unit tests for stochastic mask behavior/seed reproducibility, gradients only in allowed modules, frozen backbone, diagnosis exclusion, and exact baseline forward equivalence at p=0.
3. Add a dedicated 02B config and training entrypoint/versioned wrapper; enforce strict CUDA/Colab boundary, frozen manifest/package hash, 11-epoch/3,600-update cap, early stop, atomic per-epoch Drive sync and metadata.
4. Add an evaluation harness reusing frozen O/Z evaluators without editing historical artifacts; matched/shifted audio paired outputs, canonical prior, fixed threshold class diagnostics, CTC PER, utterance-cluster bootstrap.
5. Run local CPU unit/regression tests and manifests-only audits; review configuration, hashes, test isolation, and dry-run output. GPU execution remains a separate authorization gate.
