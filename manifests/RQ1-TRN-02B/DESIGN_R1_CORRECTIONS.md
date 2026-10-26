# RQ1-TRN-02B DESIGN-R1 Corrections

**Scope:** additive design correction only. No implementation, training, optimizer update, GPU allocation, or official-test access occurred or is authorized.

## 1. Original design defect

The prior matrix compared B0, a frozen production `best.pt` with no optimizer updates, against B1, a fine-tuned model with `p=.25` prompt dropout. That contrast confounded dropout with additional training, so it could not identify the effect attributable to dropout. It also stated 11 epochs / 3,600 updates, inconsistent with the frozen Fold 1 training size and batch/accumulation schedule.

## 2. Revised controlled comparison

The revised design defines immutable historical **Reference R** (02A `best.pt`, unchanged, no update), no-dropout fine-tuned **Control C0** (`p=0.00`), and Binary-only prompt-dropout fine-tuned **Treatment C1** (`p=.25`). Both C0 and C1 initialize from the identical checkpoint weights and reset optimizer, scheduler, and scaler under the same policy. They share seed 1337, data/split/order, objectives, optimizer policy, training schedule, and evaluation. C1−C0 is the primary controlled dropout contrast. C0−R measures additional no-dropout fine-tuning; C1−R combines fine-tuning and dropout and is not a causal dropout estimate.

The intervention remains minimal: full-vector zero masking without inverted rescaling, only on valid positions and only on the direct Binary MLP lexical-embedding path after embeddings have been consumed as cross-attention queries. Aligned acoustic features and query embeddings remain unchanged; evaluation is unmasked; p=0 retains original behavior. CTC + Binary BCE and existing loss weights stay unchanged; backbone remains frozen and diagnosis training disabled.

## 3. Corrected optimizer-update arithmetic

For 2,399 TRAIN utterances, batch size 2 gives approximately 1,200 microbatches per epoch. Accumulation 2 yields approximately 600 successful updates per epoch when no AMP update is skipped. Therefore 6 epochs are approximately 3,600 successful updates; 11 epochs are approximately 6,600, not 3,600. Actual counts must log optimizer-step attempts separately from successful updates, including accumulation boundaries, short final batches, and AMP-skipped steps.

## 4. Revised compute budget

Each arm is limited to six epochs and 3,600 successful optimizer updates. Across C0 and C1: two runs, at most 12 arm-epochs and 7,200 successful updates combined. Initial authorization excludes p=.50, additional seeds, and other folds. Planning estimate: approximately 3–5 T4 GPU-hours per arm / 6–10 hours total, including startup/validation/evaluation/transfer; actual estimate and quota must be checked before any separately authorized allocation. If both arms cannot be supported, stop before allocation; C1 alone is not a successful controlled experiment. No GPU was allocated for this revision.

Both arms use the same six-epoch exposure and schedule. Epoch 6 is the predeclared common primary comparison checkpoint; best-validation-AP checkpoints are exploratory and reported separately. Patience=5 is monitored but cannot end an arm before the common endpoint; safety stops are disclosed and may invalidate the matched comparison. This avoids unequal exposure being presented as the intervention effect.

## 5. Experimental validity improvements

The corrected estimands separate historical anchoring, fine-tuning, and dropout effects. Evaluation now explicitly pairs C0/C1 on identical positions and reports matched and deterministic mismatched-audio AP, their gap, train-only canonical prior, paired AP deltas, class-conditional decisions, speaker results, and CTC PER guardrail. Protocol Z remains secondary, zero-oracle at inference, with gold annotations required for offline scoring. Validation/checkpoint optimism, one initial seed, two validation speakers, and limits of paired bootstrap generalization are explicit. Baseline ledger facts, protocol contracts, acoustic swaps, and sealed test are preserved.

## 6. Outstanding execution prerequisites

This is not implementation authorization. Separate authorization is required before implementation preflight, GPU allocation/training, official-test access, diagnosis training, or multi-fold work. Before any authorized run: verify code/config/evaluator and source provenance; prepare paired isolated C0/C1 configurations and output/Drive destinations; establish deterministic data/mask RNG streams and audit nondeterminism; verify authorized fold package and hashes on VM-local disk; confirm quota/time can support both arms; validate atomic checkpoint persistence/resume; and keep the official test partition sealed. The frozen baseline ledger is unchanged.
