# LANGFENS — RQ1-TRN-03A / Cross-Fold Speaker-Independent Generalization Planning

**Status:** `PLANNING_COMPLETE; GATE_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-10-29T14:30:00+07:00

**Evidence source:** Commits `7a3b9ecf5e9efedc480b477f531ebe9318ae3415`, `manifests/RQ1-TRN-02B/FINAL_THREE_MODEL_ASSESSMENT.md`, `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Static Planning & Architecture Audit)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Objective
Design and specify a mathematically rigorous, leak-free 4-fold cross-validation phase for Folds 2, 3, and 4 (Phase RQ1-TRN-03A). Audit speaker overlap across splits, establish an authoritative test-isolation policy, plan independent Baseline R training across folds without checkpoint reuse, define cross-fold evaluation metrics and confidence interval aggregation, and structure downstream phases (model improvement and final testing) while preserving all completed Fold 1 R/C0/C1 findings and the `NOT_SUPPORTED` whole-phone prompt dropout verdict.

---

## 2. Inputs & Dependencies
- **Source Commit:** `7a3b9ecf5e9efedc480b477f531ebe9318ae3415` (docs: Step 16 final three-model scientific assessment and research gate closure).
- **Split Manifest:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- **Validated Baseline Architecture:** Two-head `LangfensWav2Vec2MDD` defined in `src/langfens_prep/model.py` (frozen Wav2Vec2 backbone `facebook/wav2vec2-base-960h` rev `22aad52d...`, trainable CTC head, trainable cross-attention + binary head, disabled diagnosis head).
- **Baseline R Training Contract:** `manifests/RQ1-TRN-02A/TWO_HEAD_TRAINING_CONTRACT.md` and `audit_scripts/train_fold1_ctc_binary.py`.
- **Pretrained Weights:** `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`).
- **Platform Credentials:** Kaggle CLI (`machine_shape: NvidiaTeslaT4`, weekly quota 30h) and Hugging Face API (`HF_TOKEN` in environment).

---

## 3. Work Performed
1. **Preserved Fold 1 R/C0/C1 Findings:** Reaffirmed that Reference R (`best.pt`, AP 0.317021, PER 23.68%, acoustic gap 0.022299) remains the primary production baseline. Preserved the scientific verdict that Treatment C1 ($p=0.25$ whole-phone binary prompt dropout) is `NOT_SUPPORTED` ($\Delta_{\text{dropout}} = -0.007918$, 95% bootstrap CI `[-0.018328, +0.000385]`).
2. **Audited Speaker Overlap & Staging Deadlock:**
   - Evaluated the speaker distribution across all 4 folds in `l2arctic_4fold_splits.json`.
   - Identified that Fold 1 test speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`) are required training speakers for Folds 2, 3, and 4.
   - Clarified that the "Official Test Sealing" rule in `AGENTS.md` applies to the active fold's evaluation split, not as a permanent global corpus blacklist.
   - Proved mathematically that within every fold $k$, $S_{\text{train}}^{(k)} \cap S_{\text{val}}^{(k)} = \emptyset$, $S_{\text{train}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$, and $S_{\text{val}}^{(k)} \cap S_{\text{test}}^{(k)} = \emptyset$.
3. **Formulated Authoritative Three-Tier Test-Isolation Policy:**
   - *Tier 1 (Cross-Fold Validation):* All 4 test partitions ($S_{\text{test}}^{(1..4)}$) remain 100% sealed. Fold packages staged to remote VMs will contain **strictly** Train and Val data (18 speakers per package), with test files physically omitted.
   - *Tier 2 (Model Improvement):* Model exploration evaluated strictly on pooled 4-fold validation sets ($4 \times 2 = 8$ speakers, 39,576 tokens). Test partitions remain sealed.
   - *Tier 3 (Final Testing):* One-shot evaluation on the 4 test partitions using frozen decision thresholds $\tau_k^*$ calibrated on validation.
4. **Planned Independent Baseline R Training Protocol:**
   - Mandated that Folds 2, 3, and 4 initialize from scratch using the original pretrained Wav2Vec2 checkpoint.
   - Strictly prohibited reusing, warm-starting, or fine-tuning from Fold 1 `best.pt`.
   - Specified deterministic seeding (`master_seed = 42`), standard AdamW optimizer ($lr = 10^{-3}$ for heads), cosine decay, CUDA FP16 AMP, and 6 epochs / 3,600 updates budget (max 10 epochs with early stopping patience 5).
5. **Defined Cross-Fold Evaluation Metrics & Aggregation Framework:**
   - Defined per-fold metrics: Matched Binary AP, Mismatched AP, Acoustic Reliance Gap, Greedy CTC PER, Precision, Recall, Specificity, F1 at operational ($\tau \in \{0.20, 0.50\}$) and calibrated thresholds $\tau_k^*$, and Protocol Z G2P zero-oracle AP.
   - Defined macro-averaging with Student's $t$ 95% CI ($df=3, t_{\text{crit}}=3.182$), micro-averaging across 39,576 pooled validation tokens, and 1,000-resample speaker-stratified cluster bootstrapping.
   - Structured per-speaker and L1-stratified reporting across the 8 validation speakers.
6. **Synthesized Comprehensive Research Plan:** Generated [CROSS_FOLD_RESEARCH_PLAN.md](CROSS_FOLD_RESEARCH_PLAN.md).
7. **Executed Static Verifications:** Verified that all existing unit and regression tests pass (236 passed, 4 skipped).

---

## 4. Results & Verification
- **Split Invariance:** Verified `l2arctic_4fold_splits.json` SHA-256 is `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- **Automated Split Suite:** `python3 audit_scripts/validate_splits.py` passed all 4 core checks with 0 errors.
- **Repository Regression Suite:** `PYTHONPATH=. .venv/bin/pytest tests/` passed 236 tests with 0 failures.
- **Test Isolation Maintained:** Zero test audio or TextGrid files accessed, extracted, staged, or evaluated.

---

## 5. Output Artifacts
- **Research Plan:** `manifests/RQ1-TRN-03A/CROSS_FOLD_RESEARCH_PLAN.md`
- **Process Checkpoint:** `manifests/RQ1-TRN-03A/PROCESS_CHECKPOINT.md`
- **Updated Global Registry:** `docs/process-checkpoints/INDEX.md`

---

## 6. Restart / Recovery Instructions
- **Active Branch:** `review/rq1-trn-03a-cross-fold-planning`
- **Current State:** Awaiting operator authorization to begin implementation.
- **Next Safe Step:** Upon receiving operator authorization:
  1. Implement `audit_scripts/build_fold_packages.py` to create `authorized_RQ1-TRN_fold{2,3,4}_train_val_only.zip`.
  2. Implement `configs/rq1_trn_03a_fold{2,3,4}.yaml` configuration files.
  3. Verify staging packages with automated partition audit tests.
  4. Submit Kaggle GPU training kernels.

---

## 7. Remaining Work & Safety Boundaries
- **Prohibited Actions:**
  - DO NOT run model training or backward passes on local CPU/GPU (`AGENTS.md`).
  - DO NOT access, unseal, or stage test split files for any fold.
  - DO NOT warm-start Folds 2, 3, or 4 from Fold 1 `best.pt`.
  - DO NOT alter the frozen model architecture or loss contracts.
- **Pending Approvals:** Operator authorization packet detailed in Section 10 of `CROSS_FOLD_RESEARCH_PLAN.md`.
