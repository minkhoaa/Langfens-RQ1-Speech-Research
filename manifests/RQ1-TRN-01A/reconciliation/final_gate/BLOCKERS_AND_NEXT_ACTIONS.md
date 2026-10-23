# LANGFENS — RQ1-TRN-01A-R2: Blockers, Limitations & Next Actions Report

**Gate Reference:** RQ1-TRN-01A-R2 — Final Training Correctness & Evidence Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01A/reconciliation/final_gate/BLOCKERS_AND_NEXT_ACTIONS.md`  
**Inspected Branch / PR:** `feat/rq1-trn-01a-training-smoke-test` (PR #1)  
**Lead Auditor / Speech ML Engineer:** Senior Speech ML Research Engineer & PyTorch Training Systems Reviewer  
**Overall Gate Evaluation Status:** `RQ1-TRN-01A-R2: PASS`  

---

## 1. Acceptance Gate Status & Executive Decision

| Gate Requirement | Criteria Description | Verification Status | Evidence Reference |
| :--- | :--- | :---: | :--- |
| **A. Acoustic Padding Correctness** | Validated variable-duration audio batching without GroupNorm contamination; output invariant to batching within $10^{-5}$. | **PASS** | `ACOUSTIC_PADDING_VALIDATION.md` |
| **B. Remove Silent Label Truncation** | Explicit tensor-shape validation; zero legitimate targets discarded; all-masked losses finite and differentiable. | **PASS** | `LOSS_AND_CTC_CONTRACT_AUDIT.md` |
| **C. Enforce Runtime CTC Contracts** | Runtime verification of shapes, repeats, blank-invariance, and feasibility; `zero_infinity=False` default; policy-based handling. | **PASS** | `LOSS_AND_CTC_CONTRACT_AUDIT.md` |
| **D. GPU Benchmark Reconciliation** | Discrepancies between script, JSON, and report reconciled; unverified historical claims downgraded to `NOT_VERIFIED`; bounded synthetic scaling verified. | **PASS** | `GPU_BENCHMARK_EVIDENCE_RECONCILIATION.md` |
| **E. Training-Step Accounting** | Corrected calculation using actual per-fold train utterances ($2,400$), micro-batch ($4$), and accumulation ($4$); $60,000$ steps total. | **PASS** | `TRAINING_STEP_ACCOUNTING.md` |
| **F. Regression Test Integrity** | 100% test pass rate in both CPU (`.venv`, 58 passed) and CUDA (`.venv-cuda`, 60 passed) environments. | **PASS** | `FINAL_REGRESSION_TEST_REPORT.md` |
| **G. Pretraining & Licensing Checks** | Pinned checkpoint revision (`22aad52d`), Apache 2.0 / CC BY-NC 4.0 distinctions preserved; gold boundaries unused at inference. | **PASS** | `FINAL_REGRESSION_TEST_REPORT.md` |

### Final Formal Evaluation:
$$\mathbf{RQ1-TRN-01A-R2: PASS}$$

All mathematical, architectural, and data contract blockers identified prior to RQ1-TRN-01B have been completely resolved.

---

## 2. Status of Blockers & Remediation Summary

### Previous Blockers:
1. **[RESOLVED] Acoustic Padding Contamination:**  
   *Root Cause:* `facebook/wav2vec2-base-960h` uses temporal GroupNorm in layer 0. Zero-padding in variable-length batches contaminated valid acoustic representations by up to $10.5$ in logits.  
   *Resolution:* Implemented unpadded-segment feature extraction for variable-length batches, zero-padded latent features, and applied boolean attention masking across the 12-layer context Transformer. Numerical invariance between isolated and batched evaluation is verified at $\Delta \le 4.58 \times 10^{-7}$.
2. **[RESOLVED] Silent Label Truncation:**  
   *Root Cause:* `min(target.shape[1], logits.shape[1])` silently dropped phonemes on sequence length mismatches.  
   *Resolution:* Implemented strict shape equality assertions and validated that padding dimensions contain strictly `ignore_index = -100`.
3. **[RESOLVED] Unenforced CTC Contracts & Silent Infinity Concealment:**  
   *Root Cause:* `validate_ctc_contract()` was never called at runtime, and `zero_infinity=True` hid infeasible alignments as zero loss.  
   *Resolution:* `validate_ctc_contract()` is now executed on every forward pass; `zero_infinity=False` is set by default; policy-based violation counters and empty-target handlers are in place.
4. **[RESOLVED] GPU Feasibility Evidence Discrepancy:**  
   *Root Cause:* Checked-in script and JSON contained 11 configs up to $B=4$, while report Table 3 claimed $B=8..48$ without machine-readable evidence.  
   *Resolution:* Historical claims beyond $B=4$ in commit `96aedc0` were formally marked `NOT_VERIFIED`. An explicit `--batch-scaling` synthetic routine was added and verified on the target physical GPU, establishing machine-readable evidence up to $B=40$ ($2,937.9\text{ MiB}$) and clean OOM at $B=48$.
5. **[RESOLVED] Training Step Accounting Overestimation:**  
   *Root Cause:* Report asserted ~360,000 steps based on 3,599 corpus utterances at batch size 1.  
   *Resolution:* Recomputed exact cardinalities ($N_{\text{train}} = 2,400$, $B=4$, $A=4$). Complete 4-fold cross-validation involves exactly **$60,000$ forward-backward steps** and **$15,000$ optimizer updates**.

---

## 3. Acknowledged Limitations

1. **Local Hardware VRAM Bound:**  
   The target physical GPU (NVIDIA GeForce GTX 1650 Mobile, 3,730.4 MiB usable VRAM) operates reliably at micro-batches of $B \in [2, 4]$ under AMP fp16 and gradient checkpointing. It cannot accommodate native batches of $B \ge 48$. Effective batch sizes of $16$ require gradient accumulation ($A=4$).
2. **Local Wall-Clock Duration:**  
   On local hardware, executing full 4-fold cross-validation (25 epochs per fold, 100 epochs total) requires approximately **$28.9\text{ hours}$** of continuous compute. For multi-seed sweeps and Vietnamese Leave-One-Speaker-Out (LOSO) rounds, cloud instances (NVIDIA A10G / A100) are recommended to compress runtime to ~19 minutes.
3. **Synthetic-Only Validation Scope:**  
   In compliance with gate instructions, all validations in this phase were conducted strictly on synthetic waveforms and mock phoneme targets. **Zero real speech audio from L2-ARCTIC or SpeechOcean762 was ingested for parameter optimization during this gate.**

---

## 4. Next Actions: Transition to RQ1-TRN-01B

With all training correctness contracts hardened, verified, and test-covered, the project is cleared to initiate **RQ1-TRN-01B Real-Data Training**:

1. **Commit & Push Corrections:** Commit all modified source files, test suites, and audit reports to feature branch `feat/rq1-trn-01a-training-smoke-test` and update PR #1. Do not merge PR #1 automatically.
2. **Phase Transition to RQ1-TRN-01B:**  
   - Configure training orchestrator using verified parameters ($B_{\text{micro}}=4$, $A=4$, lr=$1\times 10^{-4}$ with warmup).
   - Ingest frozen training split manifests (`manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`).
   - Run Fold 1 baseline fine-tuning under strict cross-entropy and masked BCE MDD objectives.
   - Enforce checkpointing, early stopping, and validation loss monitoring on disjoint validation speakers.
