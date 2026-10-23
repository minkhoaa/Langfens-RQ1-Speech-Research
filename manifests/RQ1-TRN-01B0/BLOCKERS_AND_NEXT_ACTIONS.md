# LANGFENS — RQ1-TRN-01B0: Blockers, Limitations & Next Actions Report

**Gate Reference:** RQ1-TRN-01B0 — Controlled Real-Data Training Sanity Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01B0/BLOCKERS_AND_NEXT_ACTIONS.md`  
**Inspected Git Commit:** `1549edd` on branch `feat/rq1-trn-01b0-training-sanity-gate`  
**Lead Auditor / Senior Speech ML Research Engineer:** Senior Speech ML Research Engineer & PyTorch Systems Reviewer  
**Overall Gate Evaluation Status:** `RQ1-TRN-01B0: PASS`  

---

## 1. Acceptance Gate Audit & Executive Decision

The formal criteria required for `RQ1-TRN-01B0: PASS` have been completely fulfilled:

| Acceptance Criterion | Verification Evidence & Location | Status |
| :--- | :--- | :---: |
| **1. Correct Pretrained Model Loading** | Hugging Face `facebook/wav2vec2-base-960h` loaded at pinned commit `22aad52d`; 210 loaded keys; 94.37M backbone / 97.41M total parameters. | **PASS** |
| **2. Correct Training-Only Data Selection** | Deterministic selection (seed 42); exactly 32 utterances from all 16 Fold 1 train speakers (2 utts/spk); 1,054 canonical phonemes; 39/39 ARPAbet covered. | **PASS** |
| **3. Finite Losses & Gradients** | Zero NaN or Inf values across all 216 micro-steps; gradients clipped at $\tau=1.0$; 8 AMP overflow events safely handled by dynamic scaling ($16,384 \to 64$). | **PASS** |
| **4. Verified Parameter Updates** | AdamW with cosine warmup; all 3 loss components improved: Total $-71.94\%$, CTC $-80.01\%$, Binary BCE $-30.28\%$, Diagnosis CE $-7.99\%$. | **PASS** |
| **5. Working Checkpoint Save/Resume** | Local checkpoint saved ($1.14\text{ GB}$, ignored by Git); reloaded model matches expected loss with bit-identical precision ($\Delta = 0.000000$). | **PASS** |
| **6. No Train/Val/Test Leakage** | Mathematical and programmatic proof of 0% overlap with Fold 1 val (300) or test (900); disjoint speakers certified; zero gold boundaries at inference. | **PASS** |
| **7. Documented GPU & Runtime Metrics** | Measured on NVIDIA GTX 1650 Mobile: $269.08\text{ s}$ wall clock (100 updates, $1.25\text{ s}$/micro-step), $1,535.7\text{ MiB}$ peak alloc, $43.1\%$ VRAM headroom. | **PASS** |
| **8. Transparent Learning Diagnostics** | Comprehensive step-by-step trajectories logged and analyzed; PR-AUC increased from $0.1762 \to 0.4066$; Diagnosis accuracy tripled ($2.7\% \to 8.0\%$). | **PASS** |
| **9. No Unresolved Correctness Blockers** | GroupNorm padding isolation verified; silent label truncation prevented; CTC contracts verified; split checksums immutable; licensing compliant. | **PASS** |

### Final Formal Evaluation:
$$\mathbf{RQ1-TRN-01B0: PASS}$$

---

## 2. Status of Blockers & Remediation Summary

### Previous Preflight Concerns & Outstanding Issues:
1. **[RESOLVED] Exact 4-Fold Sample Exposure Accounting:**  
   *Issue:* Nominal cross-validation step accounting rounded fold train sets to 2,400 utterances, stating 240,000 exposures.  
   *Resolution:* Recomputed exact cardinalities ($2,399 + 2,400 + 2,400 + 2,399 = 9,598$ unique train utterances). At 25 epochs per fold, the exact cumulative exposure is **$239,950\text{ samples}$**. Updated `TRAINING_STEP_ACCOUNTING.md` and synced to Google Drive.
2. **[RESOLVED] Theoretical Cloud Benchmarks Status:**  
   *Issue:* Historical claims of 150 ms/batch and 18.75-minute 4-fold CV completion on NVIDIA A10G were unverified analytical projections.  
   *Resolution:* Reclassified in all documentation as unverified theoretical projections based on published FLOPS ratings. Zero physical cloud passes have been run.
3. **[RESOLVED] Google Drive Final Gate Synchronization:**  
   *Issue:* Artifacts in `manifests/RQ1-TRN-01A/reconciliation/final_gate/` were present locally but missing remotely on Google Drive.  
   *Resolution:* Synchronized all 7 artifacts via `rclone copy` to `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01A/reconciliation/final_gate/` and verified with `rclone lsf`.
4. **[RESOLVED] Controlled Training Sanity Optimization:**  
   *Issue:* Real speech audio and gold phonetic targets had never been ingested for gradient backpropagation.  
   *Resolution:* Executed a strictly bounded 100-update controlled optimization run on 32 deterministic Fold 1 train utterances under AMP fp16, gradient checkpointing, and frozen feature extractor. Monotonic loss reduction verified across all three heads without instability.

### Active Blockers:
$$\mathbf{ZERO\text{ Active Technical or Scientific Blockers Remain.}}$$

---

## 3. Acknowledged Constraints & Non-Goals

1. **Local Hardware Throughput Bound:**  
   The target physical GPU (NVIDIA GeForce GTX 1650 Mobile, 3,911.6 MiB VRAM) operated reliably at micro-batch size $B=2$ and gradient accumulation $A=2$ (effective batch size 4), with $43.1\%$ VRAM headroom. Executing full Fold 1 training (25 epochs, 2,399 utterances) locally will require approximately **$7.2\text{ hours}$**, and full 4-fold cross-validation will require **$28.9\text{ hours}$**.
2. **Small-Set Overfitting Scope Limitation:**  
   The loss reductions and metric improvements achieved during this gate ($-71.94\%$ total loss, PR-AUC $0.1762 \to 0.4066$, diagnosis accuracy $2.7\% \to 8.0\%$) demonstrate that the multi-task model architecture, autograd graph, and loss functions can successfully fit real speech data. **They do not demonstrate generalization to unseen speakers or accents.**
3. **Strict Non-Goals Enforced in this Gate:**  
   - **NO full Fold 1 training** was initiated.
   - **NO held-out test data** (Fold 1 test speakers `ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`) was loaded, forwarded, or evaluated.
   - **NO 4-fold cross-validation** or Vietnamese LOSO rounds were conducted.
   - **NO model weights** were committed to Git or published to Hugging Face.

---

## 4. Next Actions: Transition to Authorized Full Fold 1 Training (RQ1-TRN-01B)

Upon receipt of explicit operator authorization to proceed to full training:

1. **Commit & Push Deliverables:**  
   Commit code changes, configuration files, and safe markdown/JSON reports in `manifests/RQ1-TRN-01B0/` to feature branch `feat/rq1-trn-01b0-training-sanity-gate`. Ensure checkpoints and datasets remain outside Git.
2. **Synchronize Deliverables to Google Drive:**  
   Sync `manifests/RQ1-TRN-01B0/` to `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01B0/` via `rclone`.
3. **RQ1-TRN-01B Full Training Protocol (Pending Authorization):**  
   - Configure training orchestrator for Fold 1 ($N_{\text{train}} = 2,399$ utterances).
   - Ingest all 16 Fold 1 train speakers with micro-batch $B=4$, gradient accumulation $A=4$ (effective batch size 16), lr=$1.0 \times 10^{-4}$ with warmup.
   - Run 25 epochs ($15,000$ forward-backward steps, $3,750$ optimizer updates).
   - Enforce epoch-level validation monitoring on disjoint validation speakers (`NJS`, `RRBI`, 300 utterances) to calibrate the binary classification decision threshold and apply early stopping.
   - Halt execution and await explicit user command before initiating full training.
