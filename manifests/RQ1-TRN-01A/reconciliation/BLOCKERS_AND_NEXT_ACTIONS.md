# LANGFENS — RQ1-TRN-01A-R: Blockers & Next Authorized Actions Report

**Execution Gate:** RQ1-TRN-01A-R — Pretrained Backbone & Inference Validity Gate  
**Deliverable Associated Path:** `manifests/RQ1-TRN-01A/reconciliation/BLOCKERS_AND_NEXT_ACTIONS.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01A/reconciliation/`  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & PyTorch Architect  

---

## 1. Definitive Gate Verdict

```text
================================================================================
FINAL EVALUATION GATE: RQ1-TRN-01A-R
STATUS: PASS
================================================================================
```

### Verification Criteria Checklist:
- [x] **Verified Pretrained Initialization:** Genuine `facebook/wav2vec2-base-960h` backbone loaded from Hugging Face ($94,371,712$ backbone params, $97,405,648$ total params). Uninitialized scratch model preserved as `LangfensCustomMDDReference`.
- [x] **Inference Path Without Gold Boundaries:** Implemented differentiable `CanonicalAcousticCrossAttention` alignment. Mandatory negative test verified that varying, removing, or scrambling gold TextGrid boundaries produces strictly identical deployment predictions.
- [x] **Correct Realized-Phone CTC Semantics:** Chronological reconstruction across all 3,599 Gold TextGrids includes 1,081 clean epenthetic additions, excludes 3,420 acoustic deletions, excludes 15,647 silence pauses, and masks 258 `'err'` substitutions. Exact identity: $101,633 + 13,840 + 1,081 = 116,554$ active CTC tokens. Zero CTC blank violations.
- [x] **Working Dataset-to-Model Integration:** Complete pipeline `Manifest -> Dataset -> Collate -> Model.forward -> Masked Losses` validated end-to-end on both CPU and GPU.
- [x] **Passing Regression Tests:** All 12 mandatory regression items passed (`12/12`). Pre-existing test suite passed with 100% success (`51/51` in CUDA environment, `49/49` passed and 2 skipped in CPU environment). Zero assertions weakened.
- [x] **Documented Full-Backbone GPU Feasibility:** GTX 1650 Mobile (3.73 GB usable VRAM) empirically benchmarked on full 12-layer architecture. Bounded local micro-batches ($B \in [2, 8]$) pass safely with $>60\%$ memory headroom. Out-of-memory boundary identified at $B \ge 48$.
- [x] **Consistent Experiment Specifications:** Reconciled $S_{\text{eval}} = 13,840$, Vietnamese LOSO Regime A (21/2/1), Vietnamese Zero-Shot Regime B (18/2/4), Fold 1 Test denominator ($N = 29,769$), strict/lenient binary policies, validation-only calibration, speaker-cluster bootstrap CIs, and licensing.

---

## 2. Inventory of Source Files & Cryptographic Checksums

| File Path | Status | SHA-256 Digest |
| :--- | :---: | :--- |
| `src/langfens_prep/model.py` | Modified | `5512ee43e427ddf28735d85f69565f89a2958fc7385f0febc675d9658015f443` |
| `src/langfens_prep/dataset.py` | Modified | `f08078ec0d8ee317f2a752b231a6a30f16373278642554e7d905671edeead5ed` |
| `src/langfens_prep/collate.py` | Modified | `18c40ed02846ea89838c1371085e7c6effc8edbb182a99685c52f13033c26066` |
| `src/langfens_prep/phoneme_normalizer.py` | Modified | `1096d8d95c9e3b3b3f2bacf0fa0d6109c36d4027e40cf832bf6b6cbc045df9fe` |
| `tests/test_reconciliation_regression.py` | Created | `7370836bdaacda9b19eca541e2914fb84e3eeb3f6d5e34f5a10a4c1ea35ddf97` |
| `tests/test_dataset_and_loader.py` | Modified | `4db1ec6ea85793af4dce2e5f0bfdb060e11850b96419ef63360f7b0493dadfea` |
| `tests/test_normalizer.py` | Modified | `055c17ff40902aadb90e8c72e094462fde58a0513d87eec017a98c416abed998` |
| `audit_scripts/gpu_feasibility_bench.py` | Created | `f8e2dd3eda4cf3502c736826f55d7e8e3a5eeed24be15362e76be4c483088f0f` |
| `audit_scripts/audit_ctc_realized_accounting.py` | Created | `9b4a7a8934c1c3b9956497b313172138cca43dcdcdb164270732f129d2140d9d` |
| `configs/huggingface_model_card.md` | Modified | `83ddeea0658294aa3e0ffdaf34c4c6fd0affff7838ab11eca7b8e368473be1c8` |
| `configs/huggingface_registry.yaml` | Modified | `e46f2de6842c59071c9e5090634b4c74d34e1a8790e5e2fd8e16bef1985f63e6` |

---

## 3. Explicit Unresolved Limitations & Risk Boundaries

1. **Local Workstation Throughput Ceiling:**
   - The NVIDIA GeForce GTX 1650 Mobile provides $3.73\text{ GB}$ usable VRAM. While sufficient for micro-batches ($B \in [2, 8]$) with AMP fp16 and gradient checkpointing, full 4-fold training across 25 epochs will require ~45 minutes per epoch locally. Cloud GPU instances ($\ge 16\text{ GB}$) are strongly recommended for full multi-fold training.
2. **Non-Commercial Data Licensing:**
   - L2-ARCTIC v5.0 is licensed under CC BY-NC 4.0. Fine-tuned model checkpoints cannot be utilized for commercial purposes without independent acoustic data licensing.

---

## 4. Next Authorized Research Actions

**CRITICAL POLICY ENFORCEMENT:**
Real-data fine-tuning (`RQ1-TRN-01B`) remains **STRICTLY UNAUTHORIZED** in this phase. Zero model weights were trained or updated on real corpus speech.

Upon formal review and approval of the `RQ1-TRN-01A-R` deliverables:
1. Push all reconciled commits to `origin/feat/rq1-trn-01a-training-smoke-test`.
2. Update Pull Request #1 with the reconciliation deliverables summary.
3. Submit for formal authorization to transition to `RQ1-TRN-01B` (Single-Fold Real-Data Fine-Tuning Execution).
