# LANGFENS — Global Process Checkpoint Registry

**Repository:** `minkhoaa/Langfens-RQ1-Speech-Research`  
**Last Verified Documentation Commit:** `7a3b9ecf5e9efedc480b477f531ebe9318ae3415`  
**Branch:** `review/rq1-trn-03a-cross-fold-planning`  
**Registry Maintenance Lead:** Senior ML Research Engineer & Reproducibility Auditor  
**Policy Authority:** `AGENTS.md` (Mandatory Process Checkpoint Policy)

---

## 1. Master Research Phase Registry

| Phase ID | Phase Objective | Verified Status | Last Completed Step | Last Verified Commit | Primary Checkpoint | Next Valid Continuation | Remaining Blocker |
|---|---|:---:|---|:---:|:---:|---|---|
| **RQ1-AUD-01** | Dataset acquisition, archive integrity, and deep annotation audit across L2-ARCTIC and SpeechOcean762. | `COMPLETED` | Step 03: Scientific Consistency Reconciliation | `5ccb4dc` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-AUD-01/PROCESS_CHECKPOINT.md) | Proceed to RQ1-STD-01 | None |
| **RQ1-STD-01** | Phoneme mapping, MDD label schemas, evaluation metrics, and speaker-independent split protocols. | `COMPLETED` | Step 02: Specification Consistency Reconciliation | `5ccb4dc` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-STD-01/PROCESS_CHECKPOINT.md) | Proceed to RQ1-SPLIT-01 | None |
| **RQ1-SPLIT-01** | Deterministic, leak-free speaker-independent split manifest generation under CC BY-NC 4.0. | `COMPLETED` | Step 02: License Provenance & Ledger | `085bfcd` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-SPLIT-01/PROCESS_CHECKPOINT.md) | Proceed to RQ1-PREP-01 | None |
| **RQ1-PREP-01** | Ingestion pipeline implementation, full 3,599-utterance census verification, and CPU throughput profiling. | `COMPLETED` | Step 02: Full Corpus Census & Profiling | `64a1c4d` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-PREP-01/PROCESS_CHECKPOINT.md) | Proceed to RQ1-TRN-00 | None |
| **RQ1-TRN-00** | GOP vs Wav2Vec2 comparative assessment, compute resource planning, and loss contract hardening. | `COMPLETED` | Step 02: Compute Planning & Contract Hardening | `175d68f` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-00/PROCESS_CHECKPOINT.md) | Proceed to RQ1-TRN-01A | None |
| **RQ1-TRN-01A** | Multi-task architecture implementation, frozen backbone integrity verification, and contract hardening. | `COMPLETED` | Step 03: Training Correctness & Padding | `5942338` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-01A/PROCESS_CHECKPOINT.md) | Proceed to RQ1-TRN-01B0 | None |
| **RQ1-TRN-01B0** | Controlled real-data sanity run (100 utterances), gradient verification, and learning diagnostics. | `COMPLETED_WITH_LIMITATIONS` | Step 02: Learning Diagnostics & Forensics | `d63f25f` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-01B0/PROCESS_CHECKPOINT.md) | Proceed to RQ1-TRN-01B1 | CTC blank collapse mitigation required |
| **RQ1-TRN-01B1** | CTC blank collapse stabilization, 2x2 factorial ablation, and acoustic feature dependence validation. | `COMPLETED` | Step 03: Acoustic Conditioning Validation | `73eae7e` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-01B1/PROCESS_CHECKPOINT.md) | Proceed to RQ1-TRN-02A | None |
| **RQ1-TRN-02A** | Two-head Fold 1 baseline training on Colab T4, CPU replay, and Protocol Z zero-oracle numerical closure. | `COMPLETED` | Step 03: Protocol Z & Numerical Closure | `c4a658f` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-02A/PROCESS_CHECKPOINT.md) | Proceed to RQ1-ARTIFACT-01 & RQ1-TRN-02B | None |
| **RQ1-ARTIFACT-01** | Baseline R model archival on Google Drive, native safetensors export, and private Hugging Face release. | `COMPLETED` | Step 02: Safetensors Export & Hugging Face Release | `a200b73` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-ARTIFACT-01/PROCESS_CHECKPOINT.md) | Ready for downstream acoustic ablations | Public release & HF dataset metadata upload pending |
| **RQ1-TRN-02B** | Controlled acoustic conditioning ablation (C0 vs C1), preflight R2 safety closure, and Colab launch gate. | `COMPLETED` | Step 16: Final three-model scientific assessment and research gate closure | `7a3b9ec` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-02B/PROCESS_CHECKPOINT.md) | Proceed to RQ1-TRN-03A (Cross-Fold Speaker-Independent Generalization Planning) | None |
| **RQ1-TRN-03A** | Cross-fold speaker-independent generalization planning for Folds 2, 3, and 4; test-isolation policy; independent Baseline R training contract; macro/micro metrics. | `READINESS_VERIFIED` | Step 05: Dedicated Kaggle Fold 2 Deployment Package | `ddd090a` | [PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-03A/PROCESS_CHECKPOINT.md) | Await operator authorization; stage fold packages to Kaggle and launch Fold 2 training | Operator authorization required for staging package upload and Kaggle GPU launch |

---

## 2. Detailed Phase Inventories & Step Checkpoints

### RQ1-AUD-01 — Dataset Acquisition and Annotation Audit
- **Overview Checkpoint:** [manifests/RQ1-AUD-01/PROCESS_CHECKPOINT.md](../../manifests/RQ1-AUD-01/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `5ccb4dc2cc986036175161c660b73a206de188d2`
- **Step Checkpoints:**
  1. [checkpoints/01-archive-integrity-and-inventory.md](../../manifests/RQ1-AUD-01/checkpoints/01-archive-integrity-and-inventory.md) — Raw archives cryptographic ledger, non-destructive ZIP CRC-32 and TAR stream integrity scans, resource library inventory.
  2. [checkpoints/02-annotation-and-phoneme-audit.md](../../manifests/RQ1-AUD-01/checkpoints/02-annotation-and-phoneme-audit.md) — Deep audit across 3,599 Praat TextGrids and 5,000 SpeechOcean762 utterances, Vietnamese deletion concentration discovery.
  3. [checkpoints/03-scientific-consistency-reconciliation.md](../../manifests/RQ1-AUD-01/checkpoints/03-scientific-consistency-reconciliation.md) — RQ1-AUD-01R reconciliation of canonical denominator (119,151 tokens, 17,518 errors), 245 orphan audio exclusions, Python 3.14.8 runtime rectification.

### RQ1-STD-01 — Data and Phoneme Standardization
- **Overview Checkpoint:** [manifests/RQ1-STD-01/PROCESS_CHECKPOINT.md](../../manifests/RQ1-STD-01/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `5ccb4dc2cc986036175161c660b73a206de188d2`
- **Step Checkpoints:**
  1. [checkpoints/01-standardization-policies-and-schema.md](../../manifests/RQ1-STD-01/checkpoints/01-standardization-policies-and-schema.md) — Eight foundational standardization policies: label schema, 39-ARPAbet mapping, audio prep, split protocol, and evaluation rules.
  2. [checkpoints/02-specification-consistency-reconciliation.md](../../manifests/RQ1-STD-01/checkpoints/02-specification-consistency-reconciliation.md) — RQ1-STD-01R pre-implementation gate: substitution decomposition ($14,098 = 12,368 + 1,472 + 258$), 4-fold 3M/3F demographic parity proof, Vietnamese LOSO dual-regime formulation, unverified claim downgrade.

### RQ1-SPLIT-01 — Deterministic Speaker-Independent Splits
- **Overview Checkpoint:** [manifests/RQ1-SPLIT-01/PROCESS_CHECKPOINT.md](../../manifests/RQ1-SPLIT-01/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `085bfcd68f6bf7de7ca6709848f0802773f8e705`
- **Step Checkpoints:**
  1. [checkpoints/01-deterministic-split-generation.md](../../manifests/RQ1-SPLIT-01/checkpoints/01-deterministic-split-generation.md) — Partition generation and automated mathematical validation: zero speaker leakage ($S_1 \cap S_2 = \emptyset$), demographic balance, 245 orphan audio exclusions.
  2. [checkpoints/02-license-provenance-and-provenance-ledger.md](../../manifests/RQ1-SPLIT-01/checkpoints/02-license-provenance-and-provenance-ledger.md) — Authoritative CC BY-NC 4.0 license erratum, cryptographic split manifest digest freeze, and timestamp erratum reconciliation.

### RQ1-PREP-01 — Audio Preparation and Dataset Validation
- **Overview Checkpoint:** [manifests/RQ1-PREP-01/PROCESS_CHECKPOINT.md](../../manifests/RQ1-PREP-01/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `64a1c4d`
- **Step Checkpoints:**
  1. [checkpoints/01-ingestion-pipeline-and-parsers.md](../../manifests/RQ1-PREP-01/checkpoints/01-ingestion-pipeline-and-parsers.md) — Implementation of `src/langfens_prep/` (phoneme normalizer, Praat textgrid parser, speechocean parser, audio loader with 16 kHz polyphase sinc resampling, PyTorch datasets).
  2. [checkpoints/02-full-corpus-census-and-profiling.md](../../manifests/RQ1-PREP-01/checkpoints/02-full-corpus-census-and-profiling.md) — Exhaustive 3,599-utterance census verification ($\Delta = 0$ variance against RQ1-STD-01R), worker multiprocessing lazy instantiation bugfix, CPU throughput benchmark (251.8 utt/s).

### RQ1-TRN-00 — Baseline Experiment Design and Environment Planning
- **Overview Checkpoint:** [manifests/RQ1-TRN-00/PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-00/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `175d68f`
- **Step Checkpoints:**
  1. [checkpoints/01-baseline-architecture-and-comparative-design.md](../../manifests/RQ1-TRN-00/checkpoints/01-baseline-architecture-and-comparative-design.md) — GOP vs Wav2Vec 2.0 MDD comparative assessment, research order recommendation, and experiment protocol.
  2. [checkpoints/02-compute-planning-and-contract-hardening.md](../../manifests/RQ1-TRN-00/checkpoints/02-compute-planning-and-contract-hardening.md) — Colab T4 compute budget, dual environment planning (`.venv` CPU / `.venv-cuda` GPU), gitignore protections, explicit loss masking contracts (`MaskedBCEWithLogitsLoss`).

### RQ1-TRN-01A — Model Architecture and Training Contract
- **Overview Checkpoint:** [manifests/RQ1-TRN-01A/PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-01A/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `5942338`
- **Step Checkpoints:**
  1. [checkpoints/01-architecture-implementation-and-synthetic-smoke.md](../../manifests/RQ1-TRN-01A/checkpoints/01-architecture-implementation-and-synthetic-smoke.md) — Multi-task `LangfensWav2Vec2MDD` implementation (CTC, Binary, Diagnosis), cross-attention alignment mechanism, synthetic forward-backward smoke tests.
  2. [checkpoints/02-backbone-integrity-and-inference-validity.md](../../manifests/RQ1-TRN-01A/checkpoints/02-backbone-integrity-and-inference-validity.md) — Pretrained Wav2Vec2 backbone freeze verification (`facebook/wav2vec2-base-960h`, 94.4M params), CTC vocabulary reconciliation, GPU VRAM profiling.
  3. [checkpoints/03-training-correctness-and-acoustic-padding.md](../../manifests/RQ1-TRN-01A/checkpoints/03-training-correctness-and-acoustic-padding.md) — Acoustic padding validation, CTC contract zero-loss fallback on length violations, optimizer micro-step vs update accounting.

### RQ1-TRN-01B0 — Controlled Training Sanity Gate
- **Overview Checkpoint:** [manifests/RQ1-TRN-01B0/PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-01B0/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED_WITH_LIMITATIONS` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `d63f25f`
- **Known Limitations:** CTC blank collapse discovered (>98% blank emission, greedy PER ~100%); 39-class substitution diagnosis head showed chance-level accuracy on 100 utterances.
- **Step Checkpoints:**
  1. [checkpoints/01-train-only-subset-sanity-execution.md](../../manifests/RQ1-TRN-01B0/checkpoints/01-train-only-subset-sanity-execution.md) — 100-utterance train-only B0-SUBSET curation, 5-epoch training run, gradient backward verification, zero data leakage audit.
  2. [checkpoints/02-learning-diagnostics-and-reconciliation.md](../../manifests/RQ1-TRN-01B0/checkpoints/02-learning-diagnostics-and-reconciliation.md) — Post-hoc forensics diagnosing CTC blank collapse mechanism, step accounting erratum (25 updates vs 50 micro-steps), diagnosis metric audit.

### RQ1-TRN-01B1 — CTC Stabilization and Pilot Experiments
- **Overview Checkpoint:** [manifests/RQ1-TRN-01B1/PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-01B1/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `73eae7e`
- **Step Checkpoints:**
  1. [checkpoints/01-ctc-stabilization-pilots.md](../../manifests/RQ1-TRN-01B1/checkpoints/01-ctc-stabilization-pilots.md) — Hyperparameter pilots (Variants A–D), complete elimination of blank collapse via Variant D (cosine warmup, peak lr=1e-3), decoded length restoration, resume determinism.
  2. [checkpoints/02-factorial-generalization-and-metric-erratum.md](../../manifests/RQ1-TRN-01B1/checkpoints/02-factorial-generalization-and-metric-erratum.md) — 2x2 factorial ablation confirming frozen backbone + lr 1e-3 superiority, majority baseline erratum correction, internal holdout protocol.
  3. [checkpoints/03-acoustic-conditioning-validation.md](../../manifests/RQ1-TRN-01B1/checkpoints/03-acoustic-conditioning-validation.md) — Acoustic conditioning validation: proved multi-task Wav2Vec2 MDD outperforms canonical text-only prior; greedy CTC protocol freeze.

### RQ1-TRN-02A — Full Fold 1 Baseline Training and Validation
- **Overview Checkpoint:** [manifests/RQ1-TRN-02A/PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-02A/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `c4a658f` (training run at commit `caa5678`)
- **Key Provenance Distinctions:**
  - Selected Baseline Checkpoint: Epoch 6 (0-indexed epoch 5), 3,600 updates, val AP 0.316062 (`best.pt`, SHA-256 `a2445c8d...`).
  - Full Training Run Termination: Epoch 11 (0-indexed epoch 10), 6,600 updates, val AP 0.304382 (`last.pt`, early stopping patience 5).
  - Diagnosis Head: Untrained / frozen throughout baseline training (two-head contract only: CTC + Binary).
  - Evaluation Protocols:
    - Protocol O (Oracle TextGrid tokens, 9,956 phones): Binary AP = **0.316062**, prevalence = 0.116211 (2.72× ratio), optimal F1 = 0.3775 at $\tau=0.20$, CTC PER = 23.64%.
    - Protocol Z (Zero-Oracle G2P tokens, 9,919 aligned pairs): Binary AP = **0.313530**, prevalence = 0.115435, F1 = 0.3712 at $\tau=0.20$, net acoustic gain = +0.012942.
- **Step Checkpoints:**
  1. [checkpoints/01-two-head-colab-training-run.md](../../manifests/RQ1-TRN-02A/checkpoints/01-two-head-colab-training-run.md) — Two-head Fold 1 training execution on Colab T4, per-epoch Google Drive synchronization, 11 epochs completed, early stopping at 6,600 updates, selection of epoch 6 `best.pt`.
  2. [checkpoints/02-checkpoint-recovery-and-cpu-replay.md](../../manifests/RQ1-TRN-02A/checkpoints/02-checkpoint-recovery-and-cpu-replay.md) — Checkpoint recovery (`best.pt`), standalone CPU evaluation pipeline, bit-identical Protocol O validation AP reproduction.
  3. [checkpoints/03-protocol-z-zero-oracle-and-numerical-closure.md](../../manifests/RQ1-TRN-02A/checkpoints/03-protocol-z-zero-oracle-and-numerical-closure.md) — Protocol Z (Zero-Oracle) verification without gold TextGrids, DP tie-break repair, acoustic controls, final errata and numerical closure.

### RQ1-ARTIFACT-01 — Baseline R Model Archival and Private Hugging Face Release
- **Overview Checkpoint:** [manifests/RQ1-ARTIFACT-01/PROCESS_CHECKPOINT.md](../../manifests/RQ1-ARTIFACT-01/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `a200b73b90a137316817bc42df0ba6e0db2c24c5`
- **Step Checkpoints:**
  1. [checkpoints/01-baseline-r-storage-and-drive-archival.md](../../manifests/RQ1-ARTIFACT-01/checkpoints/01-baseline-r-storage-and-drive-archival.md) — Storage hierarchy audit under Google Drive folder `11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx`, cryptographic verification of `best.pt`, `last.pt`, `fold1_train_val_only.zip`, and `BASELINE_R_ARTIFACT_MANIFEST.json`.
  2. [checkpoints/02-safetensors-export-and-huggingface-release.md](../../manifests/RQ1-ARTIFACT-01/checkpoints/02-safetensors-export-and-huggingface-release.md) — Native FP32 `model.safetensors` export, bit-for-bit numerical parity verification (max diff 0.000000e+00), private Hugging Face release (`minkhoaa/langfens-rq1-fold1-baseline-r` rev `6c31f719...`). Dataset metadata repo pending.

### RQ1-TRN-02B — Acoustic Conditioning Design, Implementation and Preflight Safety Closure
- **Overview Checkpoint:** [manifests/RQ1-TRN-02B/PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-02B/PROCESS_CHECKPOINT.md)
- **Status:** `COMPLETED` | **Evidence Level:** `INDEPENDENTLY_VERIFIED` (C0 & C1 3,600-update training on Kaggle T4×2 complete; private Hugging Face mirroring verified; standalone CPU paired evaluation complete; final three-model scientific assessment complete; prompt dropout hypothesis NOT_SUPPORTED; Reference R retained as primary baseline; research gate closed)
- **Source Commit:** Commits `8486cbb` through `review/rq1-trn-02b-kaggle-official-runs`
- **Current Operational State:**
  1. Training completed: Control C0 ($p=0.00$) and Treatment C1 ($p=0.25$) completed strictly 3,600 successful updates on Kaggle Tesla T4 $\times$ 2.
  2. Persistence verified: All 8 checkpoints and history files atomically mirrored to private Hugging Face repository `minkhoaa/langfens-rq1-trn-02b-checkpoints` with SHA-256 readback.
  3. Scientific evaluation completed: Standalone paired evaluation on local CPU executed under Protocol O and Protocol Z; 1,000 speaker-stratified cluster bootstrap resamples computed; prompt dropout hypothesis falsified.
- **Step Checkpoints:**
  1. [checkpoints/01-acoustic-conditioning-design.md](../../manifests/RQ1-TRN-02B/checkpoints/01-acoustic-conditioning-design.md) — Controlled prompt-dropout ablation design (Arm C0 p=0 vs Arm C1 p=0.25), compute budget (6 epochs / 3,600 updates max per arm), primary contrast metric.
  2. [checkpoints/02-implementation-and-preflight-audit.md](../../manifests/RQ1-TRN-02B/checkpoints/02-implementation-and-preflight-audit.md) — Binary-only prompt dropout implementation in `src/langfens_prep/model.py`, contract regression tests, implementation preflight report.
  3. [checkpoints/03-preflight-r2-safety-closure-and-auth-packet.md](../../manifests/RQ1-TRN-02B/checkpoints/03-preflight-r2-safety-closure-and-auth-packet.md) — Preflight R1/R2 defect resolution, strict resume validation, dual-arm endpoint integrity, unique run-ID fail-closed guards, Colab execution authorization packet.
  4. [checkpoints/04-colab-auth-r1-operational-corrections.md](../../manifests/RQ1-TRN-02B/checkpoints/04-colab-auth-r1-operational-corrections.md) — Colab Auth R1 operational corrections: Fold 1 speaker partition & disjointness proof, commit provenance tiering, non-interactive Colab launch/staging scripts, Google Drive root & file ID ledger, Colab CLI quota audit (0.00 balance blocker), 18 new regression tests.
  5. [checkpoints/05-colab-auth-r2-launcher-safety-fixes.md](../../manifests/RQ1-TRN-02B/checkpoints/05-colab-auth-r2-launcher-safety-fixes.md) — Colab Auth R2 launcher safety fixes: fresh VM deterministic bootstrap (colab_bootstrap.py), notebook kernel execution compatibility (no __file__), duplicate run-ID elimination, safe bounded zip archive inspection, fail-closed layered readiness queries, authentic Drive persistence checks, primary endpoint classification (182 passed tests).
  6. [checkpoints/06-colab-auth-r2-final-safety-hotfix.md](../../manifests/RQ1-TRN-02B/checkpoints/06-colab-auth-r2-final-safety-hotfix.md) — Colab Auth R2 final safety hotfix: strict primary endpoint invariants (9/9), negative regression suite (6/6), strict 5-layer executable readiness and fail-closed checks, immutable Git HEAD validation, authenticated Drive hierarchy & API verification (187 passed CPU tests).
  7. [checkpoints/07-colab-auth-r3-trust-boundary-closure.md](../../manifests/RQ1-TRN-02B/checkpoints/07-colab-auth-r3-trust-boundary-closure.md) — Colab Auth R3 trust-boundary closure: fail-closed on unverified Drive identity (no training launch on pending identity), elimination of user-asserted readiness flags, strict descendant Drive containment, removal of --mock-gpu from production launchers (190 passed CPU tests).
  8. [checkpoints/08-live-gpu-runtime-and-staging-verification.md](../../manifests/RQ1-TRN-02B/checkpoints/08-live-gpu-runtime-and-staging-verification.md) — Live GPU runtime and staging verification: probed Colab CCU API (0.00 compute units, 5,836 free tokens), provisioned and verified Tesla T4 GPU (CUDA 13.0, 15GB VRAM), probed credentials propagation endpoint showing unmounted Drive requiring interactive OAuth (`colab drivemount`), demonstrated /dev/tty subshell limitation, released idle GPU session, and provided operator restart instructions.
  9. [checkpoints/09-colab-cpu-fp32-feasibility-static-gate.md](../../manifests/RQ1-TRN-02B/checkpoints/09-colab-cpu-fp32-feasibility-static-gate.md) — Pinned-source CPU FP32 static feasibility gate and benchmark acceptance plan; no CPU runtime/optimizer step executed, operator authorization pending.
  10. [checkpoints/10-kaggle-t4x2-migration-feasibility-static-review.md](../../manifests/RQ1-TRN-02B/checkpoints/10-kaggle-t4x2-migration-feasibility-static-review.md) — Kaggle T4×2 GPU capacity, preservation of CUDA FP16 regime, licensing/staging/persistence gates and historical-T4-time extrapolation; no Kaggle benchmark or training.
  11. [checkpoints/11-kaggle-t4x2-platform-port-and-benchmark-gate.md](../../manifests/RQ1-TRN-02B/checkpoints/11-kaggle-t4x2-platform-port-and-benchmark-gate.md) — Kaggle T4×2 platform port implementation, verified 30.00h GPU quota, AGENTS.md amendment, auto-extracted zip handling, durable checkpoint mirror with SHA-256 readback, kernel metadata push with `--no-run`, 32 new regression tests (222 passed total), benchmark gate pending.
  12. [checkpoints/12-kaggle-direct-c0-launch-blockers-and-benchmark-waiver.md](../../manifests/RQ1-TRN-02B/checkpoints/12-kaggle-direct-c0-launch-blockers-and-benchmark-waiver.md) — Owner waived disposable 50-step benchmark; source review found false remote SYNCED backends, post-run-only sync, wrong trainer arguments and unpinned launch; official C0/C1 remain blocked/not started.
  13. [checkpoints/13-kaggle-c0-execution-blockers-and-persistence-closure.md](../../manifests/RQ1-TRN-02B/checkpoints/13-kaggle-c0-execution-blockers-and-persistence-closure.md) — Remediation of all 8 Step 12 defects, verified private dataset staging (minkhoaa/langfens-rq1-fold1-staging), genuine Hugging Face remote checkpoint mirror with live write-readback probe, C0 execution attempt, fail-closed Phase 1 GPU check (0 updates), Kaggle phone verification blocker documented.
  14. [checkpoints/14-kaggle-c0-c1-training-completion-and-paired-mirroring.md](../../manifests/RQ1-TRN-02B/checkpoints/14-kaggle-c0-c1-training-completion-and-paired-mirroring.md) — Official Control Arm C0 (3,600 updates, peak AP 0.312170, final PER 22.09%) and Treatment Arm C1 (3,600 updates, peak AP 0.320398, final PER 22.12%) executed on Kaggle Tesla T4x2 GPU; both arms verified COMPLETED_3600_UPDATES; all 8 deliverable checkpoints and history ledgers atomically mirrored to private Hugging Face repository (minkhoaa/langfens-rq1-trn-02b-checkpoints) with SHA-256 readback.
  15. [checkpoints/15-cpu-paired-scientific-evaluation.md](../../manifests/RQ1-TRN-02B/checkpoints/15-cpu-paired-scientific-evaluation.md) — Standalone paired scientific evaluation on local CPU under Protocol O and Protocol Z; 1,000 paired speaker-stratified cluster bootstrap resamples; evaluated primary causal contrast $\Delta_{\text{dropout}} = \text{C1} - \text{C0} = -0.007918$ (95% CI: `[-0.018328, +0.000385]`); acoustic reliance gap change $\Delta_{\text{gap}} = -0.006120$; zero-oracle contrast $\Delta_Z = -0.006590$; CTC PER guardrails preserved; hypothesis determined NOT_SUPPORTED; research gate concluded.
  16. [checkpoints/16-final-three-model-assessment-and-gate-closure.md](../../manifests/RQ1-TRN-02B/checkpoints/16-final-three-model-assessment-and-gate-closure.md) — Final three-model scientific assessment across Reference R, Control C0, and Treatment C1; audited and corrected NJS/RRBI token-count discrepancy (5,011 / 4,945); verified training commit parity; consolidated Protocol O, Protocol Z, bootstrap 95% CIs, and pairwise decision agreements; evaluated exploratory best.pt vs fixed-budget primary results; stated hypothesis verdict as NOT_SUPPORTED on Fold 1 without overstating negative certainty; retained Reference R as primary baseline; closed RQ1-TRN-02B research gate.

### RQ1-TRN-03A — Cross-Fold Speaker-Independent Generalization (Folds 2, 3, 4)
- **Overview Checkpoint:** [manifests/RQ1-TRN-03A/PROCESS_CHECKPOINT.md](../../manifests/RQ1-TRN-03A/PROCESS_CHECKPOINT.md)
- **Status:** `PLANNING_COMPLETE; GATE_PENDING_OPERATOR_AUTHORIZATION` | **Evidence Level:** `INDEPENDENTLY_VERIFIED`
- **Source Commit:** `05796e25b1f745ddb7772e0d5bda2edd5d0a7f1c`
- **Research Plan:** [manifests/RQ1-TRN-03A/CROSS_FOLD_RESEARCH_PLAN.md](../../manifests/RQ1-TRN-03A/CROSS_FOLD_RESEARCH_PLAN.md)
- **Key Methodological Tenets:**
  - Strict preservation of Fold 1 R/C0/C1 findings and NOT_SUPPORTED dropout verdict.
  - Per-fold test isolation: For model $M_k$, $S_{\text{test}}^{(k)}$ is strictly excluded from training, validation, early stopping, threshold calibration, and remote staging (`authorized_RQ1-TRN_fold{k}_train_val_only.zip`). Speakers rotate legitimately across folds according to `l2arctic_4fold_splits.json`.
  - Exact Fold 1 Baseline R parity: Independent initialization from pretrained Wav2Vec2 (`facebook/wav2vec2-base-960h`); `seed: 1337`, `batch_size: 2`, `gradient_accumulation_steps: 2` (effective batch size 4), `warmup_updates: 100`, cosine schedule over `max_epochs: 25` horizon (~15,000 updates), early stopping patience 5 on validation AP with `blank_fraction < 0.99`.
  - Dual checkpoint accounting: Independently save and evaluate Fixed-Budget (Epoch 6 / 3,600 updates) and validation `best.pt`.
  - Acoustic controls: Strictly mandate deterministic within-speaker cyclic shift-1 protocol (`within_speaker_cyclic`).
  - Metrics & aggregation: Mathematical definitions of FNR ($1 - \text{Recall}$) and FPR ($1 - \text{Specificity}$); Macro AP (Student's $t$ 95% CI, $df=3$), Micro AP across 39,576 pooled validation tokens, speaker-cluster bootstrap (disclosing $G=8$ small-cluster limitations).
  - Thesis scope demarcation: SpeechOcean762 cross-corpus evaluation and pedagogical pronunciation feedback belong to RQ1; RQ2 is reserved exclusively for Multi-Agent Automated Test Generation.
- **Step Checkpoints:**
  1. [checkpoints/01-cross-fold-research-plan-and-test-isolation.md](../../manifests/RQ1-TRN-03A/checkpoints/01-cross-fold-research-plan-and-test-isolation.md) — Initial research plan, mathematical speaker partition audit, staging conflict resolution, test-isolation policy, compute resource budgeting, and operator approval packet.
  2. [checkpoints/02-cross-fold-protocol-parity-and-isolation-corrections.md](../../manifests/RQ1-TRN-03A/checkpoints/02-cross-fold-protocol-parity-and-isolation-corrections.md) — Protocol parity audit against Fold 1 Baseline R implementation, per-fold test isolation refinement, dual checkpoint accounting, within-speaker shift-1 derangement, FNR/FPR disambiguation, small-sample bootstrap limitations, thesis scope demarcation, and automated regression test suite (`tests/test_rq1_trn_03a_protocol_parity_and_isolation.py`).
  3. [checkpoints/03-fold-readiness-and-trainer-generalization.md](../../manifests/RQ1-TRN-03A/checkpoints/03-fold-readiness-and-trainer-generalization.md) — Trainer and evaluator generalization across Folds 1–4, deterministic train/val-only zip builder implementation (`audit_scripts/build_fold_packages.py`), physical test partition exclusion, generation of packages for Folds 2–4, pinned configs `configs/rq1_trn_03a_fold{2,3,4}.yaml`, Kaggle and Hugging Face staging destination audit, and readiness integration tests (250 passed tests).
  4. [checkpoints/04-kaggle-deployment-blockers-closure.md](../../manifests/RQ1-TRN-03A/checkpoints/04-kaggle-deployment-blockers-closure.md) — Remediation of trainer resume bug (outdir variable ordering and fixed_budget_saved restoration), dedicated Kaggle execution driver (`audit_scripts/kaggle_fold_train_exec.py`) with fresh pretrained initialization enforcement, entrypoint mode integration, Kaggle Secrets credential hygiene, mocked failure-path integration tests, and 100% successful non-training preflight dry-run (258 passed tests).
  5. [checkpoints/04b-kaggle-durability-and-recovery-closure.md](../../manifests/RQ1-TRN-03A/checkpoints/04b-kaggle-durability-and-recovery-closure.md) — Durability and recovery closure: immutable fixed-budget checkpoint lock at strictly 3,600 updates and Epoch 6 (`is_fixed_budget_endpoint=True`), `validate_resume_checkpoint()` with cross-fold resume rejection, recovery of `fixed_budget_checkpoint.pt` from `remote_mirror`, mandatory private remote persistence (`huggingface`) rejecting empty/missing targets, per-epoch `history.json` upload with SHA-256 readback, removal of credential-file fallbacks and CLI token arguments, executable mocked recovery test suite, and independent speaker-leak audit (261 passed tests).
  6. [checkpoints/04c-final-launch-gate-correction.md](../../manifests/RQ1-TRN-03A/checkpoints/04c-final-launch-gate-correction.md) — Final launch gate correction: verified Git HEAD, strengthened resume validation (fold identity, manifest digest, pinned config, update counters, RNG states, ancestry check), separation of 25-epoch cosine scheduler horizon from 10-epoch execution cap, elimination of CLI token argument path, child trainer strict secrets, Kaggle auto-extracted zip mounting handling, and 20 comprehensive deployment tests (270 passed tests total).
  7. [checkpoints/05-dedicated-kaggle-fold2-deployment-package.md](../../manifests/RQ1-TRN-03A/checkpoints/05-dedicated-kaggle-fold2-deployment-package.md) — Dedicated Fold 2 Kaggle deployment package: kernel metadata (Tesla T4x2, private, internet enabled, minkhoaa/langfens-rq1-fold2-staging source), dataset metadata (CC BY-NC 4.0, test speakers excluded), 7-phase audited execution notebook (pinned commit 407cf04..., strict Kaggle Secrets, verify_fold2 mode, EXECUTE_TRAINING=False gate, zero disk token searches, zero CLI token arguments), and 24 comprehensive deployment tests (274 passed tests total).

---

## 3. Cross-Platform Storage & Artifact Index

| Artifact Identifier | Role / Description | Canonical Location / Identifier | Cryptographic SHA-256 Digest | Verification Status |
|---|---|---|---|:---:|
| `l2arctic_release_v5.0.zip` | Raw L2-ARCTIC Corpus Archive | Google Drive Root (`1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`) | `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda` | Verified |
| `speechocean762.tar.gz` | Raw SpeechOcean762 Archive | Google Drive Root (`1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`) | `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5` | Verified |
| `l2arctic_4fold_splits.json` | Authoritative 4-Fold Splits | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | Verified |
| `fold1_train_val_only.zip` | Authorized Fold 1 Train/Val Audio | Google Drive (`1xdypDWopd585OjuQeHjxlWOJfU2oiJiY`) | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | Verified |
| `authorized_RQ1-TRN_fold2_train_val_only.zip` | Authorized Fold 2 Train/Val Package (18 spk) | `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip` | `112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9` | Verified |
| `authorized_RQ1-TRN_fold3_train_val_only.zip` | Authorized Fold 3 Train/Val Package (18 spk) | `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip` | `e81ac9e5d95a68d7f02c7b2ea8e5f1e9fd89b047926ffeeb4d096b0a3a5378a2` | Verified |
| `authorized_RQ1-TRN_fold4_train_val_only.zip` | Authorized Fold 4 Train/Val Package (18 spk) | `datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip` | `653bdad079ee951d6b6d6a3f19964143aa5f4290a3f8c232e9b2524abdbe7f94` | Verified |
| `best.pt` | Production Baseline R Weights | Google Drive (`10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I`) | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | Verified |
| `last.pt` | Epoch 11 Training Run Endpoint | Google Drive (`1VdlhoQHfd9MqjzIkMUEqdqMxhLlJiO-s`) | `bf46fab366363d46082082cf7f088b218f6d8d12851f5a9de706276b789efcb6` | Verified |
| `model.safetensors` | Exported Native Model Weights | Hugging Face: `minkhoaa/langfens-rq1-fold1-baseline-r` | `a5b9d41f1431ad7a8b9718b4d84a7886e39fe5aff45fe742fbe572819f8babc1` | Verified (Private) |
| `C0_fixed_budget_checkpoint.pt` | Control C0 3,600-Update Primary Weights | HF: `minkhoaa/langfens-rq1-trn-02b-checkpoints` (`C0/fixed_budget_checkpoint.pt`) | `0dc1d4fbc9a107d4c6bccc92c32b028ebe2ed70647b89520a15dcd80be29e24e` | Verified (Private) |
| `C1_fixed_budget_checkpoint.pt` | Treatment C1 3,600-Update Primary Weights | HF: `minkhoaa/langfens-rq1-trn-02b-checkpoints` (`C1/fixed_budget_checkpoint.pt`) | `52b0e6c46e46439cbdbd5b977dd9fe42e90db35a14437f210f2b173c40151ea4` | Verified (Private) |
| `BASELINE_R_ARTIFACT_MANIFEST.json` | Remote Release Ledger | Google Drive (`1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP`) | `6222c92b63ea58f584267b39fb3ffc51394562086f90de21bbab8c8c1abe1e22` | Verified |

---

## 4. Policy for Continuity & Handoff

In compliance with `AGENTS.md`:
1. Every new or resuming agent must consult this master index before proposing or initiating changes.
2. Under no circumstances may an agent skip from `RQ1-TRN-02B` preflight directly to model training without explicit operator authorization and resolution of operational blockers.
3. Checkpoint documentation must be committed in the same reviewable feature branch as the underlying research deliverables.
