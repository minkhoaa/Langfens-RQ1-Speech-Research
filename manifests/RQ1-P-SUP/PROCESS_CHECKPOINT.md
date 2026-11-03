# P-SUP preparation checkpoint

State: **WORKSPACE_CREATED; DOCUMENTATION_READY; DISTINCT_OBJECTIVE_POLICY_AMENDMENT_AND_IMPLEMENTATION_PENDING; EXECUTION_NO_GO.** Continuation of RQ-RESEARCH-01. Operator authorized setup/documentation only, not model/trainer/label code changes. No label extraction, synthetic model tests, inference/fitting/backward/training/GPU/deployment. Training NOT_STARTED; 0 successful updates, 0 skipped updates/no attempts; no new model artifact.

## Identity and outputs

- Workspace `/home/khoa/Projects/KLTN/langfens-rq1-p-sup`; branch `prep/rq1-p-sup`.
- Exact frozen common base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`, preserving newer local research closure/addenda rather than choosing an older remote. Fresh `git worktree add -b` without force; no existing branch/path replaced.
- Contract `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`, SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- `EXPERIMENT_SPEC.md`: compact natural-operation/existence alternatives, contemporaneous matched binary comparison, objective/initialization/schedule and fixed/best accounting. ONE treatment/capacity control still requires registration; no complex architecture selected or implicit AGENTS amendment.
- `LABEL_COLLATOR_AND_OUTPUT_DESIGN.md`: natural strict operation labels/masks, optional realized labels, blinded-loader separation, output schema and synthetic CPU no-model/no-training test plan (NOT RUN).
- `KAGGLE_T4_DEPLOYMENT_DESIGN.md`: private T4/FP16, matched states/update budget, permanent CLI/UI workflow, atomic durable per-epoch checkpoints and safe recovery (DESIGN ONLY).
- `PROVENANCE_MANIFEST.json`: exact base/shared hash, input selectors, archived weight references, future output identities and authorization limits.

## Verification actually executed

- Fresh worktree initially clean on exact common base; distinct expected branch and shared contract SHA verified in both worktrees.
- After docs: all **57** shared source/input/archive fingerprints unchanged; other arm directory absent. No `.env`, `datasets` or `checkpoints` in this worktree; nested repo not initialized/copied. No original ignored local datasets/weights/credentials or nested dirt copied.
- Only this arm's text documentation/metadata and an additive registry entry are staged. Final lightweight JSON/schema-reference/hash/link/size/Git checks, exact arm commit SHAs and merge-base are saved in coordinator `manifests/RQ1-WORKSPACE-PREP-01/WORKSPACE_VERIFICATION.md` on `prep/rq1-shared-base`, not automatically merged here.
- One coordinator document-audit assertion failed before arm commits because it incorrectly counted new staged arm files as pre-existing tracked-file changes. Predicate corrected to distinguish additions; no source/data mutation or test execution occurred. Rerun required before commit, not marked PASS from failed attempt.
- Model/trainer/collator/label/evaluator/config/E0/E1 archive bytes unchanged. Current test status NOT_RUN, not scientific PASS. No remote/API/quota/auth/secret verification attempted; no kernel created/pushed.

## Blockers and exact safe restart

Missing latest distinct omission proposal or explicit acceptance of limitation; explicit new-objective policy authorization; ONE compact parameterization and parameter-matched binary control; loss/mask/module initialization and schedule registration; prospective effect/noninferiority/false-alarm margins; subtype support and synthetic CPU tests; actual private package/weight/readback/storage/quota/T4/AMP and recovery readiness. Existing blinded-collator defect is documented but not fixed. Baseline parity limitations remain; no P-SCORE dependency.

Read-only resume commands:

```bash
cd /home/khoa/Projects/KLTN/langfens-rq1-p-sup
git status --short
git log -1 --format='%H %s'
git merge-base HEAD 0d268ff8c444e16fd76aca2f4ad6ece966908edc
(cd manifests/RQ1-WORKSPACE-PREP-01 && sha256sum -c CONTRACT_SHA256)
```

Read AGENTS, global index, shared addendum/contract/provenance and this checkpoint/spec/label/deployment design. Confirm exact base, clean state and hashes. Request objective-policy amendment plus separate bounded label/collator implementation authorization first; no backward/model fitting locally even for a smoke test. Remote training/deployment needs its own later authorization and verified allocation. **STOP after documentation.** Archived E0/E1 remain last valid references; no model produced here. DIAG-02B PARTIAL and E1 rollout NO-GO retained.

---

## Append-only continuation — Luna implementation (2026-11-03)

The above documents the historical preparation-only state. At the explicit user's later implementation authorization, Luna resumed the intentionally paused partial source on branch `prep/rq1-p-sup` at `d6ef18443752b0add3f3dfcfff42f7fa7be967c1`; no prior work was discarded. Initial Git state, handoff and safety boundaries: [checkpoint 01](checkpoints/01-luna-handoff-and-initial-state.md). Additive prototype and validation: [checkpoint 02](checkpoints/02-implementation-and-validation.md).

Bounded isolated implementation is complete in tested source commit `d2b9702627fa03c7d2de97ff8e825bb3ffbe0b86`; the synthetic CPU forward-only suite passed **88 tests**; preflight explicitly reports EXPERIMENT_HOLD and no trainer. This is prototype code review GO only, not experiment/training/deployment GO. The preflight package contains no data/model/remote imports and was not pushed or run remotely. Full actual update counts remain NOT_STARTED, 0 successful/0 skipped/0 attempts, no artifact. Missing remote data allowlist adapter, per-epoch durable checkpoint/resume, authorized T4 trainer, scientific objective and execution approvals, missing-proposal/margin/license/support verification remain explicit blockers. No real data/labels/TEST contents, weights, backward, optimizer, GPU, API or remote service were accessed. The original historical state above is retained; use latest [source receipt](checkpoints/03-source-commit-receipt.md) and checkpoint for current state.

---

## Append-only continuation — P-SUP Pipeline Implementation & Kaggle Staging (2026-11-03)

State: **PIPELINE_IMPLEMENTED; TESTS_PASS (97 passed); PREFLIGHT_AUDITED; KAGGLE_STAGED_NO_RUN; AWAITING_WEB_UI_EXECUTION.**
Carried out under explicit execution authorization for Agent B (P-SUP Execution) on branch `prep/rq1-p-sup`.

### 1. Objective Policy Amendment
Documented in [OBJECTIVE_POLICY_AMENDMENT.md](OBJECTIVE_POLICY_AMENDMENT.md). Formally authorizes:
- **Treatment Arm `T-OP`:** `CompactOperationHead` (266,667 parameters) supervised by `objective="operation_ce"` (3-way categorical Cross-Entropy: correct=0, substitution=1, deletion=2) + chronological realized CTC loss.
- **Capacity-Matched Control Arm `B-CONTROL`:** Identical `CompactOperationHead` supervised by `objective="binary_collapsed"` ($\text{LSE}(z_1, z_2) - z_0$ with BCE loss) + realized CTC loss. Subtype logits in control are explicitly unidentifiable under binary supervision.
- **Acoustic Backbone:** `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`) kept strictly frozen in eval mode (`requires_grad=False`).

### 2. Dataset Allowlist & Loader Implementation
Implemented in `src/langfens_prep/p_sup/dataset.py`:
- `PSupL2ArcticDataset` and `collate_psup_batch`.
- Connects Fold 1 TRAIN (2,399 utterances across 16 speakers: BWC, ERMS, HKK, HQTV, MBMPS, NCC, SKA, SVBI, THV, TLV, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA).
- Subpartitions: FIT (1,799 utterances), SELECTION (300 utterances), CALIBRATION (300 utterances).
- Connects Fold 1 VAL (300 utterances across NJS, RRBI).
- **Fail-Closed Test Sealing:** Active Fold 1 TEST speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`) are physically absent from `authorized_RQ1-TRN-02A_fold1_train_val_only.zip` (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`) and strictly rejected with `SealedSpeakerContaminationError` upon any attempted access.
- Cryptographic verification of split manifest (`7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).

### 3. Training & Evaluation Pipelines
- **Trainer (`src/langfens_prep/p_sup/trainer.py`, `audit_scripts/train_p_sup.py`):**
  - Contracts enforced: seed 1337, batch_size 2, accumulation 2 (effective batch 4), tail group divisor `actual_group_size`, AdamW lr 1e-3, weight decay 1e-2, gradient clipping 1.0, warmup 100 updates, cosine schedule over 15,000 horizon, max 3,600 successful updates, CUDA FP16 AMP.
  - Dual checkpointing: fixed-budget checkpoint (`fixed_budget_3600.pt`) at exactly 3,600 updates, validation-selected `best.pt` (strict binary AP, blank_fraction < 0.99, patience 5 logged), and per-epoch `last.pt`. All checkpoints save model weights, optimizer, scheduler, scaler, full RNG states, metadata, and compute SHA-256.
- **Evaluator (`src/langfens_prep/p_sup/evaluator.py`, `audit_scripts/evaluate_p_sup.py`):**
  - Conforms to Evaluation Contract V1 (`rq1-runtime-v1` schema).
  - Computes binary AP, legacy-comparable omission/substitution AP, dedicated omission AP, PER, blank fraction, and per-speaker breakdowns (NJS, RRBI).
  - Validates full run via `validate_complete_run` and `validate_runtime_row`.

### 4. Verification Evidence
- Scoped test suite: **97 passed** (`pytest -q tests/test_p_sup_labels.py tests/test_p_sup_model.py tests/test_p_sup_runtime.py tests/test_p_sup_pipeline.py`).
- Preflight audit: `python audit_scripts/p_sup_preflight.py` verified and reports `STAGING_VERIFIED_PASSED` for `deploy/kaggle_p_sup_train`.

### 5. Kaggle T4 Deployment Package & Staging
- Deployment directory: `deploy/kaggle_p_sup_train/`.
- Target: `machine_shape: NvidiaTeslaT4`, `enable_gpu: true`, `is_private: true`, `dataset_sources: ["minkhoaa/langfens-rq1-fold1-staging"]`.
- Staged via official CLI workflow:
  ```bash
  kaggle kernels push -p deploy/kaggle_p_sup_train --no-run
  ```
- Kernel slug: `minkhoaa/langfens-rq1-p-sup-t4-training`.
- Verification status: `KernelWorkerStatus.COMPLETE` (kernel version saved without background execution).

### 6. Restart & Execution Instructions (Permanent Kaggle Workflow)
Per `AGENTS.md` and `KAGGLE_T4_DEPLOYMENT_DESIGN.md`:
1. Open Kaggle Web UI editor: `https://www.kaggle.com/code/minkhoaa/langfens-rq1-p-sup-t4-training`.
2. Attach user secret `HF_TOKEN` under Add-ons -> Secrets.
3. Start official training execution via **Save & Run All**.
4. Monitor progress and logs via CLI:
   ```bash
   kaggle kernels status minkhoaa/langfens-rq1-p-sup-t4-training
   kaggle kernels output minkhoaa/langfens-rq1-p-sup-t4-training
   ```
5. Checkpoint durability: Notebook mirrors checkpoints atomically to private Hugging Face repository `minkhoaa/langfens-rq1-cross-fold-checkpoints:p_sup/` with SHA-256 readback.
6. Post-execution evaluation: Once completed, evaluate both endpoints using `audit_scripts/evaluate_p_sup.py`.

---

## Append-only continuation — P-SUP Data Partitioning Fix (2026-11-03)

State: **DATA_PARTITIONING_FIXED; ALL_TESTS_PASS (107 passed); PREFLIGHT_AUDITED.**
Carried out under Agent A authorization on branch `prep/rq1-p-sup`.

### 1. Partition Corrections
- **Training dataset (`partition="fit"`):** Exactly 1,799 utterances across 12 speakers: `BWC`, `ERMS`, `HKK`, `HQTV`, `SKA`, `SVBI`, `TNI`, `TXHC`, `YBAA`, `YDCK`, `YKWK`, `ZHAA`.
- **Selection dataset (`partition="selection"` / `"select"`):** Exactly 300 utterances across 2 speakers: `MBMPS`, `TLV`. Used for validation loss, AP tracking, early stopping, and saving `best.pt` during training.
- **Calibration dataset (`partition="calibration"` / `"cal"`):** Exactly 300 utterances across 2 speakers: `NCC`, `THV`. Reserved for threshold/temperature calibration.
- **Evaluation dataset (`partition="val"` / `"evaluation"`):** Exactly 300 utterances across 2 speakers: `NJS`, `RRBI`. Preserved strictly unseen during training and selection; used ONLY for final candidate comparison under Evaluation Contract V1.
- **Sealed TEST partition (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`: 900 utterances):** Strictly prohibited. Fail-closed with `SealedSpeakerContaminationError` upon any access or attempt to instantiate with `split="test"` or `partition="test"`.

### 2. Implementation Updates
- `src/langfens_prep/p_sup/dataset.py`: Updated `PSupL2ArcticDataset` to support `partition` ("fit", "selection"/"select", "calibration"/"cal", "val"/"evaluation") and `split`. Enforced fail-closed `SealedSpeakerContaminationError` for test split/partition requests and sealed test speaker encounters. Added `FIT_SPEAKERS`, `SELECT_SPEAKERS`, `CALIBRATION_SPEAKERS`, `VAL_SPEAKERS`, `PARTITION_SPEAKERS`, and `PARTITION_UTTERANCE_COUNTS`. Added `speakers` and `speaker_counts()` helpers.
- `src/langfens_prep/p_sup/trainer.py`: Enforced training on `partition="fit"` (1,799 utterances) and validation/AP tracking/checkpoint selection on `partition="selection"` (300 utterances). Verified strict VAL and TEST speaker isolation. Updated metadata and logging to reflect SELECT AP.
- `audit_scripts/train_p_sup.py`: Added preflight partition verification verifying FIT (1,799), SELECT (300), CAL (300), VAL (300), disjointness, and sealed test speaker rejection.
- `configs/rq1_p_sup_v1.json`: Verified `fit_speakers` (12), `selection_speakers` (2), `calibration_speakers` (2), `evaluation_speakers` (2), and utterance counts (1799, 300, 300, 300).
- `tests/test_p_sup_partitions.py`: Added comprehensive regression test suite verifying exact utterance counts, speaker memberships, pairwise speaker/utterance disjointness, sealed test speaker fail-closed rejection, config reflection, and bit-exact SHA-256 alignment with `SHARED_PROVENANCE.json`.

### 3. Verification Evidence
- Partition and pipeline test suite: **19 passed** in 2.85s (`pytest tests/test_p_sup_partitions.py tests/test_p_sup_pipeline.py`).
- Full P-SUP scoped test suite: **107 passed** in 2.61s (`pytest tests/test_p_sup_*.py`).
- Preflight audit: `python audit_scripts/p_sup_preflight.py` reports `STAGING_VERIFIED_PASSED`.

