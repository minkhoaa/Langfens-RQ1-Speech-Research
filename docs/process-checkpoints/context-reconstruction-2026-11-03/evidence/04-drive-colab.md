# Forensic Reconstruction Report: SEAM 4 — Google Drive, Colab, Thesis Proposal & Ecosystem Compute Evolution

**Auditor / Role:** Independent Read-Only Systems & Research Forensics Auditor  
**Date of Observation:** 2026-11-03T23:55:17+07:00 (Asia/Ho_Chi_Minh)  
**Assigned Seam:** SEAM 4 — Drive, Colab, Thesis Proposal Conformance, and Ecosystem Compute Evolution  
**Authoritative Artifact Path:** `/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/ea49dbe6-c72f-494d-b433-77701f7d9e45/evidence/04-drive-colab.md`  
**Repositories Inspected:**  
- Speech Research: `/home/khoa/Projects/KLTN/Dataset` (HEAD `1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6`, branch `prep/rq1-shared-base`)  
- Core Backend: `/home/khoa/Projects/langfens/Project_Langfens_Microservice` (HEAD `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48`, branch `generation-harness`)  
- Frontend App: `/home/khoa/Projects/langfens/langfens-fe-app` (HEAD `2d9fe94a54e65680eca56aafbe40e691018113da`, branch `autonomous-generation`)  
**Evidence Standard:** Strict demarcation between DIRECT VERIFIED (live primary observation), CORROBORATED (cross-document consistency), INFERRED, and UNKNOWN.  

---

## 1. Scope & Source Inventory

### 1.1 Inspected Primary Sources (Directly Observed)
1. **Original Thesis Proposal Document (`DC KLTN0127`):**  
   - File Path: `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`  
   - Verification: Full read of 18 markdown blocks including the 50 KB main table cell (line 14) and references/signatures (line 18).  
   - Formal metadata: Faculty of Computer Science, University of Information Technology (VNU-HCM); signed 2026-08-18 by Advisor ThS. Trần Thị Hồng Yến and students Nguyễn Bá Trọng Khôi (MSSV: 23520772) & Từ Minh Khoa (MSSV: 23520757).  
2. **Google Drive Remote Storage (`kltn_gdrive:`):**  
   - Verified via authenticated non-interactive `rclone listremotes`, `rclone lsd`, and `rclone lsf` (read-only queries).  
   - Top-level folders listed (timestamps reflect Google Drive directory metadata, not creation):  
     - `Colab Notebooks` (Drive metadata: 2026-10-21 18:07:09 +07:00) -> contains `Untitled/`, `Untitled0.ipynb`  
     - `KLTN` (Drive metadata: 2026-10-21 17:56:50 +07:00) -> contains `l2arctic_release_v5.0.zip`, `Bản sao của l2arctic_release_v5.0.zip`  
     - `KLTN-Collab` (Drive metadata: 2026-10-21 18:04:36 +07:00) -> contains `KLTN-Dataset.ipynb`  
     - `KLTN-Dataset` (Drive metadata: 2026-10-21 18:16:34 +07:00) -> Root Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`, subtrees `datasets/`, `manifests/`, `references/`, `resources/`  
     - `langfens-projects-2026-08-20` (Drive metadata: 2026-08-20 16:26:30 +07:00)  
3. **Google Colab CLI (`/usr/bin/colab`):**  
   - `colab usage`: Balance = `0.00 compute units`, Usage rate = `0.00/hr`, Active assignments = `0`.  
   - `colab sessions`: `No active sessions found on server`.  
4. **Speech Research Manifests & Tooling (`/home/khoa/Projects/KLTN/Dataset`):**  
   - Checkpoints: `RQ1-AUD-01`, `RQ1-STD-01`, `RQ1-SPLIT-01`, `RQ1-PREP-01`, `RQ1-TRN-00`, `RQ1-TRN-01A`, `RQ1-TRN-01B0`, `RQ1-TRN-01B1`, `RQ1-TRN-02A`, `RQ1-ARTIFACT-01`, `RQ1-TRN-02B` (Steps 01–16), `RQ1-TRN-03A` (Steps 01–19), `RQ1-DIAG-01`, `RQ1-DIAG-02`, `RQ1-DIAG-02B`, `RQ-MODEL-01`, `RQ-RESEARCH-01`.  
   - Colab launch tooling: `audit_scripts/colab_launch_workflow.py`, `audit_scripts/colab_bootstrap.py`, `audit_scripts/colab_stage_and_verify.py`, `audit_scripts/colab_train_exec.py`.  
5. **Backend & Frontend Codebases:**  
   - Backend: IELTS Reading Multi-Agent Generation Harness (`services/generation-harness/`, `services/attempt-service/`).  
   - Frontend: Exam rendering and timer components (`src/features/exam/`, `src/app/do-test/`).  

### 1.2 Inaccessible or Restricted Sources
1. **Google Colab Active GPU Hardware:** Allocation blocked due to 0.00 CCU balance.  
2. **Automated Google Drive FUSE Mount (`/content/drive`):** Blocked in automated CLI subshells because `colab drivemount` calls `open('/dev/tty')`, which raises `OSError: [Errno 6] No such device or address: '/dev/tty'`.  
3. **Google Drive Binary Payloads:** Raw archives and weights (`best.pt`, `l2arctic_release_v5.0.zip`) were not downloaded or extracted; rclone directory listing confirms presence by name, not cryptographic payload checksum.  
4. **Interactive Authentication:** OAuth flows and browser-based token prompts were strictly bypassed.  

---

## 2. Dated Chronological Event Table

| Date / Time (Asia/Ho_Chi_Minh) | Milestone / Phase | Goal | Decision & Rationale | Alternatives Considered | Action, Result & Operational Limitation | Current Scientific & Operational Validity |
|---|---|---|---|---|---|---|
| **2026-08-18** | Thesis Proposal Sign-off | Formalize degree proposal at UIT. | Establish dual research axes: RQ1 (Acoustic MDD & pronunciation feedback) and RQ2 (Multi-Agent educational test/exercise generation). | Monolithic single-LLM generation; end-to-end ASR pronunciation grading without forced alignment. | Approved by ThS. Trần Thị Hồng Yến. Outlined 5-stage timeline (09/2026–01/2027), IELTS reading/grammar targets, and .NET 8 / FastAPI / Next.js / PostgreSQL / Qdrant stack. | **AUTHORITATIVE INTENT** |
| **2026-10-21 ~ 2026-10-23** | Drive Staging & Ingestion | Stage raw speech corpora on persistent Google Drive. | Store immutable archives under Google Drive folder `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK` (`KLTN-Dataset`). | Local laptop disk only; streaming audio across Drive FUSE. | Uploaded `l2arctic_release_v5.0.zip` and `speechocean762.tar.gz`. Proved zero variance across 3,599 Praat TextGrids in manifest audit. | **VALID & FROZEN** |
| **2026-10-24 14:00** | RQ1-TRN-01B0 | 100-utterance sanity run on Colab GPU. | Verify gradient backward pass and padding contracts before full training. | Full 4-fold training immediately; local CPU execution. | Executed 5 epochs on Colab T4. Discovered CTC blank collapse (>98% blank emissions) and chance-level substitution diagnosis. | **HISTORICAL LIMITATION (Resolved)** |
| **2026-10-25 09:30** | RQ1-TRN-01B1 | Mitigate CTC blank collapse on Colab GPU. | Introduce Variant D (cosine warmup, peak lr 1e-3, frozen backbone) to restore phonetic emissions. | Unfreezing Wav2Vec2 context encoder; higher initial lr. | Fully restored greedy decoded phoneme length; validated acoustic conditioning over canonical text-only prior. | **VALID FOUNDATION** |
| **2026-10-26 18:22** | RQ1-TRN-02A (Run commit `caa5678`) | Train full Fold 1 Baseline R on Colab T4. | Train two-head CTC+binary model across 2,399 train utts; evaluate on 300 val utts (NJS/RRBI). | Multi-task 3-head training (diagnosis head frozen due to chance accuracy). | 11 epochs completed (6,600 updates, ~855s runtime). Early stopping patience 5 selected Epoch 6 (3,600 updates, `best.pt`, val AP 0.316062). Direct Drive synchronization succeeded. | **AUTHORITATIVE BASELINE** |
| **2026-10-27 12:00** | RQ1-ARTIFACT-01 | Baseline R release & safetensors export. | Ensure multi-platform persistence and reproducibility of Baseline R weights. | Single-store Drive retention. | Serialized native `model.safetensors` (SHA-256 `a5b9d41f...`); mirrored to Hugging Face (`minkhoaa/langfens-rq1-fold1-baseline-r`). Published `BASELINE_R_ARTIFACT_MANIFEST.json` on Drive (`1yrTbDR9...`). | **VALID ARTIFACT** |
| **2026-10-28 07:56** | RQ1-TRN-02B Step 08 | Live Colab GPU runtime & staging probe for C0/C1. | Provision authorized Colab T4 GPU to begin 7,200-update paired ablation. | Manual browser training. | T4 GPU provisioned (`remainingTokens: 5836`), but account showed `0.00 compute units`. `colab drivemount` failed with `/dev/tty` missing. Session released cleanly. | **BLOCKED ON COLAB** |
| **2026-10-28 11:15** | RQ1-TRN-02B Step 09 | Static review of Colab CPU FP32 feasibility. | Evaluate whether C0/C1 could run on Colab CPU in FP32 without AMP. | Fall back to local laptop CPU; reduce update budget. | Audit proved CPU FP32 training would take 18–36s per update (>36 hours), risk preemption, and introduce numerical confounding. CPU training rejected; strict fallback prohibition enacted. | **REJECTED (NO-GO)** |
| **2026-10-28 14:00** | RQ1-TRN-02B Step 10 | Static review of Kaggle Tesla T4x2 GPU migration. | Explore Kaggle GPU (30h/week floating quota, 2x T4, CUDA FP16 AMP) to execute acoustic conditioning ablations. | Purchasing Colab CCUs; abandoning ablation. | Platform feasibility established. Required private staging, CC BY-NC 4.0 check, and Hugging Face Hub atomic checkpoint mirroring to replace Google Drive FUSE. | **VALID PIVOT** |
| **2026-10-29 16:45** | RQ1-TRN-02B Steps 13–14 | Execute C0 and C1 on Kaggle Tesla T4x2. | Execute Control C0 (p=0.0) and Treatment C1 (p=0.25) for 3,600 updates each under CUDA FP16 AMP. | Run on Colab; run concurrently on 2 GPUs. | Both arms completed strictly 3,600 updates. Checkpoints mirrored to private HF repo `minkhoaa/langfens-rq1-trn-02b-checkpoints`. CPU paired evaluation falsified prompt-dropout hypothesis ($\Delta_{\text{dropout}} = -0.007918$). | **VALID ABLATION** |
| **2026-10-30 ~ 2026-10-31** | RQ1-TRN-03A Steps 09–12 | Cross-Fold Baseline R training (Folds 2, 3, 4) on Kaggle. | Establish speaker-independent 4-fold generalization benchmarks. | Warm-starting from Fold 1 checkpoint. | Independent cold starts from pretrained Wav2Vec2 (`seed: 1337`). All folds completed 3,600 updates on Kaggle; mirrored to HF `minkhoaa/langfens-rq1-cross-fold-checkpoints`. Macro AP reached 0.279207. | **VALID CROSS-FOLD** |
| **2026-11-01 ~ 2026-11-02** | RQ1-DIAG-01 & DIAG-02B | Diagnose omission performance chasm (0.0305 omission AP). | Investigate why Binary MDD misses 97.46% of omissions and test counterfactual acoustic sensitivity. | Unconditional acceptance of DIAG-02 causal claims. | DIAG-02 forced-attention claims superseded by DIAG-02B. Proved DIAG-02 forced-attention counterfactual was invalid; CTC blank discrimination is modest (AUC ~0.653); training GO revoked. | **VALID DIAGNOSTIC** |
| **2026-11-03** | Ecosystem Forensic Reconstruction | Reconstruct full research provenance across speech, microservice backend, and frontend. | Audit repository refs, thesis alignment, remote storage, and execution contracts. | Rely on secondary agent summaries or unexecuted test claims. | Proved RQ2 original scope is educational exercise generation (not software tests); audited Drive/Colab ledger; confirmed 0.00 Colab CCU balance. | **ACTIVE RECONSTRUCTION** |

---

## 3. Architecture & Intent Evolution

### 3.1 Thesis Proposal Conformance: Planned Scope vs Current Guidance
- **Original Research Blueprint (`DC KLTN0127`, signed 2026-08-18):**
  - **Vietnamese Topic:** *Phát triển hệ thống học tiếng Anh tích hợp trợ lý AI hội thoại và tự động tạo nội dung bằng đa tác nhân*.
  - **English Topic:** *Developing an English Learning System Integrated with Conversational AI Assistant and Multi-Agent Automated Test Generation*.
  - **Five Implementation Stages:**
    - Giai đoạn 1 (09/2026): Problem survey, requirements analysis, system architecture, database design.
    - Giai đoạn 2 (10/2026): Core web features, conversational AI assistant (STT Whisper, TTS), AI runtime environment, Multi-Agent orchestration harness.
    - Giai đoạn 3 (11/2026): Pronunciation analysis module, Knowledge Base & RAG, Writing/Grammar support, content generation/review/validation components.
    - Giai đoạn 4 (Đầu–giữa 12/2026): Multi-Agent evaluation-repair-verification integration, representative exercise sets (IELTS reading comprehension & grammar).
    - Giai đoạn 5 (Cuối 12/2026–01/01/2027): Controlled comparative experiments (Speaking and Multi-Agent: Single LLM vs Self-Review vs Multi-Agent), thesis documentation.
- **RQ1 and RQ2 Formulation in Thesis:**
  - **RQ1 (Speaking & Acoustic MDD):** Measures the effectiveness of phoneme-level acoustic representations (Wav2Vec2, GOP, forced alignment) in identifying mispronunciations on reference-labeled corpora (L2-ARCTIC), and evaluates pedagogical feedback quality.
  - **RQ2 (Multi-Agent Automated Content Generation):** Investigates whether specialized task decomposition (Coordinator, Generator, Reviewer, Validator with independent blinded answering) improves educational content quality over Single-LLM and Sequential Self-Review, evaluating quality vs time/token cost tradeoffs.
- **Remediation of RQ2 Scope Misconception (Coordinator C-01):**
  - A previous child subagent incorrectly assumed `RQ2 Multi-Agent Automated Test Generation` referred to *software test automation* (e.g. test harnesses or unit tests), asserting that the IELTS Reading generation harness was out of scope.
  - Line 14 of the thesis proposal definitively refutes this: RQ2 explicitly specifies *educational test and exercise generation* (tạo bài tập đọc hiểu và ngữ pháp). The IELTS Reading generation harness in the backend repository directly embodies the intended RQ2 architecture.
- **Scope Demarcation in `AGENTS.md`:**
  - `AGENTS.md` mandates: "SpeechOcean762 cross-corpus evaluation and pedagogical pronunciation feedback belong to RQ1. RQ2 is reserved exclusively for Multi-Agent Automated Test Generation." This faithfully reflects the UIT proposal.

### 3.2 Compute Platform Migration: From Google Colab to Kaggle GPU
1. **Initial Paradigm (Colab T4 + Google Drive FUSE):**
   - In RQ1-TRN-01B through RQ1-TRN-02A, training ran on Colab Tesla T4 with per-epoch FUSE synchronization directly to Google Drive (`/content/drive/MyDrive/KLTN-Dataset`).
   - This provided simple persistent storage for `best.pt`, `last.pt`, and `history.json`.
2. **The Colab Operational Deadlock (RQ1-TRN-02B Step 08):**
   - Account compute unit balance reached **`0.00 compute units`**.
   - Automated non-interactive staging via `colab drivemount` failed due to the `/dev/tty` requirement in `colab_cli`, blocking automated Drive staging.
3. **The Historical CPU Fallback Debate (RQ1-TRN-02B Step 09):**
   - The team formally analyzed whether to fall back to Colab CPU FP32 (no AMP).
   - Static analysis demonstrated that CPU training would require 18–36 seconds per update (total >36 hours), exceeding safe runtime limits, risking uncompleted sessions, and introducing numerical confounders against GPU FP16 AMP.
   - **The Strict Fallback Prohibition Policy was enacted in `AGENTS.md`**: "Strict fallback prohibition: Never fall back to local GPU/CPU training when remote GPU platforms are unavailable. Local training remains strictly prohibited. Stop and report blockers."
4. **The Kaggle + Hugging Face Paradigm (RQ1-TRN-02B Step 10 onwards):**
   - Migrated GPU execution to Kaggle Tesla T4x2 (30h/week free quota).
   - Pinned training source commits: `audit_scripts/kaggle_fold_train_exec.py` and active Kaggle notebooks pin `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`.
   - Because Kaggle VM disk (`/kaggle/working`) is ephemeral (20 GB limit, wiped on completion) and lacks Google Drive FUSE mounting, persistence was re-engineered:
     - Dataset staging: Private Kaggle Datasets (`minkhoaa/langfens-rq1-fold*-staging`).
     - Checkpoint mirroring: Authenticated atomic API upload to private Hugging Face repositories (`minkhoaa/langfens-rq1-trn-02b-checkpoints` and `minkhoaa/langfens-rq1-cross-fold-checkpoints`) with SHA-256 readback.

---

## 4. Latest Actual State & Historical Commitments

### 4.1 Google Drive Artifact Inventory

| Asset Name | Canonical Role | Google Drive Identifier / Path | Cryptographic SHA-256 Digest | File Size (Bytes) | Verification Status |
|---|---|---|---|---|:---:|
| `KLTN-Dataset` | Persistent Root Folder | Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK` | Directory | N/A | **VERIFIED (rclone lsd)** |
| `l2arctic_release_v5.0.zip` | Raw L2-ARCTIC Archive | Folder `KLTN` & `KLTN-Dataset/datasets/L2-ARCTIC/original/` | `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda` | 2,875,876,825 | **VERIFIED (presence by rclone)** |
| `speechocean762.tar.gz` | Raw SpeechOcean762 Archive | `KLTN-Dataset/datasets/SpeechOcean762/original/` | `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5` | 1,180,696,576 | **CORROBORATED** |
| `KLTN-Dataset.ipynb` | Historical Colab Notebook | `kltn_gdrive:KLTN-Collab/KLTN-Dataset.ipynb` | Unknown (unfetched) | N/A | **VERIFIED (presence by rclone)** |
| `fold1_train_val_only.zip` | Authorized Fold 1 Audio | Drive File ID: `1xdypDWopd585OjuQeHjxlWOJfU2oiJiY` | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | 761,922,918 | **CORROBORATED** |
| `l2arctic_4fold_splits.json` | 4-Fold Split Manifest | Drive File ID: `1vlQtqZqGAQUdEY0vQxGLZW3U7C9cdfbQ` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | 448,900 | **VERIFIED (Local hash)** |
| `best.pt` | Production Baseline R | Drive File ID: `10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | 412,366,029 | **CORROBORATED** |
| `last.pt` | Epoch 11 Endpoint Weights | Drive File ID: `1VdlhoQHfd9MqjzIkMUEqdqMxhLlJiO-s` | `bf46fab366363d46082082cf7f088b218f6d8d12851f5a9de706276b789efcb6` | 412,367,181 | **CORROBORATED** |
| `BASELINE_R_ARTIFACT_MANIFEST.json` | Remote Release Ledger | Drive File ID: `1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP` | `6222c92b63ea58f584267b39fb3ffc51394562086f90de21bbab8c8c1abe1e22` | 7,486 | **CORROBORATED** |
| `history.json` | Baseline R 11-Epoch History | Drive File ID: `128mkF6EFTHl7ztiVVy56rUEwmoqlKy-M` | `270dab59626bf665c0e52964b6f3c5a90e46fbaede856df987046c641dbceff5` | 8,819 | **CORROBORATED** |
| `EVALUATION_REPORT.json` | Baseline R Evaluation Report | Drive File ID: `1vehhJp8Q23uIFY7hIEtjrIZ_czFki1wV` | `e4d093dd48bcfa9ecc0c7a2f452a5db9a86ff92b7e6702099cc504c85611778f` | 5,611 | **CORROBORATED** |
| `BASELINE_COMPARISON.json` | Baseline R Comparison Report | Drive File ID: `1PR5HLpyCJMNYM-TblNqTOPnpg0EQaNmQ` | `e08f4328115e968b8dd2f18979e660ef5a89d884b2e3e6449ef6b050597f11f9` | 1,227 | **CORROBORATED** |
| `model.safetensors` | Safetensors Export | Hugging Face: `langfens-rq1-fold1-baseline-r` | `a5b9d41f1431ad7a8b9718b4d84a7886e39fe5aff45fe742fbe572819f8babc1` | 389,650,424 | **CORROBORATED** |
| `L2Arctic-KoelLabs` | KoelLabs Phoneme Alignments | `KLTN-Dataset/datasets/L2Arctic-KoelLabs/original/` | Unknown (multi-file archive) | N/A | **VERIFIED (presence by rclone)** |

### 4.2 Live Colab & Compute State
- **CLI Quota (`colab usage`):** Balance is **0.00 CCU**. Automated execution cannot provision GPU runtimes without interactive CCU payment (`colab pay`).
- **Active Sessions (`colab sessions`):** 0 active sessions running.
- **Local Fallback Prohibition:** Active and binding under `AGENTS.md`. No local GPU/CPU training updates permitted.

---

## 5. Contradictions & Discrepancies

1. **RQ2 Scope Interpretation:**
   - *Conflict:* Prior child subagent (07-integration) asserted RQ2 was an unstarted "software test generation" effort and declared IELTS Reading generation harness out of scope.
   - *Resolution:* Primary UIT thesis proposal (`DC KLTN0127`, lines 14 & 18) explicitly specifies that RQ2 is automated *educational exercise generation* (IELTS reading comprehension questions and grammar exercises) evaluated across A (single LLM), B (self-review), and C (multi-agent). The reading harness directly addresses RQ2.
2. **Hugging Face Baseline R Visibility Drift:**
   - *Conflict:* Historical release manifest (`BASELINE_RELEASE_MANIFEST.md`) records `minkhoaa/langfens-rq1-fold1-baseline-r` as strictly private (`is_private: true`).
   - *Resolution:* Parent direct observation (`HF_METADATA_OBSERVATION.json`) reveals the repository is currently public (`private: false`). The C0/C1 and cross-fold checkpoint stores remain strictly private.
3. **Kaggle Kernel Visibility Metadata:**
   - *Conflict:* Internal training guidelines specify `is_private: true` across all Kaggle notebooks.
   - *Resolution:* Live API query receipts captured by coordinator report `is_private: false` for certain pushed kernel slugs, while all four attached staging datasets (`minkhoaa/langfens-rq1-fold*-staging`) remain private.
4. **Execution Platform Divergence (Thesis vs Implementation):**
   - *Conflict:* Original thesis proposal planned Python FastAPI as the AI microservice and did not specify external cloud training infrastructure.
   - *Resolution:* Training scale required remote GPU platforms (Colab T4 initially, then Kaggle Tesla T4x2), creating a separation between offline model training pipelines (Kaggle/HF) and online web runtime microservices (.NET 8 / FastAPI).
5. **Hardware Topology Discrepancy (Single-GPU vs Dual-GPU):**
   - *Conflict:* Initial Colab configs (`configs/rq1_trn_02a_fold1.yaml`) targeted single Tesla T4 GPUs (`device: cuda:0`).
   - *Resolution:* Kaggle provisioned dual Tesla T4 GPUs (`NvidiaTeslaT4` shape). To preserve deterministic batch size and gradient accumulation without distributed data parallel complexity, project drivers enforced sequential single-GPU execution (`audit_scripts/kaggle_fold_train_exec.py`).

---

## 6. Claim / Evidence Registry

| ID | Claim Statement | Evidence Level | Verifying Source & Line / Command |
|---|---|:---:|---|
| **CLR-01** | Original thesis topic is "Developing an English Learning System Integrated with Conversational AI Assistant and Multi-Agent Automated Test Generation" | **VERIFIED** | `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md:3` |
| **CLR-02** | RQ2 is defined as Multi-Agent educational content/test generation (reading comprehension & grammar), not software tests | **VERIFIED** | `DC KLTN0127...md:14` (Sections 2.2(e), 2.3 RQ2, 4(e)) |
| **CLR-03** | Google Drive root folder `KLTN-Dataset` exists at Folder ID `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK` | **VERIFIED** | `rclone lsd kltn_gdrive:` & `audit_scripts/colab_launch_workflow.py:35` |
| **CLR-04** | Remote Google Drive contains `KLTN-Collab/KLTN-Dataset.ipynb` and `KLTN/l2arctic_release_v5.0.zip` | **VERIFIED** | `rclone lsf --max-depth 1 kltn_gdrive:KLTN-Collab` & `rclone lsf kltn_gdrive:KLTN` |
| **CLR-05** | Colab compute quota is currently 0.00 compute units with 0 active sessions | **VERIFIED** | Live execution: `colab usage` and `colab sessions` |
| **CLR-06** | Baseline R weights `best.pt` match SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **CORROBORATED** | `manifests/RQ1-ARTIFACT-01/BASELINE_RELEASE_MANIFEST.md:40` & `docs/process-checkpoints/INDEX.md:374` |
| **CLR-07** | Automated Colab Drive mounting fails due to missing `/dev/tty` in agent subshells | **CORROBORATED** | `manifests/RQ1-TRN-02B/checkpoints/08-live-gpu-runtime-and-staging-verification.md:162-171` |
| **CLR-08** | Colab CPU FP32 training fallback was formally evaluated and rejected | **CORROBORATED** | `manifests/RQ1-TRN-02B/checkpoints/09-colab-cpu-fp32-feasibility-static-gate.md:1-120` |
| **CLR-09** | Remote training shifted from Colab to Kaggle T4x2 with Hugging Face Hub atomic checkpoint mirroring | **CORROBORATED** | `manifests/RQ1-TRN-02B/checkpoints/10-kaggle-t4x2-migration-feasibility-static-review.md` & `AGENTS.md` |
| **CLR-10** | Local GPU/CPU training remains strictly prohibited by project policy | **VERIFIED** | `AGENTS.md:13-14` |

---

## 7. Unresolved Questions & Exact Safe Restart Instructions

### 7.1 Unresolved Questions
1. **Public Visibility of Baseline R Model on Hugging Face:** `minkhoaa/langfens-rq1-fold1-baseline-r` was observed as public (`private: false`), whereas repository policy requires private storage for derived assets under CC BY-NC 4.0. Operator confirmation is required to determine whether this was intentional.
2. **Drive Remote Access Throttling:** `rclone` calls against `kltn_gdrive:` frequently time out due to Google Drive shared API client rate limiting. A dedicated Google Drive client ID is recommended if high-throughput Drive auditing is required.
3. **RQ2 Experimental Validation Execution:** While the IELTS Reading Multi-Agent generation harness exists in the backend repository, formal three-way experimental comparison (Single LLM vs Sequential Self-Review vs Multi-Agent across 25–40 requests with cost/time metrics) remains unexecuted.

### 7.2 Exact Safe Restart Instructions
1. **Never initiate local training:** Maintain strict compliance with `AGENTS.md`. Do not execute backward passes, optimizer updates, or local training scripts on CPU or local GPU.
2. **Do not run interactive Colab auth:** Never invoke `colab drivemount`, `colab auth`, or `colab repl` in automated subagent runs.
3. **Verify Git state:** Confirm HEAD on `prep/rq1-shared-base` (`1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6`) and preserve all unstaged changes across all repositories.
4. **Consult Master Registry:** Review `docs/process-checkpoints/INDEX.md` and `docs/process-checkpoints/context-reconstruction-2026-11-03/PROCESS_CHECKPOINT.md` before taking any engineering actions.
