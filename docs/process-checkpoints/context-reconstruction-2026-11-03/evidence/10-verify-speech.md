# Forensic Reconstruction & Independent Verification: SEAM 10 — Speech Research, Experiment Provenance & Intent Cross-Check

**Auditor / Role:** Independent Read-Only Systems & Research Forensics Auditor  
**Date of Observation:** 2026-11-03 (Discovery normalized: Asia/Ho_Chi_Minh, +07:00; receipts timestamped 23:45–23:55)  
**Authoritative Artifact Path:** `/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/ea49dbe6-c72f-494d-b433-77701f7d9e45/evidence/10-verify-speech.md`  
**Primary Repositories & Examined HEADs:**  
- Speech Research: `/home/khoa/Projects/KLTN/Dataset` (`prep/rq1-shared-base` @ `1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6`, dirty: `INDEX.md`, nested `repo`, untracked reconstruction docs)  
- Backend Microservice: `/home/khoa/Projects/langfens/Project_Langfens_Microservice` (`generation-harness` @ `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48`, dirty)  
- Frontend Application: `/home/khoa/Projects/langfens/langfens-fe-app` (`autonomous-generation` @ `2d9fe94a54e65680eca56aafbe40e691018113da`, dirty)  
**Assigned Mission:** Independent cross-check of research methodology, remote compute evolution, experimental provenance, and intent reconstruction against primary source files, Git trees, and live remote receipts.  

---

## 1. Scope, Source Inventory & Access Boundaries

### 1.1 Directly Observed Primary Sources & Artifact Counts
1. **Local Repository Tracked/Untracked Inventory:** 5,196 paths indexed in `SOURCE_FILE_INVENTORY.tsv` across the three local working checkouts.
2. **Thesis Proposal Document (`DC KLTN0127`):** Primary graduation thesis proposal (`/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`), signed 2026-08-18 by advisor ThS. Trần Thị Hồng Yến. Lines 1–18 directly inspected, establishing authoritative thesis scope.
3. **Literature Reference Library:** 12 research works inventoried in `manifests/RESEARCH_LIBRARY.md`:
   - *8 Verified Accessible Local PDFs:* Baevski et al. (2020, SHA-256 `4a881708...`), Zhang et al. (2021, `5edd602a...`), Zhao et al. (2018, `03a4b836...`), Peng et al. (2021, `8c56fc60...`), Gong et al. (2022, `44c5f27e...`), Shahin et al. (2023/2025, `202d722d...`), Radford et al. (2023, `84496ed0...`), Tu et al. (2025/2026, `2eb40576...`).
   - *4 Access-Restricted / Paywalled Works:* Witt & Young (2000, Elsevier paywall, bibliographic entry only); Amrate & Tsai (2024, Cambridge Core paywall); Joint-Align IEEE 420 (IEEE Xplore paywall); Cao et al. (2026, code repo license unresolved, clean-room reimplemented under `ctc-union-language-v1`).
4. **Direct Remote Receipts (Read-Only API Queries):**
   - *Google Colab:* `DRIVE_COLAB_OBSERVATION.json` records `colab usage` balance at **0.00 compute units**, active assignments = 0, sessions = 0.
   - *Kaggle Kernels & Datasets:* `KAGGLE_METADATA_OBSERVATION.json` and `KAGGLE_DATASET_METADATA.json` record 29 project kernels (P-SUP treatment & control status `COMPLETE`, server metadata `is_private: false`, pinned commit `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`) and 4 staging datasets (Folds 1–4, all strictly `is_private: true`, CC BY-NC 4.0).
   - *Hugging Face Hub:* `HF_METADATA_OBSERVATION.json` and `P_SUP_REMOTE_METADATA.json` record baseline repository `langfens-rq1-fold1-baseline-r` (currently public, `private: false`, rev `6c31f719...`) and private stores `langfens-rq1-trn-02b-checkpoints` (`private: true`) and `langfens-rq1-cross-fold-checkpoints` (`private: true`, rev `4aa9f2a6...`).
   - *GitHub Remote Visibility:* `GITHUB_METADATA_OBSERVATION.json` confirms all 3 primary checkouts and `minkhoaa/langfens-ielts-dataset` are currently public on GitHub.

### 1.2 Restricted / Inaccessible Sources
- Raw audio/annotation payloads in remote Google Drive archives and Hugging Face `.pt` tensors were NOT downloaded or executed (no inference, forward passes, or test set leakage).
- Google Drive automated subshell mounting via `colab drivemount` remains blocked by `/dev/tty` requirement.
- Paywalled literature full text (Witt & Young 2000, Amrate & Tsai 2024, Joint-Align IEEE 420) remains barred without institutional authentication.

---

## 2. Chronological Research & Decision Event Table

| Date / Time (+07:00) | Milestone / Phase | Goal | Decision & Rationale | Alternative Considered | Action, Result & Operational Limitation | Scientific Validity |
|---|---|---|---|---|---|:---:|
| **2026-08-18** | Thesis Sign-off | Formalize UIT graduation proposal | Establish RQ1 (acoustic MDD/feedback) and RQ2 (multi-agent educational test generation) | Monolithic LLM; software test generation | Approved by ThS. Trần Thị Hồng Yến. 5 stages (09/2026–01/2027) targeting IELTS reading/grammar. | **AUTHORITATIVE INTENT** |
| **2026-10-23 15:54** | RQ1-TRN-00 | Baseline paradigm selection | Adopt Wav2Vec 2.0 multi-task MDD to avoid forced-alignment collapse on non-native deletions ($D=3,420$) | Classical Kaldi GOP | Comparative design published (`BASELINE_EXPERIMENT_DESIGN.md`). Eliminated Kaldi C++ dependency. | **EMPIRICALLY VALID** |
| **2026-10-23 23:52** | RQ1-TRN-01B0 | 100-utt sanity check | Expose diagnosis metric flaw: 87.6% evaluated tokens were correct phones, masking true substitution accuracy of 3.23% (4/124) | Retain mixed 1,000-token diagnosis metric | Enforced strict 124-token denominator for substitutions (`SUBSTITUTION_DIAGNOSIS_METRIC_AUDIT.md`). | **VALID AUDIT** |
| **2026-10-25 19:30** | RQ1-TRN-01B1 | Mitigate CTC blank collapse | Introduce Variant D (cosine warmup 100 updates, peak lr 1e-3, frozen backbone) to restore phonetic emissions | Unfreezing context encoder; constant lr | Fully restored phoneme sequence length; greedy PER dropped to ~23%. Acoustic conditioning confirmed. | **VALID FOUNDATION** |
| **2026-10-25 22:34** | RQ1-TRN-02A | Fold 1 Baseline R Colab run | Enact **Two-Head Contract (CTC + Binary only)**; freeze diagnosis head (`lambda_diag = 0.0`) due to gradient interference | Full 3-head training on sparse substitutions | 11 epochs completed (6,600 updates) on Colab T4. Selected Epoch 6 / 3,600 updates (`best.pt`, AP 0.316062). | **AUTHORITATIVE BASELINE** |
| **2026-10-26 16:01** | RQ1-TRN-02A-EVAL | Zero-oracle deployment | Prohibit gold TextGrid timestamps at inference; evaluate Protocol O (oracle tokens, AP 0.317021) vs Protocol Z (G2P, AP 0.313530) | Supplying human TextGrid boundaries | Protocol Z aligned 9,919/9,956 tokens (99.63% coverage), proving zero-oracle feasibility. | **VALID PROTOCOL** |
| **2026-10-28 07:56** | RQ1-TRN-02B Step 08 | Colab GPU probe for C0/C1 | Provision Colab T4 for paired ablation; encountered 0.00 CCU balance and `colab drivemount` missing `/dev/tty` | Manual browser training | Session released cleanly; static audit rejected local CPU fallback; strict fallback prohibition codified in `AGENTS.md`. | **BLOCKED ON COLAB** |
| **2026-10-29 16:45** | RQ1-TRN-02B Steps 13–15 | C0/C1 Kaggle GPU ablation | Test prompt dropout ($p=0.25$) over 3,600 updates on Kaggle T4x2; primary contrast $\Delta = -0.007918$ (95% CI `[-0.0183, +0.0004]`) | Adopting C1 without paired contrast | Hypothesis NOT_SUPPORTED; prompt dropout degraded detection without expanding acoustic reliance. | **VALID ABLATION** |
| **2026-10-31 22:30** | RQ1-TRN-03A Steps 09–13 | 4-Fold cross-fold training | Train Folds 2, 3, 4 Baseline R independently from pretrained init (`seed: 1337`, Kaggle T4x2); macro AP 0.2792 across 39,576 tokens | Warm-starting from Fold 1 | Sealed test sets preserved; dual checkpointing (`fixed_budget.pt` and `best.pt`) mirrored to private HF repo. | **VALID BASELINE** |
| **2026-11-02 15:08** | RQ1-DIAG-02B | Counterfactual validity audit | Re-evaluate DIAG-02 forced attention with projection-correct MHA; disproved +0.25 causal drift claim (true drift was +0.023) | Accepting DIAG-02 causal training GO | **Issued STRICT NO-GO for causal training authorization**. Matched CTC blank AUC is only ~0.653; 9/80 archive parity fail. | **VALID AUDIT (NO-GO)** |
| **2026-11-02 17:15** | RQ-MODEL-01 | E1 Skip-MDD benchmark | Benchmark learnable acoustic skip token $\mathbf{s} \in \mathbb{R}^{768}$ on Kaggle GPU | Advancing to cross-fold rollout | **E1 FAILED:** Omission AP degraded -7.90% (0.0350 to 0.0323); skip mass on deletions completely collapsed to 0.000000. | **CLOSED (NO-GO)** |
| **2026-11-03 17:51** | RQ1-P-SCORE | Graph scoring benchmark | Benchmark segmentation-free GOP (`SF-SD` vs `SF-S`) on CPU log-emissions cache: Omission AP +0.0216, but Substitution AP -0.0243 | Deploying SF-SD as Baseline replacement | Fails noninferiority ($\ge -0.005$); $\epsilon$-deletion shortcut bypasses acoustic energy on substitutions. INCONCLUSIVE_TRADEOFF. | **VALID TRADEOFF** |
| **2026-11-03 23:45** | RQ1-P-SUP | Subtype supervision probe | Pinned Kaggle T4x2 kernels (`dd2dd89`); remote summaries report 3,600 updates / 0 skips; selection AP: Control 0.4676 vs Treatment 0.4650 | Inferring Fold 1 VAL outcome | Remote training completed on FIT (1,799 utts); final Fold 1 VAL subtype evaluation remains unperformed. | **AWAITING VAL EVAL** |

---

## 3. Architecture & Intent Evolution

### 3.1 Thesis Proposal Conformance: Clarification of RQ2 Scope
- **Primary Proposal Evidence (`DC KLTN0127` Line 14, signed 2026-08-18):** Vietnamese text explicitly defines RQ2 as *Multi-Agent Automated Content / Exercise Generation* (tạo bài tập đọc hiểu và ngữ pháp), evaluating whether specialized task decomposition (Coordinator, Generator, Reviewer, Validator with blinded answering) outperforms Single-LLM and Sequential Self-Review in content quality versus time/token cost.
- **Correction of Subagent Drift:** Previous child subagents incorrectly hypothesized that RQ2 referred to *software test automation* (e.g. pytest or test harness engineering). Primary proposal evidence refutes this; the IELTS Reading Multi-Agent Generation Harness in `Project_Langfens_Microservice` directly embodies the intended RQ2 architecture.
- **RQ1 vs RQ2 Boundary:** All acoustic processing, Wav2Vec 2.0 fine-tuning, GOP scoring, SpeechOcean762 cross-corpus evaluation, and pronunciation pedagogical feedback belong exclusively to RQ1.

### 3.2 Compute Platform Migration & Execution Discipline
1. **Colab Tesla T4 + Google Drive FUSE (RQ1-TRN-01B to TRN-02A):** Initial single-GPU training utilized direct FUSE mounting to `/content/drive/MyDrive/KLTN-Dataset`.
2. **Colab Deadlock (TRN-02B Step 08):** Account CCU quota reached **0.00 compute units**, and `colab drivemount` failed with `/dev/tty` missing in automation. Local CPU fallback was evaluated and rejected (estimated >36 hours, risking preemption and numerical confounding). `AGENTS.md` enacted the **Strict Fallback Prohibition Policy**, banning local GPU/CPU training updates.
3. **Kaggle Tesla T4x2 + Hugging Face Hub Mirroring (TRN-02B Step 10 onwards):** Training migrated to Kaggle dual Tesla T4 GPUs (30h/week free quota). Because Kaggle disk (`/kaggle/working`) is ephemeral, persistence was re-engineered: private Kaggle Datasets for immutable staging, and authenticated atomic API mirroring to private Hugging Face repositories with cryptographic SHA-256 readback.

### 3.3 Diagnostic Evolution & Inductive Bias Collapse
- **DIAG-01 (Root-Cause Audit):** Identified an omission performance chasm (Deletion AP collapses to 0.0305 vs 0.2643 for substitutions). Cross-attention mass is only 2.8% in-window, cycling over 6–7 global anchor frames >1.4s away.
- **DIAG-02 Fallacy & DIAG-02B Scientific Correction:** DIAG-02 claimed downstream MLPs collapse when forced to look at acoustic frames (+0.25 drift). DIAG-02B audited the code and proved the intervention bypassed learned MHA projections. When projection-correct MHA was evaluated, drift was only +0.023. DIAG-02B proved matched CTC blank discrimination is modest (AUC ~0.653) and issued a **STRICT NO-GO for causal training authorization**.
- **E1 Collapse:** Adding a static acoustic skip token $\mathbf{s} \in \mathbb{R}^{768}$ resulted in complete collapse: skip mass on deletions was exactly **0.000000**, and omission AP dropped -7.90%.
- **P-SCORE vs P-SUP:** P-SCORE demonstrated that post-hoc $\epsilon$-deletion graph scoring rescues omissions (+0.0216 AP) but severely degrades substitutions (-0.0243 AP, failing noninferiority). P-SUP was formulated to test whether 3-way subtype supervision (`operation_ce`) during training resolves this tradeoff against an exactly capacity-matched binary control (`binary_collapsed`, 266,667 parameters).

---

## 4. Latest Actual State & Historical Commitments

| Component / Subsystem | Historical Commitment / Registered Baseline | Latest Verified Actual State | Verification Evidence / Primary Receipt | Status / Compliance |
|---|---|---|---|:---:|
| **Local Working Trees** | Clean feature branches | Dirty working trees across all 3 checkouts; preserved intact | `watchdog_diff` & `SOURCE_DISCOVERY_SNAPSHOT.json` | **PRESERVED DIRT** |
| **Speech Research Checkpoint** | Epoch 6 / 3,600 updates `best.pt` (AP 0.316062) | Bit-identical across Drive, HF, and local caches | Drive File ID `10SBhwWd...`; SHA-256 `a2445c8d...` | **VERIFIED** |
| **Split & Data Isolation** | 4-fold split manifest; sealed test partitions | Manifest SHA-256 `7d6bfdf6...`; test audio excluded from staging | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | **VERIFIED** |
| **Colab Compute Allocation** | T4 GPU runtime for remote training | Balance is **0.00 CCU**; 0 active assignments | `DRIVE_COLAB_OBSERVATION.json` (`colab usage`) | **BLOCKED (0 CCU)** |
| **Kaggle Kernel Visibility** | Mandatory `is_private: true` under CC BY-NC 4.0 | Server metadata reports `is_private: false` for P-SUP notebooks | `KAGGLE_METADATA_OBSERVATION.json` | **VISIBILITY DRIFT** |
| **Kaggle Staging Datasets** | Private dataset containment | All 4 staging datasets verified strictly `is_private: true` | `KAGGLE_DATASET_METADATA.json` | **COMPLIANT** |
| **Hugging Face Model Visibility** | Private release for derived research weights | Baseline R repo is public (`private: false`); C0/C1 & Cross-fold private | `HF_METADATA_OBSERVATION.json` | **DRIFT NOTED** |
| **P-SUP Kernel Execution** | Staged `--no-run` awaiting Web UI secret attachment | Kernels status `COMPLETE`; source pins `dd2dd89` | `KAGGLE_METADATA_OBSERVATION.json` | **COMPLETED** |
| **P-SUP Training Provenance** | P-SUP 3,600 updates / 0 skips on Kaggle T4x2 | HF summaries confirm 3,600 updates / 0 skips at rev `4aa9f2a6` | `P_SUP_REMOTE_METADATA.json` | **PERSISTED (SELECT)** |
| **P-SUP Validation Evaluation** | Prospective Evaluation Contract V1 gates | Final Fold 1 VAL subtype evaluation unperformed / unverified | `COORDINATOR_VERIFICATION_NOTES.md` C-04 | **PENDING VAL EVAL** |

---

## 5. Contradictions & Discrepancies Discovered

1. **RQ2 Scope Misinterpretation:** Child subagent 07 claimed RQ2 was software test generation and declared the IELTS reading harness off-scope. Primary proposal (`DC KLTN0127` Line 14) confirms RQ2 is educational test/exercise generation; the reading harness directly addresses RQ2.
2. **P-SUP Operational State vs Registry:** Registry index records P-SUP as staged awaiting Web UI execution. Direct Kaggle API receipts confirm status `COMPLETE`, and authenticated Hugging Face metadata confirms both arms completed 3,600 updates with 0 skips at revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`.
3. **P-SUP Kernel Commit Pin:** Staging documentation and commit titles cite pinned commit `df55f125f5e0e26211986dae32a2dac78c364aaa`. Direct inspection of deployed notebook sources and local notebooks proves they pin `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`.
4. **Kaggle Notebook vs Dataset Privacy:** Internal guidelines mandate private execution. Direct Kaggle API queries return `is_private: false` for deployed P-SUP notebooks, whereas all 4 attached staging datasets remain private (`is_private: true`).
5. **Hugging Face Baseline R Visibility Drift:** `BASELINE_RELEASE_MANIFEST.md` records Baseline R as private. Live metadata query reveals `minkhoaa/langfens-rq1-fold1-baseline-r` is currently public (`private: false`).
6. **DIAG-02 vs DIAG-02B Causal Precedence:** DIAG-02 claimed causal proof of attention bypass (+0.25 drift) and issued a training GO. DIAG-02B proved the intervention bypassed learned projections (true drift +0.023) and issued a **STRICT NO-GO**. Furthermore, DIAG-02B failed archive parity on 9/80 utterances (max drift 0.001305), precluding training authorization.
7. **Overclaims in Research Literature Reviews (Lane 04/06):**
   - *Census Sufficiency Overclaim:* A census showing >2,300 omissions does not prove dataset sufficiency or architectural blame.
   - *Protocol Z Overclaim:* Aligning 9,919 G2P tokens proves zero-oracle offline feasibility, NOT production deployment readiness.

---

## 6. Material Claim-Verification Matrix

| ID | Material Claim Statement | Primary Source & Line / Receipt | Direct Evidence & Exact Quantities | Audit Verdict & Confidence |
|---|---|---|---|:---:|
| **CLM-01** | DIAG-02 causal proof is invalidated by projection-correct MHA; training gate is NO-GO | `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md:1-12, 105-118` | Forced-attention bypassed learned projections; corrected drift is +0.023 (not +0.25). Status: `PARTIAL; NO_GO_TRAINING_CAUSAL_GATE`. | **VERIFIED** |
| **CLM-02** | DIAG-02B fails archive parity on 9/80 utterances | `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md:46-48` | Maximum probability drift is **0.001304954** on HKK/arctic_a0105 exceeding 0.000200 tolerance. Sample AP drift: -0.000061292. | **VERIFIED** |
| **CLM-03** | Matched CTC blank discrimination on omissions is modest (AUC ~0.653), not reliable omission evidence | `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md:78-98` | 66 matched pairs: Mean P(blank) on omissions is 0.8244 vs 0.7322 on controls. Blank ROC-AUC is 0.653 (95% CI `[0.586, 0.746]`). | **VERIFIED** |
| **CLM-04** | CTC target excludes deletions, providing zero loss gradient on omissions | `src/langfens_prep/dataset.py:105-118, 132` | `chronological_realized_phones` explicitly excludes deletions; CTC loss is computed purely on realized phonemes. | **VERIFIED** |
| **CLM-05** | Protocol Z unaligned token count is exactly 37, decomposing into 25 correct, 4 substitutions, and 8 deletions | `manifests/RQ1-WORKSPACE-PREP-01/P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md:21, 59-61, 307` | Full canonical population $N=9,956$; aligned population $N=9,919$. Exactly 37 unaligned tokens mapped to wh-aspiration, contractions, and weak vowels. | **VERIFIED** |
| **CLM-06** | Evaluation denominators differ strictly: Binary (9,956), Omission (8,944), Substitution (9,811) | `manifests/RQ1-WORKSPACE-PREP-01/P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md:149-155` | Omission AP strictly excludes substitutions ($N=8,944$); Substitution AP strictly excludes deletions ($N=9,811$); PER ref $N=9,881$. | **VERIFIED** |
| **CLM-07** | P-SCORE benchmark completed on CPU cache, but fails prospective noninferiority on substitutions | `manifests/RQ1-WORKSPACE-PREP-01/P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md:23, 136-138, 183-186` | Omission AP $+0.021615$ (PASS), Recall@FPR15 $+0.182482$ (PASS), but Substitution AP $-0.024256$ (**FAIL** noninferiority $\ge -0.005$). INCONCLUSIVE. | **VERIFIED** |
| **CLM-08** | P-SUP remote kernels executed 3,600 updates with 0 skips, but Fold 1 VAL comparison remains unverified | `P_SUP_REMOTE_METADATA.json` & `COORDINATOR_VERIFICATION_NOTES.md` C-04 | HF summaries at rev `4aa9f2a6` confirm 3,600 updates / 0 skips. Selection AP: Control 0.4676 vs Treatment 0.4650 on SELECT (300 utts). VAL unperformed. | **VERIFIED (SELECT ONLY)** |
| **CLM-09** | Deployed P-SUP Kaggle notebooks pin dd2dd89, not documented df55f12 | `deploy/kaggle_psup_control_train/kaggle_train.ipynb:111` & `KAGGLE_METADATA_OBSERVATION.json` | Fetched notebook source confirms `PINNED_COMMIT_SHA = "dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f"`. | **VERIFIED** |
| **CLM-10** | Folds 2–4 Baseline R trained independently from pretrained init, with warm-starting strictly prohibited | `AGENTS.md:73-77` & `configs/rq1_trn_03a_fold2.yaml:1-12` | Pretrained `facebook/wav2vec2-base-960h` rev `22aad52...`, `seed: 1337`, cosine schedule over 25 epochs. Dual checkpoint accounting enforced. | **VERIFIED** |
| **CLM-11** | Prompt dropout ($p=0.25$) ablation hypothesis is NOT_SUPPORTED ($\Delta = -0.007918$) | `manifests/RQ1-TRN-02B/checkpoints/15-cpu-paired-scientific-evaluation.md:20-26, 80-85` | 3,600 updates on Kaggle T4x2: C1 vs C0 contrast $\Delta = -0.007918$ (95% CI `[-0.018328, +0.000385]`); acoustic gap change $\Delta = -0.006120$. | **VERIFIED** |
| **CLM-12** | E1 Skip-MDD completely collapsed on omissions (skip mass 0.000000; AP degraded -7.90%) | `manifests/RQ-MODEL-01/PROCESS_CHECKPOINT.md:21-26, 65-67` | Kaggle GPU benchmark: Omission AP dropped 0.0350 to 0.0323; skip mass on deletions was strictly 0.000000. NO-GO for cross-fold rollout. | **VERIFIED** |

---

## 7. Exact Corrections & Reconciliation

1. **Thesis Level Correction:** Primary proposal `DC KLTN0127` establishes this is an undergraduate graduation thesis (Khóa luận tốt nghiệp đại học), NOT a Master's thesis.
2. **RQ2 Scope Correction:** RQ2 is passage-grounded reading comprehension and grammar exercise generation by multi-agent coordination, NOT software testing.
3. **P-SUP Operational State Correction:** Kaggle kernels are not awaiting execution; both arms executed to completion (3,600 updates, 0 skips). However, this represents training completion and SELECT partition validation, NOT prospective Fold 1 VAL evaluation.
4. **Visibility Status Drift Corrections:**
   - `minkhoaa/langfens-rq1-fold1-baseline-r` is currently public (`private: false`).
   - Kaggle P-SUP kernels report `is_private: false`, while staging datasets remain private.
   - All 3 GitHub repositories are currently public.

---

## 8. Unresolved Questions & Exact Safe Restart Procedure

### 8.1 Unresolved Questions
1. **P-SUP Fold 1 VAL Subtype Evaluation:** Standalone CPU evaluation replay on Fold 1 VAL (`NJS`, `RRBI`, 300 TextGrids) comparing `T-OP` vs `B-CONTROL` fixed-budget checkpoints under Protocol O and Protocol Z has not been executed. Does subtype supervision resolve the omission-substitution tradeoff?
2. **DIAG-02B Cross-Backend Archive Parity:** What explains the 9/80 archived-score discrepancies (max drift 0.001305) between historical GPU checkpoints and CPU inference? Full 1,199-utterance forward replay remains unperformed.
3. **Public Visibility Intent:** Operators must confirm whether public visibility on GitHub, Hugging Face Baseline R, and Kaggle notebook metadata complies with academic licensing.

### 8.2 Exact Safe Restart Procedure
1. **Preserve All Checked-out Dirt:** Maintain HEAD `1fa4a5b` on `prep/rq1-shared-base` in speech, `2ba12e5` in backend, and `2d9fe94` in frontend. Do not reset, fetch, pull, or checkout.
2. **Strict Fallback Prohibition:** Maintain total prohibition on local GPU/CPU training updates or backward passes.
3. **Awaiting Operator Authorization for P-SUP VAL Replay:**
   - Execute standalone read-only CPU evaluation script on Fold 1 VAL (300 TextGrids) using Protocol Z CMUdict alignments.
   - Evaluate primary contrast `T-OP` vs `B-CONTROL` against prospective Evaluation Contract V1 criteria ($\Delta \text{AP}_{\text{omit}} \ge +0.030$, $\Delta \text{AP}_{\text{sub}} \ge -0.005$).
4. **Preserve Test Set Sealing:** Ensure sealed test partitions ($S_{\text{test}}^{(k)}$) are strictly excluded from all evaluations.