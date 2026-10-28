# LANGFENS — RQ1-TRN-02B: Colab Execution Authorization Corrections (Revision R1)

**Document ID:** `RQ1-TRN-02B-COLAB-AUTH-R1-CORRECTIONS`<br>
**Recorded at:** 2026-10-28T10:45:00+07:00<br>
**Author:** Senior Speech ML Engineer, Colab Infrastructure Engineer & Reproducibility Auditor<br>
**Review Branch:** `fix/rq1-trn-02b-colab-auth-r1`<br>
**Parent Documentation Commit:** `aa6415a8d3a073a3550f209a53e1baa44461ce97`<br>
**Preflight R2 Base Commit:** `8486cbb3596a0f5d2603dcdf46fd891b5c849840`<br>
**Operational Status:** `CORRECTED_PENDING_RUNTIME_VERIFICATION`

---

## 1. Executive Summary & Audit Findings

A rigorous reproducibility and operational safety audit of the historical execution packet (`manifests/RQ1-TRN-02B/COLAB_EXECUTION_AUTHORIZATION_PACKET.md` at commit `8486cbb3596a0f5d2603dcdf46fd891b5c849840`) identified four critical operational and provenance discrepancies prior to GPU runtime allocation:

1. **Defect 1 — Scientific Provenance Corruption in Documentation:**
   Section 2.3 of the packet reported a corrupt Fold 1 TRAIN speaker partition list (`ABA`, `BWC`, `ERMS`, `HJK`, `HKK`, `LXC`, `MBMD`, `NCC`, `PNV`, `RRBI`, `SKA`, `THV`, `TLV`, `TNI`, `TXHC`, `YDCK`). This list erroneously incorporated official SEALED TEST speakers (`ABA`, `HJK`, `LXC`, `PNV`), validation speaker `RRBI`, and a misspelled speaker name (`MBMD`). The underlying JSON split manifest (`manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`) was verified to be correct and untouched, requiring a formal documentation correction and disjointness proof.

2. **Defect 2 — Self-Referential Phantom Commit SHA:**
   The packet header referenced an invalid embedded Git commit SHA (`96729fe10153bf143b9c15eada477a0c0ba0c89c`), created prior to committing. Provenance levels (scientific baseline, preflight R2, registry backfill, operational corrections, and future execution commit) were conflated.

3. **Defect 3 — Unsupported Colab CLI Syntax:**
   The execution command sequence proposed running arbitrary shell command strings via `colab exec -s S -- "git clone ... && ..."`. Colab CLI v0.7.4 does NOT support positional shell command strings. Running this command results in a CLI parsing failure or silent failure. Additionally, the default timeout for `colab exec` is 30.0 seconds, which causes long-running processes to terminate prematurely unless `--timeout` is explicitly provided.

4. **Defect 4 — Google Drive Staging & Mounting Flaws:**
   The historical packet assumed `/content/drive/MyDrive/` is mounted automatically on fresh Colab VMs. In reality, Google Drive FUSE mounting via `colab drivemount` is an interactive browser/OAuth flow that requires terminal interaction by the human operator. Furthermore, the path in the old script looked for `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` instead of the canonical Google Drive path under `KLTN-Dataset` (`1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`), and failed to stage the baseline checkpoint `best.pt`.

Revision R1 resolves all four defects with tested, verified scripts and updated documentation.

---

## 2. Scientific Provenance Corrections & Disjointness Proof

### 2.1 Authoritative Split Manifest
- **Manifest Path:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`
- **Cryptographic SHA-256:** `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`
- **Corpus Census:** 3,599 utterances across 24 speakers, 119,151 canonical tokens, 17,518 errors (14.70% error rate).

### 2.2 Corrected Fold 1 Partitions
- **TRAIN Partition:** Exactly 2,399 utterances across **16 speakers**:
  ```text
  BWC, ERMS, HKK, HQTV, MBMPS, NCC, SKA, SVBI, THV, TLV, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA
  ```
- **VALIDATION Partition:** Exactly 300 utterances across **2 speakers**:
  - `NJS`: Utterances 0 to 149 (150 utterances)
  - `RRBI`: Utterances 150 to 299 (150 utterances)
- **SEALED TEST Partition:** Exactly 900 utterances across **6 speakers**:
  ```text
  ABA, ASI, EBVS, HJK, LXC, PNV
  ```

### 2.3 Mathematical Proof of Speaker Disjointness
Let $S_{\text{train}}$, $S_{\text{val}}$, and $S_{\text{test}}$ denote the speaker sets for Fold 1:
$$S_{\text{train}} \cap S_{\text{val}} = \emptyset$$
$$S_{\text{train}} \cap S_{\text{test}} = \emptyset$$
$$S_{\text{val}} \cap S_{\text{test}} = \emptyset$$
$$|S_{\text{train}}| + |S_{\text{val}}| + |S_{\text{test}}| = 16 + 2 + 6 = 24$$
$$N_{\text{train}} + N_{\text{val}} + N_{\text{test}} = 2,399 + 300 + 900 = 3,599$$

Automated mathematical verification of speaker disjointness and utterance totals is codified and verified in `tests/test_rq1_trn_02b_colab_launch.py::test_fold1_speaker_disjointness_and_counts`.

### 2.4 Official Test Sealing Audit
The authorized audio archive `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`) was inspected:
- It contains exactly 18 speaker directories corresponding to $S_{\text{train}} \cup S_{\text{val}}$.
- All 6 test speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`) are strictly absent ($\text{Leakage} = \emptyset$).
- Verified in `tests/test_rq1_trn_02b_colab_launch.py::test_authorized_zip_contains_zero_test_speakers`.

---

## 3. Commit Provenance Reconciliation

To prevent phantom commit references and maintain an auditable chain of custody, the Git commit hierarchy is formally classified into six distinct tiers:

```
[Tier 1: Frozen Scientific Baseline]
  caa5678 (Fold 1 Colab T4 baseline training run)
    |
  c4a658f (Protocol Z zero-oracle numerical closure & evaluation verification)
    |
  a200b73 (RQ1-ARTIFACT-01 Drive archival & private Hugging Face release)
    |
[Tier 2: Preflight R2 Implementation]
  8486cbb (Preflight R2 safety closure, unique run-ID, fail-closed guards)
    |     (Erratum: embedded phantom SHA 96729fe... revoked)
    |
[Tier 3: Checkpoint Registry Base]
  aa6415a (Global process checkpoint registry backfill & AGENTS.md policy)
    |
[Tier 4: Operational Correction Review Branch]
  fix/rq1-trn-02b-colab-auth-r1 (current feature branch)
    |
[Tier 5: Future Immutable Execution Commit]
  <IMMUTABLE_GIT_COMMIT_SHA> (to be pinned upon operator review & branch merge)
```

**Execution Invariant:** Prior to launching C0 or C1 training, the exact Git commit SHA must be determined. Colab execution must check out this single immutable SHA, never a floating branch name like `fix/rq1-trn-02b-preflight-r2` or `main`.

---

## 4. Colab CLI Verification & Command Protocol Repair

### 4.1 CLI Inspection Findings (Colab CLI v0.7.4)
Inspection of `colab --help`, `colab exec --help`, `colab run --help`, and `colab skill`:
1. `colab exec` executes code inside the session's live Python kernel.
2. It accepts `-f <file>` (local Python script read and sent to remote VM) and `--env KEY=VALUE`.
3. It does NOT accept positional command strings (`colab exec -s S -- "cmd"` is invalid).
4. The default execution timeout is **30.0 seconds**. Training takes 3.5–4.5 hours per arm, so omitting `--timeout` causes immediate timeout failure.
5. `colab run` is an ephemeral runner (`new` + `exec` + `stop`). For multi-stage workflows involving interactive Drive mounting, `colab new`, `colab exec`, and `colab stop` provide the required lifecycle isolation.

### 4.2 Implemented Operational Package
Three verified scripts have been implemented and regression-tested:

1. **`audit_scripts/colab_stage_and_verify.py`:**
   - Runs in Colab VM kernel via `colab exec -s <session> -f audit_scripts/colab_stage_and_verify.py --timeout 1800`.
   - Validates ARM (`C0` or `C1`).
   - Verifies target commit SHA against local git clone (`git rev-parse HEAD`).
   - Discovers KLTN-Dataset root in Google Drive. Fails closed with instructions if unmounted.
   - Stages authorized dataset archive to `/content/data/fold1_train_val_only.zip` and verifies SHA-256 (`b750eb3...`).
   - Stages baseline weights to `/content/checkpoints/RQ1-TRN-02A/recovered/best.pt` and verifies SHA-256 (`a2445c8...`).
   - Prepares Drive output directory, verifies write access, and enforces fail-closed checks against pre-existing conflicting checkpoints.

2. **`audit_scripts/colab_train_exec.py`:**
   - Runs in Colab VM kernel via `colab exec -s <session> -f audit_scripts/colab_train_exec.py --timeout 28800`.
   - Verifies CUDA availability. Strictly prohibits CPU training fallback.
   - Verifies git commit SHA matches target commit.
   - Verifies local staged files exist and match cryptographic hashes.
   - Launches `train_rq1_trn_02b.py` with explicit local paths and remote Drive output directory.
   - Captures full stdout/stderr to local log and syncs to Google Drive.
   - Verifies remote output checkpoints (`last.pt`, `fixed_budget_checkpoint.pt`) and history on Drive upon completion.

3. **`audit_scripts/colab_launch_workflow.py`:**
   - Local validation and orchestrator tool.
   - Validates local git cleanliness and commit SHA.
   - Queries `colab usage` and parses compute unit balance.
   - Generates verified, copy-pasteable CLI execution sequences with `--env` and `--timeout`.

---

## 5. Google Drive Architecture & Staging Protocol

### 5.1 Storage Ledger Under `KLTN-Dataset`
Mandatory project Drive root: `KLTN-Dataset` (Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`)
- **Baseline R Folder:** `https://drive.google.com/drive/folders/11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx`
- **Baseline R Weights:** `manifests/RQ1-TRN-02A/checkpoints/best.pt`
  - File ID: `10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I`
  - SHA-256: `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`
- **Authorized Dataset Archive:** `manifests/RQ1-TRN-02A/authorized_data/fold1_train_val_only.zip`
  - File ID: `1xdypDWopd585OjuQeHjxlWOJfU2oiJiY`
  - SHA-256: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`

### 5.2 Persistent Output Hierarchy
Output checkpoints for the upcoming experiment will be isolated under:
- Control Arm C0: `manifests/RQ1-TRN-02B/checkpoints/C0/`
- Treatment Arm C1: `manifests/RQ1-TRN-02B/checkpoints/C1/`
- Optional Run-ID namespacing: `manifests/RQ1-TRN-02B/checkpoints/<ARM>/<RUN_ID>/`

### 5.3 Interactive Drive Mount Protocol
Because `colab drivemount` initiates an interactive OAuth consent flow, it cannot be run unattended by an automated script. The staging protocol explicitly models this:
1. Operator runs `colab new -s langfens-rq1-c0 --gpu T4`.
2. Operator runs `colab drivemount -s langfens-rq1-c0` in their terminal once (or opens `colab url -s langfens-rq1-c0` in a browser).
3. All subsequent staging and training commands execute non-interactively via `colab exec -s ... -f script.py`.

---

## 6. Colab Resource Readiness & Quota Blocker

Read-only inspection via `colab usage` and `colab sessions` executed on 2026-10-28:
- **CLI Version:** `0.7.4`
- **Authentication Provider:** OAuth2 (`tukhoa040505@gmail.com`), token valid.
- **Measured Balance:** `0.00 compute units`
- **Usage Rate:** `0.00/hr`
- **Active Sessions:** `0` (`[colab] No active sessions found on server.`)
- **Estimated Requirement:**
  - Control Arm C0: 6 epochs $\times$ ~35–45 min $\approx$ 3.5–4.5 T4 GPU-hours.
  - Treatment Arm C1: 6 epochs $\times$ ~35–45 min $\approx$ 3.5–4.5 T4 GPU-hours.
  - Total: 7.0–9.0 T4 GPU-hours ($\approx$ 15–20 compute units on Google Colab standard T4 tier).
- **BLOCKER:** Available compute units are zero. The operator must purchase/assign compute units before any runtime provisioning.

---

## 7. Verification Evidence & Regression Testing

### 7.1 Regression Test Census
- **18 New Unit Tests:** Added in `tests/test_rq1_trn_02b_colab_launch.py` covering arm validation, commit SHA validation, usage parsing, quota evaluation, command generation syntax, strict GPU enforcement, staging fail-closed guards, conflicting checkpoint detection, and Fold 1 speaker disjointness.
- **Full Repository Test Suite:** **175 passed, 4 skipped** (0 failures).

### 7.2 Static Verification of Training Entrypoint
Preflight dry-run of `train_rq1_trn_02b.py` verified with `--initial-checkpoint` override support:
- `configs/rq1_trn_02b_c0.yaml` (`--dry-run`): **PASS** (Manifest SHA, Initial Checkpoint SHA, AdamW/Cosine cold reset verified).
- `configs/rq1_trn_02b_c1.yaml` (`--dry-run`): **PASS** (Manifest SHA, Initial Checkpoint SHA, AdamW/Cosine cold reset verified).

---

## 8. Operational Decision Gate

```text
================================================================================
GATE: RQ1-TRN-02B-COLAB-AUTHORIZATION
STATUS: CORRECTED_PENDING_RUNTIME_VERIFICATION

All operational scripts, command syntaxes, data hashes, and documentation defects
have been corrected, static-inspected, and regression-verified (175 tests pass).

REMAINING OPERATOR ACTIONS:
1. Top-up / allocate ~15-20 Colab compute units (colab pay).
2. Pin the final approved Git commit SHA from this branch.
3. Provision T4 session (colab new -s langfens-rq1-c0 --gpu T4).
4. Authorize Google Drive mount (colab drivemount -s langfens-rq1-c0).
5. Execute non-interactive staging script (audit_scripts/colab_stage_and_verify.py).
6. Execute non-interactive training script (audit_scripts/colab_train_exec.py).

GPU training backward passes remain strictly BLOCKED until these operator actions
are completed.
================================================================================
```
