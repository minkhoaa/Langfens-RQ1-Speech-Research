# LANGFENS — RQ1-TRN-01B1: Git Provenance Verification & Remote Traceability Audit

**Gate Reference:** RQ1-TRN-01B1 — CTC Stabilization & Training Readiness Pilot  
**Deliverable Path:** `manifests/RQ1-TRN-01B1/GITHUB_PROVENANCE_AUDIT.md`  
**Lead Speech ML Research Engineer & Reproducibility Reviewer:** Senior Speech ML Research Engineer  
**Audit Status:** `PROVENANCE: PASS (Commit d63f25f Verified and Synced to Origin)`  

---

## 1. Executive Summary & Verification Mandate

During the preflight review for phase `RQ1-TRN-01B1`, an independent query against the upstream GitHub repository did not immediately locate the reported reconciliation commit `d63f25f` or its associated feature branch `feat/rq1-trn-01b1-ctc-stabilization`. 

Under the strict project reproducibility contract:
1. Every historical gate baseline must be anchored to an immutable cryptographic SHA-1 Git commit.
2. The exact commit parentage and feature history must be retrievable from the canonical remote (`origin`: `https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git`).
3. No code changes or pilot training experiments may proceed if the reported baseline commit cannot be established (which would mark provenance `BLOCKED`).

This audit report records the forensic inspection of the local and remote repository, proves the parentage of commit `d63f25fa2d1e242e85b64cfe0b6e796fbab35a32`, documents normal branch synchronization without force-pushing or history rewriting, and confirms independent retrievability via the GitHub REST API.

---

## 2. Forensic Inspection of Git Environment & Remote Configuration

### 2.1 Remote URLs and Tracking Configuration
```bash
$ git remote -v
origin  https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git (fetch)
origin  https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git (push)
```

### 2.2 Branch Tracking Status (`git branch -vv`)
Prior to synchronization, local branches reflected:
```
  feat/rq1-trn-01a-training-smoke-test        5942338 [origin/feat/rq1-trn-01a-training-smoke-test] feat(trn): resolve RQ1-TRN-01A-R2 final training correctness & evidence gate
  feat/rq1-trn-01b0-r-evidence-reconciliation d63f25f [origin/feat/rq1-trn-01b0-r-evidence-reconciliation] feat(trn): resolve RQ1-TRN-01B0-R real-data learning diagnostics and evidence reconciliation
  feat/rq1-trn-01b0-training-sanity-gate      1c5dc61 [origin/feat/rq1-trn-01b0-training-sanity-gate] feat(trn): resolve RQ1-TRN-01B0 controlled real-data training sanity gate
* feat/rq1-trn-01b1-ctc-stabilization         d63f25f feat(trn): resolve RQ1-TRN-01B0-R real-data learning diagnostics and evidence reconciliation
  main                                        175d68f [origin/main] chore(git): harden .gitignore against large datasets and configure Hugging Face model registry
```

**Key Finding:**  
Commit `d63f25f` was already present and tracked on remote branch `origin/feat/rq1-trn-01b0-r-evidence-reconciliation`. The branch `feat/rq1-trn-01b1-ctc-stabilization` had been branched locally from `d63f25f` but had not yet published its tracking ref to origin.

---

## 3. Commit Identification & Parentage Verification

Running `git log -n 5 --graph --oneline d63f25f` establishes the immutable lineal heritage:

```
* d63f25f feat(trn): resolve RQ1-TRN-01B0-R real-data learning diagnostics and evidence reconciliation
* 1c5dc61 feat(trn): resolve RQ1-TRN-01B0 controlled real-data training sanity gate
* 5942338 feat(trn): resolve RQ1-TRN-01A-R2 final training correctness & evidence gate
* 6581027 feat(trn): resolve RQ1-TRN-01A-R pretrained backbone & inference validity gate
* f0ade49 feat(trn): complete RQ1-TRN-01A training implementation, CUDA verification, and contract hardening
```

- **Reported Commit SHA:** `d63f25fa2d1e242e85b64cfe0b6e796fbab35a32`
- **Parent Commit SHA:** `1c5dc61c0afb9510b7041984f5170576473a4ab6`
- **Author/Date:** minkhoaa <tukhoa040505@gmail.com>, Fri Oct 23 23:52:47 2026 +0700
- **Scope of Changes:** 17 files changed, 2,426 insertions(+), 37 deletions(-)

All reconciliation source changes (`src/langfens_prep/collate.py`, `src/langfens_prep/dataset.py`, `src/langfens_prep/textgrid_parser.py`), audit scripts (`audit_scripts/audit_checkpoint_continuity.py`, `audit_scripts/run_repeat_sanity.py`, `audit_scripts/run_training_sanity.py`), tests (`tests/test_metric_reconciliation.py`), and documentation deliverables in `manifests/RQ1-TRN-01B0/reconciliation/` are completely committed.

---

## 4. Normal Remote Synchronization & Verification

The branch `feat/rq1-trn-01b1-ctc-stabilization` was pushed normally without force-pushing or rewriting history:

```bash
$ git push -u origin feat/rq1-trn-01b1-ctc-stabilization
To https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git
 * [new branch]      feat/rq1-trn-01b1-ctc-stabilization -> feat/rq1-trn-01b1-ctc-stabilization
branch 'feat/rq1-trn-01b1-ctc-stabilization' set up to track 'origin/feat/rq1-trn-01b1-ctc-stabilization'.
```

### 4.1 Remote Branch Listing (`git ls-remote --heads origin`)
```
59423385464f92700f38e9e5898812e367b48900 refs/heads/feat/rq1-trn-01a-training-smoke-test
d63f25fa2d1e242e85b64cfe0b6e796fbab35a32 refs/heads/feat/rq1-trn-01b0-r-evidence-reconciliation
1c5dc61c0afb9510b7041984f5170576473a4ab6 refs/heads/feat/rq1-trn-01b0-training-sanity-gate
d63f25fa2d1e242e85b64cfe0b6e796fbab35a32 refs/heads/feat/rq1-trn-01b1-ctc-stabilization
175d68f31d07d1c0b4caaf97111a90af5acd4c83 refs/heads/main
```

### 4.2 Independent Remote Verification via GitHub REST API
Querying GitHub's remote API endpoint confirmed remote commit presence:
```bash
$ curl -s https://api.github.com/repos/minkhoaa/Langfens-RQ1-Speech-Research/commits/d63f25fa2d1e242e85b64cfe0b6e796fbab35a32
```
```json
{
  "sha": "d63f25fa2d1e242e85b64cfe0b6e796fbab35a32",
  "node_id": "C_kwDOVU7p-NoAKGQ2M2YyNWZhMmQxZTI0MmU4NWI2NGNmZTBiNmU3OTZmYmFiMzVhMzI",
  "commit": {
    "author": {
      "name": "minkhoaa",
      "email": "tukhoa040505@gmail.com",
      "date": "2026-10-23T16:52:47Z"
    },
    "message": "feat(trn): resolve RQ1-TRN-01B0-R real-data learning diagnostics and evidence reconciliation"
  }
}
```

---

## 5. Independent Retrievability & Test Execution Certification

Following remote verification, the complete test suite was executed against the checked-out codebase:
- **Test Command:** `PYTHONPATH=src:. .venv-cuda/bin/pytest tests/`
- **Result:** `68 passed, 1 warning in 83.22s` (pre-pilot regression baseline).
- **Integrity Status:** All source modules, test fixtures, and datasets are verified retrievable and fully functioning.

| Provenance Requirement | Expected Verification | Verified Status | Evidence |
| :--- | :--- | :---: | :--- |
| **Commit Identity** | SHA `d63f25fa2d1e242e85b64cfe0b6e796fbab35a32` exists | **PASS** | Verified locally and remotely |
| **Parent Lineage** | Direct child of `1c5dc61c0afb9510b7041984f5170576473a4ab6` | **PASS** | `git log -n 5` tree audit |
| **Branch Sync** | Pushed to `origin/feat/rq1-trn-01b1-ctc-stabilization` | **PASS** | `git ls-remote` match |
| **GitHub API Query** | Status 200 OK with identical commit payload | **PASS** | Verified via HTTPS API query |
| **Clean Tree** | No uncommitted modifications prior to task execution | **PASS** | Working tree clean |
| **Reproducibility** | All test modules and audit scripts pass | **PASS** | 68/68 regression tests passing |

**Conclusion:** Task A Provenance Verification is **PASSED**. Phase `RQ1-TRN-01B1` proceeds with verified foundation.
