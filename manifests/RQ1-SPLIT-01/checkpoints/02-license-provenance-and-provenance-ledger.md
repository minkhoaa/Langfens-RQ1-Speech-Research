# LANGFENS — RQ1-SPLIT-01 / 02-License Provenance and Provenance Ledger

**Status:** COMPLETED

**Recorded at:** 2026-10-23T13:25:00+07:00

**Evidence source:** Commits `5ccb4dc2cc986036175161c660b73a206de188d2` and `085bfcd68f6bf7de7ca6709848f0802773f8e705`, `manifests/RQ1-SPLIT-01/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Establish authoritative CC BY-NC 4.0 license provenance for L2-ARCTIC derived splits, rectify historical timestamp inconsistencies in manifest generation logs, and freeze cryptographic split digests.

## 2. Inputs & Dependencies
- Split manifests: `manifests/RQ1-SPLIT-01/*.json`.
- Texas A&M University PSI Lab L2-ARCTIC official licensing terms (`CC BY-NC 4.0`).
- Provenance ledger: `manifests/RQ1-SPLIT-01/SPLIT_PROVENANCE.json`.

## 3. Work Performed
1. Authored `LICENSE_PROVENANCE_ERRATUM.md` correcting earlier informal citations of CC BY 4.0 and formally applying CC BY-NC 4.0 to all derived manifests and model checkpoints.
2. Verified non-commercial graduation thesis research scope (Khóa Luận Tốt Nghiệp — KLTN).
3. Issued `SPLIT_PROVENANCE_TIME_ERRATUM.md` (commit `085bfcd`) reconciling local system timestamps against commit UTC timestamps.
4. Sealed cryptographic hashes for all split artifacts.

## 4. Results & Verification
- Acceptance gate: `PROVENANCE_FREEZE: PASS`.
- Frozen SHA-256 digests:
  - `l2arctic_4fold_splits.json`: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
  - `l2arctic_loso_vietnamese_splits.json`: `6056d56dfd86088863f69fc25a2df5ca8cf99fe2533ba03bfa86fa736b485984`.
  - `speechocean762_official_splits.json`: `f829ec96c3d8a7c2e3074b8fb1737e1bf110c73336fbf42a35368fe0ef844510`.
- Documented in [LICENSE_PROVENANCE_ERRATUM.md](../LICENSE_PROVENANCE_ERRATUM.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-SPLIT-01/LICENSE_PROVENANCE_ERRATUM.md`, `manifests/RQ1-PREP-01/SPLIT_PROVENANCE_TIME_ERRATUM.md`.
- Provenance JSON: `manifests/RQ1-SPLIT-01/SPLIT_PROVENANCE.json`.

## 6. Restart / Recovery Instructions
- Verify split hashes against ledger:
  ```bash
  sha256sum manifests/RQ1-SPLIT-01/*.json
  ```
- No user authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: None.
- Prohibited actions: Never distribute model checkpoints or splits under commercial licenses.
- Next phase: [RQ1-PREP-01](../../RQ1-PREP-01/PROCESS_CHECKPOINT.md).
