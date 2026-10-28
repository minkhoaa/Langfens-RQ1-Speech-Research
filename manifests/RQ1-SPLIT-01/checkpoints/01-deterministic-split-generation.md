# LANGFENS — RQ1-SPLIT-01 / 01-Deterministic Split Generation

**Status:** COMPLETED

**Recorded at:** 2026-10-23T13:20:00+07:00

**Evidence source:** Commit `5ccb4dc2cc986036175161c660b73a206de188d2`, `manifests/RQ1-SPLIT-01/SPLIT_VALIDATION_REPORT.md`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Generate deterministic split manifests for L2-ARCTIC 4-Fold CV, Vietnamese LOSO/Transfer, and SpeechOcean762, mathematically verifying speaker disjointness, demographic balance, and token conservation.

## 2. Inputs & Dependencies
- Allocation specification: `manifests/RQ1-STD-01/reconciliation/VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md`.
- Raw audio census from `manifests/RQ1-AUD-01/`.
- Generator script: `audit_scripts/generate_rq1_splits.py`.

## 3. Work Performed
1. Serialized `l2arctic_4fold_splits.json` allocating 24 speakers across 4 balanced folds.
2. Serialized `l2arctic_loso_vietnamese_splits.json` allocating Vietnamese speakers with fixed non-Vietnamese validation (`NJS`, `RRBI`).
3. Serialized `speechocean762_official_splits.json` preserving 2,500 official test utterances while carving 500 validation utterances from train.
4. Excluded 245 unannotated orphan WAVs from SpeechOcean762.
5. Ran verification asserting mutual exclusivity ($S_1 \cap S_2 = \emptyset$).

## 4. Results & Verification
- Acceptance gate: `SPLIT_VALIDATION: PASS`.
- Fold 1 verified: 2,399 train (16 speakers), 300 val (`NJS`, `RRBI`), 900 test (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`).
- Demographic parity: Exactly 3M/3F, 1-per-L1 in all test folds.
- 119,151 tokens and 17,518 errors conserved with 0 variance.
- Documented in [SPLIT_VALIDATION_REPORT.md](../SPLIT_VALIDATION_REPORT.md).

## 5. Output Artifacts
- Manifests: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`, `l2arctic_loso_vietnamese_splits.json`, `speechocean762_official_splits.json`.
- Provenance ledger: `manifests/RQ1-SPLIT-01/SPLIT_PROVENANCE.json`.

## 6. Restart / Recovery Instructions
- Run split validator:
  ```bash
  python audit_scripts/validate_splits.py
  ```
- Fast read-only check; zero authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Verification of license provenance metadata.
- Prohibited actions: Never alter serialized partition boundaries.
- Next step: [02-license-provenance-and-provenance-ledger.md](02-license-provenance-and-provenance-ledger.md).
