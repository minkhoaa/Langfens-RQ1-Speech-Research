# LANGFENS — RQ1-SPLIT-01 / Deterministic Speaker-Independent Splits

**Status:** COMPLETED

**Recorded at:** 2026-10-23T13:30:00+07:00

**Evidence source:** Commits `5ccb4dc2cc986036175161c660b73a206de188d2` and `085bfcd68f6bf7de7ca6709848f0802773f8e705`, `manifests/RQ1-SPLIT-01/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Generate, serialize, and validate deterministic, speaker-independent data splits for L2-ARCTIC (4-Fold CV and Vietnamese LOSO/Transfer regimes) and SpeechOcean762, proving strict partition disjointness, demographic balance, and conservation of tokens under CC BY-NC 4.0 license provenance.

## 2. Inputs & Dependencies
- Source branch: `main` (commit `5ccb4dc2cc986036175161c660b73a206de188d2`).
- Specifications: `manifests/RQ1-STD-01/reconciliation/VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md`.
- Raw metadata from `manifests/RQ1-AUD-01/`.
- Generator script: `audit_scripts/generate_rq1_splits.py`.

## 3. Work Performed
1. Generated `l2arctic_4fold_splits.json` implementing the validated 3M/3F, 1-per-L1 demographic matrix.
2. Generated `l2arctic_loso_vietnamese_splits.json` for Cross-Vietnamese LOSO (Regime A) and Zero-Shot Transfer (Regime B).
3. Generated `speechocean762_official_splits.json` preserving the official 2,500-utterance test partition while stratifying validation from train.
4. Executed automated mathematical validation proving zero speaker leakage across all partitions ($S_{train} \cap S_{val} = \emptyset$, etc.).
5. Documented CC BY-NC 4.0 license provenance erratum and rectified split provenance timestamp metadata.

## 4. Results & Verification
- Acceptance gate: `RQ1-SPLIT-01: PASS`.
- L2-ARCTIC 4-Fold CV Fold 1 contract:
  - TRAIN: 2,399 utterances (16 speakers: 8M/8F).
  - VALIDATION: 300 utterances (`NJS`, `RRBI`: 1M/1F).
  - SEALED TEST: 900 utterances (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`: 3M/3F).
- Zero speaker leakage proven; prompt tokens (119,151) and errors (17,518) perfectly conserved.
- Documented in [SPLIT_VALIDATION_REPORT.md](SPLIT_VALIDATION_REPORT.md) and [LICENSE_PROVENANCE_ERRATUM.md](LICENSE_PROVENANCE_ERRATUM.md).

## 5. Output Artifacts
- Manifests:
  - `l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
  - `l2arctic_loso_vietnamese_splits.json` (SHA-256 `6056d56dfd86088863f69fc25a2df5ca8cf99fe2533ba03bfa86fa736b485984`).
  - `speechocean762_official_splits.json` (SHA-256 `f829ec96c3d8a7c2e3074b8fb1737e1bf110c73336fbf42a35368fe0ef844510`).
- Validation scripts: `audit_scripts/validate_splits.py`.

## 6. Restart / Recovery Instructions
- Environment: Python standard library (`json`, `hashlib`).
- Re-run split validation suite:
  ```bash
  python audit_scripts/validate_splits.py
  ```
- Validation must pass with zero errors before any data loader or training pipeline reads splits.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: None for split definitions.
- Prohibited actions: Never modify frozen JSON split files; never expose sealed test utterances.
- Next phase: [RQ1-PREP-01](../RQ1-PREP-01/PROCESS_CHECKPOINT.md).
