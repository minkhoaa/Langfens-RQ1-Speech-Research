# LANGFENS — RQ1-STD-01 / Data and Phoneme Standardization

**Status:** COMPLETED

**Recorded at:** 2026-10-23T11:00:00+07:00

**Evidence source:** Commit `5ccb4dc2cc986036175161c660b73a206de188d2`, `manifests/RQ1-STD-01/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Establish comprehensive, binding specifications for phoneme mapping, MDD binary and diagnostic label schemas, audio preprocessing standards, speaker-independent split protocols, and evaluation metrics for Project Langfens.

## 2. Inputs & Dependencies
- Source branch: `main` (commit `5ccb4dc2cc986036175161c660b73a206de188d2`).
- Audit baselines: `manifests/RQ1-AUD-01/` and `manifests/RQ1-AUD-01/reconciliation/`.
- CMUDict v0.7b and 39-ARPAbet phone inventory.

## 3. Work Performed
1. Authored eight primary standardization specifications defining label accounting, audio preparation, evaluation protocols, and speaker partitioning.
2. Conducted rigorous pre-implementation consistency gate (`RQ1-STD-01R`).
3. Reconciled phonemic substitution decomposition ($14,098 = 12,368 \text{ clean} + 1,472 \text{ asterisk} + 258 \text{ err}$).
4. Disproved initial 4-fold test gender imbalance (5M/1F in Fold 1) and proved combinatorial feasibility of balanced 3M/3F, 1-per-L1 partitions.
5. Resolved 26-speaker Vietnamese LOSO contradiction by defining two disjoint regimes (Cross-Vietnamese LOSO and Zero-Shot Accent Transfer).
6. Audited and downgraded unverified CPU resampling throughput claims.

## 4. Results & Verification
- Acceptance gate: `RQ1-STD-01: PASS` / `RQ1-STD-01R: PASS`.
- Mathematical reconciliation: zero unexplained variance across L2-ARCTIC annotations ($\Delta = 0$).
- Proved 4-fold allocation table satisfies 3M/3F, 1 per L1 in test, 1M/1F in val, 8M/8F in train.
- Formally cataloged and downgraded unsupported performance claims to `NOT_BENCHMARKED`.
- Documented in [STANDARDIZATION_POLICY.md](STANDARDIZATION_POLICY.md) and [SPECIFICATION_CONSISTENCY_REPORT.md](reconciliation/SPECIFICATION_CONSISTENCY_REPORT.md).

## 5. Output Artifacts
- Core specifications: `manifests/RQ1-STD-01/STANDARDIZATION_POLICY.md`, `MDD_LABEL_SCHEMA.md`, `PHONEME_MAPPING_SPEC.md`, `AUDIO_PREPARATION_SPEC.md`, `SPEAKER_SPLIT_PROTOCOL.md`, `EVALUATION_PROTOCOL.md`, `DATA_CONTRACT_SPEC.md`, `OPEN_DECISIONS_AND_RISKS.md`.
- Reconciliation addenda: `manifests/RQ1-STD-01/reconciliation/CORRECTED_LABEL_ACCOUNTING.md`, `VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md`, `UNVERIFIED_CLAIMS_AND_DECISIONS.md`, `SPECIFICATION_CONSISTENCY_REPORT.md`.

## 6. Restart / Recovery Instructions
- Reference specifications are frozen and version-controlled.
- Review specifications before pipeline development or manifest generation.
- No execution scripts required for policy specifications.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Concrete manifest JSON generation deferred to RQ1-SPLIT-01.
- Prohibited actions: Never implement training pipelines using unverified label definitions.
- Next phase: [RQ1-SPLIT-01](../RQ1-SPLIT-01/PROCESS_CHECKPOINT.md).
