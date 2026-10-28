# LANGFENS — RQ1-STD-01 / 01-Standardization Policies and Schema

**Status:** COMPLETED

**Recorded at:** 2026-10-23T10:20:00+07:00

**Evidence source:** Commit `5ccb4dc2cc986036175161c660b73a206de188d2`, `manifests/RQ1-STD-01/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Formulate the foundational standardization policies for phoneme mappings, audio pre-processing contracts, evaluation protocols, and speaker split requirements across Project Langfens.

## 2. Inputs & Dependencies
- Audit deliverables from `manifests/RQ1-AUD-01/`.
- CMUDict v0.7b and 39-ARPAbet specification.

## 3. Work Performed
1. Authored `STANDARDIZATION_POLICY.md` establishing non-destructive pre-processing and frozen split requirements.
2. Drafted `MDD_LABEL_SCHEMA.md` defining strict binary error labeling and 39-class diagnosis.
3. Created `PHONEME_MAPPING_SPEC.md` establishing stress-stripping rules and AX schwa folding.
4. Specified 16 kHz polyphase bandlimited resampling in `AUDIO_PREPARATION_SPEC.md`.
5. Formulated candidate speaker-independent partitioning in `SPEAKER_SPLIT_PROTOCOL.md`.

## 4. Results & Verification
- Acceptance gate: Initial specification drafting complete.
- Defined uniform 39-class ARPAbet target space with `<BLANK>` for CTC alignment.
- Established strict speaker disjointness principles ($S_{train} \cap S_{val} = \emptyset, \dots$).
- Identified open decisions regarding 4-fold gender balance and Vietnamese LOSO structure.

## 5. Output Artifacts
- Initial specifications: `manifests/RQ1-STD-01/STANDARDIZATION_POLICY.md`, `MDD_LABEL_SCHEMA.md`, `PHONEME_MAPPING_SPEC.md`, `AUDIO_PREPARATION_SPEC.md`, `SPEAKER_SPLIT_PROTOCOL.md`, `EVALUATION_PROTOCOL.md`, `DATA_CONTRACT_SPEC.md`, `OPEN_DECISIONS_AND_RISKS.md`.

## 6. Restart / Recovery Instructions
- Inspect specifications under `manifests/RQ1-STD-01/`.
- Specifications are static documentation; no code execution required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Gender imbalance and speaker count ambiguities identified in initial drafting.
- Prohibited actions: Do not proceed to dataset splitting without resolving protocol inconsistencies.
- Next step: [02-specification-consistency-reconciliation.md](02-specification-consistency-reconciliation.md).
