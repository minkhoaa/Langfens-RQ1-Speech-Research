# LANGFENS — RQ1-TRN-00 / 01-Baseline Architecture and Comparative Design

**Status:** COMPLETED

**Recorded at:** 2026-10-23T15:55:00+07:00

**Evidence source:** Commit `9382cca`, `manifests/RQ1-TRN-00/BASELINE_EXPERIMENT_DESIGN.md`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Compare the theoretical and operational characteristics of Goodness of Pronunciation (GOP) against Wav2Vec 2.0 multi-task MDD, and establish an evidence-based research roadmap for Project Langfens.

## 2. Inputs & Dependencies
- Corpus characteristics from `manifests/RQ1-AUD-01/`.
- Scientific literature: Witt & Young (2000) [P02], Baevski et al. (2020) [P04], Gong et al. (2022) [P06].

## 3. Work Performed
1. Formalized GOP log-likelihood ratio formulation and identified critical failure mode under phonemic deletions ($D = 3,420$).
2. Formalized end-to-end Wav2Vec 2.0 MDD architecture combining CTC phoneme alignment, binary error detection, and 39-class diagnosis.
3. Formulated research execution roadmap prioritizing Wav2Vec 2.0 as the primary foundation model while retaining GOP as comparative reference.

## 4. Results & Verification
- Acceptance gate: Comparative assessment approved.
- Decided research order: Baseline B (Wav2Vec 2.0) implemented first to eliminate external forced-alignment dependencies.
- Documented in [BASELINE_EXPERIMENT_DESIGN.md](../BASELINE_EXPERIMENT_DESIGN.md) and [EXPERIMENT_PROTOCOL.md](../EXPERIMENT_PROTOCOL.md).

## 5. Output Artifacts
- Reports: `manifests/RQ1-TRN-00/BASELINE_EXPERIMENT_DESIGN.md`, `EXPERIMENT_PROTOCOL.md`, `SOURCE_REPRODUCIBILITY_AUDIT.md`.

## 6. Restart / Recovery Instructions
- Review comparative design documents before initiating model development.
- Static documentation review; no command execution required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Concrete compute infrastructure planning and loss contract implementation.
- Prohibited actions: Do not claim GOP superiority without empirical ablation.
- Next step: [02-compute-planning-and-contract-hardening.md](02-compute-planning-and-contract-hardening.md).
