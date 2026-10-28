# LANGFENS — RQ1-TRN-02B / 01-Acoustic Conditioning Design

**Status:** COMPLETED

**Recorded at:** 2026-10-26T23:50:00+07:00

**Evidence source:** Commits `6836367`, `4d6b868`, `428061d`, `manifests/RQ1-TRN-02B/`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Design the controlled acoustic conditioning experiment (RQ1-TRN-02B) to test whether Binary Prompt Dropout forces the mispronunciation detection head to rely on acoustic representations rather than linguistic prompt priors.

## 2. Inputs & Dependencies
- Baseline R checkpoint: `best.pt` (SHA-256 `a2445c8d...`).
- Baseline ledger: `manifests/RQ1-TRN-02B/BASELINE_FREEZE_LEDGER.md`.
- Prior design documents: `manifests/RQ1-TRN-02B/ACOUSTIC_CONDITIONING_DESIGN.md`.

## 3. Work Performed
1. Formalized dual-arm design: Control Arm C0 ($p=0$, no dropout) versus Treatment Arm C1 ($p=0.25$ Binary prompt dropout).
2. Defined compute budget: Maximum 6 epochs and 3,600 optimizer updates per arm (ceiling 7,200 updates total across C0/C1).
3. Pinned invariant hyperparameters: Seed 1337, AdamW lr=1e-3, batch size 2, gradient accumulation 2, CTC + Binary loss, frozen Wav2Vec2 backbone.
4. Corrected initial budget calculations in `DESIGN_R1_CORRECTIONS.md` (commit `428061d`).

## 4. Results & Verification
- Acceptance gate: `DESIGN_R1: PASS`.
- Experimental contrast formally specified: Primary metric is $C1 - C0$ AP delta on Protocol O matched/mismatched audio.
- Documented in [ACOUSTIC_CONDITIONING_DESIGN.md](../ACOUSTIC_CONDITIONING_DESIGN.md) and [DESIGN_R1_CORRECTIONS.md](../DESIGN_R1_CORRECTIONS.md).

## 5. Output Artifacts
- Design reports: `manifests/RQ1-TRN-02B/ACOUSTIC_CONDITIONING_DESIGN.md`, `DESIGN_R1_CORRECTIONS.md`, `BASELINE_FREEZE_LEDGER.md`.

## 6. Restart / Recovery Instructions
- Review experimental design and budget ledger before implementing model changes.
- Static documentation review; zero execution required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Code implementation and preflight testing.
- Prohibited actions: Never alter training budget without documenting design revisions.
- Next step: [02-implementation-and-preflight-audit.md](02-implementation-and-preflight-audit.md).
