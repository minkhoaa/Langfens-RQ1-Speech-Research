# LANGFENS Research Cycle 2 — state

Updated: 2026-11-07

## Campaign state

**ACTIVE — CUDA restored; baseline completed; new method screens underway.** The initial NVIDIA device-node exposure issue has been resolved in the current local Linux session. Driver `615.78.08`, GTX 1650, CUDA 13.0 in `.venv-cuda`; CUDA smoke tests passed. The Phase 3 Gold-Only control ran as `RC2-GOLD-ONLY-CUDA130-S101`, returned 0, and released its lease. Receipt: `baseline_control/RC2-GOLD-ONLY-CUDA130-S101-receipt.json`. Three Luna agents are currently handling SSL, multi-teacher/articulatory, and unified-design follow-up. No Phase 4 process was observed or interrupted. See `CYCLE2_FAILURES.md` for historical blockers and their resolution.

## Immutable starting point and pre-existing work

- Git HEAD: `aa4e6ea2147180a4a820d77a71911f333fcb6721`.
- Initial `git status --short`: 107 pre-existing untracked paths. Preserved; no reset, clean, overwrite or historical-output edits were performed.
- Research Cycle 1 checkpoint: `reports/long_running_method_discovery/NEXT_CYCLE.md` and `RESEARCH_STATE.md`.
- EVAL-V3 closure: `reports/EVAL_V3_CLOSURE_REPORT.md`; final release under `experiments/evaluation_v3/closure/release_final/`.
- TMCI, Phase 3 and Phase 4 source reports were inspected. Their scores remain development evidence; DEV_SEARCH is exposed and not an independent test.
- No TEST_SEALED/protected data or encrypted archive was accessed.

## Dataset / evaluation contract carried forward

Published authorized populations: TRAIN 1,800 utterances / 12 speakers; DEV_CAL 300 / 2; DEV_SEARCH 599 / 4. Published DEV_SEARCH support: 19,755 target events, 20,354 exact insertion gaps, S/O/I positives 1,768/209/125. Preserve N target slots and N+1 gaps, including leading and trailing slots. Use the authoritative `UnifiedEvaluatorV3`; raw AP is discrimination, while F1/policies are threshold-dependent. DEV_SEARCH is development-exposed.

Intended-reference independence remains unverified: canonical phones are annotation-derived, although inspected feature extraction uses intended phones, acoustic data and model posteriors, not gold timing/realized phones/error labels for X. This provenance limitation must accompany any result.

## Workstream status

All six original research hypotheses were dispatched to `gpt-6-luna`; current CUDA follow-ups use the same small model. H-001 has both the earlier frozen-emission scoring result and a CUDA optional-score + acoustic student screen; the combined model is not supported for discrimination. H-002 is executing a corrected locally cached ASR-finetuned Wav2Vec2 run using the checkpoint-required waveform normalization; the first raw-input cache and failed head run are retained and disqualified. H-003 CPU local gap pooling probe is not supported. H-004 raw OOF scalar KD completed with small mixed deltas; separate feature-level KD preparation is active. H-005 remains an existing-artifact calibration audit. H-006 one-seed articulatory auxiliary/control screen showed small mixed ranking shifts and is exploratory only. All SEARCH metrics are development-exposed; no test generalization claim.

## Current resume plan

1. Keep new training under unique RC2 arm names and use `/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python` through `coordination/phase3_gpu_job.py`; inspect leases and VRAM before each job.
2. H-002 is testing locally cached ASR-finetuned Wav2Vec2 hidden states as a clearly labeled sensitivity experiment. H-004/H-006 are screening one seed each with their own controls. Unified design is auditing EXP24/29/36/43 comparability.
3. Add results only after execution with EVAL-V3, receipts, exact hashes, and explicit split exposure. DEV_SEARCH is descriptive and cannot select a winner.
4. Do not modify Phase 4. Do not claim TEST_SEALED generalization.

Current agents are active only in the present session; this checkpoint makes no claim of background continuation.
