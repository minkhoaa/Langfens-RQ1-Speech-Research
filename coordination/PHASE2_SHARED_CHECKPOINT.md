# LANGFENS PHASE 2: SHARED CHECKPOINT & PROVENANCE LEDGER

- **Research Root:** `/home/khoa/Projects/New-Method-Research`
- **Git Commit Provenance:** `41f02ad9ad4ea2df4734ebc38bc2052503586fb6`
- **Execution Date:** 2026-11-06
- **Authorized Dataset:** L2-ARCTIC canonical subset (Train: 1800 utts, Dev-Cal: 300 utts, Dev-Search: 599 utts)
- **DEV_SEARCH Population:** 19,755 reference-phone targets (17,778 Correct, 1,768 Substitution, 209 Omission), 20,354 inter-phone gaps (20,229 Normal, 125 Insertion)

## FROZEN TEACHER CHECKPOINTS & ARTIFACT FINGERPRINTS

| Specialist Teacher | Checkpoint File | Checkpoint SHA-256 | Prediction Ledger SHA-256 (Dev-Search) |
|---|---|---|---|
| **Teacher S Champion** | `models/teachers/TEACHER_S_CHAMPION_checkpoint.pt` | `47e804011f6fa744bf09dbd13896cfcd9ef874534d9eaa9f403e37265b000510` | `6816101dafb88268f3c725e74ab4407f01ec063e5811f4ebb5ddb90f455082e4` |
| **Teacher O Champion** | `models/teachers/TEACHER_O_CHAMPION_checkpoint.pt` | `80618baaefe59eff0da3db0a9a64941fb2bdb1a10146c60ee93b286f9634db76` | `2a1f1f31ee4df8c9fac33890b0d7027851f73184659a7e89bee5e7b766f1ba66` |
| **Teacher I-Precision** | `models/teachers/TEACHER_I_CHAMPION_checkpoint.pt` | `301a3fc7e8f627a058c165be53cd024a9aaa3e60b0f598e36170322e1d0f31bd` | `a1de03b87f528a1b1cf68c77b3ca5f4194562ed8c5cabaa8c9a260d6ef030e53` |
| **Teacher I-F1** | `models/teachers/TEACHER_I_EXP01_checkpoint.pt` | `98da8bb5ff756df026e7ec17e17b4c5c60b8149c6cdba8c269005831238c6220` | `0621a6d4a0cd55b69844f7901b06bd91554f22fd9b4d9788281704deaba8bcd2` |
| **UPR03 Baseline** | `experiments/predictions/UPR03_SYNTHESIS_CHAMPION_predictions.jsonl` | N/A | `bc3f2b6a55157e9778eba5b5b414e91ee72e254575c287872746e4ed2d4d71c3` |

## ACTIVE RESEARCH WORKSTREAMS

1. **AGENT A (CPL-01):** Joint Evidence & Overlap Analysis.
2. **AGENT B (CPL-02):** Joint Score Fusion & Pareto Decision Policies.
3. **AGENT C (CPL-03):** Distillation Feasibility & Multi-Teacher Student Contract.
