# LANGFENS RQ1-TRN-04 — Master Research Process Checkpoint

**Document ID:** `LANGFENS-RQ1-TRN-04-PROCESS-CHECKPOINT-01`  
**Repository:** `minkhoaa/Langfens-RQ1-Speech-Research`  
**Branch:** `review/rq1-trn-04-factorial-training`  
**Base Pinned Commit:** `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`  
**Author:** Principal Speech ML Research Engineer, Phonetic Scientist, Reproducibility Auditor & Coordinator  
**Date:** 2026-11-04  
**Disposition:** **COMPLETED_ALL_FOUR_ARMS_AND_SYNTHESIS — ALL EXPERIMENTAL HYPOTHESES CONCLUSIVELY REJECTED**  
**Policy Authority:** Repository `AGENTS.md` (Mandatory Process Checkpoint Policy)  

---

## 1. Process Status & Executive Summary

The **LANGFENS RQ1-TRN-04** parallel factorial training experiment has fully executed to completion. All four matched training arms were trained to their prospectively frozen fixed-budget endpoint of **exactly 3,600 successful optimizer updates** (with 3 skipped AMP attempts each, representing 3,603 total optimization attempts across 9 completed epochs) on authorized remote GPU hardware (Nvidia Tesla T4, CUDA FP16 AMP):

1. **Gate 0 Verification (PASS):**
   - 7/7 automated contract tests passed (`tests/trn04/test_gate0_factorial_contracts.py`).
   - All 16 parameter tensors verified bit-identical at initialization from pinned seed `1337` (`COMMON_INITIALIZATION_MANIFEST.json`).
   - Identical capacity across all arms: **exactly 267,179 trainable parameters**.
   - Strict speaker isolation verified across FIT (1,799), SELECT (300), CAL (300), VAL (300), and sealed TEST (900).
2. **Independent Remote GPU Training (COMPLETED):**
   - **Arm C0 (`langfens-rq1-trn-04-c0-ce-nofeat`):** 3,600 successful updates in 507.3s. SELECT best AP: 0.476465, PER: 0.240204.
   - **Arm T1 (`langfens-rq1-trn-04-t1-focal-nofeat`):** 3,600 successful updates in 449.4s. SELECT best AP: 0.461918, PER: 0.240516.
   - **Arm T2 (`langfens-rq1-trn-04-t2-ce-feat`):** 3,600 successful updates in 466.5s. SELECT best AP: 0.470217, PER: 0.240516.
   - **Arm T3 (`langfens-rq1-trn-04-t3-focal-feat`):** 3,600 successful updates in 493.4s. SELECT best AP: 0.467060, PER: 0.240619.
   - All 12 checkpoint artifacts mirrored to Hugging Face (`minkhoaa/langfens-rq1-cross-fold-checkpoints/trn04/`) with cryptographic SHA-256 readback.
3. **Centralized Evaluation under Protocol Z (COMPLETED):**
   - Benchmarked deterministically on Fold-1 VAL (300 utterances, 9,956 gold canonical tokens) with CAL-selected thresholds from NCC/THV.
   - **Hypothesis $H_1$ (Objective Effect): REJECTED.** Class-conditioned focal loss collapsed Dedicated Omission AP from $0.1133$ to $0.0276$ and exploded false alarms to $985.4$ / 1k.
   - **Hypothesis $H_2$ (Acoustic Feature Effect): REJECTED.** 4D waveform acoustic evidence improved acoustic reliance gap ($+0.0225$ vs $+0.0199$) and reduced false alarms by $9.1\%$, but degraded recall from $60.0\%$ to $51.0\%$ and reduced precision from $7.55\%$ to $7.09\%$, being strictly Pareto-dominated by C0.
   - **Hypothesis $H_3$ (Interaction Effect): REJECTED.** Zero beneficial synergy ($T_3 \approx T_1$).
   - **Prospective Acceptance Gates:** All four arms FAILED both Level A (Pilot Feasibility, $\ge 25\%$ prec) and Level B (Research Target, $\ge 30\%$ prec).
4. **Final Research Decision (Outcome A / Outcome E):**
   - Retain Baseline C0 (`CompactFactorialOperationHead`, uniform CE) as the authoritative champion architecture.
   - Terminate class-conditioned focal down-weighting.
   - Enforce the Horvitz-Thompson Adjudicated Pronunciation Event Horizon (which achieves $32.18\%$ broad precision on C0) as the mandatory reporting horizon for pedagogical feedback.

---

## 2. Cryptographic Artifacts & Provenance Inventory

All model artifacts are preserved in approved private persistent storage with SHA-256 verification:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             CHECKPOINT DURABILITY & INTEGRITY LEDGER                                   │
├─────┬──────────────────────┬─────────────┬─────────────┬───────────────────────────────────────────────────────────────┤
│ Arm │ Endpoint Artifact    │ Size (bytes)│ Updates     │ Cryptographic SHA-256 Checksum                                │
├─────┼──────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────────────────────┤
│ C0  │ fixed_budget_3600.pt │ 3,243,408   │ 3,600 / 3   │ ca4cff18ff38729ae553483017cf70678d975db353bb895e6f6a7d5fa21c4 │
│ C0  │ best.pt (Epoch 5)    │ 3,239,079   │ 2,250 / 3   │ 0b7092d44bae013fa89369e96f1fa2c263795ba429b6f849ff0dc4d830b56 │
│ C0  │ training_summary.json│ 4,152       │ 3,600       │ e41e99add01882e9fa56417531776cebf88e1bb342e4313f8c5b59ebec7da │
├─────┼──────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────────────────────┤
│ T1  │ fixed_budget_3600.pt │ 3,243,408   │ 3,600 / 3   │ 0d3cdad4a0ab786b51eb5825a071d227dddb5eb8d197607d727b1ee6c7647 │
│ T1  │ best.pt (Epoch 2)    │ 3,238,631   │   900 / 3   │ 88a39b3617219db3f28cf67484df463aa1f40212f453c072c4161bb740ca3 │
│ T1  │ training_summary.json│ 4,169       │ 3,600       │ 88ff27256c1041066795f59fa0f4e38e6e5db4d18805f1df7ef14cf3fa201 │
├─────┼──────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────────────────────┤
│ T2  │ fixed_budget_3600.pt │ 3,243,408   │ 3,600 / 3   │ e8b00002aef9ee1587990a69162d5a5438b092e02c83d5654ae5aea0e1ec9 │
│ T2  │ best.pt (Epoch 9)    │ 3,239,591   │ 3,600 / 3   │ 98d3fc1ecf6a5539d211028e32c9cbd2d5668b3942ad4d8fcae8226a99927 │
│ T2  │ training_summary.json│ 4,141       │ 3,600       │ a5e1f69581d789fc5fe45fa8f1fcbfe4c1a7a13c32b55b96ee8d20e74f3ff │
├─────┼──────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────────────────────┤
│ T3  │ fixed_budget_3600.pt │ 3,243,408   │ 3,600 / 3   │ ca32da07eac2094676451eebc95350c33748be7c4c1a719c8366ce2b451ec │
│ T3  │ best.pt (Epoch 5)    │ 3,239,079   │ 2,250 / 3   │ 1feb311ec9f88569c735d487cb1f912e75e1b6f00db1bbad478bf11979ad7 │
│ T3  │ training_summary.json│ 4,160       │ 3,600       │ 247bcaeb32fd44b64f33b1e77f041ff3c4adff7ec2882250106201e5df707 │
└─────┴──────────────────────┴─────────────┴─────────────┴───────────────────────────────────────────────────────────────┘
```

---

## 3. Master Deliverables Directory

All required experiment artifacts are versioned in `manifests/RQ1-TRN-04/`:
1. `EXPERIMENT_REGISTRATION.md`
2. `FACTORIAL_DESIGN_AND_CONTROLS.md`
3. `DATA_AND_LABEL_READINESS.md`
4. `COMMON_INITIALIZATION_MANIFEST.json`
5. `c0/TRAINING_RUN_MANIFEST.json`, `t1/`, `t2/`, `t3/`
6. `c0/TRAINING_SUMMARY.json`, `t1/`, `t2/`, `t3/`
7. `c0/CHECKPOINT_INTEGRITY.json`, `t1/`, `t2/`, `t3/`
8. `c0/EVALUATION_METRICS.json`, `t1/`, `t2/`, `t3/`
9. `FIXED_BUDGET_COMPARISON.md`
10. `FACTORIAL_EFFECT_ANALYSIS.md`
11. `PRECISION_RECALL_TRADEOFF.md`
12. `ACOUSTIC_GROUNDING_REPORT.md`
13. `SPEAKER_AND_PHONEME_ERROR_ANALYSIS.md`
14. `SCIENTIFIC_ACCEPTANCE_REPORT.md`
15. `NEXT_RESEARCH_DECISION.md`
16. `PROCESS_CHECKPOINT.md`
