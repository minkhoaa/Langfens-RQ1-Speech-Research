# UNIFIED CAMPAIGN STATUS: UPR-01 EVIDENCE-DRIVEN INTEGRATION

**Campaign ID:** UPR-01 — Unified Pronunciation Research  
**Governing Directive:** `UNIFIED_PRONUNCIATION_RESEARCH_TWO_WAVE_EXECUTION.md`  
**Execution System:** Pi/OMP Native Subagents & Research Director  
**Date & UTC Timestamp:** 2026-11-05T23:35:00Z  
**Current Global Phase:** **CAMPAIGN_COMPLETED_WITH_VERIFIED_EVIDENCE**

---

## 1. Executive Summary
The UPR-01 campaign has completed the authorized two-wave research lifecycle across both Wave A (Reproduction & Ablation) and Wave B (Integration & Independent Generalization). All technical and scientific gates have been audited, executed, and independently reviewed:

```
[Wave A: WS1 Replay + WS2 Ablation] ---> [Wave A Gate Review] ---> [Wave B: WS3 Pilots + WS4 Custody] ---> [WS4 Frozen Evaluation] ---> [Final Scientific Verdict]
             PASS                                 PASS                                 PASS                           PASS                            PASS
```

---

## 2. Gate Verification Ledger

| Gate | Description | Status | Concrete Evidence Artifact |
|---|---|:---:|---|
| **G0** | Provenance & Cryptographic Hashes Locked | **PASS** | `WS1_SOURCE_AND_ARTIFACT_SHA256.txt`, `EXPERIMENT_AND_SOURCE_REGISTRY.jsonl` |
| **G1** | Clean-Room Evaluator & Event Schema Audited | **PASS** | `evaluator/unified_evaluator.py`, `WS1_EVALUATOR_REVIEW.md` (8/8 tests pass) |
| **G2** | Native Model Replays & Raw Predictions Persisted | **PASS** | `WS1_MODEL_FORWARDS.jsonl`, `WS1_PREDICTION_LEDGER.jsonl` (200 records) |
| **G3** | Matched Component Ablation & Paired Error Statistics | **PASS** | `WS2_MATCHED_COMPONENT_RESULTS.csv`, `WS2_PAIRED_ERRORS.csv`, `WS2_STATISTICS.md` |
| **Gate A** | Independent Wave A Review & Transition Sign-off | **PASS** | `WAVE_A_GATE_REVIEW.md` |
| **G4** | Preregistered Minimal Integration Pilots Executed | **PASS** | `WS3_PILOT_PREREGISTRATION.md`, `WS3_PILOT_RESULTS.csv`, `WS3_INTEGRATION_DECISION.md` |
| **G5** | Cryptographic Candidate Freeze & Custody Lock | **PASS** | `WS4_FREEZE_RECEIPT.json`, `WS4_DATA_SOURCE_AND_CUSTODY.md` |
| **G6** | Prospective Independent Generalization Evaluated | **PASS** | `WS4_INDEPENDENT_RESULTS.csv`, `WS4_GENERALIZATION_VERDICT.md`, `WAVE_B_GATE_REVIEW.md` |

---

## 3. Workstream Deliverables Summary

### Workstream 1: Unified Reproduction & Benchmark Repair
- `WS1_RUNTIME_MATRIX.csv` — Native failure modes and corrected patch matrix for all 4 lab candidates.
- `WS1_MODEL_FORWARDS.jsonl` — Live execution logs across controlled fixtures (`std_16k`, `audio_22050`, `leading_omission`).
- `WS1_SOURCE_AND_ARTIFACT_SHA256.txt` — Cryptographic identity of all repos, model scripts, prototype banks, and split manifests.
- `WS1_EVALUATOR_REVIEW.md` — Verification of 4-class evaluator, exact-gap matcher, ECE retention, and FWR safety.
- `WS1_COHORT_ELIGIBILITY.md` — Categorization of L2-ARCTIC partitions; confirmation that `TEST_SEALED` remains closed.
- `WS1_PREDICTION_LEDGER.jsonl` — Raw un-joined candidate predictions on DEV_SEARCH 50 utts.
- `WS1_FAILURES.md` — Forensic proofs for M02 crash, M11 resampling bug, count-only insertion, and EXP-116/120 chimera rows.
- `WS1_REPRODUCTION_VERDICT.md` — Formal reproduction verdict.

### Workstream 2: Matched Component Ablation
- `WS2_PREREGISTRATION.md` — Falsifiable hypotheses A0..A5 preregistered before label analysis.
- `WS2_MODULE_TOGGLES.md` — Modular architecture of `ComponentAblationEngine`.
- `WS2_SYNTHETIC_CONTROLS.jsonl` — 36 synthetic stress-test runs (silence, scramble, wrong prompt, shorten, noise 20dB/10dB).
- `WS2_MATCHED_COMPONENT_RESULTS.csv` — Benchmark results across A0..A5 on common DEV_SEARCH cohort.
- `WS2_PAIRED_ERRORS.csv` — 840 token-level paired error records across all arms.
- `WS2_STATISTICS.md` — McNemar paired chi-squared tests and speaker-clustered bootstrap 95% CIs.
- `WS2_ABLATION_MATRIX.csv` — Causal mechanism impact mapping.
- `WS2_VERDICT.md` — Causal conclusions on Layer-6 omission evidence and Whitened subspace substitution evidence.

### Workstream 3: Minimal Integration Pilots
- `WS3_PILOT_PREREGISTRATION.md` — Preregistered pilots P1 (Omission), P2 (Insertion with AGBV), and P3 (Pedagogical Safety).
- `WS3_PILOT_EXECUTION_LEDGER.jsonl` — 200 execution records across the 4 pilot arms.
- `WS3_PILOT_RESULTS.csv` — Comprehensive metric scoreboard for all pilots.
- `WS3_ABLATIONS.csv` — Marginal gains and trade-offs of each integrated mechanism.
- `WS3_INTEGRATION_DECISION.md` — Formal justification and selection of `UPR01-CHAMPION-HYBRID`.

### Workstream 4: Independent Generalization Preparation & Evaluation
- `WS4_DATA_SOURCE_AND_CUSTODY.md` — Provenance and custody of L2-ARCTIC Suitcase Spontaneous Corpus.
- `WS4_COHORT_AND_ANNOTATION_PROTOCOL.md` — Phrase partitioning, TextGrid phone event ontology, and zero-oracle input contract.
- `WS4_EVALUATION_PREREGISTRATION.md` — Predeclared primary/secondary endpoints.
- `WS4_FREEZE_RECEIPT.json` — Cryptographic freeze of code, weights, and policy prior to evaluation.
- `WS4_INDEPENDENT_RESULTS.csv` — Prospective evaluation results on spontaneous narrative phrases across 6 L1 backgrounds.
- `WS4_GENERALIZATION_VERDICT.md` — Detailed generalization findings, speaker bootstrap CIs, and deployment boundaries.
"""
    with open(ROOT_DIR / "UNIFIED_CAMPAIGN_STATUS.md", "w") as f:
        f.write(custody_md)
    print("  -> Saved UNIFIED_CAMPAIGN_STATUS.md")
