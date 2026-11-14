# Research Plan: Hypothesis H35 / Experiment EXP-40

## 1. Objective
Build an automated standalone reproduction script (`tools/reproduce_champions.py`), execute full cryptographic compaction across all 39 completed experiments, and generate `COMPACTED_ARCHIVE_RECEIPT.json`.

## 2. Experimental Steps
1. **Develop `tools/reproduce_champions.py`:**
   - Evaluates:
     * System 1: EXP-39 (Octad-Pareto Super-Ensemble)
     * System 2: EXP-38 (HTC-Net Hierarchical Cascade)
     * System 3: EXP-33 (INT8 Edge APED-Net)
   - Checks reproduced metrics against logged receipts to ensure $|\Delta| < 10^{-4}$.
2. **Execute Full Cryptographic Archive Sweep:**
   - Scan all directories `experiments/EXP-01` through `experiments/EXP-39`.
   - Validate presence of `RUN_SPEC.yaml`, `RUN_STATE.json`, `TRAIN.log`, and `ARTIFACT_HASHES.sha256`.
   - Confirm `test_sealed.jsonl` SHA-256 (`a5636249f7...`).
3. **Receipt and Ledger Archival:**
   - Save compilation receipt under `experiments/EXP-40-COMPREHENSIVE-ARCHIVE-COMPACTION/METRICS/COMPACTED_ARCHIVE_RECEIPT.json`.
   - Update `experiments/REGISTRY.jsonl`, `experiments/COMPARISONS.csv`, and all summary reports.
