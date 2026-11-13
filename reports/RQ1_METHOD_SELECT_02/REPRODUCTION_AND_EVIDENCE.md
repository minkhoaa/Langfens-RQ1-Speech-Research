# LANGFENS RQ1-METHOD-SELECT-02 — Reproduction Guide & Evidence Receipts

**Campaign:** LANGFENS RQ1-METHOD-SELECT-02  
**Git HEAD:** `aa4e6ea2147180a4a820d77a71911f333fcb6721`  
**Execution Environment:** Standardized PyTorch 2.6.0+cu124, Python 3.11.17, CUDA 12.4, GTX 1650 4GB  
**Protocol:** EVAL-V3.1, Speaker-Grouped TRAIN (1,800 utterances, 12 speakers)  

---

## 1. Quickstep Reproduction Runbook

All experiments in this campaign are 100% deterministic, self-contained, and reproducible from the repository root:

```bash
cd /home/khoa/Projects/New-Method-Research

# 1. Run Phase 0 integrity preflight gate
.venv/bin/python experiments/rq1_method_select_02/common/verify_integrity.py

# 2. Run EXP-01 discrepancy resolution and multi-layer scalar mixing campaign (Seeds 101, 202, 303)
.venv/bin/python experiments/rq1_method_select_02/exp01_encoder_adaptation/audit_discrepancy.py
.venv/bin/python experiments/rq1_method_select_02/exp01_encoder_adaptation/run_exp01.py

# 3. Run EXP-02 reference-conditioned joint alignment campaign (Controls 1, 2, 3)
.venv/bin/python experiments/rq1_method_select_02/exp02_joint_alignment/run_exp02.py

# 4. Run EXP-03 structured joint S/O/I learning campaign (Configs 1, 2, 3, 4, 5)
.venv/bin/python experiments/rq1_method_select_02/exp03_structured_learning/run_exp03.py

# 5. Run EXP-04 literature baselines and perturbation audit (GOP/CTC, Attention MDD, Jitter Audit)
.venv/bin/python experiments/rq1_method_select_02/exp04_external_controls/run_exp04.py

# 6. Run Supervisor consolidation and metric table synthesis
.venv/bin/python experiments/rq1_method_select_02/supervisor/synthesize_results.py
```

Total execution wall time for the complete campaign is **under 15 minutes** on the single GTX 1650 GPU.

---

## 2. Cryptographic Hashes of Key Manifests & Sources

```json
{
  "dataset_manifests": {
    "train_manifest": "data/derived/l2arctic/manifests/train.jsonl",
    "train_manifest_sha256": "1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff",
    "split_spec": "data/split_specs/L2_ARCTIC_SPLIT.yaml",
    "split_spec_sha256": "27f00d3d4bdf7d4f8ff4b09cfbe5f8e6b1856cf784aa926ea7247a8ee934f07a"
  },
  "precomputed_features": {
    "alignment_centers": "experiments/rq1_encoder_bench_01/features/alignment_centers.npz",
    "alignment_centers_sha256": "63fb37060ba10cbe6a85816999a38f72ad5611cfab5d43cae235e16b9cbbe2f7",
    "target_refs": "research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/train_reference.npy",
    "target_refs_sha256": "a353906ea19fbff4b24e6522c7a72d31c4f9bc2c002cb00fe277fc4e797fc796",
    "gap_refs": "experiments/rq1_encoder_bench_01/features/gap_references.npy",
    "gap_refs_sha256": "3cb6c8d7e008a4f91ba7dc00966a98dfc38f1f727c62bb95f87b8f066b5936ad"
  },
  "experiment_scripts": {
    "verify_integrity": "experiments/rq1_method_select_02/common/verify_integrity.py",
    "audit_discrepancy": "experiments/rq1_method_select_02/exp01_encoder_adaptation/audit_discrepancy.py",
    "run_exp01": "experiments/rq1_method_select_02/exp01_encoder_adaptation/run_exp01.py",
    "run_exp02": "experiments/rq1_method_select_02/exp02_joint_alignment/run_exp02.py",
    "run_exp03": "experiments/rq1_method_select_02/exp03_structured_learning/run_exp03.py",
    "run_exp04": "experiments/rq1_method_select_02/exp04_external_controls/run_exp04.py",
    "synthesize_results": "experiments/rq1_method_select_02/supervisor/synthesize_results.py"
  },
  "results_receipts": {
    "exp01_results": "experiments/rq1_method_select_02/exp01_encoder_adaptation/EXP01_RESULTS.json",
    "exp02_results": "experiments/rq1_method_select_02/exp02_joint_alignment/EXP02_RESULTS.json",
    "exp03_results": "experiments/rq1_method_select_02/exp03_structured_learning/EXP03_RESULTS.json",
    "exp04_results": "experiments/rq1_method_select_02/exp04_external_controls/EXP04_RESULTS.json",
    "master_results_csv": "reports/RQ1_METHOD_SELECT_02/MASTER_RESULTS.csv",
    "classwise_fwr_csv": "reports/RQ1_METHOD_SELECT_02/CLASSWISE_AND_FWR_ANALYSIS.csv",
    "ablation_matrix_csv": "reports/RQ1_METHOD_SELECT_02/ABLATION_MATRIX.csv"
  }
}
```

---

## 3. Evidence Audit Receipt & Firewalls

- **Firewalled Sets:** `DEV_SEARCH`, `DEV_CAL`, and `TEST_SEALED` manifests were **zero-touched**. Not a single inference pass, feature extraction, or threshold was derived from protected holdouts.
- **Leakage Prevention:**
  - Standardizers fit strictly on fit-partition speakers ($\mu, \sigma$ derived only on 9 training speakers per outer fold).
  - Operating policies and warning thresholds $(\tau_S, \tau_O, \tau_I)$ frozen strictly on inner 3-fold OOF predictions before outer held fold evaluation.
  - Zero test-label tuning or metric-based checkpoint selection.
- **Hardware Isolation:** Execution confined strictly to local GTX 1650 4GB GPU; peak memory never exceeded 780 MB VRAM.
