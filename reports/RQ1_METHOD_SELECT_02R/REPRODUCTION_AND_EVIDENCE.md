# LANGFENS RQ1-METHOD-SELECT-02R — Reproduction and Evidence Package

**Standard:** EVAL-V3.1, Exact N target / N+1 gap evaluation contract  
**Repository:** `/home/khoa/Projects/New-Method-Research`  
**Git HEAD:** `aa4e6ea2147180a4a820d77a71911f333fcb6721`  
**Execution Environment:** Standardized PyTorch 2.6.0+cu124, Python 3.11.17, CUDA 12.4, NVIDIA GeForce GTX 1650 4GB  

---

## 1. Quick-Start Exact Reproduction Commands

All experimentation and audits are fully automated, deterministic, and self-contained. To replicate all results from scratch:

```bash
# Activate virtual environment
source .venv/bin/activate

# 1. Reproduce Agent A: SM13 Training, Evaluation & Checkpointing across all seeds (101, 202, 303)
python experiments/rq1_method_select_02r/agent_a/reproduce_sm13.py

# 2. Generate Agent A Paired Head-to-Head Comparison Table
python experiments/rq1_method_select_02r/agent_a/generate_paired_comparison.py

# 3. Reproduce Agent B: Bounded Encoder Fine-Tuning & Profiling on GPU
python experiments/rq1_method_select_02r/agent_b/train_fine_tuned.py
python experiments/rq1_method_select_02r/agent_b/evaluate_controls.py

# 4. Reproduce Agent C: Matched-FWR and Classwise Speaker Audit
python experiments/rq1_method_select_02r/agent_c/audit_matched_fwr.py

# 5. Run Supervisor Final Acceptance & Statistical Inference
python experiments/rq1_method_select_02r/supervisor/synthesize_final_acceptance.py
```

---

## 2. Checkpoint & Prediction Artifact Inventory

### 2.1 Model Checkpoints (`experiments/rq1_method_select_02r/agent_a/checkpoints/`)
All 24 outer fold checkpoints (12 Target Head models + 12 Gap Insertion Head models) are retained:

| File Name | SHA256 Hash | Size | Architecture Role |
|---|---|---:|---|
| `target_seed101_fold0.pt` | `ba36b9e4a8b792e3a19b8df790eb994191d4e2df4ae89e701ea7a67f1ee0cb65` | 2.5 MB | Target Head (Seed 101, Outer Fold 0) |
| `target_seed101_fold1.pt` | `0ea90184e62241b7dd77670e1ee100e495fe6f6343542283921312386fe8e227` | 2.5 MB | Target Head (Seed 101, Outer Fold 1) |
| `target_seed101_fold2.pt` | `ff15ce3a6c986963caec7cf402ceec561d368e7ec89fbaae3e2dfb102b37e8c3` | 2.5 MB | Target Head (Seed 101, Outer Fold 2) |
| `target_seed101_fold3.pt` | `9b360b64d1f5686be5048d88e0409da64243aeb2125bb23f338d781b0a8523c9` | 2.5 MB | Target Head (Seed 101, Outer Fold 3) |
| `gap_seed101_fold0.pt` | `c0fcce47b74f3b622c813589b8d0ca2c4fcdd316986fa8542b87fcf3ec4c2e07` | 844 KB | Gap Head (Seed 101, Outer Fold 0) |
| `gap_seed101_fold1.pt` | `88c6919ebdd8bcbe5510fc70e301d51c36f56e0996841e05d0e2e5c8e37cf712` | 844 KB | Gap Head (Seed 101, Outer Fold 1) |
| `gap_seed101_fold2.pt` | `f312cb2462e7aa222cf61f4fa55bcab6a6dfa3ff67341eaae718dfb5f9024f0c` | 844 KB | Gap Head (Seed 101, Outer Fold 2) |
| `gap_seed101_fold3.pt` | `0d2dbbb76db1a03e1e5b41ea9b0cb51bb4c483d97d02ba0646c0d80c0c66d9f7` | 844 KB | Gap Head (Seed 101, Outer Fold 3) |
| `target_seed202_fold0.pt` | `105fe8b6938a9daae9d4f0df40fc6017b3a985f5e27a6f30e66ea8d6fb7a7509` | 2.5 MB | Target Head (Seed 202, Outer Fold 0) |
| `target_seed202_fold1.pt` | `b2e4b0c7c8bc1e61eaebbb4bc8ba201e7655073e6b7c093caae5a53e925fa0a7` | 2.5 MB | Target Head (Seed 202, Outer Fold 1) |
| `target_seed202_fold2.pt` | `844ea1b0f5898d5c589ba572d4c06f3630f40d344d4715f3ec77826279f06316` | 2.5 MB | Target Head (Seed 202, Outer Fold 2) |
| `target_seed202_fold3.pt` | `e2a4be6562092cc7a22409ae03d98c2573229bca63a566c79ecbe750e3ce183a` | 2.5 MB | Target Head (Seed 202, Outer Fold 3) |
| `gap_seed202_fold0.pt` | `cf61ca2e0a293ce1842bc42168fa5534ec4c1f51f47b2b8e3ea970c64eb3f8a4` | 844 KB | Gap Head (Seed 202, Outer Fold 0) |
| `gap_seed202_fold1.pt` | `fc2d1bf439775fbf104ea264fa740e53a3a4cbe2001a4db4cbfa0210ee276188` | 844 KB | Gap Head (Seed 202, Outer Fold 1) |
| `gap_seed202_fold2.pt` | `b6a12b9beee76d6537305928d3efaa976dbce7494a530f2c418f77435f3b610c` | 844 KB | Gap Head (Seed 202, Outer Fold 2) |
| `gap_seed202_fold3.pt` | `8e6c7102e3b2e9569b97779b5b91b9b1e9cb14c46fbe8353cfc5c494ca0e3860` | 844 KB | Gap Head (Seed 202, Outer Fold 3) |
| `target_seed303_fold0.pt` | `9fa73932e6584288019e1e9c56baef1ca77dfd08c5c7e1452243d4e0ce05f6eb` | 2.5 MB | Target Head (Seed 303, Outer Fold 0) |
| `target_seed303_fold1.pt` | `ad48f7eb482a17058d88df17ae28d9ffba297d2836262cb4aa519446d3e86c03` | 2.5 MB | Target Head (Seed 303, Outer Fold 1) |
| `target_seed303_fold2.pt` | `037b524795c47936a5fc1a629fb665d914ae72bbdaeb5cf8d71206f671d4670c` | 2.5 MB | Target Head (Seed 303, Outer Fold 2) |
| `target_seed303_fold3.pt` | `0c0e7fe7582b9e693172a5a54dbbb472faee440536ee1d923769910a307c4d51` | 2.5 MB | Target Head (Seed 303, Outer Fold 3) |
| `gap_seed303_fold0.pt` | `e2a775ebcfc4c47844081c7e90956429219b265d3ec62b5bf46b2b489bf946e6` | 844 KB | Gap Head (Seed 303, Outer Fold 0) |
| `gap_seed303_fold1.pt` | `432cfa3065d6666ba4706aa7c458319f61b3fc023a1372dbf3cf9f14300ec0ea` | 844 KB | Gap Head (Seed 303, Outer Fold 1) |
| `gap_seed303_fold2.pt` | `b4916a0322ba63f8dbdb0374e2d3df24cf8f65751d2f78bdf0eb38c439f00d23` | 844 KB | Gap Head (Seed 303, Outer Fold 2) |
| `gap_seed303_fold3.pt` | `47171d9bbdb9dfa44ffae9d691eb22115167aa887cbf5a92cfd35c1566497f25` | 844 KB | Gap Head (Seed 303, Outer Fold 3) |

### 2.2 Prediction Archives (`experiments/rq1_method_select_02r/agent_a/predictions/`)

| File Name | SHA256 Hash | Size | Contents |
|---|---|---:|---|
| `predictions_sm13_seed101.npz` | `610cb5511b8b80fc27715ecb8dc51e5953e5fbf477e68e42db629094195195b0` | 3.6 MB | All 59,562 target + 61,362 gap keyed predictions, gold labels, and policy decisions (Seed 101) |
| `predictions_sm13_seed202.npz` | `9842a229a43a059d3e843f550972b2c14041e065bc598a39a03191f6f6983fc1` | 3.6 MB | All 59,562 target + 61,362 gap keyed predictions, gold labels, and policy decisions (Seed 202) |
| `predictions_sm13_seed303.npz` | `bc65bcf418bf1ce1c7908b26118b6287c88b9bf35a4d4452097ef7ff42131976` | 3.6 MB | All 59,562 target + 61,362 gap keyed predictions, gold labels, and policy decisions (Seed 303) |

### 2.3 Fine-Tuning Checkpoints (`experiments/rq1_method_select_02r/agent_b/checkpoints/`)

| File Name | Size | Contents |
|---|---:|---|
| `finetune_upper_1_block_l11_epoch_1.pt` | 30 MB | Candidate B1 (Layer 11 unfrozen, Epoch 1) |
| `finetune_upper_1_block_l11_epoch_2.pt` | 30 MB | Candidate B1 (Layer 11 unfrozen, Epoch 2) |
| `finetune_upper_2_blocks_l10_l11_epoch_1.pt` | 57 MB | Candidate B2 (Layers 10-11 unfrozen, Epoch 1) |
| `finetune_upper_2_blocks_l10_l11_epoch_2.pt` | 57 MB | Candidate B2 (Layers 10-11 unfrozen, Epoch 2) |

---

## 3. Custodial Firewall Audit Certification

The custodial firewall protecting holdout sets was maintained 100% intact throughout the campaign.

| Partition | Manifest Path | SHA256 Hash | Custody Status | Probes / Access Count |
|---|---|---|:---:|:---:|
| **TRAIN** | `data/derived/l2arctic/manifests/train.jsonl` | `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff` | Active Authorized | Evaluated in Cross-Validation |
| **DEV_SEARCH** | `data/derived/l2arctic/manifests/dev_search.jsonl` | `a4ab6c7df7bd65b5d213d306c2fdfde6c316cbf7768a3eb63d056775351f99d2` | **LOCKED PRESERVED** | **ZERO PROBES (Untouched)** |
| **DEV_CAL** | `data/derived/l2arctic/manifests/dev_cal.jsonl` | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` | **LOCKED PRESERVED** | **ZERO PROBES (Untouched)** |
| **TEST_SEALED**| `data/derived/l2arctic/manifests/test_sealed.jsonl`| `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | **STRICTLY SEALED** | **ZERO PROBES (Untouched)** |

---

## 4. Deliverable File Registry

The deliverables required under the mission mandate are deposited in `reports/RQ1_METHOD_SELECT_02R/`:
1. `MASTER_CONFIRMATION_REPORT.md`: Comprehensive synthesis of the entire confirmation campaign.
2. `SM13_REPRODUCTION_AUDIT.md`: Agent A's bit-level reproduction and forensic lineage audit.
3. `FINE_TUNING_RESULTS.md`: Agent B's hardware profiling and bounded encoder adaptation findings.
4. `MATCHED_FWR_ANALYSIS.csv`: Agent C's posthoc realized-FWR matching analysis across operating points.
5. `CLASSWISE_SPEAKER_ANALYSIS.csv`: Agent C's speaker-by-speaker breakdown across all 12 L2 speakers.
6. `PAIRED_SEED_FOLD_COMPARISON.csv`: Paired head-to-head comparison table across all 12 outer folds.
7. `FINAL_METHOD_DECISION.md`: Formal gate document adopting `LANGFENS-SM13-FUSION-V1`.
8. `FINAL_METHOD_SPEC.md`: Complete immutable architecture and engineering specification.
9. `REPRODUCTION_AND_EVIDENCE.md`: This reproduction recipe, artifact registry, and custody certification.
