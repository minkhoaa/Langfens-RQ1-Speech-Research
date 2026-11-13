# RQ1-METHOD-BENCH-02 — Reproduction and provenance

## Environment and boundaries

- Repository: `/home/khoa/Projects/New-Method-Research`.
- Evaluator: official `research.evaluation_v3.evaluator.UnifiedEvaluatorV3`, EVAL-V3.1.
- Split manifests: `data/derived/l2arctic/manifests/{train,dev_cal,dev_search}.jsonl`.
- DEV_CAL verified population: 300 utterances, 10,043 targets and 10,343 N+1 gaps. DEV_SEARCH: 599 utterances, 19,755 targets and 20,354 gaps. DEV_SEARCH was exploratory only; no parameter, threshold or model selection used it.
- TEST_SEALED/protected data were not opened. Workspace changes that predated this campaign were preserved.
- `nvidia-smi` could not communicate with the NVIDIA driver, and the historical lease files alone do not grant a safe GPU lease. Runs in this campaign used CPU.

## Commands

Run from the repository root:

```bash
./.venv/bin/python experiments/rq1_method_bench_02/agent_a/single_gap_ablation.py
./.venv/bin/python experiments/rq1_method_bench_02/agent_a/frame_gop_substitution.py
./.venv/bin/python experiments/rq1_method_bench_02/agent_a/frame_decode_insertions.py
./.venv/bin/python experiments/rq1_method_bench_02/agent_b/discrete_unit_context_screen.py
./.venv/bin/python experiments/rq1_method_bench_02/agent_b/ottc_subphonetic_screen.py
./.venv/bin/python experiments/rq1_method_bench_02/agent_c/build_ledger.py
./.venv/bin/python experiments/rq1_method_bench_02/agent_c/h05_cal_operating_and_pr.py
```

H04 command writes its receipt under `experiments/rq1_method_bench_02/agent_a/frame_gop_run_v2/`.

## Artifacts and identities

- H04: full local CharSiU frame-classifier logits and target-span scoring identities are recorded in the Agent A receipt. Do not interpret these frame scores as CTC posteriors.
- H05 topology ablation: [`single_gap_ablation.py`](../experiments/rq1_method_bench_02/agent_a/single_gap_ablation.py), [`EVALUATION_RECEIPT.json`](../experiments/rq1_method_bench_02/agent_a/single_gap_run/EVALUATION_RECEIPT.json), and `event_scores.csv`.
- H05 insertion decode: `frame_decode_insertions.py` and ambiguity-audited `frame_decode_run_v5/EVALUATION_RECEIPT.json`; every gap has a score and tied optimal mappings are marked ambiguous.
- Discrete-unit transition screen: [`discrete_unit_context_screen.py`](../experiments/rq1_method_bench_02/agent_b/discrete_unit_context_screen.py), `results.json`, and target prediction JSONLs. It reuses the MB01 K=64 codebook; receipt includes input hashes.
- Ordered-state transport adaptation: [`ottc_subphonetic_screen.py`](../experiments/rq1_method_bench_02/agent_b/ottc_subphonetic_screen.py), `ottc_results.json`, target prediction JSONLs, and checkpoint `MB02-B-OTTC-ORDERED-STATES-S101.pt` (SHA256 `6c830bd653fc078b738d370c1b0e7a59a20e2529ea8f699f628d15cbffb41da8`).
- Baseline/manifest ledger: `experiments/rq1_method_bench_02/agent_c/{manifest_audit.csv,benchmark_ledger.csv}`. Rebuild with `./.venv/bin/python experiments/rq1_method_bench_02/agent_c/build_ledger.py`.
- CAL-selected H05 thresholds and class-wise PR points: `experiments/rq1_method_bench_02/agent_c/{h05_operating_policy_metrics.csv,h05_cal_policy_receipt.json,classwise_pr_curves.csv}`. The reproduction command above uses the unchanged closure policy; the frame-count insertion arm has no CAL vector and uses only the predeclared `count > 0` operating rule.
- Full source model metadata, training procedures, population coverage, and limitations are in per-agent receipts and reports. No external code or weights were downloaded.

## Execution and limitations

All new scores were evaluated against the official manifest populations. The H04 score cache is a frame-classification model (`Wav2Vec2ForFrameClassification`), not a phoneme CTC model. The optional-phone experiment is max-sum discriminative path scoring. Only H01-derived target scores and the inherited Gold-Only gap head provide the dimensions needed for full S/O/I rows; specialist methods are reported only for supported classes. No new method was trained or evaluated on TEST.
