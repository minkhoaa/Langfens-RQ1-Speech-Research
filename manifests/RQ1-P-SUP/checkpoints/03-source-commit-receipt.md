# P-SUP source commit and final validation receipt

- Source implementation commit: `d2b9702627fa03c7d2de97ff8e825bb3ffbe0b86` (`feat(rq1-p-sup): implement isolated synthetic prototype`), branch `prep/rq1-p-sup`, parent was handoff HEAD `d6ef18443752b0add3f3dfcfff42f7fa7be967c1`.
- Exact source commit above was tested after commit with `PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_p_sup_labels.py tests/test_p_sup_model.py tests/test_p_sup_runtime.py`: 88 passed (27+11+50 collected tests).
- On the same commit, `.../python audit_scripts/p_sup_preflight.py` returned `PREPARATION_AUDITED_EXPERIMENT_HOLD`, `training=NOT_STARTED`, 0 successful updates, 0 skipped updates, 0 attempts, no model artifact and no training entrypoint. Missing integration and authorization blockers are enumerated in that output and checkpoint 02.
- `git diff --check HEAD^ HEAD` passed. The diff contains 19 isolated P-SUP files; no changes in historical E0/E1 model/parser/normalizer/trainer/evaluator/metrics/config or frozen shared contract. Contract SHA-256 verified as `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- After source commit, `git status --porcelain=v1` was empty. Documentation receipt/index update is a separate additive commit; do not confuse that documentation commit with the exact tested source SHA.

No remote command is provided: current tree has a static source-tree audit, not a runnable GPU preflight/trainer or deploymentable training package. Never push or run the Kaggle metadata artifact without separate approvals and missing implementation gates.
