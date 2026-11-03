# Independent QA — P-SCORE / P-SUP Luna implementation handoff

## Disposition

- **P-SCORE:** isolated math/runtime prototype is reviewable, but **implementation integration HOLD**. All authored unit tests pass; independent application of the shared strict runtime validator rejects real emitted P-SCORE rows because `raw_scores.posterior_target_occupancy` violates the frozen runtime-name rule banning metadata names containing `target`. Rename to a non-leakage name (e.g. `posterior_phone_occupancy`) in a reviewed arm-source correction, then rerun the shared integration test and scoped suite. QA did not patch arm code.
- **P-SUP:** **bounded synthetic prototype review GO**. Labels/control/objective/model-forward behavior and actual runtime rows pass scoped tests and independent shared-validator checks. This does not establish a trainer, experiment, scientific or deployment readiness.
- **Both experiments/training/deployment:** **HOLD**. No winner/ranking/performance claim. Separate authorization and all scientific/provenance/privacy/persistence gates remain required.

## Repository and immutable-source audit

QA initial heads: coordinator `prep/rq1-shared-base` `21999dcf57f3e6f72c39158f1a959441b80095ed`; P-SCORE `prep/rq1-p-score` `e9a97c620b7cf4e01284583ef5fee2cfdd35dc4e`; P-SUP `prep/rq1-p-sup` `599c9e65da3bf156f5ebc6a936372bd0c09c46ac`. All exact merge-bases are frozen base `0d268ff8c444e16fd76aca2f4ad6ece966908edc`. The 57 entries in `SHARED_PROVENANCE.json` independently hash-match in each of the three checkouts (171/171). All three V1 contract bytes hash to `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`. The eight newly added shared files were copied byte-identically to both arms; generated `__pycache__` and inventories/checkpoints were not copied as shared source. The distinct latest omission proposal remains unlocated.

`git diff` against the frozen base is empty for protected V1, historical model/parser/normalizer/trainer/evaluator/metrics/config files in all three repositories. Arm additions remain isolated; no new dataset, audio, label/timestamp, checkpoint/weight, credential, secret or large binary was introduced. P-SCORE independently implements named `ctc-union-language-v1` S/SD with collapsed output-string deduplication and ordinary CTC repeat rules; it does not claim author restricted topology, author code/license permission, or Eq.13 parity. Exhaustive tiny acoustic-path tests independently verify path masses, repeats, deletion, and occupancy. P-SUP adds strict TRAIN-only labels separated from runtime input, one three-logit CE treatment, three-logit collapsed binary control, cloned identical initial tensors with separate storage/RNG, feature routing, padding masks, and frozen/no-grad injected encoder. Tests use synthetic fixtures only; no backward/optimizer call.

P-SCORE has no metric/evaluation module (no grouped AP/tie policy, PER calculation, or empirical coverage analysis); coverage accounting is runtime row/status counts only. P-SUP similarly provides no experiment evaluator. Do not mistake source tests or metadata for evaluation evidence.

## Commands and evidence

Environment for pytest runs: repository-local source paths; `/home/khoa/Projects/KLTN/Dataset/.venv/bin/python` (Python 3.14.9, pytest 9.1.1, torch 2.14.1+cpu); `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. All executions were local CPU, synthetic-only or metadata-only.

- P-SCORE suite (**251 passed**, 245 CTC + 6 runtime/cache/CLI), exact command:
  `cd /home/khoa/Projects/KLTN/langfens-rq1-p-score && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_p_score_ctc.py tests/test_p_score_runtime.py`
  Exhaustive tiny-lattice brute-force cases included. CLI help command `cd /home/khoa/Projects/KLTN/langfens-rq1-p-score && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python audit_scripts/p_score_infer.py --help` passed. The suite's end-to-end CLI fixture is synthetic and verifies all four scorer coverage keys, `training=NOT_STARTED`, `weights_loaded=false`, and no-overwrite behavior.
- P-SUP suite (**88 passed**, 27 label + 11 model + 50 runtime/preflight), exact command:
  `cd /home/khoa/Projects/KLTN/langfens-rq1-p-sup && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_p_sup_labels.py tests/test_p_sup_model.py tests/test_p_sup_runtime.py`
  Forward/loss algebra ran under `torch.no_grad()`; preflight reported `PREPARATION_AUDITED_EXPERIMENT_HOLD`, `training_entrypoint_available=false`, 0 attempts/updates, no artifact. Exact safe preflight command: `cd /home/khoa/Projects/KLTN/langfens-rq1-p-sup && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python audit_scripts/p_sup_preflight.py`. It reads only frozen split/provenance metadata for its audit.
- Shared schema (**22 passed**, JSON-only), exact command:
  `cd /home/khoa/Projects/KLTN/Dataset/manifests/RQ1-WORKSPACE-PREP-01/implementation-readiness && PYTHONPATH=.:/home/khoa/Projects/KLTN/Dataset/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m unittest -v test_shared_runtime`.
- Existing safe regression (**9 passed**), exact command:
  `cd /home/khoa/Projects/KLTN/Dataset && PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_normalizer.py tests/test_parsers.py::test_parse_gold_textgrid_intervals tests/test_protocol_z_alignment_and_scoring.py::test_alignment_correctness_synthetic_cases tests/test_protocol_z_alignment_and_scoring.py::test_ambiguous_alignment_tie_break_divergence`. No other inherited tests run.
- Fingerprint audit: 57/57 in all trees; byte comparison: 8/8 shared artifacts identical in each arm.
- Independent shared-validator bridge: P-SUP two treatment/control repeated-position outputs and explicit failure row pass. P-SCORE generated rows for SF-S, SF-SD, greedy-DP and GOP-SA; validator rejects on `posterior_target_occupancy`. Removing that one key allows the SF-SD row to pass, confirming the reported blocker.
- P-SUP offline preflight CLI passed with exact HOLD status. No Kaggle/Colab/GPU/API command was run.

The initial test invocation from coordinator root for arm-relative paths failed with “file or directory not found”; rerunning from each correct arm worktree passed. The first cross-arm schema bridge tried importing both same-named `langfens_prep` packages in one interpreter and failed namespace resolution; isolated arm interpreters exposed the actual P-SCORE schema mismatch and P-SUP pass. These are recorded as invocation/inspection evidence, not product test passes.

## Remaining gates and resources

- Missing latest distinct UIT omission proposal; no substitute claim.
- P-SCORE schema-name mismatch above; source/license permission, exact author topology/parity, prospective effect/noninferiority/false-alarm/coverage margins, frozen-cache provenance/readback, and later authorized validation remain open.
- P-SUP has no trainer, data adapter/allowlist+hash ingestion, atomic private per-epoch checkpoint durability/recovery, or executable remote package. Distinct-objective policy and execution authorization, subtype support/margins/license/missing proposal, private T4/FP16/storage readiness remain open.
- DIAG-02B remains PARTIAL with 9/80 parity discrepancies and causal-training NO-GO; E1 rollout remains NO-GO.
- Resource estimates are qualitative, not measurements: P-SCORE cost scales with frames, prompt positions, vocabulary alternatives and CTC path-state work; P-SUP future GPU resource use is unbenchmarked. No performance or compute benchmark was run.

Training/inference state: **NOT_STARTED; 0 successful updates, 0 skipped updates, 0 optimizer attempts; no new artifact.** No real audio/data/labels/timestamps, weights, corpus inference, active TEST contents, fitting/backward/training, GPU, remote API, staging, deployment or push was accessed. Historical E0/E1 hashes are archive metadata only; no actual weight or durable-store verification. Existing nested subrepository at `repo` was untouched; original status and binary diff digest `68e0959819ed0673918da0d097bb365cd5f1a749055399f9fc477c44c8ec2b50` remain preserved.

Safe next work: correct only the named P-SCORE runtime field after reviewer agreement; run that bridge test and listed synthetic suite; update exact tested SHA/checkpoints. Then independent review and separately scoped operator authorization are still required before any real-data experiment or remote execution. P-SUP has no runnable training command; do not stage or run its metadata-only Kaggle preflight package as training.
