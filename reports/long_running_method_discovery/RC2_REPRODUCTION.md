# LANGFENS RC2 — reproduction and evidence provenance

All commands assume repository root `/home/khoa/Projects/New-Method-Research`. No test suite was run for this cycle.

## H-001 optional-phone frozen-emission comparison

- Experiment: `RC2-H001-A-B-OPTIONAL-PHONE-CPU-20261107`.
- Run artifact: `research/long_running_method_discovery/cycle_02/h001_alignment/run_20261107_viterbi_optional_cpu_v3/`.
- Script SHA-256: `c0e9602c130e24b18681f0ae792fd3574e1d5a2102e45f02a536edd5cd54050a`.
- EVAL-V3 receipt SHA-256: `a33af22380141b174c86d790a0d2c4959e1a31ffcca9ebb7b6c197c91ab1ced8`.
- Frame model weights SHA-256 `6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1`; config `7406aa4f917267640865688aa62f2337664a3abb9a49a2f204d932b53aeb6cb7`.
- Full details, event hashes, predictions and run command: [H-001 report](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H001_ALIGNMENT.md).
- CTC forward-sum was deliberately not run because this cache is from a frame classifier, not a verified CTC-trained phone model.

## H-003 local temporal insertion probe

```bash
.venv/bin/python research/long_running_method_discovery/cycle_02/h003_insertion/run_h003_linear_probe.py
```

Experiment: `RC2-H003-CPU-LOGREG-001`. Runner SHA-256 `066ae9bcee39fc82e491a6b42aa58e5a540e286234dd862a9ba50f3c5a29dd40`; result JSON SHA-256 `a7f201851570fb79569fcd72f2a84a0c66c46cff8503c2d39ca655d01187ded4`. Frozen checkpoint SHA-256 `8ce4ebe5f5e4ebb6fcbd02a9599328059e51d5ecc3bd9f52bdb66c822f994090`. The runner writes inside its dedicated `linear_probe/` namespace; to reproduce without replacing this record, first copy the script/artifact inputs to a new versioned output folder and change its `OUT` path. Do not rerun into the archived result directory.

An independent AP calculation can be reproduced from each saved score NPZ and `data/derived/l2arctic/manifests/dev_search.jsonl`; it must join exact `(utterance_id, gap_position)` keys and sum precision at grouped descending-score ties. The check in this session found exact-gap population 20,354, positives 125, and AP `0.040022890777950` static / `0.011320187270797` local.

## H-005 existing reliability artifact

Read-only inputs: `experiments/evaluation_v3/closure/confidence_recovered_v1/CAL_SEARCH_SHIFT.csv`, `calibrator_recipes.json`, and `research/evaluation_v3/evaluator.py` (EVAL-V3.1). SHA-256 values and row-selection procedure are recorded in [H-005 report](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H005_CALIBRATION.md). No calibration fit was rerun.

## Other workstreams

H-002, H-004 and H-006 are CPU source/provenance audits and experiment specifications; they have no Cycle 2 prediction/checkpoint metrics. Their six technical reports link from [the final synthesis](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_FINAL_SCIENTIFIC_REPORT.md).

## Runtime constraints

`nvidia-smi` could not communicate with the NVIDIA driver. No neural training or model inference was run in Cycle 2, no GPU lease was claimed, and there are no Cycle 2 CUDA receipts. No TEST_SEALED/protected data were accessed. DEV_SEARCH remains development-exposed and is not independent validation.
