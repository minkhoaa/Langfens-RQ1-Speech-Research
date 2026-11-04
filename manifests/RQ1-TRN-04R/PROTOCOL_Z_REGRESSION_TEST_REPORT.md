# RQ1-TRN-04R Protocol Z regression tests (Workstream D)

## Scope and status

This is a synthetic/no-training regression suite, not a real-data benchmark. Frozen contract inspected: `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`, SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`. Historical runner inspected: `audit_scripts/evaluate_protocol_z.py`; recovered API inspected read-only in sibling worktree: P-SUP `runtime.py`, `dataset.py`, and `evaluator.py`. Workstream A's new `trn04r/runtime.py` and B's new metrics API were not present in this lane at test authoring. Tests target available recovered runtime serializer plus old alignment/AP helpers; they do not claim integrated production-runner coverage.

## Regression coverage added

`tests/trn04/test_protocol_z_regression.py` covers: prompt-only deterministic inference stand-in with changed counterfactual gold; forbidden gold fields, nonfinite output and sealed-speaker rejection; T-OP versus B-Control score semantics; expected prompt coverage including missing/duplicate/extra positions; deterministic alignment and unmapped positions; archival 145-deletion denominator accounting; grouped-tie full-precision AP; CAL-only threshold fixture; separation of CTC recognition deletion from learner omission; mismatch control changing audio while holding prompt IDs constant; repeated-phone position order.

The counterexample is explicitly a tiny deterministic stand-in, not A-runtime or model integration. The test therefore demonstrates the required causal separation at the API boundary but does not verify the new integrated acoustic model call path. A follow-up regression asserts `class_focal` retains three-way subtype scores while `binary_collapsed` retains null subtype semantics. It currently FAILS against the available serializer because `class_focal` is treated as collapsed; A has been notified to fix the new serializer while preserving historical files. Rerun after integration.

The audit also records two pending evidence checks: establish FIT provenance for acoustic-feature constants from an inspectable receipt, and verify frozen/hash-sealed runtime outputs for both CAL and VAL before any gold labels are joined.

## Execution

Pending integration: run `pytest -q tests/trn04/test_protocol_z_regression.py` after parent integration resolves imports/API. No training, backward, optimizer, model loading/mutation, real audio inference, dataset archive access, VAL tuning, TEST payload access, artifact publication, or privacy change was performed in this lane.
