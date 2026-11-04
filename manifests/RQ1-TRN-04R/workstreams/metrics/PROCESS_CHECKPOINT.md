# Workstream B — independent metric accounting checkpoint

**State: IMPLEMENTED / REAL-DATA RECOMPUTATION PENDING.** Campaign LANGFENS-RQ1-TRN-04R; branch `review/trn04r-b`; base `d332997e732315292d673a3bd42a03698182f5f6`. This lane did no inference, model loading/mutation, training, backward pass, optimizer update, audio/TextGrid/gold-source read, TEST access, tuning, or publication/privacy change.

## Inputs and scope verified

Read `AGENTS.md`, `docs/process-checkpoints/INDEX.md`, `manifests/RQ1-TRN-04/PROCESS_CHECKPOINT.md`, `manifests/RQ1-P-SUP-BENCHMARK-V1/final-v2/PROCESS_CHECKPOINT.md`, `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`, parent `manifests/RQ1-TRN-04R/PROTOCOL_Z_CORRECTION_REPORT.md`, and inspected the recovered runner/recompute path for exchange-field conventions. Contract separates runtime Z from offline gold joins, CAL thresholds from VAL diagnostics, and recognition edits from human error labels. Coordinator has exclusive responsibility for final benchmark execution.

## Deliverables in this lane

- `src/langfens_prep/trn04r/metrics.py`: exact grouped-tie AP; CAL correct-only thresholds; deployed full-gold counts/coverage/abstention, omission FP_correct/FP_sub/FN and full-gold FA/1k; label/speaker/phone summaries; extra-prompt alarm separation; saved CTC edit/PER accounting; factorial contrasts. Follow-up objective handling pins C0/T2 to `operation_ce`, T1/T3 to `class_focal`, requires dedicated channels for both three-way objectives, and reserves null subtype scores for genuine `binary_collapsed` records.
- `audit_scripts/recompute_trn04r_metrics.py`: model-free private-record CLI with `--endpoint fixed_budget_3600|best`, correct endpoint-prefixed directories, runtime/offline recognition separation, strict speaker census, frozen Level A/B gates on fixed-budget only, and explicit primary scalars from dedicated omission at CAL 15%.
- `tests/trn04r/test_metrics_accounting.py`: synthetic tests of tie handling, binary/omission/substitution FP disjointness, FP_other_subtype, PER by speaker, CAL/VAL identities, recognition-runtime preservation, dedicated omission CAL-15 scalar, objective semantics, gate thresholds, abstentions, extras, failures and contrast direction.
- `manifests/RQ1-TRN-04R/METRIC_ACCOUNTING_CORRECTION.md`, `INDEPENDENT_METRIC_RECOMPUTATION.json`, `CORRECTED_FACTORIAL_COMPARISON.md`: correction methods and explicit pending execution/gate status; no numbers imported from earlier reports.

## Evidence and safe restart

Executed targeted synthetic suite with `/home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/trn04r/test_metrics_accounting.py` (16 passed including accounting, endpoint-layout, identity and gate regressions), Python compilation of the new source/CLI/test, and `git diff --check`. Initial test iterations revealed and fixed separate CAL-vs-VAL join handling and full-gold omission false alarm accounting; passing tests exercise both. No real exchange rows were available or loaded, so no numerical metrics or scientific gates are reported.

After coordinator freezes the integrated source and private rows, validate the supplied manifest/receipt hashes and cell identities. Run `audit_scripts/recompute_trn04r_metrics.py --input-root <immutable-private-root> --endpoint fixed_budget_3600 --output <new-private-output>` (or `--endpoint best` for separate non-gated metrics) only on the coordinator-owned final execution path; do not overwrite outputs. The CLI resolves exactly one of `PRIVATE_ROOT/<CELL>_<endpoint>`, `PRIVATE_ROOT/arms/<CELL>/<CELL>_<endpoint>`, or `PRIVATE_ROOT/<CELL>/<CELL>_<endpoint>`; ambiguity and symlinks fail, and it creates no copies or symlinks. It validates the distinct `recognition_runtime_Z.jsonl` and offline `recognition_Z.jsonl` pair. It validates actual Z census `9956=9919+37` and actual labels `8799/1012/145`, aligned labels `8774/1008/137`, and per-operating-point `FN=145-TP`, `TP<=137`. Compare metrics independently and report absolute Level A/B gates separately from factorial contrasts; all measured outcomes remain pending until the coordinator runs on immutable inputs. No global index update was made (parent owns that file).

## State and boundaries

Implemented metric logic and synthetic validation only. Real-data recomputation, final contrasts, registered-gate evaluation, external-reference comparison and noninferiority assessment are **pending**, not executed. No source predictions, checkpoints, archives or datasets were copied/changed. Preserve originals and keep prediction/result rows in approved private storage.
