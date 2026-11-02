# DIAG-02B initial execution interruption

Base/HEAD: `dd2dee9a2063284d2a0f9fbf1c18315f3a7cd774`; branch `feat/rq1-diag-02b-counterfactual-validity`.

Initial new regression-test collection failed on an unclosed test dictionary; corrected before inference; then 7 tests passed. No backward/optimizer/training calls were executed.

First CPU inference command: `PYTHONPATH=.:src HF_HUB_OFFLINE=1 .venv/bin/python -u audit_scripts/diagnose_counterfactual_validity.py --outdir manifests/RQ1-DIAG-02B/evaluation`.

Stopped at Fold 1 RRBI (after 18 printed utterances) with:

```
AssertionError: Archived prediction parity failed: 0.0003131777048110962
```

Declared archived probability tolerance: `< 0.000200`. Diagnostic head and all-open masking matched `model.forward()` at `atol=rtol=2e-5` before this check; repeat forwards were exact. CPU PyTorch `2.14.1+cpu`; historical scores are CUDA FP32. Backend drift is plausible but NOT established as the exclusive cause. No relaxed-tolerance PASS is claimed.

The original `evaluation/RUN_STATE.json` is a historical start marker, NOT evidence of ongoing execution or completion. `evaluation/` contains only pre-inference matching/duration tables; it contains no completed inference report. It is preserved without replacement.

Safe continuation: use a NEW output directory, explicitly opt into `--continue-on-archive-drift` to collect non-confirmatory sensitivity diagnostics and a full drift ledger while retaining the original threshold. This does not waive parity acceptance. Final status must be PARTIAL if any archived-score failures remain. All input checkpoints remain read-only, selected endpoints remain 3,600 successful training updates; zero training/skipped training updates in this diagnostic. No sealed TEST audio or annotations read.
