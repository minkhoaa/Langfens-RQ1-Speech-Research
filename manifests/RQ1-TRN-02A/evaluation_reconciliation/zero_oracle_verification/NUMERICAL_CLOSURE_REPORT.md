# Protocol Z Numerical Closure

**Status:** `RQ1-TRN-02A-NUMERICAL-CLOSURE: PASS`

## Findings and corrections

- Reconstructed aligned-only labels and scores directly from all 9,919 `aligned_pairs` in `PROTOCOL_Z_PREDICTION_RECORDS.json`. At threshold 0.20: **TP=605, FP=1,510, FN=540, TN=7,264; F1=0.3712**. At threshold 0.50: **TP=208, FP=213, FN=937, TN=8,561**. The original `PROTOCOL_Z_VERIFICATION_REPORT.json` is correct; the manual amendment in `final_errata/FINAL_ERRATA_METRICS.json` (614/1,491/531/7,283; F1=0.3780) was incorrect. Its cause was amending aligned-only metrics with unsupported confusion counts; direct reconstruction does not reproduce them. Downstream final errata numbers and wording are corrected.
- The evaluator now computes B3 for every independently reconstructed tie-break population. B3 APs are `sub_first=0.307811`, `del_first=0.308147`, `ins_first=0.309224` (range **0.001413**). The blanket claim that all imputation policies vary by less than 0.0006 was false and has been retracted. Other reported B3 values match the reconstruction.
- AP remains 0.313530 on the aligned-only population. Protocol O historical artifacts were not changed.

## Provenance and safeguards

The artifacts record evaluator execution SHA `03854dca2dce81fbfcf1e2715eaf399af477dd62`; this is distinct from both the supplied base source commit `9b23d8238343d7ceeb4e0c837dfe02b7c3236602` and this closure's source commit (the commit containing this report). No new model inference was run. Confusion matrices and B3 are computed from the exported prediction records; final reports are explicitly amended deliverables, not artifacts of a fresh inference execution.

Verified unchanged SHA-256:
- `checkpoints/RQ1-TRN-02A/recovered/best.pt`: `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`
- `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`

Regression verification: `PYTHONPATH=src:. .venv/bin/pytest -q tests/test_protocol_z_alignment_and_scoring.py` — **11 passed**. No training, optimizer updates, official test access, or checkpoint modification.
