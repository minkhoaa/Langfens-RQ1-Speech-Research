# Seed-303 hierarchical U3 replay — Agent B

Ran only the assigned arm under the cooperative CUDA lease:

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job RQ1-U3-HIERARCHICAL-TWOHEAD-S303-AGENTB -- .venv/bin/python experiments/rq1_unified_02/agent_b/train_twohead_local.py --seed 303 --epochs 8 --target-head hierarchical
```

Lease return code 0, duration 36.9267 s from the GPU ledger; GPU is released. Training runtime 30.3318 s, peak allocated memory 41.3257 MiB on GTX 1650. Eight epochs; 59,562 targets / 61,362 gaps. Seed-303 U2/U3 receipts have identical class counts and input hashes; parameter counts are 653,893 / 653,764.

## EVAL-V3.1 and matched seed-303 comparison

U3 SEARCH coverage 100%, zero simplex violations. AP-S **0.3649132**, AP-O **0.1769628**, AP-I **0.1157384**, macro-3 error AP **0.2192048**. CAL-selected .10/.20 macro-3 F1 **0.0372500 / 0.0715253**; .20 joint FWR **0.12187**. SEARCH is exposed/descriptive.

| Metric | U2 flat | U3 hierarchical | U3−U2 |
|---|---:|---:|---:|
| AP-S | .3710644 | .3649132 | −.0061512 |
| AP-O | .1735915 | .1769628 | +.0033713 |
| AP-I | .1614510 | .1157384 | −.0457126 |
| Macro-3 AP | .2353690 | .2192048 | −.0161642 |
| CAL20 Macro-3 F1 | .0974216 | .0715253 | −.0258962 |

This seed weakens the insertion ranking substantially and has lower aggregate AP/F1 under U3. It contrasts with seed202's small macro-3 AP gain and seed101's macro-3 AP decrease; across these three matched seeds, hierarchical-minus-flat macro-3 AP deltas are −.00411, +.00150, −.01616 and CAL20 F1 deltas are −.05308 (seed101), −.04493, −.02590. This pattern is not evidence of a stable U3 benefit. It is still only three development-exposed seeds, no independent test.

## Provenance

- Checkpoint `RQ1-U3-HIERARCHICAL-TWOHEAD-S303.pt`, SHA-256 `a2d679ea1139780dbee239e85566788032928f00073803b791a0e17dc13ea379`.
- Receipt `RQ1-U3-HIERARCHICAL-TWOHEAD-S303_receipt.json`; EVAL-V3.1 `RQ1-U3-HIERARCHICAL-TWOHEAD-S303_EVAL-V3.1.json`.
- Prediction SHA-256: CAL `fd8663b78360c292b3c7fe1171ce51c80642afa529201c7c2846c6106586d14d`; SEARCH `114a40f393865f6d5d351c327ca758607e2a705963d2e6bcd0479ccc9a8ee77b`.
- Receipt/prediction hashes and EVAL schema verified. No other job launched.
