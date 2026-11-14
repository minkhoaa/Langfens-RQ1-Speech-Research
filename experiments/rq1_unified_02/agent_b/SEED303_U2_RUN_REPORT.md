# Seed-303 flat U2 replay — Agent B

## Execution

Ran only the assigned flat U2 seed303 arm under the cooperative CUDA lease:

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job RQ1-U2-FLAT-TWOHEAD-S303-AGENTB -- .venv/bin/python experiments/rq1_unified_02/agent_b/train_twohead_local.py --seed 303 --epochs 8 --target-head flat
```

Lease return code 0; duration 29.5674 s from lease ledger. GPU is released. Training ran 8 epochs in 23.5039 s; peak allocated memory 41.3271 MiB; GTX 1650. 59,562 targets / 61,362 gaps, with same class counts and input hashes as seed-101/202. No U3 seed303 or other GPU job was started.

## EVAL-V3.1 SEARCH metrics

100% event coverage, zero simplex violations. AP-S **0.3710644**, AP-O **0.1735915**, AP-I **0.1614510**, macro-3 error AP **0.2353690**. CAL-selected .10/.20 macro-3 error F1 **0.1038516 / 0.0974216**; at .20, joint FWR is .25376, exceeding .20. SEARCH is development-exposed and descriptive only.

Seed303 U2 macro-3 AP is +.011946 vs seed202 U2 (.223423), illustrating across-seed variation; it is not an intervention comparison. U3 seed303 remains pending separate slot assignment.

## Provenance

- Checkpoint: `RQ1-U2-FLAT-TWOHEAD-S303.pt`, SHA-256 `43f1c3bd93771fc669df80fafad057b4001813538cb465fb005a8dcc86f82756`.
- Receipt: `RQ1-U2-FLAT-TWOHEAD-S303_receipt.json`.
- EVAL-V3.1: `RQ1-U2-FLAT-TWOHEAD-S303_EVAL-V3.1.json`.
- Prediction SHA-256: CAL `8c218e30fc37228dbea2c00bde32e1769a5b6cfadc6c06585ba0c1c3bb435c58`; SEARCH `997d34578e2776af5d187b93773782ee0aae78caa35c4722b06a986762b4ea7d`.
- Receipt hashes and EVAL schema verified. All output files are in this directory.
