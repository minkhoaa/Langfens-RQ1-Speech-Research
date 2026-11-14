# Seed-202 hierarchical U3 replay — Agent B

## Execution

Ran only the assigned matched U3 arm under the cooperative CUDA lease:

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job RQ1-U3-HIERARCHICAL-TWOHEAD-S202-AGENTB -- .venv/bin/python experiments/rq1_unified_02/agent_b/train_twohead_local.py --seed 202 --epochs 8 --target-head hierarchical
```

Lease ended successfully (`returncode=0`, 31.0103 s; telemetry in `coordination/PHASE3_GPU_LEDGER.jsonl`). GPU released, no additional job launched. Training runtime 25.4679 s; peak allocated memory 41.3257 MiB; 8 epochs; 59,562 targets / 61,362 gaps. U2 and U3 receipts both report identical target/gap class counts and four identical input hashes. Parameter counts: U2 653,893; U3 653,764.

## EVAL-V3.1 SEARCH results and paired comparison

U3 coverage 100% for target/gap events, zero simplex violations. AP-S **0.3586540**, AP-O **0.1723827**, AP-I **0.1437239**, macro-3 error AP **0.2249202**. CAL-selected budget .10/.20 macro-3 error F1 was **0.05100 / 0.07416**; corresponding joint FWR .04174 / .10851. SEARCH is development-exposed and descriptive.

Matched U3 minus U2 seed-202 AP deltas (U2 results in `SEED202_U2_RUN_REPORT.md`):

| Metric | U2 flat | U3 hierarchical | Delta U3−U2 |
|---|---:|---:|---:|
| AP-S | .3625582 | .3586540 | −.0039042 |
| AP-O | .1701592 | .1723827 | +.0022235 |
| AP-I | .1375528 | .1437239 | +.0061711 |
| Macro-3 AP | .2234234 | .2249202 | +.0014968 |
| CAL20 macro-3 error F1 | .1190903 | .0741588 | −.0449315 |

At seed 202, hierarchical objective slightly raised O/I AP but lowered S AP; aggregate macro-3 AP gain is small (+.00150), while CAL20 F1 is lower. This is one matched seed, and the operating thresholds are separately selected on CAL, so do not claim stable hierarchical benefit. No seed303 launched.

## Provenance

- Checkpoint `RQ1-U3-HIERARCHICAL-TWOHEAD-S202.pt`; SHA-256 `3bc4d43108db07719d873f6812c6dcb27822109ab0b4c8a5fa2b0803a9b83e2b`.
- Receipt `RQ1-U3-HIERARCHICAL-TWOHEAD-S202_receipt.json`.
- EVAL-V3.1 `RQ1-U3-HIERARCHICAL-TWOHEAD-S202_EVAL-V3.1.json`.
- Predictions: CAL `000d5c8843aba7dd42d2e05b8c7ab506372b197236baa14c9440bb27c134f31a`; SEARCH `efa1b5d3a8ae4a1e6c78d9090ba98e8e9de12079460a0a1760fee4c7f2be7d70`.
- Receipt and prediction SHA-256 checks verified; EVAL schema verified. Same train cache hashes as matched seed-202 U2.
