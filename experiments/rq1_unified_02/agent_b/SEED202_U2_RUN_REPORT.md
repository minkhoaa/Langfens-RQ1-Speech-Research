# Seed-202 flat U2 replay — Agent B

## Execution

Ran only the assigned flat U2 arm under the cooperative CUDA lease, with the byte-identical local copy of the source trainer:

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job RQ1-U2-FLAT-TWOHEAD-S202-AGENTB -- .venv/bin/python experiments/rq1_unified_02/agent_b/train_twohead_local.py --seed 202 --epochs 8 --target-head flat
```

Lease ended successfully (`returncode=0`, 32.1341 s; telemetry in `coordination/PHASE3_GPU_LEDGER.jsonl`). The GPU is now released. No U3 run was launched. Device: GTX 1650; training time 26.6632 s; peak allocated memory 41.327 MiB. Eight epochs completed; 59,562 targets / 61,362 exact gaps; class counts and data hashes in receipt.

Trainer copy SHA-256 equals source (`12ec7bf6a288acfa3178458145d0cbdf905120cc500831fa27192031a5f52736`). All outputs are in this agent directory.

## EVAL-V3.1 exploratory SEARCH metrics

Coverage was 100% for both target and gap events; zero simplex violations. AP-S **0.3625582**, AP-O **0.1701592**, AP-I **0.1375528**, macro-3 error AP **0.2234234**. Macro-3 error F1 at CAL-selected budgets 0.10 / 0.20 was **0.0928821 / 0.1190903**. At the 0.20 budget, joint FWR was 0.2170 and exceeded the 0.20 reference slightly. Unrestricted F1/AP are not operating-point recommendations. SEARCH is development-exposed and descriptive.

Compared with existing seed-101 U2 EVAL-V3.1 APs (S/O/I .3713596/.1853183/.1179922), seed-202 deltas are **−.0088014 / −.0151591 / +.0195606**, macro-3 AP delta **−.0014667**. This is cross-seed variation, not a treatment contrast; seed-202 U3 has not been run.

## Artifact provenance

- Checkpoint: `RQ1-U2-FLAT-TWOHEAD-S202.pt`; SHA-256 `59072056d761d9b178e0eb4c8956adcd3606ce2d3d9df516af459217e06108de`.
- Receipt: `RQ1-U2-FLAT-TWOHEAD-S202_receipt.json`.
- EVAL-V3.1: `RQ1-U2-FLAT-TWOHEAD-S202_EVAL-V3.1.json`.
- Predictions: CAL `RQ1-U2-FLAT-TWOHEAD-S202_dev_cal.npz`, SHA-256 `ee76ff92a2a56aefbc11338c77a9f569f1a91d6af4516446e46e8680b05370c2`; SEARCH `RQ1-U2-FLAT-TWOHEAD-S202_dev_search.npz`, SHA-256 `5c0a26386d8d63c9666aacf742d9be47096b6b929a95652610be5008a75bd35a`.
- Input cache hashes match seed-101 source receipts: target frames `eb8b23a2…`, labels `8e2be711…`, references `4a03dacc…`, train gap contexts `1e4f4727…`.

Next action requires a fresh explicit slot assignment; this report grants no authorization to start hierarchical U3.
