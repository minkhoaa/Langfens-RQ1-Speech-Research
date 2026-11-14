# Prepared seed-202 U2/U3 commands (not launched)

**GPU state:** Both assigned seed-202 arms completed in separate single-job leases; no further jobs were launched. GPU is released.

Local trainer is a byte-identical copy of `experiments/rq1_unified_01/agent_a/train_twohead.py` (`train_twohead_local.py`, SHA-256 `12ec7bf6a288acfa3178458145d0cbdf905120cc500831fa27192031a5f52736`). Its `OUT` resolves to Agent B's directory, so all generated contexts, checkpoints, prediction NPZs, logs/receipts, and EVAL-V3.1 results remain here. It reads frozen source representations and canonical features only. Same 8 epoch default, same cache paths, same seed 202, same optimizer/loss/settings for each matched arm.

The flat and hierarchical commands both completed with receipts and EVAL-V3.1 artifacts. The hierarchical run was separately assigned and leased; do not execute additional jobs without fresh authorization.

Arm names: `RQ1-U2-FLAT-TWOHEAD-S202` and `RQ1-U3-HIERARCHICAL-TWOHEAD-S202`. Training inherently evaluates on DEV_CAL and saves DEV_SEARCH predictions for descriptive EVAL-V3.1 replay only; DEV_SEARCH was not used for selection. Both receipts share identical input cache hashes. Seed 303 was not launched.
