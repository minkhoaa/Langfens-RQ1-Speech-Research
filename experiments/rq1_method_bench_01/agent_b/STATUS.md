# Agent B status — RQ1-METHOD-BENCH-01

- CUDA state inspected 2026-11-08: GPU idle at inspection, but the RQ1-UNIFIED-02 handoff reserves the sole device slot for Agent A until explicit release. No training/inference launched.
- Lease mechanism inspected: `coordination/phase3_gpu_job.py` takes a shared flock on the cross-project global lock, then an exclusive local slot flock and checks >=700 MiB free VRAM. Use this wrapper only after the shared RQ1-U02 owner releases and a fresh campaign slot is granted.
- Benchmark contract: `research.evaluation_v3.evaluator.UnifiedEvaluatorV3.evaluate(...)`, version EVAL-V3.1. Fit on TRAIN only; DEV_CAL for permitted calibration/selection; DEV_SEARCH descriptive only and never fit thresholds, weights, or calibrators.
- External method selection awaits Agent A's sourced candidate handoff. No method is being presented as reproduced before that evidence and a runnable experiment exist.
