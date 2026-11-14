# Research Plan: Experiment EXP-66 (Dual-Engine Edge Streaming Quantization Benchmark)

## 1. Technical Formulation & Profiling Protocol
Package, quantize, and profile the dual-engine edge architecture uniting:
- **Engine A (Target Phone Specialist):** INT8-DSCA-Net (EXP-55) predicting `CORRECT`, `OMISSION`, `SUBSTITUTION`.
- **Engine B (Gap Insertion Specialist):** INT8-TDIS-Net (EXP-63) predicting `INSERTION` across triad gap regimes.

### Quantization Protocol
- Technique: `torch.ao.quantization.quantize_dynamic` targeting `nn.Linear` layers with `qint8` activations and weights.
- Targets: All projection, attention, and head linear modules.

### Benchmarking Protocol
- Hardware: Intel Core i5-12450H CPU (4 execution threads).
- Cohort: Full held-out `DEV_REPLICATION` (300 utterances, 9,863 target phones, 10,163 inter-phone gaps).
- Profiling: 10 warmup runs + 300 benchmark iterations measuring Mean, Median, P90, P99 latency, and Real-Time Factor (RTF = $\text{latency} / \text{audio\_duration}$).

## 2. Experimental Steps
1. Load FP32 checkpoints of DSCA-Net (`experiments/EXP-55-.../CHECKPOINTS/dsca_net.pt`) and TDIS-Net (`experiments/EXP-63-.../CHECKPOINTS/tdis_net.pt`).
2. Quantize both models to INT8 and measure serialized file footprints.
3. Profile CPU latency and throughput for:
   - Standalone INT8-DSCA-Net
   - Standalone INT8-TDIS-Net
   - Unified Dual-Engine (INT8-DSCA + INT8-TDIS)
4. Evaluate diagnostic parity against FP32 baselines under locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
5. Package `tools/stream_edge_dual_engine.py` with standalone CLI self-test.
6. Audit against H61 falsification criteria.
