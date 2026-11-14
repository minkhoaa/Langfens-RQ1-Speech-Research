# Falsification Criteria: Hypothesis H61

Hypothesis H61 will be declared **FALSIFIED** if any of the following failure conditions occur when profiling INT8-DSCA-Net and INT8-TDIS-Net on held-out `DEV_REPLICATION`:

1. **Parity Loss Violation:**
   - Absolute drop in Macro AP under INT8 quantization exceeds 0.0100 ($|\Delta \text{Macro AP}| > 0.0100$).
   - Absolute drop in Weighted Utility exceeds 0.0100 ($|\Delta \text{Utility}| > 0.0100$).
   - Absolute drop in Insertion F1 exceeds 0.0150 ($|\Delta \text{Insertion F1}| > 0.0150$).
2. **Latency / RTF Target Miss:**
   - Mean dual-engine CPU inference latency exceeds **12.0 ms** on the host Intel Core i5-12450H CPU.
   - Real-Time Factor (RTF) exceeds **0.010** (must be at least $100\times$ faster than real time).
3. **Memory Footprint Failure:**
   - Serialized INT8 model file size exceeds **2.00 MB** for either model.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) under INT8 quantization increases by more than 0.25 false alarms per utterance compared to FP32 baselines.
5. **Numerical Instability:**
   - Any runtime failure, non-finite (NaN/Inf) score, or failure on edge-case audio inputs.
