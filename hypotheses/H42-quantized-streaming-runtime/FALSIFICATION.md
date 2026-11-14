# Falsification Criteria: Hypothesis H42

Hypothesis H42 will be declared **FALSIFIED** if any of the following failure conditions occur during empirical evaluation on the 300 held-out utterances of `DEV_REPLICATION`:

1. **Parity Loss Violation:**
   - Absolute drop in Macro AP under INT8 quantization exceeds 0.0100 ($\Delta \text{Macro AP} > 0.0100$).
   - Absolute drop in Weighted Utility exceeds 0.0100 ($\Delta \text{Utility} > 0.0100$).
2. **Latency / RTF Target Miss:**
   - Mean single-utterance CPU inference latency exceeds **15.0 ms** on the host Intel Core i5-12450H CPU.
   - Real-Time Factor (RTF) exceeds **0.005** (must be at least $200\times$ faster than real time).
3. **Memory Footprint Failure:**
   - Serialized INT8 model file size exceeds **1.50 MB**.
4. **False Alarm Explosion:**
   - False Warning Rate (FWR) under INT8 quantization increases by more than 0.20 false alarms per utterance compared to the FP32 baseline.
5. **Runtime Crash / Instability:**
   - Any runtime failure, non-finite (NaN/Inf) score, or failure on edge-case audio sequences.
