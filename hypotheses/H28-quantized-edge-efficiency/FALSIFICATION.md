# Falsification Criteria: Hypothesis H28

## 1. Primary Falsification Gates
Hypothesis H28 will be considered **FALSIFIED** if on `DEV_REPLICATION`:
1. **Accuracy Degradation:** INT8 Macro F1 drops by $> 0.0150$ compared to FP32 baseline (e.g. falling below 0.5570).
2. **Utility Collapse:** INT8 Utility drops by $> 0.0150$ (falling below 0.4080).
3. **Insufficient Compression:** Compressed model footprint remains $> 1.0\text{ MB}$ (compression ratio $< 50\%$).
4. **Latency Regression:** CPU inference latency increases by $> 25\%$ compared to FP32.

## 2. Inconclusive Conditions
1. Precision increases while Recall decreases symmetrically due to weight clipping, resulting in an ambiguous trade-off.
