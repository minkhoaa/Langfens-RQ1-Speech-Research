# Research Plan: Experiment EXP-96 (Dynamic INT8 Quantization of ADAE-Net)

## 1. Methodology
1. Load FP32 ADAE-Net checkpoint from `experiments/EXP-94-ARTICULATORY-DISENTANGLED-AUTOENCODER/CHECKPOINTS/adae_net.pt`.
2. Apply PyTorch dynamic INT8 quantization to `nn.Linear` and `nn.LSTM` layers.
3. Measure disk footprint reduction (FP32 vs INT8).
4. Profile CPU inference latency over 300 utterances on Intel Core i5-12450H CPU (4 threads) with warmup.
5. Evaluate full metrics on `dev_replication` using frozen calibration thresholds from EXP-94.
6. Compute parity deltas ($\Delta \text{Macro AP}, \Delta \text{Utility}, \Delta \text{F1}$).
7. Audit against H91 falsification criteria.
8. Save artifacts, quantized weights, latency logs, and hashes in `experiments/EXP-96-PROGRESSIVE-CONFIDENCE-DYNAMIC-QUANTIZER/`.
