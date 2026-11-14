# Research Plan: Hypothesis H28 / Experiment EXP-33

## 1. Objective
Quantify the compression ratio, inference latency, throughput, and diagnostic accuracy parity of post-training dynamic INT8 quantization on APED-Net (EXP-28) across `DEV_REPLICATION` under `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.

## 2. Experimental Steps
1. **Model Preparation & Quantization:**
   - Load pre-trained FP32 APED-Net checkpoint (`experiments/EXP-28-PROSODIC-ENERGY-DYNAMICS/CHECKPOINTS/aped_net.pt`).
   - Apply dynamic INT8 quantization using PyTorch:
     `torch.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)`.
   - Measure and record serialized model file sizes on disk (`fp32_size` vs `int8_size`).
2. **Inference Profiling (CPU Edge Emulation):**
   - Run 300 utterances of `DEV_REPLICATION` on CPU with warm-up iterations.
   - Measure total inference wall time, mean latency per utterance, and Real-Time Factor (RTF).
3. **Diagnostic Parity Evaluation:**
   - Evaluate INT8 predictions with frozen locked thresholds ($\theta_{\text{subs}}=0.50, \theta_{\text{omis}}=0.55, \theta_{\text{ins}}=0.50$).
   - Compute Macro AP, Macro Precision, Macro F1, Utility, and per-class metrics.
   - Measure numerical delta against FP32 APED-Net: $\Delta \text{Macro F1}$, $\Delta \text{Utility}$.
4. **Ledger & Artifact Audit:**
   - Save serialized INT8 checkpoint, predictions, profiling benchmarks, and SHA-256 hashes under `experiments/EXP-33-COMPACT-QUANTIZED-EDGE-DEPLOYMENT/`.
