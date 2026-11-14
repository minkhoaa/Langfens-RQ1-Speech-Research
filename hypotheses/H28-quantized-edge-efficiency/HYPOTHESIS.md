# Hypothesis H28: Post-Training Dynamic INT8 Quantization & Edge Latency Parity

## 1. Context and Problem Statement
In `AGENTS.md` Sections 2, 7, and 12, practical deployability, real-time feasibility, and memory constraints are core scientific requirements alongside raw classification accuracy.
In EXP-15, we benchmarked unquantized FP32 latency at 1.61ms (GPU) and 1.54ms (CPU), with model footprint of 1.78 MB.
In EXP-28, APED-Net established all-time standalone records for Macro F1 (0.5725), Substitution F1 (0.4461), and Insertion F1 (0.3579).

However, deploying pronunciation diagnosis on edge mobile devices (smartphones, tablets, web client runtimes) requires sub-megabyte memory footprints and low-energy integer arithmetic without floating-point coprocessors.

## 2. Formal Hypothesis
Applying post-training dynamic INT8 quantization (`torch.quantization.quantize_dynamic` across `nn.Linear` and `nn.LSTM` layers) to APED-Net will:
1. **Compress Model Footprint:** Reduce parameter size by $\ge 60\%$ (from $\sim 1.86\text{ MB}$ down to $\le 0.85\text{ MB}$).
2. **Preserve Diagnostic Parity:** Maintain diagnostic performance within a strict parity tolerance:
   $$|\text{Macro F1}_{\text{int8}} - \text{Macro F1}_{\text{fp32}}| \le 0.0100$$
   and $|\text{Utility}_{\text{int8}} - \text{Utility}_{\text{fp32}}| \le 0.0100$ on `DEV_REPLICATION`.
3. **Accelerate CPU Inference:** Maintain or reduce CPU latency ($\le 1.50\text{ ms}$ per utterance) while preserving $100\%$ evaluation coverage and zero false alarm explosion.

## 3. Predicted Evidence & Significance
If validated, this proves that physical waveform energy dynamics and multi-task articulatory heads are mathematically robust to 8-bit fixed-point quantization, establishing a production-ready edge diagnostic engine.
