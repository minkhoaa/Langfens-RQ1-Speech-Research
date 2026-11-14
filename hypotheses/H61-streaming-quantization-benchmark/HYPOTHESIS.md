# Hypothesis H61: Dual-Engine Dynamic INT8 Quantization and Sub-8ms Edge Profiling (EDGE-DUO)

## 1. Context & Motivation
Our laboratory has produced two complementary champion architectures:
1. **EXP-55 DSCA-Net:** All-time peak standalone Weighted Utility (0.4928), Macro AP (0.6159), Macro F1 (0.6026), and Omission AP (0.4427), powered by bidirectional prompt-audio cross-attention.
2. **EXP-63 TDIS-Net:** All-time peak insertion champion with 0.4321 Insertion F1, 0.3151 Insertion AP, and 51.47% Insertion Precision, powered by Triad decoupled gap gating.

However, real-world educational and clinical applications demand client-side execution on low-power hardware (smartphones, browser WebAssembly, low-cost tablets) where floating-point compute is constrained. While EXP-47 verified dynamic quantization for DMCT-Net, neither DSCA-Net (with cross-attention) nor TDIS-Net (with triad gap routing) has been quantified under INT8 edge execution.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Applying dynamic post-training INT8 quantization (`torch.ao.quantization.quantize_dynamic`) to the linear projection, attention, and classification layers of **DSCA-Net** and **TDIS-Net** will:
   - Compress serialized model footprints to $\le 1.80$ MB per model.
   - Achieve mean single-utterance CPU inference latency $\le 4.0$ ms per model on commodity CPU ($RTF \le 0.008$, $> 125\times$ faster than real time).
2. The bidirectional cross-attention in DSCA-Net and the triad decoupled heads in TDIS-Net possess inherent spectral redundancy that is resilient to 8-bit round-off noise, resulting in near-zero parity loss on held-out `DEV_REPLICATION`:
   $$|\Delta \text{Macro AP}| \le 0.0050$$
   $$|\Delta \text{Insertion F1}| \le 0.0050$$
   $$|\Delta \text{Utility}| \le 0.0050$$
3. Cascading `INT8-DSCA-Net` (for target phone diagnosis) with `INT8-TDIS-Net` (for inter-phone gap diagnosis) in a unified edge pipeline (`tools/stream_edge_dual_engine.py`) will achieve:
   - Total dual-engine CPU latency $\le 7.50$ ms.
   - Macro AP $\ge 0.6150$, Macro F1 $\ge 0.6020$, and Insertion F1 $\ge 0.4300$.
   - 100% test pass rate across deterministic self-tests with zero NaN/Inf outputs.

## 3. Predicted Experimental Outcomes
1. **Mean CPU Latency:** $\le 7.50$ ms total (Dual-engine).
2. **Parity Retention:** $|\Delta \text{Utility}| \le 0.0050$ (Zero perceptible degradation).
3. **Insertion F1 under INT8:** $\ge 0.4300$.
4. **Model Footprint:** $\le 1.80$ MB per model.
