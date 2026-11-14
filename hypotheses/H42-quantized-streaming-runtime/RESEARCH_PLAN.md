# Research Plan: Experiment EXP-47 (Edge Streaming Deployment Package)

## 1. Objective & Technical Scope
Package and profile an end-to-end edge-deployable streaming runtime for the champion **DMCT-Net (EXP-44)** using PyTorch Dynamic INT8 Post-Training Quantization:
- **Target Platform:** Commodity x86_64 / ARM CPU edge runtimes.
- **Quantization Technique:** `torch.ao.quantization.quantize_dynamic` targeting `nn.Linear` layers with `qint8` activations and weights.
- **Evaluation Surface:** Full held-out `DEV_REPLICATION` (300 utterances, 9,863 target phones, 10,163 inter-phone gaps).
- **Benchmarking Protocol:** Cold-start latency, warm-up (10 runs), 300 benchmark iterations measuring mean, median, P90, and P99 latency, and Real-Time Factor (RTF = $\text{latency} / \text{audio\_duration}$).

## 2. Experimental Steps
1. Load FP32 checkpoint of `DMCT-Net` from `experiments/EXP-44-SELF-SUPERVISED-DYNAMIC-MASKING/CHECKPOINTS/dmct_net.pt`.
2. Apply `torch.ao.quantization.quantize_dynamic` to quantize all linear projections (`t_proj`, `fusion`, `local_attn.q_proj`, `local_attn.k_proj`, `local_attn.v_proj`, `local_attn.out_proj`, `manner_head`, `place_head`, `voicing_head`, `vowel_head`, `cons_head`, `gap_net`).
3. Serialize INT8 checkpoint to `CHECKPOINTS/dmct_net_int8.pt` and measure disk footprint.
4. Run inference across all 300 utterances on CPU, measuring precise millisecond timing.
5. Score predictions against gold labels using primary evaluator `PronunciationEvaluator("evaluation/UTILITY_SPEC.yaml")`.
6. Compute parity deltas: $\Delta \text{Macro AP}$, $\Delta \text{Macro Prec}$, $\Delta \text{Utility}$, $\Delta \text{FWR}$.
7. Package an all-in-one standalone streaming inference CLI in `tools/stream_edge_infer.py` capable of accepting arbitrary 16kHz WAV and prompt text, outputting real-time JSON diagnostics.
