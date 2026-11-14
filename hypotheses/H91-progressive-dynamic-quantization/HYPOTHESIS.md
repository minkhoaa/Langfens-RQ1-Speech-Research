# Hypothesis H91: Dynamic INT8 Edge Quantization of ADAE-Net

## 1. Context & Motivation
In EXP-94, ADAE-Net was established as the undisputed standalone champion of the campaign (Weighted Utility = 0.5041, Macro F1 = 0.6285).
For edge deployment on resource-constrained devices (smartphones, local browser runtimes, Raspberry Pi), neural pronunciation diagnostic models must execute with ultra-low latency (<10ms) and minimal memory footprints.

## 2. Hypothesis Statement
We hypothesize that:
1. Applying dynamic post-training INT8 quantization (`torch.ao.quantization.quantize_dynamic`) to the linear and recurrent projection layers of ADAE-Net will:
   - Reduce model checkpoint size by $\ge 60\%$.
   - Achieve mean CPU inference latency $\le 5.0$ ms per utterance on 4 CPU threads (Real-Time Factor $\text{RTF} \le 0.0035$).
   - Preserve $\ge 99.5\%$ metric parity:
     $$\Delta \text{Macro AP} = \text{Macro AP}_{\text{INT8}} - \text{Macro AP}_{\text{FP32}} \ge -0.0050$$
     sustaining Macro AP $\ge 0.6100$ and Weighted Utility $\ge 0.5000$.
2. Falsification will occur if INT8 quantization degrades Macro AP by more than 0.0080 or CPU latency exceeds 6.0 ms per utterance.
