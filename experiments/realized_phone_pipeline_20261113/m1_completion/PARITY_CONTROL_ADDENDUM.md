# E1 export-parity control — registered before its execution

E4 uses saved fp16 frame posteriors, renormalized in float64. The original E1
used live fp32 logits. Pooled PER reconstruction exposed one additional edit
in seed101 after export. No new training or tuning is warranted; add a pure E1
replay from the exact same normalized fp16 posteriors as E4, all folds/seeds,
same calibration budgets and full held denominator. Compare E4 against this
E1-export control for inference effects, retain original fp32 E1 as context.
Record changed utterance sequences from quantization. This implements GOAL's
preprocessing-parity requirement; no threshold is chosen using held labels.
