# Superseded float32-cache controls

These original control checkpoints, predictions, and paired summaries were retained after a numerical-parity review found that the candidate used the shared float16 prefix cache while the initial control used historical full-float32 pooled features. Use `../matched_pair_seed*_fold*.json` and `encoder_adapter_control_prefix_*` for the primary comparison. The original candidate predictions/checkpoints were retained and reused; only the frozen control was retrained for the primary matched comparison.
