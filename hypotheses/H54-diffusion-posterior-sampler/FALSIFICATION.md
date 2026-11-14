# Falsification Criteria: Hypothesis H54

Hypothesis H54 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating ALDS-Net on held-out `DEV_REPLICATION`:

1. **Failure to Outperform DSCA-Net Baseline:**
   - Standalone Weighted Utility falls below 0.4850 (inferior to DSCA-Net 0.4928).
   - Standalone Macro AP falls below 0.6100.
2. **Diffusion Hallucination / Over-Denoising:**
   - Score matching pulls genuine phonetic substitutions toward the prompt target, artificially inflating false negatives and causing Substitution AP to drop below 0.4100.
3. **Omission Sensitivity Loss:**
   - Omission AP falls below 0.4200.
4. **Latency Target Miss:**
   - 5-step DDIM sampling CPU inference latency exceeds 30.0 ms per utterance.
5. **Score-Matching Non-Convergence:**
   - Diffusion loss fails to decrease or exhibits numerical NaN instability during score estimation.
