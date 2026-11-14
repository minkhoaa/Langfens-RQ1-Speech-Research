# Falsification Criteria: Hypothesis H64

Hypothesis H64 will be declared **FALSIFIED** if any of the following failure conditions occur when evaluating IPTM-Net on held-out `DEV_REPLICATION`:

1. **Failure to Outperform DSCA-Net Baseline:**
   - Standalone Weighted Utility falls below 0.4850 (inferior to DSCA-Net 0.4928).
   - Standalone Macro AP falls below 0.6120.
2. **Prior Shortcut Hallucination:**
   - The network ignores acoustic evidence and relies solely on interlanguage priors, resulting in False Warning Rate (FWR) exceeding 2.10 false alarms per utterance.
   - Correct class precision falls below 90.00%.
3. **Substitution Degradation:**
   - Standalone Substitution AP falls below 0.4200.
4. **Data Leakage Violation:**
   - Prior matrix $\pi(p)$ computed using any development, replication, or sealed evaluation split annotations (must derive strictly from `train.jsonl`).
