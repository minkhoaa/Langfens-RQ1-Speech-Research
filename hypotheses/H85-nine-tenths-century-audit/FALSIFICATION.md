# Falsification Criteria: Hypothesis H85

Hypothesis H85 will be declared **FALSIFIED** if any of the following failure conditions occur:

1. **Bootstrap Utility Failure:**
   - The 10,000-sample clustered bootstrap mean Weighted Utility for Viginti-Pareto falls below 0.5020.
   - The lower bound of the 95% Confidence Interval falls below 0.4800.
2. **Bootstrap Macro F1 Failure:**
   - The 10,000-sample clustered bootstrap mean Macro F1 falls below 0.6200.
3. **Statistical Insignificance:**
   - Paired permutation tests vs. Hexad-Pareto or Undecim-Pareto fail to reach statistical significance ($p \ge 0.05$).
4. **Manifest / Sealed Data Corruption:**
   - Any of the 89 prior experiment directories fail SHA-256 verification or have missing files.
   - The sealed test partition hash does not match `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`.
