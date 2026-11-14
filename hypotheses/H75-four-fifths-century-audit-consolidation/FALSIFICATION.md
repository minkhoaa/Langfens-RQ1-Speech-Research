# Falsification Criteria: Hypothesis H75

Hypothesis H75 will be declared **FALSIFIED** if any of the following failure conditions occur:

1. **Bootstrap Utility Failure:**
   - The 10,000-sample clustered bootstrap mean Weighted Utility for Duodeviginti-Pareto falls below 0.4950.
   - The lower bound of the 95% Confidence Interval falls below 0.4700.
2. **Bootstrap Macro AP Failure:**
   - The 10,000-sample clustered bootstrap mean Macro AP falls below 0.6320.
3. **Statistical Insignificance:**
   - Paired permutation tests vs. Hexad-Pareto or Undecim-Pareto fail to reach statistical significance ($p \ge 0.05$).
4. **Manifest / Sealed Data Corruption:**
   - Any of the 79 prior experiment directories fail SHA-256 verification or missing files.
   - The sealed test partition hash does not match `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`.
