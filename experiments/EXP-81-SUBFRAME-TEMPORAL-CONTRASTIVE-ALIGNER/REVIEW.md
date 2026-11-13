# Independent Scientific Review: Experiment EXP-81

## 1. Audit Against Falsification Criteria (Hypothesis H76)
- **Criterion 1 (Macro F1 Benchmark):**
  * Target was $\ge 0.6250$.
  * Actual on DEV_REPLICATION: Macro F1 = **0.6078 (60.78%)**.
  * Failed target benchmark.
- **Criterion 2 (Macro AP Benchmark):**
  * Target was $\ge 0.6220$.
  * Actual: Macro AP = **0.5870 (58.70%)**.
  * Failed target benchmark.
- **Criterion 3 (False Warning Rate Guardrail):**
  * Target was $\le 1.80$.
  * Actual: FWR rose to **3.167 false alarms/utterance** (736 false substitutions, 156 false omissions).
  * Failed guardrail.

## 2. Scientific Verdict & Lessons Learned
Hypothesis H76 is **FALSIFIED for Rigid Contrastive Boundary Sharpening**.
- **Root Cause Analysis:**
  The contrastive boundary sharpening loss forces the neural latent space to maximize acoustic distance across phone boundaries. However, in continuous non-native speech, natural coarticulation involves smooth formant transitions and anticipatory gestures between adjacent sounds. By penalizing transition smoothness, the model misinterprets natural phonetic blending as an error, causing false substitutions to explode along phoneme boundaries (736 false substitutions).
- **Architectural Directive for EXP-82:**
  Phonetic transitions must not be artificially separated by hard contrastive distance margins. Instead, models should rely on statistical interlanguage priors (IPTM-Net) and smooth local monotonic attention (HRTD-Net / DSCA-Net), which naturally preserve coarticulatory continuity. Standalone supremacy remains with **EXP-69 IPTM-Net** (Macro F1 = 0.6205) and **EXP-55 DSCA-Net** (Utility = 0.4928), and ensemble supremacy with **EXP-79 Duodeviginti-Pareto** (Utility = 0.5005, Macro F1 = 0.6155).
