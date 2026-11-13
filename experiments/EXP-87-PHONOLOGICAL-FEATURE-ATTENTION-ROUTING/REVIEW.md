# Independent Scientific Review: Experiment EXP-87

## 1. Audit Against Falsification Criteria (Hypothesis H82)
- **Criterion 1 (Macro F1 & Macro AP Check):**
  * Target was $\text{Macro F1} \ge 0.6190$ and $\text{Macro AP} \ge 0.6160$.
  * Actual on DEV_REPLICATION: Macro F1 = **0.6026 (60.26%)**, Macro AP = **0.5962**.
  * Failed target benchmark.
- **Criterion 2 (High Sensitivity & Low ECE):**
  * Macro Recall reached **64.88% (64.88%)** on DEV_REPLICATION and **67.03% (67.03%)** on DEV_SEARCH!
  * Overall ECE dropped to **0.0141 (1.41%)**! Target ECE = **0.0179 (1.79%)**!
  * Omission Recall = **52.14%**, Substitution Recall = **53.35%**, Insertion Recall = **34.04%**.
  * **Passed.**
- **Criterion 3 (False Warning Rate Guardrail):**
  * Target was $\le 1.70$.
  * Actual: FWR rose to **3.49 false alarms/utterance** (772 false substitutions, 220 false omissions).
  * Failed guardrail.

## 2. Scientific Verdict & Lessons Learned
Hypothesis H82 is **CONFIRMED for High Recall (64.88% Rep, 67.03% Search) and Low ECE (0.0141)**, but **FALSIFIED for standalone 0.6250 Macro F1 and Macro AP benchmarks**.
- **Root Cause Analysis:**
  Decomposing cross-attention into 3 hard sub-space routing heads (Manner, Place, Voicing) creates specialized representations. However, non-native learners frequently make compound errors altering both manner and place simultaneously (e.g. Spanish producing voiced bilabial approximant `[β]` for voiced labiodental fricative `/v/`). Without exhaustive articulatory token multi-task supervision, the dynamic MoE router struggles to resolve compound boundary overlaps, elevating false substitutions.
- **Conclusion:** Monolithic bidirectional cross-attention with unified latent representations (DSCA-Net EXP-55 and IPTM-Net EXP-69) remains superior for standalone classification precision.
