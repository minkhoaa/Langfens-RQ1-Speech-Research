# Hypothesis H58: Triad Decoupled Insertion Specializer (TDIS-Net) for Structural Epenthesis Detection

## 1. Context & Motivation
Across 62 completed experiments, the laboratory has achieved monumental breakthroughs on:
- **Macro AP:** 0.6341 (63.41% in EXP-58 Duodecim-Pareto).
- **Macro F1:** 0.6129 (61.29% in EXP-52 Undecim-Pareto).
- **Omission AP:** 0.4733 (47.33% in EXP-58).
- **Substitution AP:** 0.4540 (45.40% in EXP-58).
- **Insertion Precision:** 54.55% (EXP-59 ALDS-Net).

However, **Insertion Recall ($\approx 25 - 35\%$) and Insertion F1 ($\approx 0.30 - 0.36$)** remain the primary bottleneck in elevating all error metrics toward the 80% gate.

In linguistic phonetics, inter-phone gaps are fundamentally non-homogeneous:
1. **Intra-Word Cluster Gaps:** Consonant-consonant transitions (e.g. `/s p/`, `/t r/`) where L2 learners insert epenthetic vowels (e.g. `[e s p]`, `[t uh r]`). The acoustic signature is a brief voiced formant burst within a silent/unvoiced transition.
2. **Coda Release Gaps:** Word-final positions where learners release unreleased stops with an intrusive schwa (e.g. `[k ae t uh]`).
3. **Inter-Word Pause Gaps:** Natural acoustic silences between words, where shallow models trigger false insertion alarms due to breathing noise or glottal closure clicks.

When a single flat feedforward network predicts insertions across all three gap types, its parameters are forced into an unsatisfactory compromise between high sensitivity for intra-cluster epenthesis and high conservatism for breathing pauses.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing **TDIS-Net (Triad Decoupled Insertion Specializer Network)** with three dedicated, phonotactically routed gap prediction heads:
   $$\text{Head}_{\text{cluster}}(g) = \text{MLP}_{\text{cluster}}([g, h_{\text{left}}, h_{\text{right}}])$$
   $$\text{Head}_{\text{coda}}(g) = \text{MLP}_{\text{coda}}([g, h_{\text{left}}, h_{\text{right}}])$$
   $$\text{Head}_{\text{pause}}(g) = \text{MLP}_{\text{pause}}([g, h_{\text{left}}, h_{\text{right}}])$$
   routed by structural gap context (consonant cluster vs word-final coda vs inter-word pause) will eliminate cross-regime parameter interference.
2. TDIS-Net coupled with DSCA-Net (EXP-55) will achieve:
   - **Insertion Recall** surging to $\ge 45.00\%$ on `DEV_REPLICATION` (breaking the previous plateau of $\approx 28\%$).
   - **Insertion F1** breaking the 0.4200 (42.00%) barrier for the first time ($\text{Insertion F1} \ge 0.4200$).
   - **Insertion AP** breaking $\ge 0.3000$ (30.00%).
   - **Weighted Utility** breaking the 0.4950 milestone ($\text{Utility} \ge 0.4950$).
   - False Warning Rate maintaining $\le 1.65$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Insertion F1:** $\ge 0.4200$ (All-time campaign record).
2. **DEV_REPLICATION Insertion Recall:** $\ge 45.00\%$.
3. **DEV_REPLICATION Insertion AP:** $\ge 0.3000$.
4. **DEV_REPLICATION Weighted Utility:** $\ge 0.4950$.
