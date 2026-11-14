# Hypothesis H93: Global Consensus Uncertainty Router (GCUR-Engine)

## 1. Context & Motivation
Under `AGENTS.md` Section 1, the laboratory is mandated to deliver trustworthy diagnostic feedback, including explicit uncertainty when a phonetic event cannot be evaluated reliably.
In educational Computer-Assisted Pronunciation Training (CAPT), a false alarm (telling a student their correct sound was wrong) severely impairs learner motivation. By quantifying epistemic model disagreement across fundamentally different model families (ADAE-Net, CICA-Net, MTRC-Net, Viginti-Pareto), we can selectively abstain on ambiguous acoustic segments while preserving $\ge 90\%$ evaluation coverage.

## 2. Hypothesis Statement
We hypothesize that:
1. Formulating the **Global Consensus Uncertainty Router (GCUR-Engine)** computing epistemic entropy across diverse paradigm champions:
   $$H_{\text{consensus}}(k) = -\sum_{c} \bar{P}_c(k) \log \bar{P}_c(k)$$
   and routing high-entropy slots ($H > \tau_{\text{abstain}}$) to an explicit `ABSTAIN` diagnostic state will:
   - Preserve $\ge 90.0\%$ evaluation coverage ($\text{Coverage} \ge 0.90$).
   - On the evaluated eligible cohort, surge **Macro Precision past 70.00% (70.00%)**!
   - Drop the **False Warning Rate below 1.10 false alarms per utterance**!
   - Elevate **Macro F1 past 0.6500 (65.00%)** on confident tokens!
2. Falsification will occur if achieving $\ge 70\%$ Macro Precision requires dropping coverage below 80% (violating the `AGENTS.md` coverage guardrail).
