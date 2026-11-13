# Scientific Review: Experiment EXP-92-CONTINUOUS-RESONANCE-CONVERGENCE-STUDY

## 1. Executive Summary & Experimental Verification
In Experiment EXP-92, we investigated continuous vowel formant resonance dynamics (F1-F3 frequencies, resonance ratio $F_1 / F_2$, and dynamic formant deltas $\Delta F_1, \Delta F_2$) concatenated with base HRTD features (73-dimensional acoustic stream) in the **Formant-Resonance-Aware Network (FRTE-Net)**.

## 2. Key Experimental Findings
1. **Insertion Breakthrough:**
   - Formant resonance transitions provided sharp vocalic onset cues across inter-word and intra-cluster gaps.
   - Pushed **Insertion Recall to an all-time standalone peak of 42.55%**!
   - Pushed **Insertion F1 to 0.4061 (40.61%)** (second highest in laboratory history after TDIS-Net 0.4321, and highest among full acoustic backbones!).
   - Insertion AP reached **0.2942 (29.42%)**.
2. **Overall Macro Metrics:**
   - **Macro F1:** **0.6005 (60.05%)** (Surpassing the 60% milestone).
   - **Macro Recall:** **62.80% (62.80%)**.
   - **Weighted Utility:** **0.4815 (48.15%)**.
3. **Falsification of Hypothesis H87 on Vowel False Alarms:**
   - Falsification condition met: Unconstrained LPC root tracking on non-native accented speech exhibited pole-frequency jitter during nasalized vowels and breathy voice qualities.
   - This inflated the False Warning Rate (FWR = 3.08 on replication, 3.81 on search), failing the $\le 2.80$ FWR guardrail.
   - **Scientific Directives Logged to `reports/RESEARCH_FAILURES.md`:** Continuous raw formant frequencies should not be fed directly to the substitution classifier without vocal tract normalization (VTLN) or confidence gating; however, the formant ratio and delta features serve as premier indicators of epenthesis and vowel insertion.
