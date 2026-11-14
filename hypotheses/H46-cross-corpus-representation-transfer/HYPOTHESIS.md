# Hypothesis H46: Cross-Corpus Generalization of 5ms Micro-Dynamics and Multi-Model Consensus on SpeechOcean762

## 1. Context & Motivation
In Experiment EXP-41, the baseline BiLSTM model demonstrated a zero-shot Pearson correlation of $r = 0.2690$ ($p = 4.9 \times 10^{-76}$) and an error recall of $63.88\%$ on the SpeechOcean762 test benchmark (4,536 phonemes). However, the cascaded HTC-Net (EXP-38) suffered from vocal tract length domain shift because SpeechOcean762 includes adolescent/child speakers with higher fundamental frequencies ($F_0$) and shifted formants ($F_1/F_2$).

Since EXP-41, the laboratory has discovered two major paradigm advances:
1. **EXP-48 HRTD-Net (High-Resolution Temporal Dynamics):** Introduces 5ms sub-frame acoustic micro-dynamics (instantaneous $dE/dt$, micro-temporal crest factor, burst onset ratios) that operate independently of absolute pitch/formant locations.
2. **EXP-45 Decimal-Pareto (10-Model Consensus):** Combines 10 orthogonal models, including EXP-42 (VTLN formant warping), providing explicit compensation for vocal tract length variance.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. **HRTD-Net (EXP-48)**, owing to its pitch-invariant 5ms micro-temporal dynamics and dynamic sequence token masking, will generalize zero-shot to SpeechOcean762 without vocal tract degradation, achieving:
   - Pearson correlation with human expert scores: $r \ge 0.2800$ ($p < 10^{-80}$).
   - Spearman rank correlation: $\rho \ge 0.3200$.
2. **Decimal-Pareto (EXP-45)**, by leveraging consensus across VTLN, prosodic energy, and phonotactics, will achieve superior phoneme mispronunciation detection:
   - Mispronunciation Error AP $\ge 0.1450$ (breaking previous baseline 0.1258).
   - Mispronunciation Error Recall $\ge 65.00\%$.
3. Cross-corpus evaluation will confirm domain robustness without any dataset adaptation or fine-tuning, preserving zero-leakage standards.

## 3. Predicted Experimental Outcomes
1. **HRTD-Net Pearson $r$:** $\ge 0.2800$.
2. **Decimal-Pareto Error AP:** $\ge 0.1450$.
3. **Decimal-Pareto Error Recall:** $\ge 65.00\%$.
