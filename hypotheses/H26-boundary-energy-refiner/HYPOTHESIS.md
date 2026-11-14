# Hypothesis H26: Sub-Phonemic Boundary Energy Refinement (BER-Net)

## 1. Context and Problem Statement
In EXP-28 and EXP-29, physical waveform energy dynamics elevated Macro F1 to 0.5801 and Substitution AP to 0.4019. However, error analysis reveals that residual omissions and substitutions cluster around stop consonants (/P/, /T/, /K/, /B/, /D/, /G/) and affricates (/CH/, /JH/).
These phonemes possess rapid sub-phonemic acoustic transitions:
1. Occlusion phase (near-zero energy closure).
2. Transient burst release (5-15 ms impulse).
3. Aspiration or Voice Onset Time (VOT).

When a learner produces an unreleased stop or weakly articulates a plosive burst, segment-level RMS averaging obscures the burst transient, conflating an omitted release with a canonical closure.

## 2. Formal Hypothesis
Augmenting phone slots with sub-phonemic burst transient features:
$$f_{\text{boundary}} = \left[ \frac{\max_{t \in \text{slot}} |s(t)|}{\text{RMS}_{\text{slot}} + 10^{-6}}, \frac{E_{\text{last\_20ms}}}{E_{\text{first\_20ms}} + 10^{-6}}, \Delta \text{ZCR}_{\text{edge}} \right]$$
forming a 63-dimensional target vector for **Boundary Energy Refinement Network (BER-Net)**, will:
1. Sharpen stop consonant and affricate discrimination, raising Stop/Affricate Recall by $\ge 5.0\%$.
2. Push standalone Omission Precision on `DEV_SEARCH` above $25.0\%$ and Omission AP above $0.2000$.
3. Maintain low False Warning Rate ($\le 1.80$).

## 3. Predicted Evidence & Significance
If validated, BER-Net proves that sub-phonemic acoustic temporal resolution resolves fine-grained consonant release omissions without requiring oracle manual boundary annotations.
