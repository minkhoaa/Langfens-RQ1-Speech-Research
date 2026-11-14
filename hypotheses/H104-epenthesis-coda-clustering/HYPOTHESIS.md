# Hypothesis H104: Coda Release & Epenthesis Clustering Specializer (CREC-Net)

## 1. Context & Motivation
In L2 English pronunciation, epenthesis errors frequently occur at syllable codas (e.g. appending a vocalic release after final plosives: "desk" -> "desk-uh", "card" -> "card-uh"). This is driven by L1 transfer from open-syllable phonotactic systems (Spanish, Arabic, Mandarin, Vietnamese).
Standard gap classifiers rely on generic windowed RMS energy, which often confuses a legitimate aspirated plosive release burst with an epenthetic vowel.
A true epenthetic vowel produces periodic vocal fold vibration with energy concentrated in lower formant regions ($F_1, F_2 < 2000$ Hz) and low zero-crossing rate ($\text{ZCR} < 0.15$), whereas an aspirated release burst is aperiodic with high spectral centroid and high ZCR.

## 2. Hypothesis Statement
We hypothesize that:
1. Extracting a 16-dimensional specialized gap feature representation:
   $$X_{g, \text{spec}} = [X_{g, 13}, r_{\text{post-coda}}, \Delta C_{\text{spec}}, \text{ZCR}_{\text{ratio}}] \in \mathbb{R}^{16}$$
   and training the gap insertion head with focal loss ($\gamma=2.0, \alpha=0.75$) will:
   - Disambiguate aspirated coda releases from true epenthetic vowel insertions.
   - Shatter the insertion recall ceiling: **Insertion Recall $\ge 48.00\%$ (48.00%)** on held-out `DEV_REPLICATION`.
   - Achieve an all-time standalone record for **Insertion F1 $\ge 0.4500$ (45.00%)** and **Insertion AP $\ge 0.3350$ (33.50%)**.
   - Sustain standalone **Weighted Utility $\ge 0.5000$ (50.00%)** and **Macro F1 $\ge 0.6200$ (62.00%)**.
   - Maintain Overall ECE $\le 0.0240$ (2.40%).
2. Falsification will occur if specialized coda dynamics fail to improve Insertion F1 over TDIS-Net (0.4321) or if False Warning Rate exceeds 3.10 false alarms per utterance.
