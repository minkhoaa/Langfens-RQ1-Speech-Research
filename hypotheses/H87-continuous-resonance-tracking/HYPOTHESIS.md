# Hypothesis H87: Continuous Formant Resonance Dynamics (FRTE)

## 1. Context & Motivation
Vowel substitutions account for over 42% of all phonetic substitutions in L2-ARCTIC (e.g. /IY/ <-> /IH/, /AE/ <-> /EH/, /AA/ <-> /AH/).
While self-supervised representations (Charsiu w2v2-FC) encode broad acoustic timbre, fine vowel distinctions are fundamentally governed by vocal tract resonance frequencies:
- First Formant ($F_1$): Tongue height and pharyngeal constriction (300 Hz - 900 Hz).
- Second Formant ($F_2$): Tongue advancement and oral cavity volume (800 Hz - 2500 Hz).
- Resonance Ratio $R_{12} = F_1 / F_2$: Invariant measure of vowel quality across speaker vocal tract lengths.

## 2. Hypothesis Statement
We hypothesize that:
1. Extracting continuous 6-dimensional formant resonance trajectory features:
   $$X_{\text{res}} = [F_1, F_2, F_3, F_1/F_2, \Delta F_1, \Delta F_2]$$
   and integrating them into the acoustic representation stream will:
   - Increase Vowel Substitution F1 by $\ge +2.5\%$ relative to baseline.
   - Reduce vowel false alarms along close phonological neighbors (/IY/ vs /IH/).
   - Maintain overall Macro AP $\ge 0.6100$ and Weighted Utility $\ge 0.4800$.
2. Falsification will occur if formant resonance tracking inflates overall False Warning Rate (FWR > 2.80) due to pitch-formant interaction errors in non-native speakers.
