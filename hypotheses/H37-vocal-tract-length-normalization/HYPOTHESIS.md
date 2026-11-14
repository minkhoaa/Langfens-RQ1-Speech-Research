# Hypothesis H37: Vocal Tract Length Normalization & Pitch-Invariant Acoustic Warping (VTLN-HTC-Net)

## 1. Context and Problem Statement
In EXP-41, zero-shot transfer on SpeechOcean762 exposed a domain shift: models trained on adult speakers (L2-ARCTIC) experienced a correlation drop when evaluated on children and adolescent speakers whose vocal tract lengths are physically shorter, shifting formant resonances ($F_1-F_3$) upward by $15-25\%$.
In classical speech science, Vocal Tract Length Normalization (VTLN) standardizes speaker formant spaces by estimating an acoustic warping factor $\alpha_{\text{vtln}}$ derived from vocal tract resonance scaling:
$$\alpha_{\text{vtln}} = \frac{\bar{C}_{\text{adult}}}{\bar{C}_{\text{speaker}} + \epsilon}$$
where $\bar{C}_{\text{speaker}}$ is the mean spectral centroid of the speaker's voiced frames.

## 2. Formal Hypothesis
Augmenting the sequence representation with speaker-level VTLN resonance scaling:
$$f_{\text{vtln}} = \left[ \alpha_{\text{vtln}}, \text{centroid}_{\text{norm\_spk}}, \Delta F_{0,\text{ratio}} \right]$$
forming a 66-dimensional input vector for **VTLN-HTC-Net**, will:
1. Normalize out age-related and speaker vocal tract length differences without destroying phonetic contrast.
2. Maintain all-time high Utility on `DEV_SEARCH` ($\ge 0.3900$) and `DEV_REPLICATION` ($\ge 0.4350$).
3. Enhance cross-corpus generalization when tested on non-adult populations.

## 3. Predicted Evidence & Significance
If validated, VTLN-HTC-Net demonstrates that physical acoustic normalization provides the mathematical bridge between adult training data and diverse learner demographics.
