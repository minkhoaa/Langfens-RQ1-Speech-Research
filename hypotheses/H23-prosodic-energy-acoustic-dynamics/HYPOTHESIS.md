# Hypothesis H23: Acoustic Prosodic-Energy Dynamics (APED-Net)

## 1. Context and Problem Statement
In all previous iterations (EXP-01 through EXP-27), input representations were derived entirely from the Wav2Vec2 phonetic classifier posterior logits and duration ratios.
While this yielded outstanding performance on `CORRECT` (AP > 0.97, F1 > 0.94), record Macro Precision (0.6272), and record Substitution AP (0.3943), a persistent challenge remains in separating:
1. A **low-energy realized phone** (e.g., an unreleased consonant or a reduced schwa) from a **genuinely omitted phoneme** (OMISSION).
2. A **background acoustic pause** from an **extra inserted phonetic burst** (INSERTION).

Because neural CTC posteriors are non-linearly normalized, they can assign non-zero probability mass to silence frames when forced by the monotonic alignment trellis. Conversely, physical waveform prosodic features—such as Root-Mean-Square (RMS) energy ratio relative to the utterance mean ($E_{\text{slot}} / \bar{E}_{\text{utt}}$), zero-crossing rate (ZCR), and spectral centroid—provide direct acoustic evidence of physical vocal tract activity.

## 2. Formal Hypothesis
Augmenting the 55-dimensional contextual sequence representation with physical waveform energy and spectral dynamics:
$$f_{\text{prosodic}} = \left[ \frac{E_{\text{slot}}}{\bar{E}_{\text{utt}}}, \log\left(\frac{E_{\text{slot}}}{\bar{E}_{\text{utt}}} + 10^{-4}\right), \text{ZCR}_{\text{slot}}, \Delta E_{\text{decay}}, \text{Centroid}_{\text{norm}} \right]$$
forming a 60-dimensional input vector for **Acoustic Prosodic-Energy Diagnostic Network (APED-Net)**, will:
1. Provide the physical energy discriminant needed to sharply elevate standalone Omission Precision ($\ge 35\%$) and Omission AP ($\ge 0.2000$) on `DEV_SEARCH`.
2. Reduce false omission warnings caused by acoustic pauses, lowering False Warning Rate ($\text{FWR} \le 1.40$).
3. Maintain high overall Utility ($\ge 0.3800$) and high Macro Precision ($\ge 0.5800$) in standalone inference.

## 3. Predicted Evidence & Significance
If validated, APED-Net proves that fusing self-supervised neural posteriors with fundamental physical acoustic energy dynamics resolves the forced-alignment deletion blindspot from first principles.
