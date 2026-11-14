# Hypothesis H06: Multi-Head Self-Attention Conformer Scorer (MH-Conformer)

## 1. Proposition
While Bidirectional LSTMs (EXP-08) demonstrated that sequential context resolves local coarticulatory ambiguities, recurrent cells suffer from sequential gradient bottlenecking and cannot simultaneously model local phonetic transitions (10-30ms) and utterance-level prosodic rhythm (1-3 seconds).

A hybrid Conformer architecture integrating:
1. Multi-Head Self-Attention (4 heads) to capture non-local syllable structure, vowel stress, and consonant cluster dependencies across the prompt, combined with
2. Depthwise Separable 1D Convolution (kernel size 5) to model local acoustic-phonetic formant dynamics,
will outperform BiLSTM in separating non-native allophonic variation from genuine errors, elevating:
- `SUBSTITUTION` F1 beyond 0.40,
- `OMISSION` F1 beyond 0.25,
- `INSERTION` F1 beyond 0.28, and
- Pushing Macro-F1 toward 0.55.

## 2. Experimental Plan
- Method ID: `M10-conformer-attention-scorer`
- Run ID: `EXP-10-CONFORMER-ATTENTION`
- Train on `TRAIN` sequence batches for 30 epochs with AdamW and cosine schedule.
- Pareto calibration on `DEV_CAL`.
- Primary locked evaluation on `DEV_SEARCH`.
