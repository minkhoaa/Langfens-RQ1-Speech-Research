# Hypothesis H04: Sequence-Level Contextual Phonetic Network (BiLSTM-CPN)

## 1. Proposition
Phoneme-level mispronunciation detection in L2 speech is fundamentally constrained when phone slots are classified independently without sequential phonotactic context.
Coarticulation, vowel reduction, nasalization, and consonant cluster transitions introduce localized spectral shifts that an isolated MLP head misinterprets as errors, causing either high false alarms or suppressed recall.

By encoding the full utterance-level sequence of acoustic-phonetic tokens $\mathbf{x}_1, \dots, \mathbf{x}_N$ using a Bidirectional Long Short-Term Memory (BiLSTM) network with residual acoustic skip connections:
1. The forward and backward hidden states $[\overrightarrow{\mathbf{h}}_i ; \overleftarrow{\mathbf{h}}_i]$ integrate left and right phonetic context (place/manner transitions), enabling the model to distinguish legitimate coarticulatory allophones from true cross-phoneme `SUBSTITUTION` errors.
2. Contextual duration expectations allow the model to recognize genuine `OMISSION` events even when neighboring phones bleed energy into the target window.
3. Modeling inter-phonemic gap environments conditioned on adjacent consonant clusters provides strong inductive priors to elevate `INSERTION` Average Precision and F1.

## 2. Experimental Plan
- Method ID: `M08-bilstm-context-model`
- Run ID: `EXP-08-PHONOTACTIC-CONTEXT`
- Train on `TRAIN` (1,800 full utterance sequences) using sequence batching with padding/masking.
- Calibrate Pareto decision policy on `DEV_CAL` (300 utterance sequences).
- Evaluate on `DEV_SEARCH` (599 utterance sequences, 19,755 phones, 20,354 gaps).

## 3. Measurable Criteria
- Achieve higher Macro AP than EXP-07 ($> 0.50$).
- Increase SUBSTITUTION AP beyond 0.38 and OMISSION AP beyond 0.15.
- Improve INSERTION AP beyond 0.04.
- Increase Weighted Utility over EXP-06 ($> 0.35$).
