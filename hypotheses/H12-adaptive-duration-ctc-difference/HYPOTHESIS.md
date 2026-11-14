# Hypothesis H12: Speaking-Rate Adaptive Duration Normalization (SRAD-Net)

## 1. Proposition
Phonetic omission detection and substitution diagnosis are degraded when raw frame durations are evaluated without accounting for utterance-level speaking tempo:
- Fast L2 speech (tempo $> 4.5$ phones/sec) produces legitimate 30ms phones that fixed-duration thresholds misdiagnose as omissions.
- Slow L2 speech (tempo $< 2.5$ phones/sec) leaves 40ms pauses or coarticulation residue that fixed-duration thresholds misdiagnose as realizations.

By normalizing phoneme durations against the utterance-level mean tempo:
$$r_i = \frac{|T_i|}{\frac{1}{N}\sum_{k=1}^N |T_k|}$$
and injecting log-duration ratio $\log(r_i + 1e-4)$, an abbreviation indicator $\mathbb{I}(r_i < 0.35)$, and relative silence contrast into the sequence encoder:
1. The model becomes invariant to speaking tempo fluctuations across speakers and sentences.
2. `OMISSION` F1 will break past **0.40** on `DEV_REPLICATION`,
3. `SUBSTITUTION` F1 will improve as vowel reduction is decoupled from true substitution errors,
4. Total Weighted Utility will advance to a new campaign all-time high.

## 2. Experimental Plan
- Method ID: `M17-adaptive-ctc-duration-model`
- Run ID: `EXP-17-ADAPTIVE-DURATION-CTC`
- Augment sequences with 4 adaptive tempo features (55 target dims total).
- Train `AdaptiveHierarchicalNetwork` (HM-ADN with 55 input dims) on `TRAIN`.
- Calibrate on `DEV_CAL`.
- Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` with primary locked evaluator.
