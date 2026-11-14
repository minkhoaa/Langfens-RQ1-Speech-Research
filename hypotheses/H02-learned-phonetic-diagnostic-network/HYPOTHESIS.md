# Hypothesis H02: Learned Multimodal Phonetic Diagnostic Network (LPDN)

## 1. Proposition
Heuristic single-threshold Goodness of Pronunciation (GOP) rules suffer from severe threshold coupling: lowering the error threshold to capture rare omissions and substitutions inevitably causes thousands of false alarms on non-native accents.
By parameterizing the diagnostic function as a learned acoustic-phonetic neural head that combines:
1. Target phonological identity and context embeddings,
2. Pre-softmax logit margins ($\ell(c_i) - \max_{k \neq c_i} \ell(k)$),
3. Segmental duration and acoustic silence probabilities,
4. Sequence-level CTC delta loss ($\Delta \mathcal{L}_{\text{omission}}$), and
5. Explicit gap-interval frame representations for insertions,
and training with class-frequency focal loss on `TRAIN`, the model will simultaneously achieve:
- Marked reduction in false alarms (suppressing false warning rate below 0.50/utterance),
- Significant increase in OMISSION recall and F1,
- Class-specific AP $\ge 0.40$ on rare classes,
- Substantially higher Weighted Utility score ($> 0.50$).

## 2. Theoretical Grounding
- Eliminates the single-threshold dilemma by learning non-linear phoneme-specific boundaries (vowels vs stops vs fricatives naturally have different GOP distributions).
- Solves the insertion void by training a dedicated gap-anchored binary focal classifier.
- Preserves deployment feasibility: inference requires only the raw audio waveform and prompt text.

## 3. Measurable Criteria
- Outperform EXP-03-SSL-GOP on all four class F1-scores.
- Reduce False Warning Rate from 3.12 down to $< 0.50$ false warnings/utterance.
- Exceed Weighted Utility of 0.50.
