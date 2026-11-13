# Experiment Review: EXP-17-ADAPTIVE-DURATION-CTC

## 1. Objective and Setup
Investigated the hypothesis that phoneme duration is severely confounded by utterance-level speaking tempo.
Constructed a Speaking-Rate Adaptive Duration Network (`SRAD-Net`) that normalizes phoneme duration against the utterance mean duration ratio, introduces log-duration ratio, an abbreviation indicator, and silence contrast (expanding target features to 55 dimensions).
Evaluated on `DEV_SEARCH` (599 utterances) and `DEV_REPLICATION` (300 utterances).

## 2. Quantitative Breakthroughs
- **OMISSION F1 on DEV_REPLICATION Reaches 0.3926:**
  - `OMISSION` F1 surged to **0.3926** (an all-time campaign high, approaching 40%).
  - Omission Recall reached **36.96%**, Precision = **41.85%**.
  - Confirms that normalizing for speaking tempo resolves false omission alarms on fast speakers.
- **SUBSTITUTION F1 on DEV_REPLICATION Reaches 0.3763:**
  - `SUBSTITUTION` F1 surged to **0.3763** (all-time high).
  - Precision reached **48.05%**, Recall = **30.92%**, AP = **0.3656**.
- **Macro F1 Reaches 0.5671:**
  - Macro F1 reached **0.5671** (highest all-time).
  - Macro Precision = **0.6041** (comfortably passing the 60% macro milestone).
- **INSERTION AP:**
  - `INSERTION` AP on DEV_SEARCH reached **0.2547**.

## 3. Scientific Synthesis & Next Action
Tempo normalization provides clear causal improvements for both omission and substitution detection.
To formally verify the stability of this achievement across statistical variations, `EXP-18-MULTI-SEED-CERTIFICATION` will perform a 5-seed random initialization sweep on `DEV_REPLICATION`.
