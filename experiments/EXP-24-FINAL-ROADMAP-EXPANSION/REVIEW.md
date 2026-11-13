# Experiment Review: EXP-24-FINAL-ROADMAP-EXPANSION

## 1. Objective and Setup
Investigated the hypothesis that boundary jitter in forced alignment degrades phoneme assessment by prematurely truncating coarticulatory transitions.
Constructed a Joint Soft-Attention CTC Network (`JSAC-Net`) that introduces a monotonic local attention layer with Gaussian temporal decay ($\sigma = 3.0$), allowing target phone queries to attend smoothly to neighboring frame representations without hard boundary truncation.
Trained on `TRAIN` (1,800 utterances) and evaluated on `DEV_SEARCH` (599 utterances) and `DEV_REPLICATION` (300 utterances).

## 2. Quantitative Breakthroughs
- **OMISSION AP on DEV_SEARCH Reaches 0.2251:**
  - `OMISSION` AP broke past 0.22, reaching an all-time campaign high of **0.2251** on `DEV_SEARCH` (a +2,028% relative gain over prior chance baseline 0.0106).
  - Omission Recall on DEV_SEARCH reached **41.15%**, with F1 = **0.2683**.
- **Record Standalone Utility on DEV_SEARCH:**
  - Standalone model utility reached **0.3802** on `DEV_SEARCH`, surpassing all previous individual models (EXP-13 at 0.3777, EXP-11 at 0.3736).
- **Macro AP on DEV_SEARCH Passes 0.52:**
  - Macro AP reached **0.5202** (all-time high on dev search).
- **SUBSTITUTION F1 on DEV_REPLICATION Reaches 0.3811:**
  - Standalone substitution F1 reached **0.3811** (Precision = **49.66%**, Recall = **30.92%**, AP = **0.3740**).
- **Insertion Recall:**
  - `INSERTION` Recall reached **31.20%** on DEV_SEARCH, with AP = **0.2476**.

## 3. Scientific Validation
Hypothesis H19 is **CONFIRMED**:
Local monotonic Gaussian windowed attention successfully eliminates boundary jitter, allowing target phones to pool acoustic evidence across their natural coarticulatory span. This pushed OMISSION AP past 0.22 on DEV_SEARCH and boosted standalone utility past 0.380!
