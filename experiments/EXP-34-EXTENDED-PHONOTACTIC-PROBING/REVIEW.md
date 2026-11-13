# Experiment Review: EXP-34-EXTENDED-PHONOTACTIC-PROBING (PNC-Net)

## 1. Executive Summary
Experiment EXP-34 implemented and evaluated Hypothesis H29 by explicitly conditioning the neural diagnostic sequence model on lexical-phonotactic n-gram transition probabilities ($P(p_t \mid p_{t-1})$, $P(p_t \mid p_{t+1})$) and consonant cluster legality flags derived from `cmudict.dict`.

## 2. All-Time Campaign Records Broken on DEV_REPLICATION
- **Insertion F1:** **0.3692** (New all-time campaign high, breaking previous 0.3579 in EXP-28; Precision = **35.64%**, Recall = **38.30%**)
- **Standalone Substitution Precision:** **50.67%** (Breaking the 50% milestone in a single un-ensembled model)
- **Standalone Model Utility:** **0.4255** (Surpassed all standalone models in campaign history)
- **Omission Recall & F1:** Recall = **39.69%**, F1 = **0.3799** (AP = **0.3188**)
- **Substitution AP:** **0.3818** (F1 = **0.3889**)

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H29 Strongly Confirmed:**
  * In inter-phonemic gaps, phonotactic cluster legality provides an essential linguistic anchor: epenthesis occurs primarily inside difficult non-native consonant clusters, so incorporating cluster transition probability sharply lifts Insertion Recall without triggering false alarms inside standard diphthongs and vowel-consonant sequences.
  * In phone slots, forward and backward transition probabilities weed out acoustically plausible but phonologically impossible substitutions, driving Substitution Precision past 50% (50.67%).
