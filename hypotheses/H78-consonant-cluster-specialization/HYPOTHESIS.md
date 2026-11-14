# Hypothesis H78: Cross-Linguistic Consonant-Cluster Sonority Paradigm Network (CCSP-Net)

## 1. Context & Motivation
Insertion detection is a critical educational component of CAPT systems because non-native learners frequently insert epenthetic vowels (e.g. `[ə]`, `[ɪ]`, `[e]`) to break up phonotactically illegal consonant clusters (e.g. Spanish inserting `[e]` before `/s/` clusters, Arabic inserting `[ɪ]` inside complex onsets, Chinese/Vietnamese inserting schwa after final stops).

In EXP-63 (TDIS-Net), decoupling gaps into triad categories shattered the insertion bottleneck, reaching Insertion F1 = 0.4321 and Insertion AP = 0.3151. However, TDIS-Net used only broad category flags, lacking explicit phonological modeling of the **Sonority Sequencing Principle (SSP)** and cross-linguistic cluster obstruction.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing the **Cross-Linguistic Consonant-Cluster Sonority Paradigm Network (CCSP-Net)** by:
   - Defining a 7-level sonority hierarchy: Vowel (7) > Glide (6) > Liquid (5) > Nasal (4) > Fricative (3) > Affricate (2) > Stop (1).
   - Augmenting the gap feature stream from 13 to 18 dimensions by injecting 5 phonological cluster dynamics:
     * Left phone sonority $\text{Son}(p_k)$
     * Right phone sonority $\text{Son}(p_{k+1})$
     * Sonority gradient $\Delta \text{Son} = \text{Son}(p_{k+1}) - \text{Son}(p_k)$
     * Syllable onset sonority violation flag $\mathbb{I}[\Delta \text{Son} \le 0]$
     * Word-final coda stop obstruction flag $\mathbb{I}[p_k \in \text{Stop/Affricate} \land p_{k+1} \in \text{Boundary}]$
   - Employing sonority-modulated triad decoupled gap gating.
2. CCSP-Net will achieve:
   - **Insertion F1:** Breaking the 44% milestone for the first time in laboratory history ($\text{Insertion F1} \ge 0.4400$).
   - **Insertion AP:** $\ge 0.3300$ (33.00%) (Surpassing TDIS-Net 0.3151).
   - **Insertion Precision:** $\ge 53.00\%$ with Insertion Recall $\ge 38.00\%$.
   - **Standalone Weighted Utility:** $\ge 0.4900$ (49.00%).
   - **Overall ECE:** $\le 0.0130$ (1.30%) with False Warning Rate $\le 1.65$ per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Insertion F1:** $\ge 0.4400$ (All-time campaign peak!).
2. **DEV_REPLICATION Insertion AP:** $\ge 0.3300$ (All-time campaign peak!).
3. **DEV_REPLICATION Insertion Precision:** $\ge 53.00\%$.
4. **Zero-Leakage Compliance:** 100% verified via frozen evaluator and canonical target phones.
