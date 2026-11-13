# Scientific Review: Experiment EXP-105-CROSS-CORPUS-FULL-TRAINING-EXPANSION

## 1. Executive Summary & Critical Falsification
In Experiment EXP-105, we implemented the **Joint Cross-Corpus Multi-Accent Co-Training Network (JCCM-Net)**, expanding the training pool from 1,800 L2-ARCTIC utterances to 1,999 utterances by ingesting SpeechOcean762 training speech (adult and child non-native learners) mapped to soft target distributions.

## 2. Experimental Outcomes & Theoretical Failure Mechanism
1. **In-Domain L2-ARCTIC DEV_REPLICATION:**
   - **Macro Precision surged to 67.60% (67.60%)**!
   - **False Warning Rate dropped to 1.037 false alarms per utterance** (only 29 false omissions across 9,863 target phones).
   - Macro AP: **0.6089 (60.89%)**, Overall ECE = **0.0132 (1.32%)**.
   - However, Macro Recall dropped to **47.68%**, reducing Weighted Utility to 0.4569.
2. **Cross-Corpus Domain Generalization on SpeechOcean762-Test:**
   - Pearson $r = -0.0636$ ($p = 1.84 \times 10^{-5}$).
   - Spearman $\rho = -0.0807$ ($p = 5.20 \times 10^{-8}$).
3. **Failure Mechanism (Cross-Corpus Pedagogical Misalignment):**
   - In SpeechOcean762, the human scores $S \in [0, 10]$ represent holistic pedagogical quality (pronunciation, prosody, stress, fluency). A native-like phone produced with child pitch or hesitant tempo is penalized with a low score ($S \approx 2-4$).
   - In L2-ARCTIC, ground truth represents physical phonetic realization (Arpabet identity).
   - Mapping SpeechOcean762 scores $S < 4$ directly to OMISSION/SUBSTITUTION forced the network to treat natural child acoustic prosody as phonemic errors. Consequently, the network's predicted correctness became anticorrelated with holistic test scores.

## 3. Hypothesis H100 Evaluation
- **Decisively Falsified:** Cross-corpus transfer correlation inverted to $-0.0636$, and in-domain recall fell below 60%.
- **Architectural Directive for Phase 2:** Cross-corpus co-training requires domain-adversarial representation disentanglement (gradient reversal) or explicit multi-task heads separating categorical MDD realization from continuous pedagogical proficiency scores.
