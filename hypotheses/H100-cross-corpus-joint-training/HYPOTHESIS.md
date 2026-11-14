# Hypothesis H100: Joint Cross-Corpus Multi-Accent Co-Training (JCCM-Net)

## 1. Context & Theoretical Motivation
Under `AGENTS.md` (Section 5 and Section 14), evaluating and improving cross-dataset generalization without manufacturing artificial AP is a central scientific requirement.
All prior models (EXP-01 through EXP-104) were trained exclusively on L2-ARCTIC adult speakers (1,800 utterances). When tested zero-shot on SpeechOcean762 (Chinese child and adult learners), acoustic domain shift (shorter vocal tracts, higher fundamental frequencies, different recording acoustics) degrades phone-level ranking.

## 2. Hypothesis Statement
We hypothesize that:
1. Constructing the **Joint Cross-Corpus Multi-Accent Network (JCCM-Net)** by co-training on:
   - Primary supervised data: 1,800 L2-ARCTIC utterances (adult L2 English across 6 L1 cohorts).
   - Multi-corpus auxiliary data: 300 diverse SpeechOcean762 training utterances (ingested from original publisher archives, mapping continuous scores $S \in [0, 10]$ to soft target distributions).
   - Coupled with ADAE-Net's orthogonal articulatory latent spaces and 47-dimensional canonical IPTM prompt stream.
2. JCCM-Net will:
   - Significantly elevate SpeechOcean762 cross-corpus domain correlation: Pearson $r \ge 0.2800$ (surpassing Septendecim-Pareto 0.2292) and Spearman $\rho \ge 0.3200$ (surpassing 0.3014).
   - Preserve in-domain L2-ARCTIC `DEV_REPLICATION` performance without negative transfer: **Macro F1 $\ge 0.6150$ (61.50%)** and **Weighted Utility $\ge 0.4900 (49.00%)**.
   - Bridge the adult-to-child vocal tract gap via shared articulatory subspace representations.
3. Falsification will occur if multi-corpus co-training causes negative transfer on in-domain L2-ARCTIC, dropping Macro F1 below 0.6050 or Weighted Utility below 0.4800.
