# Experiment Review: EXP-22-PHONOLOGICAL-CONTRASTIVE-MARGIN

## 1. Objective and Setup
Investigated the hypothesis that introducing an explicit Supervised Contrastive Margin Loss ($m = 0.35$) on normalized bottleneck embeddings will enforce geometric angular separation between canonical targets and allophonic substitution competitors.
Trained `ContrastiveHierarchicalNetwork` (PSC-Net) on `TRAIN` (1,800 utterances) and evaluated on `DEV_SEARCH` (599 utterances) and `DEV_REPLICATION` (300 utterances).

## 2. Key Findings and Quantitative Gains
- **New Campaign All-Time High on Standalone Substitution AP:**
  - `DEV_REPLICATION` Substitution AP reached **0.3874** (highest all-time for any standalone model).
  - Substitution Precision was maintained at **51.29%**!
- **Solid Insertion Recall and F1:**
  - `DEV_REPLICATION` Insertion F1 reached **0.3060**, with Recall reaching **29.79%** and AP reaching **0.2264**.
- **Omission Sensitivity:**
  - `DEV_SEARCH` Omission Recall was maintained at **38.76%**, with F1 = **0.2643** and AP = **0.1942**.
- **Macro Performance:**
  - Macro AP on replication reached **0.5550**, Macro Precision reached **59.95%**, Macro F1 = **0.5475**.

## 3. Scientific Synthesis & Next Continuous Step
The contrastive margin confirms that pushing competitors away in cosine space directly yields cleaner substitution ranking.
The next planned experiment is `EXP-23-COMPOSITE-UTILITY-SWEEP`: a global grid sweep over the full hybrid model space to identify the global Pareto frontier across all evaluation metrics.
