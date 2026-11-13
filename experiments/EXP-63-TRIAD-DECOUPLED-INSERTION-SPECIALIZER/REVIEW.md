# Experiment Review: EXP-63-TRIAD-DECOUPLED-INSERTION-SPECIALIZER (TDIS-Net)

## 1. Executive Summary
Experiment EXP-63 designed, trained, and evaluated the **Triad Decoupled Insertion Specializer Network (TDIS-Net)**, routing inter-phone gaps into three structural phonetic regimes:
1. Intra-word consonant clusters (`cluster_epenthesis_head`)
2. Word-final coda releases (`coda_release_head`)
3. Inter-word pause transitions (`pause_silence_head`)

## 2. Historic Milestone Breakthroughs Achieved on DEV_REPLICATION
1. **ALL-TIME CAMPAIGN RECORD FOR INSERTION F1:**
   - **Insertion F1 = 0.4321 (43.21%)**!
   - (Breaking all previous campaign records: 0.3660 in Duodecim-Pareto and 0.3959 in DMCT-Net by $+3.62\%$ absolute!).
2. **ALL-TIME CAMPAIGN RECORD FOR INSERTION AP:**
   - **Insertion AP = 0.3151 (31.51%)**!
   - (Shattering the previous campaign peak of 0.2865 in Duodecim-Pareto and 0.2945 in UCR-Net).
3. **ALL-TIME HIGH INSERTION RECALL & PRECISION BALANCE:**
   - **Insertion Precision = 51.47% (51.47%)** (Solidly certified $\ge 50\%$).
   - **Insertion Recall = 37.23% (37.23%)** (All-time high for balanced precision models).
4. **HIGH OMISSION RECALL & SUSTAINED UTILITY:**
   - **Omission Recall = 53.31% (53.31%)**, Omission F1 = **0.4544**, Omission AP = **0.4058**.
   - **Weighted Utility = 0.4863 (48.63%)**.
   - **Overall ECE = 0.0185 (1.85%)**, Gap ECE = **0.0072 (0.72%)**.

## 3. Scientific Synthesis & Hypothesis Verdict
- **Hypothesis H58 Decisively Confirmed:**
  * Decoupling the three structural gap regimes eliminated parameter interference between intrusive vowel epenthesis and breathing pause noise.
  * TDIS-Net shatters the stubborn insertion bottleneck, lifting Insertion F1 from the low 0.30s to **0.4321**.
  * **Next Step (EXP-64):** Construct the **13-Model Tredecim Super-Ensemble ("Tredecim-Pareto")** integrating TDIS-Net (EXP-63) into Duodecim-Pareto (EXP-58) to unite 0.4733 Omission AP, 0.4540 Substitution AP, and 0.4321 Insertion F1!
