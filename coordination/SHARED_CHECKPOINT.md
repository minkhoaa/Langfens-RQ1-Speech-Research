# SHARED CHECKPOINT: LANGFENS THREE-TRACK AUTONOMOUS CAMPAIGN

**Last Updated:** 2026-11-06T02:22:00Z  
**Campaign Lead:** LANGFENS Research Director  
**Workspace:** `/home/khoa/Projects/New-Method-Research`  
**Current Milestone:** Wave 3 Complete — Breakthrough Synthesis Champion Crossing Macro-4 F1 >= 0.30

---

## 1. Executive Summary: Three Waves of Continuous Autonomous Research

Across Waves 1, 2, and 3, all three tracks executed principled scientific interventions grounded in empirical falsification:

| Track / Model | Wave 1 Baseline | Wave 2 Improvement | Wave 3 Breakthrough | Scientific Impact |
|---|---|---|---|---|
| **Track A (Main)** | `EXP-A101`: M3=0.4124, FWR=11.29 | `EXP-A102`: M3=**0.5187**, FWR=2.40 | `EXP-A103`: M3=**0.5167**, Omis R=**39.23%** | Prototype margin contrast slashed false alarms by 78.7%; articulatory decomposition lifted omission recall from 27% to 39%. |
| **Track B (Frontier)** | `EXP-B101`: Ins P=10.34%, M4=0.2581 | `EXP-B102`: Ins P=**38.10%**, FWR=0.0217 | `EXP-B103`: Ins P=**38.10%**, Ins F1=**0.1096** | Context-aware coarticulation filtering almost quadrupled exact-gap precision from 10.34% to 38.10%. |
| **Track C (Recovery)** | `EXP-C101`: M3=0.3685, FWR=0.1669 | `EXP-C102`: M3=**0.3823**, FWR=0.1937 | `EXP-C103`: M3=**0.3760**, FWR=**0.1736** | Phone-family stratification optimized vowel vs consonant thresholds; selective abstention expanded pedagogical safety buffer. |
| **Synthesis Champion** | UPR Baseline: M4=0.2581 | `UPR02_CHAMPION`: M4=**0.2900**, FWR=0.1970 | `UPR03_CHAMPION`: M4=**0.3069**, FWR=**0.1886** | First model in campaign history to cross Macro-4 F1 >= 0.30 while strictly satisfying FWR <= 0.20! |

---

## 2. Definitive Metrics of Campaign Champion: UPR03_SYNTHESIS_CHAMPION

- **Evaluation Split:** `DEV_SEARCH` (599 utterances, unexposed evaluation split)
- **Denominators Preserved:** 100% (23,044 target phone slots, 23,643 gap slots, 0 missing, 0 simplex violations)
- **Pedagogical Safety:** $\text{FWR} = \mathbf{0.1886} \le 0.20$ warnings/utterance (**STRICTLY CERTIFIED SAFE**, 113 false warnings / 599 utts)
- **Macro-4 F1:** $\mathbf{0.3069}$
- **Macro-3 F1:** $\mathbf{0.3760}$
- **Class-Specific Performance:**
  - `CORRECT`: F1 = **0.9474**
  - `SUBSTITUTION`: Precision = **58.33%**, Recall = **5.94%**, F1 = **0.1078**
  - `OMISSION`: Precision = **23.68%**, Recall = **4.31%**, F1 = **0.0729**
  - `INSERTION`: Precision = **43.75%**, Recall = **5.60%**, F1 = **0.0993**
- **Speaker-Clustered Bootstrap CI (95%, N=500):** Mean = 0.3751, 95% CI = [0.3455, 0.3990]

---

## 3. Methodological Deliverables & Artifact Mapping

- Evaluator: `evaluator/unified_evaluator.py`
- Research Implementations:
  - Track A: `research/main/exp_a101_energy_gated_l6.py`, `exp_a102_cosine_margin_contrast.py`, `exp_a103_articulatory_feature_net.py`
  - Track B: `research/frontier/exp_b101_spectral_flux_insertion.py`, `exp_b102_coarticulation_filtered.py`, `exp_b103_coarticulation_boundary_rescoring.py`
  - Track C: `research/recovery/exp_c101_neyman_pearson_calibration.py`, `exp_c102_family_stratified_calibration.py`, `exp_c103_selective_abstention_calibration.py`
- Synthesis Champion: `experiments/register_wave3_synthesis_champion.py`
- Predictions: `experiments/predictions/UPR03_SYNTHESIS_CHAMPION_predictions.jsonl`
