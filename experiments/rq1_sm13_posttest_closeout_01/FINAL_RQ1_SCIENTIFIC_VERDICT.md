# LANGFENS — FINAL RQ1 SCIENTIFIC VERDICT & MODEL SELECTION
## Protocol: LANGFENS-RQ1-POSTTEST-CLOSEOUT-01
**Research Question 1 (RQ1):** *Can scalar-mixing multi-layer acoustic representations coupled with sequence likelihoods decouple and diagnose phone-level pronunciation errors within a strict pedagogical safety ceiling?*  
**Evaluated Candidate:** `LANGFENS-SM13-ASYMMETRIC-ROUTING-V1` (Full-TRAIN Seed 101, Budget $B=0.12$)  
**Historical Baseline:** `LANGFENS-SM13-FUSION-V1` (Fixed 50/50 Target Fusion, Budget $B=0.20$)  
**Governing Standard:** EVAL-V3.1 (`UnifiedEvaluatorV3`, Exact $N$ Targets, $N+1$ Gaps)  
**Evaluated Cohort:** `TEST_SEALED` (600 utterances across 4 held-out speakers: `ZHAA`, `TNI`, `MBMPS`, `TLV`)  
**Evaluation Mode:** Final Authorized Scientific Closeout (Zero Re-Execution, Bitwise Audited)  
**Scientific Authority:** LANGFENS Research Directorate & Lead Systems Auditor  

---

## 1. Master Scientific Verdict Card

```
========================================================================================
                          LANGFENS RQ1 SCIENTIFIC VERDICT
========================================================================================
Candidate Model:     LANGFENS-SM13-ASYMMETRIC-ROUTING-V1 (Seed 101, B=0.12)
Historical Baseline: LANGFENS-SM13-FUSION-V1 (Fixed 50/50 Simplex Fusion)
Evaluation Cohort:   TEST_SEALED (600 Utterances, 19,931 Targets, 20,531 Gaps)
----------------------------------------------------------------------------------------
Preregistered Gate                     Criterion                       Realized   Status
----------------------------------------------------------------------------------------
Gate 1: Safety Ceiling                 Candidate Joint FWR <= 0.2000   0.163333    PASS
Gate 2: Substitution Cure              Rec_S >= 2.0% & Prec_S >= 40.0% R=4.34%     PASS
                                       and Rec_S > Baseline            P=75.59%
Gate 3: Omission Retention             Rec_O >= 0.70 * Base Rec_O      94.3% ret.  PASS
Gate 4: Insertion Retention            Rec_I >= 0.85 * Base Rec_I      93.7% ret.  PASS
Gate 5: Multi-Class Gain               Candidate F1 > Baseline F1      +0.022198   PASS
                                                                       (+14.86%)
----------------------------------------------------------------------------------------
MASTER VERDICT: CANDIDATE_ACCEPTED_SUPERIOR
RQ1 RESEARCH-SELECTED MODEL: LANGFENS-SM13-ASYMMETRIC-ROUTING-V1
TEST_SEALED STATUS: EVALUATED / NO LONGER BLIND (PERMANENTLY SEALED AGAINST RE-TESTING)
========================================================================================
```

---

## 2. Research-Selected Model Freezing

The candidate model **`LANGFENS-SM13-ASYMMETRIC-ROUTING-V1`** is hereby declared the official **RQ1 Research-Selected Model**. Its neural architecture, checkpoints, decision policies, operating thresholds, and tie-breaking contracts are permanently frozen.

### 2.1 Frozen Checkpoint Ledger & Cryptographic Hashes

| Component | Canonical Repository Checkpoint Path | Verified SHA-256 Checksum | Size (Bytes) |
|---|---|---|:---:|
| **Target Head** | `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt` | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` | 2,829,321 |
| **Gap Head** | `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt` | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` | 1,774,489 |
| **Linear CTC** | `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt` | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` | 134,817 |
| **Decision Policy**| `FROZEN_ASYMMETRIC_POLICY.json` | `e74ea31804f2d00d731aefffbfef2d613e6112d7965c4cbf60807b0bcee45023` | 8,258 |

### 2.2 Frozen Operating Thresholds & Routing Contract

The model operates under an asymmetric, decoupled routing contract designed to avoid the mutual-exclusion pathology of static 3-way softmax/simplex fusion:

```
                                    ┌───────────────────────┐
                                    │ Input Waveform + Text │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │ 13-Layer Wav2Vec2-960h│
                                    └─────┬───────────┬─────┘
                     Target Features (2304D)│           │Gap Features (1536D)
              ┌─────────────────────────────┘           └─────────────────────────────┐
              │                                                                       │
    ┌─────────▼──────────┐                                                  ┌─────────▼──────────┐
    │Target Head (SM13)  │                                                  │Gap Head (SM13)     │
    └─────────┬──────────┘                                                  └─────────┬──────────┘
              │                                                                       │
    p_neural[:, 1] (Sub)                                                    p_gap (Ins)
              │                                                                       │
              ▼                                                                       ▼
    s_S >= 0.949374? ───┐                                                   s_I >= 0.886658? ───► INSERTION
                        │                                                                         (else CLEAN)
    ┌───────────────────┼────────────────────────────────────────┐
    │                   │                                        │
    │                   ▼                                        │
    │    ┌──────────────────────────────┐                        │
    │    │ CTC Proj (L6) + C++ E4 Kernel│                        │
    │    └──────────────┬───────────────┘                        │
    │                   │                                        │
    │             p_fixed[:, 2] (Omi)                            │
    │                   │                                        │
    │                   ▼                                        │
    │         s_O >= 0.857454? ───┐                              │
    │                             │                              │
    │                             ▼                              │
    │              ┌─────────────────────────────┐               │
    │              │ Arbitration: score_margin   │               │
    │              │ (margin_S vs margin_O)      │               │
    │              └──────────────┬──────────────┘               │
    │                             │                              │
    │         ┌───────────────────┴───────────────────┐          │
    │         ▼                                       ▼          │
    │    SUBSTITUTION                              OMISSION      │
    │                                                            │
    └────────────────────────────────────────────────────────────┘
                        (Neither triggered -> CORRECT)
```

1. **Substitution Stream:** Evaluated directly from `ScalarMixingTargetHead` neural posterior:
   $$s_S = p_{\text{neural}}[i, 1] \ge \tau_S = 0.9493744373321533$$
   *(Tie cutoff: `12445346866078397902`)*
2. **Omission Stream:** Evaluated from the 50/50 fused sequence-acoustic simplex:
   $$s_O = p_{\text{fixed}}[i, 2] \ge \tau_O = 0.8574540615081787$$
   *(Tie cutoff: `2357324174070838657`)*
3. **Target Arbitration:** If both $s_S \ge \tau_S$ and $s_O \ge \tau_O$, resolve via `score_margin`:
   $$\Delta_S = s_S - \tau_S, \quad \Delta_O = s_O - \tau_O$$
   - If $\Delta_S > \Delta_O \implies \text{SUBSTITUTION}$
   - If $\Delta_O > \Delta_S \implies \text{OMISSION}$
   - If $\Delta_S == \Delta_O \implies \text{tie\_hash}(event\_id) \pmod 2 == 1 \ ? \ \text{SUBSTITUTION} : \text{OMISSION}$
   - If neither triggers $\implies \text{CORRECT}$ (label `0`)
4. **Insertion Stream:** Evaluated independently from `ScalarMixingGapHead`:
   $$s_I = p_{\text{gap}}[j] \ge \tau_I = 0.8866584300994873$$
   *(Tie cutoff: `9157070711166496197`)*

---

## 3. Retention of Historical Comparison Baseline

The prespecified production baseline, **Fixed 50/50 Target Fusion** (`LANGFENS-SM13-FUSION-V1`, Policy SHA-256: `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1`), is permanently retained as the benchmark reference:
- **Operating Thresholds:** $\tau_T = 0.848521411895752, \quad \tau_G = 0.8736369609832764$
- **Decision Mechanism:** Coupled 3-way argmax over the fixed 50/50 sequence-neural simplex ($0.5 \times p_{\text{neural}} + 0.5 \times p_{\text{E4}}$).
- **Fundamental Scientific Flaw:** On unseen non-native speakers, coupling substitution and omission under a single shared simplex threshold caused total suppression of substitution warnings (**0 True Positives, 0.00% recall across 2,214 gold substitution errors**), rendering the baseline incapable of multi-class error diagnosis.
- **Role in Closeout:** Preserved intact as the negative control demonstrating why decoupled asymmetric routing was theoretically and empirically mandatory.

---

## 4. Comprehensive Scientific Findings & Gate Documentations

### 4.1 Gate-by-Gate Evaluation & Verification

#### Gate 1: Safety Ceiling (PASS)
- **Standard:** Realized Joint False Warning Rate on held-out test data must satisfy $\text{FWR} \le 0.2000$ (no more than 1 false alert per 5 utterances).
- **Outcome:** Realized candidate $\text{FWR} = \mathbf{0.163333}$ (98 false warnings over 600 utterances).
- **Safety Margin:** 22 fewer false warnings than the 120 allowed under the $0.2000$ ceiling.

#### Gate 2: Substitution Pathology Recovery (PASS)
- **Standard:** The model must genuinely cure the zero-substitution pathology on held-out test data, achieving Recall $\ge 2.0\%$, Precision $\ge 40.0\%$, and strictly exceeding the baseline.
- **Outcome:** Realized Substitution Recall $= \mathbf{4.34\%}$ (96 True Positives vs. Baseline 0 TP), realized Substitution Precision $= \mathbf{75.59\%}$ (96 TP / 127 alerts).
- **Verdict:** Decoupled neural routing completely cured the pathology, generating clean, actionable alerts with $> 75\%$ precision.

#### Gate 3: Omission Retention (PASS)
- **Standard:** Omission recall must retain at least $70.0\%$ of baseline omission recall ($15.7421\% \times 0.70 = 11.0195\%$).
- **Outcome:** Realized Omission Recall $= \mathbf{14.8426\%}$ (99 TP vs. Baseline 105 TP), achieving **$94.3\%$ relative retention**. Precision improved from $67.74\%$ to $68.75\%$.

#### Gate 4: Insertion Retention (PASS)
- **Standard:** Insertion recall must retain at least $85.0\%$ of baseline insertion recall ($13.1148\% \times 0.85 = 11.1475\%$).
- **Outcome:** Realized Insertion Recall $= \mathbf{12.2951\%}$ (15 TP vs. Baseline 16 TP), achieving **$93.7\%$ relative retention**. Precision improved from $36.36\%$ to $40.54\%$.

#### Gate 5: Multi-Class Superiority (PASS)
- **Standard:** Candidate unweighted Macro-3 Error F1 must strictly exceed Baseline Macro-3 Error F1.
- **Outcome:** Candidate $\text{Macro-3 F1} = \mathbf{0.171613}$ vs. Baseline $\text{Macro-3 F1} = \mathbf{0.149415}$.
- **Net Gain:** Decisive absolute improvement of $\mathbf{+0.022198}$ ($\mathbf{+14.86\%}$ relative gain).

---

## 5. Critical Scientific Disclosures, Trade-Offs & Limitations

### 5.1 The Low-Recall Reality Under Pedagogical Safety
While all five gates were passed, scientific integrity demands explicit documentation of the model's absolute recall limitations:
- **Substitution Recall:** $4.34\%$ ($96$ detected out of $2,214$ gold substitutions; $2,118$ false negatives).
- **Omission Recall:** $14.84\%$ ($99$ detected out of $667$ gold omissions; $568$ false negatives).
- **Insertion Recall:** $12.30\%$ ($15$ detected out of $122$ gold insertions; $107$ false negatives).
- **Overall Error Coverage:** The model flags only $210$ true mispronunciations out of $3,003$ total gold errors on `TEST_SEALED` ($7.0\%$ error coverage).

**Pedagogical Root Cause:** In second-language pronunciation training, false warnings create severe cognitive distress, leading learners to second-guess correct articulations. To guarantee $\text{FWR} \le 0.20$ across diverse non-native accents, decision thresholds must be placed at the extreme upper tail of acoustic confidence ($\tau_S \approx 0.95, \tau_O \approx 0.86, \tau_I \approx 0.89$). The system intentionally trades recall for high precision ($\approx 75.6\%$ on substitutions, $\approx 68.8\%$ on omissions).  
**Handoff Directive:** Product implementations must explicitly convey to users that *absence of a warning does not guarantee native accuracy*—it merely indicates the absence of a high-confidence, pedagogically disruptive error.

### 5.2 Ranking Average Precision (AP) vs. Discrete F1 Trade-Off
A key finding of this research is the divergence between continuous ranking metrics and discrete operational utility:

| Metric | Fixed 50/50 Baseline | SM13 Asymmetric | Absolute Delta ($\Delta$) | Qualitative Meaning |
|---|:---:|:---:|:---:|---|
| **Substitution AP** | 0.413145 | 0.408628 | -0.004517 | Minor ranking loss from neural-only score |
| **Substitution F1** | **0.000000** | **0.082016** | **+0.082016** | **Pathology cure (0 TP $\to$ 96 TP)** |
| **Omission AP** | 0.488440 | 0.488440 | +0.000000 | Invariant (both route from 50/50 simplex) |
| **Omission F1** | 0.255474 | 0.244143 | -0.011331 | Slight drop (-6 TP, -5 FP) |
| **Insertion AP** | 0.184249 | 0.184249 | +0.000000 | Invariant (both route from gap head) |
| **Insertion F1** | 0.192771 | 0.188679 | -0.004092 | Slight drop (-1 TP, -6 FP) |
| **Macro-3 AP** | **0.361945** | **0.360439** | **-0.001506** | Continuous ranking essentially invariant |
| **Macro-3 F1** | **0.149415** | **0.171613** | **+0.022198** | **Net operational gain (+14.86%)** |

**Interpretation:** Although continuous Average Precision (AP) exhibits a trivial drop of $-0.0015$ (driven by routing substitution from neural scores rather than fused scores), the discrete operating point achieves a substantial $+0.0222$ Macro-3 F1 gain. In the baseline, a superior AP score of $0.413$ was completely unusable in practice because simplex argmax coupling suppressed all substitution alerts.

### 5.3 Per-Speaker Cross-Lingual Nuances & Limitations

Evaluation across four distinct L1 backgrounds reveals substantial accent-conditioned variation:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │          PER-SPEAKER JOINT FWR VS MACRO-3 F1            │
                  ├───────────────────┬──────────────┬──────────────────────┤
                  │ Speaker (L1)      │  Joint FWR   │ Macro-3 Error F1     │
                  ├───────────────────┼──────────────┼──────────────────────┤
                  │ MBMPS (Spanish M) │    0.1267    │        0.0865        │
                  │ TLV (Vietnamese M)│    0.2467    │        0.1422        │
                  │ TNI (Hindi M)     │    0.2000    │        0.0559        │
                  │ ZHAA (Arabic F)   │    0.0800    │        0.1619        │
                  └───────────────────┴──────────────┴──────────────────────┘
```

1. **Vietnamese Speaker (`TLV`):**
   - Exhibited the highest error density ($1,265$ total gold errors: $789$ Sub, $437$ Omi, $39$ Ins).
   - FWR reached $0.2467$ ($37$ false warnings), slightly exceeding the pooled ceiling due to frequent final-consonant de-voicing and glottalization.
   - High precision on detected substitutions ($86.0\%$, $43$ TP / $50$ alerts) and omissions ($79.5\%$, $89$ TP / $112$ alerts).
   - Zero insertion recall ($0/39$ detected).
2. **Hindi Speaker (`TNI`):**
   - FWR rose from $0.0933$ ($14$ FW) to $0.2000$ ($30$ FW) under asymmetric routing.
   - Lowest substitution precision ($43.33\%$, $13$ TP / $30$ alerts, $17$ FP) due to retroflex consonant confusion with dental stops.
   - Zero insertion recall ($0/8$ detected).
3. **Spanish Speaker (`MBMPS`):**
   - Highest substitution recall ($6.42\%$, $34$ TP) with high precision ($85.0\%$).
   - Lowest omission recall ($1.59\%$, $1$ TP / $63$ gold omissions).
   - Well within the safety ceiling ($\text{FWR} = 0.1267$, $19$ FW).
4. **Arabic Speaker (`ZHAA`):**
   - Lowest FWR of all speakers ($\text{FWR} = 0.0800$, $12$ FW; actually improved from Baseline $0.0867$).
   - Highest insertion recall ($26.53\%$, $13$ TP / $49$ gold insertions, $68.4\%$ precision), correctly catching vowel epenthesis in complex English consonant clusters.
   - High substitution precision ($85.71\%$, $6$ TP / $7$ alerts).

---

## 6. Formal Test Cohort Burn Notice

The held-out partition **`TEST_SEALED`** (`data/derived/l2arctic/manifests/test_sealed.jsonl`) is formally designated:
# **STATUS: EVALUATED / NO LONGER BLIND / PERMANENTLY BURNED**

### Anti-Contamination Mandate:
1. **Zero Retraining:** No future models may be trained on or incorporating any audio, alignments, or gold labels from `TEST_SEALED`.
2. **Zero Retuning:** Thresholds $\tau_S, \tau_O, \tau_I$ may never be adjusted or optimized against `TEST_SEALED`.
3. **Prohibition of Repeated Testing:** `TEST_SEALED` is formally closed. No secondary candidates, hyperparameter sweeps, or ablations may be evaluated on this cohort to preserve scientific validity and prevent publication bias.
4. **Subsequent Research:** Any future investigations (e.g., RQ2/RQ3) must construct new, independent, unexposed evaluation cohorts.

---

## 7. Official Closeout Pronouncement

The prospective scientific evaluation of **`LANGFENS-SM13-ASYMMETRIC-ROUTING-V1`** under protocol `LANGFENS-SM13-TEST-EVALUATION-PROTOCOL-V1.0` has completed in full compliance with research integrity standards.

All preregistered acceptance criteria have been satisfied. The zero-substitution pathology has been eliminated on unseen non-native speakers, pedagogical safety has been maintained, and multi-class diagnostic capability has been conclusively demonstrated.

# **FINAL OFFICIAL VERDICT: `CANDIDATE_ACCEPTED_SUPERIOR`**
### **RQ1 RESEARCH-SELECTED MODEL: `LANGFENS-SM13-ASYMMETRIC-ROUTING-V1`**

*Executed and Authorized by the LANGFENS Scientific Governance Board on November 11, 2026.*
