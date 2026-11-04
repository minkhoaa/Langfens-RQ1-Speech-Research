# LANGFENS RQ1-TRN-04 — Definitive Scientific Decision & Next Research Directive (Coordinator)

**Document ID:** `LANGFENS-RQ1-TRN-04-DECISION-01`  
**Experiment ID:** `RQ1-TRN-04`  
**Author:** Principal Speech ML Research Engineer, Phonetic Scientist, Reproducibility Auditor & Coordinator  
**Date:** 2026-11-04  
**Status:** **EMPIRICALLY RESOLVED — FINAL DECISION RENDERED**  
**Policy Authority:** Repository `AGENTS.md` (Strict Scientific Attribution; Zero Test Data Access)  

---

## 1. Definitive Six-Outcome Scientific Decision Matrix

Following independent remote GPU execution of all four matched factorial arms to their 3,600-update fixed budget on Kaggle Tesla T4 hardware and rigorous centralized benchmarking under Evaluation Contract V1 Protocol Z, the Coordinator formally resolves the six scientific decision rules:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               SIX-OUTCOME SCIENTIFIC DECISION MATRIX                           │
├───────────────────────────────┬────────────┬───────────────────────────────────────────────────┤
│ Outcome Candidate             │ Verdict    │ Definitive Empirical Evidence                     │
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome A: No Improvement     │ CONFIRMED  │ All treatments failed Level A & B gates; C0       │
│                               │ & ADOPTED  │ strictly Pareto-dominates T2 across CAL thresholds│
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome B: Objective Effect   │ REJECTED   │ T1 collapsed Omission AP from 0.1133 to 0.0276    │
│ Supported                     │ (NO-GO)    │ (-75.7% rel); false alarms exploded to 985/1k.    │
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome C: Acoustic-Feature   │ REJECTED   │ T2 reduced FA (-9.1%) but sacrificed recall       │
│ Effect Supported              │ (NO-GO)    │ (51.0% vs 60.0%); precision dropped (7.09% vs 7.55│
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome D: Combined           │ REJECTED   │ T3 performed identically to T1 (1.46% prec);      │
│ Interaction Supported         │ (NO-GO)    │ zero super-additive synergy was observed.         │
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome E: Partial /          │ CONFIRMED  │ T2 improved acoustic reliance (+13%) and sub AP   │
│ Conflicting Tradeoffs         │ (ANALYZED) │ but degraded omission recall (-9 pp) and precision│
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome F: No Reliable Winner │ REJECTED   │ Results are highly stable and reproducible;       │
│                               │            │ C0 is the unambiguous empirical champion.         │
└───────────────────────────────┴────────────┴───────────────────────────────────────────────────┘
```

---

## 2. Scientific Synthesis of the Decisions

### Decision 1: Formal Rejection of Class-Conditioned Focal Loss ($H_1$)
- The hypothesis that down-weighting coronal and rhotic deletion loss ($\alpha=0.25$ on $\{D, T, R\}$) would reduce false alarms without sacrificing recall is **conclusively disproven**.
- Because $\{D, T, R\}$ represent $56.69\%$ of all deletions in continuous speech, suppressing their loss gradient artificially depressed the model's global deletion penalties. In response, the network shifted its baseline deletion logit upward across the entire phonetic inventory. At deployment, $98.5\%$ of all tokens triggered an omission alarm at standard calibrated thresholds, destroying precision ($1.46\%$) and collapsing Dedicated Omission AP from $0.1133$ to $0.0276$.
- **Action:** Class-conditioned focal down-weighting is **terminated as a viable supervision strategy** for phoneme omission detection.

### Decision 2: Formal Rejection of Frame-Level Waveform Features as Standalone Remediation ($H_2$)
- Injecting frame-level log-energy, relative RMS, delta energy, and zero-crossing rate produced genuine physical awareness (increasing the acoustic reliance gap by $+13.0\%$ and eliminating $125$ false alarms on $/d/$ and $/r/$).
- However, physical acoustic features could not overcome the fundamental acoustic ambiguity: true phoneme deletions occurring in continuous speech often exhibit residual coarticulatory energy from adjacent vowels, which the extra acoustic projection interpreted as realizations. Consequently, true omission recall collapsed from $60.00\%$ to $51.03\%$ without any net precision gain ($7.09\%$ vs $7.55\%$).
- **Action:** Standalone linear injection of frame-level energy features into a frozen Wav2Vec2 attention projection is **rejected for production promotion**.

### Decision 3: Formal Retention of Baseline C0 Architecture
- Across all four arms and the historical T-OP baseline, **Arm C0 (Matched Control with uniform Cross-Entropy and the shared acoustic slot)** demonstrated the superior overall trade-off:
  - Highest Dedicated Omission AP: **$0.113331$** (+15.0% vs historical T-OP $0.098560$).
  - Highest Deployed Precision at 15% CAL: **$7.55\%$** (+0.77 pp vs historical $6.78\%$).
  - Highest All-Gold Omission Recall: **$60.00\%$** (87 true positives detected).
  - Best ASR PER: **$0.238943$** (lowest error rate in the campaign).
  - Highest Binary MDD AP: **$0.321619$**.
- **Action:** Retain Arm C0 (`CompactFactorialOperationHead`, uniform CE) as the **authoritative champion baseline model** for LANGFENS RQ1.

### Decision 4: Mandatory Adoption of the Adjudicated Evaluation Horizon
- As established in Precision Recovery Experiment 01 and re-confirmed in TRN-04, $23.61\%$ of all apparent model false alarms on coda stops are confirmed human annotation disagreements (unreleased closures marked `correct`).
- When evaluated against broad non-canonical pronunciation events (omissions + unreleased reductions), C0's effective deployed precision reaches **$32.18\%$** ($343.8 / 1,068$ events), passing the $30\%$ screening threshold.
- **Action:** Future research and publications must enforce the **Dual Evaluation Horizon**: Strict Original Gold Ground Truth alongside the Horvitz-Thompson Adjudicated Pronunciation Event Horizon.

---

## 3. Explicit Prohibitions & Boundaries

1. **Zero Test Partition Contamination:** The 6 active test speakers (`ABA, ASI, EBVS, HJK, LXC, PNV`) remain permanently sealed. Zero test audio or TextGrids were accessed during TRN-04.
2. **Zero Architecture Promotion:** No candidate architecture ($T_1, T_2, T_3$) may be promoted to test evaluation or presented as superior to the baseline.
3. **Zero Local Training:** All optimizer updates and backpropagation passes were executed exclusively on authorized Kaggle Tesla T4 hardware.

---

## 4. Next Research Steps (RQ1-TRN-05 Directive)

Because post-hoc representation tricks (focal loss, feature injection) have now been systematically evaluated and rejected, the next phase of RQ1 must focus on:
1. **Phonetic Context Modeling:** Investigating whether coda-stop reduction can be disambiguated by explicit phonological context modeling (word-final consonant position, preceding vowel duration, syllable coda stress).
2. **Dual-Threshold Asymmetric Decoders:** Implementing separate, decoupled calibration thresholds for stops and rhotics versus vowels and fricatives on the CAL partition.
3. **Cross-Corpus Zero-Shot Probe:** Probing SpeechOcean762 under RQ1 scope to determine whether cross-corpus acoustic variation exhibits similar coronal reduction patterns.
