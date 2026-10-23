# LANGFENS — RQ1-AUD-01: Blockers, Nuances & Next Actions

**Execution Gate:** RQ1-AUD-01 — Dataset Integrity & Annotation Audit  
**Phase:** Phase G — Audit Reports & Reproducibility  
**Audit Timestamp (UTC):** 2026-10-22T20:50:00Z  
**Scope:** Phase Transition & Research Strategy  
**Execution Lead:** Senior Speech ML Research Engineer & Research Data Engineer  

---

## 1. Categorized Findings & Blocker Status

```text
Status Classification Ledger
├── A. True Blockers (Nil — All Audit Gates Pass)
├── B. Critical Empirical Nuances (Discovered During Audit)
├── C. Methodological & Evaluation Risks (Class Imbalance & Leakage)
└── D. Recommended Next Actions (Phase RQ1-PREP-01 Roadmap)
```

---

### A. True Blockers
- **None for Phase RQ1-AUD-01.**
- All acceptance gates (G1 through G5) have passed with full empirical verification. Raw dataset archives are cryptographically verified and structurally intact. All required deliverables are generated.

---

### B. Critical Empirical Nuances Discovered During Audit

1. **Vietnamese Speaker Identity Correction:**
   - *Previous Report Nuance:* Identified `TLV`, `TNI`, `TXHC`, `BWC` as Vietnamese speakers.
   - *Audit Finding:* `BWC` and `TXHC` are Chinese L1; `TNI` is Hindi L1. The true Vietnamese L1 speakers in L2-ARCTIC are `HQTV` (Male), `PNV` (Female), `THV` (Female), and `TLV` (Male). All future scripts and reports must strictly use these four IDs.
2. **Vietnamese Consonant Deletion Concentration:**
   - *Audit Finding:* Vietnamese speakers account for **1,566 out of 3,420 deletions (45.79%)** in the entire corpus. Top deletions are alveolar stops (`D`: 275, `T`: 220), liquids (`R`: 228, `L`: 171), and fricatives (`Z`: 168, `S`: 73).
   - *Impact on Model Design:* Models evaluated on Vietnamese speech must specifically be weighted and tested on final consonant cluster deletion detection.
3. **Audio Sampling Rate Discrepancy (44.1 kHz vs 16 kHz):**
   - *Audit Finding:* L2-ARCTIC raw audio is recorded at 44.1 kHz (CD quality). Pretrained SSL models (Wav2Vec 2.0, HuBERT, WavLM) and MFA acoustic models strictly require 16 kHz single-channel audio.
   - *Impact on Next Phase:* Dynamic on-the-fly resampling (via `torchaudio.transforms.Resample(44100, 16000)`) must be incorporated into dataloaders during data preparation without permanently overwriting raw 44.1 kHz files.
4. **SpeechOcean762 Diagnostic Limitations:**
   - *Audit Finding:* OpenSLR SLR101 does not include an explicit `mispronunciations` field in `scores.json`. It provides scalar quality ratings ($0.0 - 2.0$), not explicit perceived phonetic substitutions.
   - *Impact on Next Phase:* SpeechOcean762 must be reserved for continuous scoring (GOP, GOPT regression), while L2-ARCTIC serves as the benchmark for multi-class error diagnosis.
5. **TextGrid Nonstandard Typo Tokens:**
   - *Audit Finding:* Six nonstandard typo tokens were detected in gold TextGrids: `ER)` (`ABA/arctic_a0025`), `D_` (`BWC/arctic_a0434`), `W`` (`TXHC/arctic_b0418`), `V```` (`TXHC/arctic_b0488`), `Z_` (`HJK/arctic_a0104`), `Y_` (`ERMS/arctic_a0097`). Additionally, over 100 interval labels contain unstripped whitespace (e.g., `"DH,D,s "`).
   - *Impact on Next Phase:* A preprocessing text-cleaning utility must strip whitespace and map these six typos to their base phones (`ER`, `D`, `W`, `V`, `Z`, `Y`).
6. **KoelLabs Lossy IPA Derivative:**
   - *Audit Finding:* KoelLabs discarded lexical stress and collapsed unstressed vowels into `/ə/`.
   - *Decision:* KoelLabs will not be used as ground truth; the project will maintain the canonical ARPAbet annotations of L2-ARCTIC v5.0.

---

### C. Methodological & Evaluation Risks

1. **Class Imbalance in Phoneme Annotations:**
   - L2-ARCTIC: 85.3% Correct vs 14.7% Error.
   - SpeechOcean762: 80.7% Score 2.0 vs 2.4% Score 0.0.
   - *Mitigation:* Models must be optimized using class-weighted loss (e.g. weighted CTC loss or focal loss) and evaluated using Precision, Recall, F1, and PR-AUC on the error class.
2. **Speaker & L1 Acoustic Leakage:**
   - With 24 speakers, random split partitioning leaks speaker vocal tract characteristics into the test set.
   - *Mitigation:* Future data splitting must strictly adhere to a **Speaker-Independent Leave-One-Speaker-Out (LOSO)** or 6-fold cross-validation scheme.

---

## 2. Recommended Next Actions (Roadmap to RQ1-PREP-01)

Upon official sign-off and approval of RQ1-AUD-01, the project should proceed to:

### Phase RQ1-PREP-01 — Data Preparation & Pipeline Standardization
1. **Action PREP-01 (Non-Destructive TextGrid Cleaning Pipeline):**
   - Implement an automated parser that strips trailing whitespace, standardizes three-part error tags, and safely normalizes the 6 isolated nonstandard typo symbols.
2. **Action PREP-02 (Canonical Speaker-Independent Splitting):**
   - Formally partition L2-ARCTIC into a reproducible, speaker-independent 6-fold / 8-fold cross-validation split (e.g., matching Peng et al., 2021), isolating Vietnamese speakers (`HQTV`, `PNV`, `THV`, `TLV`) for dedicated held-out evaluation.
3. **Action PREP-03 (Dynamic Audio Dataloader Scaffold):**
   - Construct PyTorch `Dataset` and `DataLoader` classes that stream audio from raw archives with on-the-fly 44.1 kHz $\rightarrow$ 16 kHz resampling.
4. **Action PREP-04 (MFA Alignment Verification for SpeechOcean762):**
   - Run a test forced alignment on 10 utterances of SpeechOcean762 using `english_us_arpa` v3.0.0 to generate acoustic time boundaries for downstream GOP feature extraction.

---

## 3. Boundary Affirmation

In strict compliance with project directives:
- No model training or fine-tuning was initiated during this audit.
- No raw datasets were altered.
- All next actions are deferred until explicit operator authorization.
