# UPR-02 RECOMMENDED NEXT EXPERIMENTS & RESEARCH ROADMAP
**Campaign:** UPR-02  
**Document ID:** UPR02-ROADMAP-20261106  
**Status:** APPROVED FOR POST-UPR02 INVESTIGATION

---

## 1. Executive Summary of UPR-02 Discoveries
UPR-02 demonstrated that raw acoustic contrast mechanisms (Layer 6 contrast, 32-D Mahalanobis whitened metric) boost raw error recall but suffer from catastrophic false warning explosion when unconstrained. By developing acoustic presence verification (energy/RMS ratio + silence floor gating), unstressed vowel protection, phonologically-bounded whitened margins, and exact-gap AGBV with class-specific calibration, UPR-02 achieved the pedagogical safety target:
- **False Warning Rate (FWR) $\le 0.20$** per utterance strictly certified ($0.1703$ false warnings/utt on full DEV_SEARCH).
- **Omission Precision improved 4x** over baseline (from 10.56% to 40.00%).
- **Expected Calibration Error (ECE) reduced to $0.0223$**.
- All 3 error detection classes (Substitution, Omission, Insertion) remain active.

---

## 2. Prioritized Technical Experiments for UPR-03

### Recommendation 1: Trainable Soft Alignment & Segment Pruning (TRAIN Split)
- **Problem:** Currently, DP sequence alignment is fixed with heuristic deletion/insertion costs ($1.2$). While effective, it generates $\approx 140$ missing slot misalignments due to CMUDict pronunciation variants (e.g. `F ER G EH T` vs `F AO R G EH T`).
- **Proposed Architecture:** Train a differentiable monotonic alignment layer (e.g. forward-backward CTC / connectionist transducer) directly on the authorized `TRAIN` subset (1,800 utterances across 12 speakers).
- **Hypothesis:** Learned alignment transitions will eliminate pronunciation variant mismatches, reducing token abstentions from 140 down to $< 20$.

### Recommendation 2: Articulatory Feature Decomposition for Substitution Calibration
- **Problem:** Substitution detection under strict FWR $\le 0.20$ relies on high confidence ($> 0.96$), which suppresses subtle minimal-pair substitutions (e.g. /IH/ vs /IY/, /S/ vs /TH/).
- **Proposed Architecture:** Instead of scalar posterior thresholding, decompose substitutions into 5 phonological feature distances (Voicing, Place, Manner, Backness, Rounding). If a detected substitution differs along only a single minor articulatory dimension (e.g. voicing only for /S/ vs /Z/), calibrate a lower confidence threshold for issuing constructive feedback without full pedagogical warning penalties.
- **Hypothesis:** Multi-task articulatory decomposition will recover 3x more true substitutions at the same FWR level.

### Recommendation 3: Multi-Frame Temporal Diffusion for Epenthetic Insertion
- **Problem:** Insertion detection achieves $F_1 = 0.1538$ in unconstrained mode and $0.0146$ under pedagogical safety, as true insertions in L2 speech are often brief epenthetic schwas (e.g., /AH/ or /IH/ inserted into consonant clusters like "stop" $\to$ "s-uh-top").
- **Proposed Architecture:** Train an epenthetic vowel resonance detector on `TRAIN` that scans inter-consonant spectral dips for formant resonance patterns ($F_1 \approx 500$ Hz, $F_2 \approx 1500$ Hz).
- **Hypothesis:** Formant-guided epenthesis detection will increase exact-gap insertion recall by $+40\%$ relative while maintaining zero false alarms on normal consonant releases.

### Recommendation 4: Controlled Independent Benchmark Runner
- **Policy:** The encrypted benchmarks (SpeechOcean, Suitcase, and TEST_SEALED) remain isolated under cryptographic custody.
- **Protocol:** Once UPR-03 freezes the candidate checkpoint and runtime container, an external, independently audited evaluation runner should be authorized to decrypt and benchmark the candidate in a sealed environment with zero feedback loop.
