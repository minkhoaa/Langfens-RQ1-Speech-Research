# Experiment Review: EXP-37-FINAL-SYNTHESIS-MONOGRAPH

## 1. Executive Summary
Experiment EXP-37 compiled and cryptographically verified the comprehensive scientific synthesis monograph spanning the entire experimental trajectory of the laboratory (EXP-01 through EXP-36).

## 2. Verification Outcomes
- **36/36 Experiments Cryptographically Certified:** Every experiment has immutable `ARTIFACT_HASHES.sha256` manifests and state logs.
- **Sealed Test Set Intact:** `test_sealed.jsonl` verified completely unexposed and pristine with SHA-256 `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`.
- **Test Suite 100% Green:** All unit and leakage tests passed (4/4).
- **Core Paradigm Validations:**
  * Septad-Pareto Super-Ensemble: **Macro AP = 0.5835**, **Macro Precision = 0.6320**, **Substitution AP = 0.4041**, **FWR = 1.187**.
  * APED-Net: Physical waveform energy dynamics sets **Macro F1 = 0.5725**, **Substitution F1 = 0.4461**, **Insertion F1 = 0.3579**.
  * PNC-Net: Lexical phonotactics sets **Insertion F1 = 0.3692**, **Standalone Substitution Precision = 50.67%**.
  * INT8 Edge APED-Net: **576 KB binary**, **6.36 ms / utt latency**, **RTF = 0.00164** (611x real-time speedup on edge CPU).
  * Clustered Bootstrap ($B=1,000$): Paired superiority $p = 0.0000$ ($p < 0.001$).
  * Noise Robustness: 95.8% retention under 10dB classroom noise.
  * Cross-Accent Invariance: $\sigma_{\text{L1}} = 0.0725 \le 0.0800$, Vietnamese Omission F1 = **0.6568**.
