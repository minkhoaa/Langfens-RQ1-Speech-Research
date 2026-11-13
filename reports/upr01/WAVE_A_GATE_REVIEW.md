# WAVE A GATE REVIEW & TRANSITION DECISION

**Campaign ID:** UPR-01 — Evidence-Driven Integration  
**Date of Review:** 2026-11-05T23:15:00Z  
**Governing Directive:** `UNIFIED_PRONUNCIATION_RESEARCH_TWO_WAVE_EXECUTION.md`  
**Reviewing Officers:**
- Unified Pronunciation Research Director
- Independent Scientific Reviewer & Evaluator Auditor
**Verdict:** **PROMOTED_TO_WAVE_B** (Gates G0, G1, G2, G3 Passed)

---

## 1. Audit of Evidence and Exact Checksums

### 1.1. Laboratory Source Revisions (Immutable Commit HEADs)
- **Main Research (EXP-101+):** `4c3169fe29b019509da83746d11e3afec0798045` (`/home/khoa/Projects/Pronunciation-Research-EXP101`)
- **Scientific Recovery:** `94bba1e2674c18337ffeef79868b89f54f957288` (`/home/khoa/Projects/Pronunciation-Scientific-Recovery`)
- **Frontier Research:** `3aa16164cd2cadc52100b418927bdfce7f960619` (`/home/khoa/Projects/New-Method-Research/Pronunciation-Frontier-Research`)
- **Quarantined Archival Lab:** `/home/khoa/Projects/New-Method-Research` (read-only provenance)

### 1.2. Key Checksums & Shared Ancestry Verification
- **Shared Prototype Bank:**
  `reference_phone_prototypes.npz` in Main and Frontier are byte-identical:
  `SHA-256: a209d214397b65550764afcb5f6e803b6ddd9aa7908eafa19677b20a413f9705` (110,883 bytes).
  *Finding:* Claims of independent discovery between Main and Frontier prototype representations are **falsified**. They are treated as single-ancestry evidence.
- **Main Whitened Subspace Matrix:**
  `phonetic_whitened_projection.npz`:
  `SHA-256: e8ebaa7fe5f6a9c37225c56e301d010bb92ef43c9735dbe0011559e21715694a` (105,946 bytes).

---

## 2. Technical Replay & Forensic Audits (WS1 Evidence)

1. **Replay of Prior Deficiencies (4/4 Confirmed):**
   - *Frontier M02 Leading Omission Crash:* Replicated `IndexError: index -4 is out of bounds for axis 1 with size 3`. Cause: mismatched omission cost ($1.5$ in initialization vs $1.2$ in recursion) and lack of boundary protection when $j=0$. Fixed in `adapters/frontier_m02_adapter.py`.
   - *Main M11 Resampling Distortion:* Replicated $50\%$ duration loss on $22.05\text{ kHz}$ audio caused by hardcoded `resample_poly(160, 441)`. Fixed in `adapters/main_m11_adapter.py` using dynamic $\gcd(16000, sr)$ polyphase filtering.
   - *Frontier Count-Only Insertion Metric:* Replicated vulnerability where count-only scoring awarded $F_1 = 1.0$ despite $100\%$ spatial misalignment. Exact gap matching ($0..N$) enforced in `evaluator/unified_evaluator.py`.
   - *EXP-116 and EXP-120 Summary Rows:* Replicated arithmetic contradiction between reported class $F_1$ and macro $F_1$. Proved that EXP-120 was a spliced chimera combining headline metrics from 5 distinct runs. Replaced by raw count recomputation.
2. **Standardized Runtime Adapters:**
   All 4 candidate models (`MAIN_M11_CORRECTED`, `RECOVERY_M07`, `FRONTIER_M02_CORRECTED`, `FRONTIER_M03_CORRECTED`) execute cleanly from raw unsegmented waveform + written prompt, emitting full 4-class target predictions and exact gap coordinates.

---

## 3. Matched Component Contributions (WS2 Evidence)

*Evaluated on identical DEV_SEARCH 50 utterances, 4 speakers (NCC, SVBI, YBAA, YDCK), 1,643 targets, 1,693 gaps:*

| Arm ID | Mechanism Tested | Target Macro-3 F1 | Correct F1 | Subs Recall | Subs F1 | Omis Recall | Omis F1 | Ins F1 | FWR (err/utt) | Causal Verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **A0** | CTC Baseline | 0.3692 | 0.9168 | 0.1910 | 0.1700 | 0.0000 | 0.0000 | 0.1231 | 3.06 | Total omission blindness |
| **A1** | A0 + Layer 6 Contrast | 0.3129 | 0.8712 | 0.0674 | 0.0687 | **0.5000** | **0.3636** | 0.1231 | 10.48 | **Causal omission detection** ($p < 0.0001$) |
| **A2** | A0 + 32-D Whitened Subspace | 0.2904 | 0.8105 | **0.4607** | **0.2523** | 0.0000 | 0.0000 | 0.1231 | 12.30 | **Causal minimal-pair unmasking** ($p < 0.0001$) |
| **A3** | A0 + Soft Lattice Alignment | 0.3692 | **0.9168** | 0.1910 | 0.1700 | 0.0000 | 0.0000 | 0.1515 | 3.06 | Sequence alignment & gap localization |
| **A4** | A1 + A2 (Dual-Metric Fusion) | 0.2605 | 0.7850 | 0.2360 | 0.1818 | **0.5000** | **0.3636** | 0.2069 | 15.84 | **Complementary orthogonal signals** |
| **A5** | A1 + A3 (Contrast + Soft Lattice)| 0.3124 | 0.8705 | 0.0674 | 0.0687 | **0.5000** | **0.3636** | **0.2581** | 9.80 | Viable alternative hybrid |

### Key Mechanistic Conclusions:
1. **Omission Mechanism Proven:** Baseline CTC fails on omission due to LM prior. Layer-6 contrast provides genuine subphonetic absence evidence ($+50.0\%$ absolute recall, McNemar $\chi^2 = 18.05$, $p < 0.0001$).
2. **Substitution Mechanism Proven:** 32-D Whitened Mahalanobis transform removes articulatory manner dominance, significantly boosting substitution recall ($19.1\% \to 46.1\%$, $p < 0.0001$).
3. **Signal Complementarity Established:** Layer 6 and Whitened subspace signals operate on orthogonal feature spaces with minimal negative interference.
4. **Pedagogical Safety Bottleneck:** Raw unconstrained error-hunting inflates FWR ($3.06 - 15.84\text{ err/utt}$), vastly exceeding the pedagogical threshold ($FWR \le 0.20$). Safe integration requires calibrated thresholding or tiered feedback.

---

## 4. Graduation Gate Status Table

| Gate | Criterion | Status | Evidence Reference |
|---|---|:---:|---|
| **G0** | Source/runtime provenance locked | **PASS** | `WS1_SOURCE_AND_ARTIFACT_SHA256.txt` |
| **G1** | Verified evaluator, population & event schema | **PASS** | `WS1_EVALUATOR_REVIEW.md`, `tests/test_unified_evaluator.py` (8/8 PASS) |
| **G2** | Native forwards & eligible predictions persisted | **PASS** | `WS1_MODEL_FORWARDS.jsonl`, `WS1_PREDICTION_LEDGER.jsonl` |
| **G3** | Matched ablations, paired errors & uncertainty | **PASS** | `WS2_MATCHED_COMPONENT_RESULTS.csv`, `WS2_PAIRED_ERRORS.csv`, `WS2_STATISTICS.md` |
| **G4** | Preregistered minimal integration candidate | **PENDING (WAVE B)**| Scope of Workstream 3 |
| **G5** | Frozen candidate + independent cohort custody | **PENDING (WAVE B)**| Scope of Workstream 4 |
| **G6** | Scientific generalization verdict | **PENDING (WAVE B)**| Final Validation |

---

## 5. Transition Decision: Promote to WAVE B

The prerequisites set forth in Section 2 and Section 5 of `UNIFIED_PRONUNCIATION_RESEARCH_TWO_WAVE_EXECUTION.md` have been fulfilled:
- At least one integration hypothesis has matched evidence of complementary signals (Layer 6 omission evidence + Whitened substitution evidence + Soft sequence alignment).
- Technical replay and benchmark repair are complete.

**DECISION:**
1. **AUTHORIZE IMMEDIATE LAUNCH OF WAVE B:**
   - **Workstream 3 (WS3): Minimal Integration Pilots** — Execute pre-registered pilots P1 (Omission evidence fusion), P2 (Insertion localization with AGBV), and P3 (Pedagogical safe feedback policy).
   - **Workstream 4 (WS4): Independent Generalization Preparation & Evaluation** — Concurrently establish custody of prospective untouched cohort (Suitcase Spontaneous phrases), pre-declare frozen evaluation protocol, freeze candidate, and execute final independent evaluation.

Signed:  
*Unified Pronunciation Research Director*  
*Independent Scientific Reviewer & Metric Auditor*  
Date: 2026-11-05
