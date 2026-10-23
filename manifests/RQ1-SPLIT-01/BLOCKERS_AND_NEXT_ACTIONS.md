# LANGFENS — RQ1-SPLIT-01: Blockers, Handover & Next Actions

**Milestone:** RQ1-SPLIT-01 — Deterministic Speaker-Independent Split Manifest Generation  
**Engineering Leads:** Senior Speech ML Research Engineer & Research Data Engineer  
**Gate Status:** **RQ1-SPLIT-01: PASS**  
**Deliverable File Path:** `manifests/RQ1-SPLIT-01/BLOCKERS_AND_NEXT_ACTIONS.md`  
**Remote Sync Destination:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-SPLIT-01/BLOCKERS_AND_NEXT_ACTIONS.md`  
**Date of Certification:** October 23, 2026 (UTC)  

---

## 1. Acceptance Checklist & Gate Determination

| Gate Requirement | Specific Requirement Criteria | Audit Status | Evidence / Artifact |
| :--- | :--- | :---: | :--- |
| **1. License Provenance** | Document official CC BY-NC 4.0 license, non-commercial restriction, citation, and derived asset impact. | **MET** | `LICENSE_PROVENANCE_ERRATUM.md` |
| **2. Speaker Disjointness** | Zero speaker overlap between train, val, and test across all folds and regimes. | **MET** | `l2arctic_4fold_splits.json`, `speechocean762_official_splits.json` |
| **3. 4-Fold Constraints** | Exactly 16 train, 2 val, 6 test per fold; 3M/3F and 1 per L1 in test; 1M/1F in val; 8M/8F in train. | **MET** | `SPLIT_VALIDATION_REPORT.md` (Section 3) |
| **4. 4-Fold Speaker Coverage** | All 24 L2-ARCTIC speakers appear in exactly one test fold across the 4 folds. | **MET** | Verified via `audit_scripts/validate_splits.py` |
| **5. Vietnamese Regimes** | Regime A (4 rounds LOSO, 21 train, 2 val, 1 test) & Regime B (Zero-Shot, 18 train, 2 val, 4 test). | **MET** | `l2arctic_loso_vietnamese_splits.json` |
| **6. SpeechOcean762 Splits** | Preserve official 2,500 test set; carve out 25-speaker stratified val (500 utts); exclude 245 orphan WAVs. | **MET** | `speechocean762_official_splits.json` |
| **7. Conservation Laws** | Canonical tokens ($119,151$) and errors ($17,518$) invariant across all partitions. | **MET** | Verified across all folds and rounds |
| **8. Deterministic Reproduction** | Repeat-run deterministic equality proven using fixed master seed 42. | **MET** | Subprocess bit-for-bit SHA-256 match |
| **9. Drive Synchronization** | All generated manifests and reports synchronized to `kltn_gdrive:` and verified. | **MET** | Verified via rclone remote checksums |

### Formal Milestone Gate Status
$$\mathbf{RQ1\text{-}SPLIT\text{-}01: \ PASS}$$

---

## 2. Active Blockers Audit

There are **zero blocking defects** preventing the transition from Phase `RQ1-SPLIT-01` to Phase `RQ1-PREP-01`.

### Summary of Rectified Defects from Prior Gates
1. **Defect 1 (4-Fold Demographic Skew):** Rectified from the superseded 5M/1F and 1M/5F distribution to exact $3\text{M} / 3\text{F}$ parity across all four folds.
2. **Defect 2 (26-Speaker Vietnamese LOSO Contradiction):** Rectified from the invalid $23 + 2 + 1 = 26$ double-counting to strictly disjoint $21 / 2 / 1$ (Regime A) and $18 / 2 / 4$ (Regime B).
3. **Defect 3 (License Provenance Mismatch):** Rectified citation from CC BY 4.0 to official CC BY-NC 4.0 with comprehensive legal analysis.
4. **Defect 4 (SpeechOcean 245 Orphan WAV Contamination):** Explicitly excluded from supervised manifests; dataloaders restricted to manifest utterance IDs.

---

## 3. Handover Protocol to Phase RQ1-PREP-01

With all split manifests deterministically generated and frozen, Phase **RQ1-PREP-01 (Audio Standardization & Feature Pipeline)** may now be planned.

### Immediate Next Actions for Phase RQ1-PREP-01
1. **Action 1 — Audio Resampling Micro-Benchmark Gate:**
   - Execute standalone micro-benchmark (`benchmark_resampling_pipeline.py`) measuring Kaiser sinc resampling latency ($44.1\text{ kHz} \rightarrow 16\text{ kHz}$) and multi-worker PyTorch DataLoader throughput across 1,000 sample WAVs.
   - Verify whether on-the-fly resampling satisfies latency $< 5\text{ ms}$ and throughput $> 500\text{ utts/s}$ on host CPU, or whether offline pre-computation is required.
2. **Action 2 — PyTorch Dataset & DataLoader Implementation:**
   - Implement `L2ArcticDataset` and `SpeechOceanDataset` reading exclusively from `l2arctic_4fold_splits.json`, `l2arctic_loso_vietnamese_splits.json`, and `speechocean762_official_splits.json`.
   - Implement zero-leakage collation and batch padding for variable-length speech waveforms.
3. **Action 3 — Dynamic Audio Normalization:**
   - Implement DC offset removal and $[-1.0, +1.0]$ peak amplitude normalization.
4. **Action 4 — Acoustic Feature Validation:**
   - Validate input tensor representations for baseline Wav2Vec 2.0 fine-tuning and GOP feature extraction.

### Explicit Operational Boundary
In strict accordance with the execution protocol:
- **No model training, fine-tuning, hyperparameter sweeps, or audio feature caching are initiated in this phase.**
- Execution halts upon completion and verification of the split manifest deliverables.
