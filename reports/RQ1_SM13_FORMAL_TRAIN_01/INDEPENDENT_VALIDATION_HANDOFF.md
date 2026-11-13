# LANGFENS RQ1-SM13-FORMAL-TRAIN-01 — Independent Validation Handoff Specification

**From:** Principal Speech ML Engineer & Formal Training Lead  
**To:** Independent Scientific Validation Team & External Evaluation Gate  
**Campaign:** `LANGFENS-RQ1-SM13-FORMAL-TRAIN-01`  
**Winning Method:** `LANGFENS-SM13-FUSION-V1`  
**Evaluation Contract:** EVAL-V3.1 (Exact $N$ targets, $N+1$ gaps)  
**Primary Predetermined Model Artifact:** `Seed 101`  
**Handoff Date:** November 10, 2026  
**Status:** **READY FOR INDEPENDENT VALIDATION GATE — DO NOT UNLOCK PROTECTED SETS BEFORE AUTHORIZATION**  

---

## 1. Executive Purpose & Scope of Handoff

The formal training of `LANGFENS-SM13-FUSION-V1` on the complete L2-ARCTIC TRAIN partition (1,800 utterances, 12 speakers) is finished. All engineering and training criteria have passed.

In accordance with strict reproducibility protocols, **the training team is prohibited from evaluating held-out evaluation sets (`DEV_SEARCH`, `DEV_CAL`, and `TEST_SEALED`)**. This document establishes the formal handoff boundary, cataloging verified model artifacts, frozen operating policies, cryptographic checksums, and execution instructions for the independent validation gate.

---

## 2. Handoff Artifact Package Catalog

All delivered model files reside under `experiments/rq1_sm13_formal_train_01/`:

### 2.1 Primary Predetermined Production Model (Seed 101)
- **Target Head Checkpoint:** `checkpoints/target_sm13_seed101_fulltrain.pt`
  - *Parameters:* 633,361 (14 scalar mixing + 633,347 MLP)
  - *SHA256:* `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6`
- **Gap Head Checkpoint:** `checkpoints/gap_sm13_seed101_fulltrain.pt`
  - *Parameters:* 215,120 (14 scalar mixing + 215,106 MLP)
  - *SHA256:* `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98`
- **Retained CTC Branch Checkpoint:** `checkpoints/ctc_registered_seed101.pt`
  - *Parameters:* 33,067 (Linear 768 $\to$ 43)
  - *SHA256:* `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a`
- **Keyed In-Sample Predictions:** `predictions/predictions_sm13_seed101_fulltrain.npz`
  - *SHA256:* `dabdee231457275fc8be006465f6c4c78c833b38c0077ddf23e49cbd5adbb946`

### 2.2 Confirmatory Replication Models (Seeds 202 & 303)
- **Seed 202 Target Head:** `checkpoints/target_sm13_seed202_fulltrain.pt` (SHA256: `e237332cbf99f2f5...`)
- **Seed 202 Gap Head:** `checkpoints/gap_sm13_seed202_fulltrain.pt` (SHA256: `50a69aadafce42ee...`)
- **Seed 202 CTC Branch:** `checkpoints/ctc_registered_seed202.pt` (SHA256: `dd31d79119bddb7d...`)
- **Seed 303 Target Head:** `checkpoints/target_sm13_seed303_fulltrain.pt` (SHA256: `278a464dc51b363c...`)
- **Seed 303 Gap Head:** `checkpoints/gap_sm13_seed303_fulltrain.pt` (SHA256: `4ec235e713303186...`)
- **Seed 303 CTC Branch:** `checkpoints/ctc_registered_seed303.pt` (SHA256: `57c9a26a63c77bc6...`)

---

## 3. Operational Policy & Threshold Deployment

The independent validation team must evaluate the primary model using the frozen reference policy derived on TRAIN inner OOF splits:
- **Nominal Operating Budget:** $B = 0.20$ false warnings per utterance.
- **Budget Partition:**
  - Target error false warning budget: $0.160$ false alarms / utterance ($80\%$).
  - Gap insertion false warning budget: $0.040$ false alarms / utterance ($20\%$).
- **Reference Operating Thresholds:**
  - Target Warning Threshold ($\tau_T$): Trigger error if $\max(p_S, p_O) \ge \tau_T$.
  - Gap Warning Threshold ($\tau_G$): Trigger insertion if $p_I \ge \tau_G$.
  - Tie-breaking: SHA256 canonical hash descending order on event keys.
- **Strict Protocol Requirement:** **Zero threshold tuning on the validation partitions.** The validation run must be executed strictly once, blind, with the frozen thresholds.

---

## 4. Key Scientific Invariants to Audit During Validation

The independent validation team is requested to specifically audit:
1. **Macro-3 AP Generalization:**
   - On TRAIN cross-validation, SM13 achieved Macro-3 AP of **$0.327300 \pm 0.0083$** ($+7.61\%$ relative gain over the Phase-01 baseline $0.304154$).
   - Confirm whether the $+0.023$ AP improvement transfers to unseen speakers in `DEV_CAL` and `DEV_SEARCH`.
2. **Operational False Warning Rate (FWR) Control:**
   - On TRAIN cross-validation, nominal $B=0.20$ produced an achieved development FWR of **$0.235370$** ($+17.7\%$ overshoot).
   - Evaluate whether the achieved FWR on external partitions stays within an acceptable pedagogical envelope ($\text{FWR} \le 0.25$) or requires posthoc recalibration.
3. **Classwise Error Balance:**
   - Substitution AP (Dev Reference: $0.422551$).
   - Omission AP (Dev Reference: $0.383154$).
   - Insertion AP (Dev Reference: $0.176194$) and Insertion $P@560$ (Dev Reference: $0.262500$).
4. **Speaker Robustness:**
   - Cross-speaker variance across different non-native L1 accents (e.g. Vietnamese, Mandarin, Arabic, Spanish).
5. **Event Cardinality Parity:**
   - Confirm that the validation engine receives exactly $N$ targets and $N+1$ gaps per utterance without dropping or creating events.

---

## 5. Execution Environment & Reproducibility Receipt

To reproduce evaluation:
```bash
# 1. Activate isolated Python environment
source /home/khoa/Projects/New-Method-Research/.venv/bin/activate

# 2. Verify all artifact hashes against the manifest
python3 -c "
import json, hashlib
from pathlib import Path
manifest = json.loads(Path('reports/RQ1_SM13_FORMAL_TRAIN_01/FINAL_ARTIFACT_MANIFEST.json').read_text())
for p_str, entry in manifest.items():
    p = Path(p_str)
    assert p.exists(), f'Missing {p}'
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == entry['sha256'], f'Checksum mismatch on {p}'
print('All artifact hashes verified!')
"
```

---

## 6. Formal Validation Gate Acceptance Criteria

The independent validation gate shall issue one of four formal determinations:
1. **VALIDATED_PRODUCTION_READY:** Held-out Macro-3 AP $\ge 0.3150$, strictly superior to baseline, empirical $\text{FWR} \le 0.25$, and zero regressions on individual error classes.
2. **VALIDATED_CALIBRATION_REQUIRED:** Macro-3 AP superiority confirmed, but empirical FWR on held-out data deviates from nominal by $> 25\%$, requiring threshold recalibration on `DEV_CAL`.
3. **GENERALIZATION_DEGRADED:** Significant performance regression on held-out evaluation speakers compared to cross-validation development figures.
4. **REJECTED:** Contract, cardinality, or numerical integrity violations.

---

## 7. Final Training Team Directive

Formal training is **COMPLETE**. The model artifacts, training telemetry, and cryptographic manifests are sealed.

**STOP AT ARTIFACT READINESS.**  
**DO NOT UNLOCK PROTECTED EVALUATION DATASETS.**  
**DO NOT RUN BLIND TESTING PRIOR TO FORMAL VALIDATION AUTHORIZATION.**
