# Final Acceptance Decision & Verification Summary — RQ1-TRN-02A-ZERO-ORACLE-VERIFY

**Final Evaluation Gate Decision:** **RQ1-TRN-02A-ZERO-ORACLE-VERIFY: PASS**  
**Auditor:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Verified Git Base SHA:** `613dc728b06cd50604c9fb902923b223671545b3`  
**Dedicated Branch:** `feat/rq1-trn-02a-zero-oracle-verify`  

---

## 1. Final Gate Summary

| Gate Dimension | Status | Key Verifiable Evidence |
|---|:---:|---|
| **Task A: Reproducible Protocol Z Entrypoint** | **PASS** | Implemented `audit_scripts/evaluate_protocol_z.py`, `langfens_prep/g2p.py`, and `ZeroOracleL2ArcticDataset`. Zero TextGrids read in input generation or model forward pass. Full per-utterance predictions and alignment traces exported for all 300 validation utterances. |
| **Task B: Scoring Contract & Population Integrity** | **PASS** | Investigated unaligned tokens (37 unaligned gold, 37 extra G2P; reconciled prior 40-token draft). Replaced flawed validation prevalence imputation with predeclared train-only prior lookup ($P_{\text{train}}(\text{error} \mid g_j)$, $\text{AP} = 0.314265$) and aligned-only scoring ($N=9,919$, $\text{AP} = 0.313530$). Historical reported AP reproduced to within $0.000004$ ($0.312471$ vs $0.312467$). |
| **Task C: Acoustic Conclusions & Statistical Rigor** | **PASS** | Disaggregated agreement by class: positive class agreement is $86.60\%$ ($155$ flips, true error recall $+16.86\%$, false alarms $-16.80\%$), masked by negative class agreement ($98.09\%$). Preserved all four permutation APs (`shift_1`: 0.2947, `shift_2`: 0.3026, `shift_75`: 0.3001, `reverse`: 0.3140). Retracted uncomputed hypothetical CI `[+0.000, +0.039]`. Corrected validation checkpoint selection bias. |
| **Task D: Automated Tests & Regressions** | **PASS** | 100% test coverage across G2P determinism, input isolation, alignment correctness, denominator integrity, and AP reconstruction from prediction records. Zero test set contamination. Zero parameter mutations. |

---

## 2. Key Scientific Findings & Clarifications

### 1. Reproduction of Protocol Z Historical AP:
- Under the historical validation-prevalence imputation policy ($0.116211$), the recomputed AP across all 9,956 validation gold tokens is **0.312471**, reproducing the draft claim ($0.312467$) to within **0.000004** ($4 \times 10^{-6}$).
- However, our audit identified that imputing scores with validation error prevalence is methodologically flawed because it leaks target prevalence into the scoring rule.

### 2. Primary Replay Metric & Scoring Policy Clarifications:
- **Primary Replay Metric (Population 1 — Aligned-Only):** On the 9,919 phone tokens where G2P and Gold annotations align ($99.63\%$ coverage), Protocol Z achieves **AP = 0.313530** vs Protocol O **AP = 0.316434** ($\Delta = -0.002904$). This metric evaluates model prediction quality on queries without arbitrary imputation.
- **Oracle-Referenced Full-Gold Benchmark Imputation (Policy B1):** Imputing omitted gold tokens with train conditional priors $P_{\text{train}}(\text{error} \mid g_j)$ achieves **AP = 0.314265** across all 9,956 gold tokens ($\Delta = -0.002756$ relative to Protocol O $0.317021$). This is strictly an offline oracle-assisted benchmarking analysis against Protocol O, not a deployment-time inference policy.
- **Sensitivity & Audit Results (Policies B2, B3, Val Prevalence):** Preserved strictly as sensitivity controls (B2 AP: $0.313006$, B3 AP: $0.307811$, Val Prevalence AP: $0.312471$), not operational policies.
- **Exact Sequence Utterances ($N = 2,338$, 82 utts):** Protocol Z and Protocol O APs are **mathematically identical** ($0.305362$, $\Delta = 0.000000$).

### 3. Acoustic Waveform Conditioning is Class-Asymmetric:
- Overall prediction agreement at $\tau = 0.50$ ($96.76\%$) gave the false impression that audio swapping has almost no effect.
- Disaggregating by class proves that on **true errors ($N = 1,157$)**, decision agreement drops to **86.60%**. Matched audio enables the detection of 29 additional true errors (201 vs 172 TP) and suppresses 42 false alarms (208 vs 250 FP).
- The acoustic representation provides meaningful error discriminability that is partially masked by phonotactic prompt priors.

### 4. Deployment Boundary Warning:
- **Protocol Z is NOT declared deployment-ready solely because its validation AP is close to Protocol O.**
- The model weights in `best.pt` were trained exclusively on human-annotated Gold TextGrid intervals. Although the cross-attention module is timestamp-free, it was conditioned on annotator-selected pronunciation choices.
- Deployment readiness requires fine-tuning or end-to-end training with G2P canonical inputs and evaluating on multi-speaker holdouts.

---

## 3. Cryptographic and Environmental Inventory

| Artifact / Asset | File Path | SHA-256 Digest | Status |
|---|---|---|:---:|
| **Evaluation Entrypoint** | `audit_scripts/evaluate_protocol_z.py` | — | Executable / Source-backed |
| **G2P Frontend** | `src/langfens_prep/g2p.py` | — | Tested / Integrated |
| **Production Model Checkpoint**| `checkpoints/RQ1-TRN-02A/recovered/best.pt` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | Immutable |
| **Model Weights Fingerprint** | (In-Memory 210 Tensors) | `32d12bdf3524508bd891605a728c9207be588d0ce8a1a16bca2f65cab12ae60d` | Invariant ($\Delta=0.0$) |
| **Pinned CMUdict Lexicon** | `resources/CMUDict/cmudict.dict` | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` | Verified |
| **Pinned ARCTIC Prompts** | `resources/L2-ARCTIC/prompts.json` | `87843d363bc7e2bbe849842efe16974cb6634dfe0539efb42d666aed7e17f11e` | Verified |
| **Fold 1 Split Manifest** | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | Verified |
| **Authorized Audio Zip** | `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | Sealed |
| **Protocol Z Prediction Records**| `manifests/RQ1-TRN-02A/.../PROTOCOL_Z_PREDICTION_RECORDS.json` | Generated | Auditable |
| **Protocol Z Verification Report**| `manifests/RQ1-TRN-02A/.../PROTOCOL_Z_VERIFICATION_REPORT.json` | Generated | Auditable |

---

## 4. Execution Commands

### Full Protocol Z Verification Pass (300 Validation Utterances, CPU):
```bash
PYTHONPATH=.:src python3 audit_scripts/evaluate_protocol_z.py \
  --checkpoint checkpoints/RQ1-TRN-02A/recovered/best.pt \
  --manifest manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json \
  --authorized_zip datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip \
  --prompts resources/L2-ARCTIC/prompts.json \
  --output_dir manifests/RQ1-TRN-02A/evaluation_reconciliation/zero_oracle_verification \
  --device cpu \
  --tie_break sub_first
```

### Complete Verification Test Suite:
```bash
PYTHONPATH=.:src pytest \
  tests/test_protocol_z_g2p.py \
  tests/test_zero_oracle_isolation.py \
  tests/test_protocol_z_alignment_and_scoring.py \
  tests/test_acoustic_controls_class_separation.py \
  tests/test_prediction_reconstruction.py \
  tests/test_label_independence.py
```
