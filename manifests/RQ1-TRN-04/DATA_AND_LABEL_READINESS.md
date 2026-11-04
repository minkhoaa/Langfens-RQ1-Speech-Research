# LANGFENS RQ1-TRN-04 — Data & Label Readiness Audit

**Document ID:** `LANGFENS-RQ1-TRN-04-DATA-READINESS-01`  
**Experiment ID:** `RQ1-TRN-04`  
**Author:** Principal Speech ML Research Engineer & Reproducibility Auditor  
**Date:** 2026-11-04  
**Status:** **AUDITED, FROZEN & CERTIFIED**  

---

## 1. Cryptographic Archive Verification

In compliance with repository `AGENTS.md`, all training and evaluation data originate strictly from frozen, immutable archives with SHA-256 integrity verification:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             CRYPTOGRAPHIC DATASET VERIFICATION                                        │
├──────────────────────┬──────────────────────────────────────────────────────────────────┬─────────────┬───────────────┤
│ Asset Name           │ Expected SHA-256 Checksum                                        │ Size (bytes)│ Verification  │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼─────────────┼───────────────┤
│ Split Manifest       │ 7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b │ 4,792       │ PASSED        │
│ Authorized Train/Val │ b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9 │ 761,922,918 │ PASSED        │
│ Contract SHA-256     │ 4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2 │ 64          │ PASSED        │
└──────────────────────┴──────────────────────────────────────────────────────────────────┴─────────────┴───────────────┘
```

---

## 2. Speaker Partition Census & Mutual Isolation

All 24 L2-ARCTIC speakers are partitioned strictly without overlap ($S_i \cap S_j = \emptyset$):

| Partition Role | Speaker Count | Speaker IDs | Utterances | Purpose | Access Invariant |
|---|:---:|---|:---:|---|---|
| **FIT** | 12 | `BWC, ERMS, HKK, HQTV, SKA, SVBI, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA` | 1,799 | Parameter optimization / gradient descent | Train-only |
| **SELECT** | 2 | `MBMPS, TLV` | 300 | Epoch AP tracking & checkpoint selection | Val-during-training |
| **CAL** | 2 | `NCC, THV` | 300 | Operational threshold selection | Calibration-only |
| **VAL** | 2 | `NJS, RRBI` | 300 | Final candidate evaluation under Protocol Z | Unseen evaluation |
| **TEST** | 6 | `ABA, ASI, EBVS, HJK, LXC, PNV` | 900 | Generalization benchmark (future sealed test) | **SEALED & PROHIBITED** |

### Test Partition Sealing Verification
1. `authorized_RQ1-TRN-02A_fold1_train_val_only.zip` contains exactly 18 speaker packages. Zero audio or TextGrid files from `ABA, ASI, EBVS, HJK, LXC, PNV` exist within the archive.
2. Dataset loader raises `SealedSpeakerContaminationError` fail-closed if any test speaker ID is queried.
3. Automated regression tests (`test_gate0_6_speaker_disjoint_partitions_and_sealed_test`) certified 100% test sealing.

---

## 3. Label Accounting & Evaluation Denominators

### 3.1 Token Distribution on Fold-1 VAL (Protocol Z)
Evaluation is performed across the complete Fold-1 VAL cohort (300 utterances across speakers NJS and RRBI):

| Population Tier | Total Canonical Tokens | Correct Tokens ($y=0$) | Substitution Tokens ($y=1$) | Omission Tokens ($y=2$) |
|---|---:|---:|---:|---:|
| **Gold Canonical Census** | **9,956** | 8,799 | 1,012 | **145** |
| **Protocol Z Aligned** | **9,919** | 8,774 | 1,008 | **137** |
| **Unmapped / Abstentions** | **37** | 25 | 4 | **8** |

### 3.2 Evaluation Accounting Invariants
- **All-Gold Omission Recall Denominator:** Stated against the full gold census ($N_{\text{pos}} = 145$). All 8 unmapped gold omissions count strictly as False Negatives ($\text{FN}$).
- **Aligned Omission Recall Denominator:** Stated against aligned tokens ($N_{\text{pos}} = 137$).
- **Deployed Omission Precision:** Evaluated over all emitted alarms:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}_{\text{cor}} + \text{FP}_{\text{sub}}}$$
  where $\text{FP}_{\text{cor}}$ are alarms on correct phones ($y=0$) and $\text{FP}_{\text{sub}}$ are alarms on substitutions ($y=1$).
- **False Alarm Normalization:** Evaluated as $(\text{FP}_{\text{cor}} + \text{FP}_{\text{sub}}) / 9.956$ gold canonical thousand-phone units.
- **Dedicated Omission Population:** Evaluated exclusively on correct ($y=0$) and omission ($y=2$) tokens ($N = 8,944$ gold, $N = 8,911$ aligned). Substitutions ($y=1$) are strictly excluded from the Dedicated Omission PR curve to prevent label confusion.
