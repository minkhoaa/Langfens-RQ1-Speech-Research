# LANGFENS — RQ1-TRN-01B0-R: Substitution Diagnosis Metric Audit & Denominator Reconciliation

**Gate Reference:** RQ1-TRN-01B0-R — Real-Data Learning Diagnostics & Evidence Reconciliation  
**Deliverable Path:** `manifests/RQ1-TRN-01B0/reconciliation/SUBSTITUTION_DIAGNOSIS_METRIC_AUDIT.md`  
**Lead Research Data Engineer & Speech ML Reviewer:** Senior Speech ML Research Engineer  
**Audit Status:** `DIAGNOSIS_METRIC_AUDIT: PASS (Corrected Semantics & Denominators Enforced)`  

---

## 1. Executive Summary & Defect Statement

In speech Mispronunciation Detection & Diagnosis (MDD), a fundamental distinction exists between:
1. **Mispronunciation Detection (Binary MDD):** Classifying whether a canonical phoneme in prompt text was produced correctly ($y = 0$) or erroneously ($y = 1$).
2. **Mispronunciation Diagnosis (Multi-Class MDD):** Given an erroneous phoneme, identifying the specific erroneous realized phone substitute produced by the non-native speaker.
3. **General Realized-Phone Classification (Auxiliary Acoustic Recognition):** Classifying the realized acoustic sound for all prompt phonemes (both correct productions and substitutions).

### The Critical Defect in Prior Gate RQ1-TRN-01B0:
- `src/langfens_prep/textgrid_parser.py` line 284 assigned `diagnosis_target = base_phone` to **correct productions** as well as substitutions.
- `compute_metrics()` in `audit_scripts/run_training_sanity.py` evaluated diagnosis accuracy over all positions where `target_diagnosis != -100`.
- The evaluation denominator was reported as **1,000 tokens**.
- However, in the 32-utterance training subset, there were **only 126 substitutions** (124 eligible, 2 uncertain `err` tags) and **876 correct productions**.
- Consequently, 87.6% of the evaluated tokens in the "diagnosis" denominator were **correct phonemes**. The reported metric measured realized phone classification dominated by correct tokens, concealing that true substitution diagnosis accuracy was vastly different.
- In the original checkpoint, the reported "8.0% diagnosis accuracy" was driven by 76 correct predictions on correct phones and only **4 correct predictions on actual substitutions** (a meager $4 / 124 = 3.23\%$).

---

## 2. Rigorous Denominator and Token Accounting

Across the fixed 32-utterance training subset (`TRAIN_ONLY_SUBSET_MANIFEST.json`):

| Phonetic Category | Raw Count | Strict Masking Policy | Target Diagnosis Value | Included in Substitution Diagnosis? | Included in Auxiliary Classification? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Correct Productions** | **876** | Masked | `ignore_index = -100` | **NO (Masked)** | **YES** |
| **Eligible Substitutions** | **124** | Evaluated | Standard 39-ARPAbet Class ($0 \dots 38$) | **YES (Primary)** | **YES** |
| **Uncertain Substitutions (`err`)** | **2** | Masked | `ignore_index = -100` | **NO (Masked)** | **NO (Masked)** |
| **Acoustic Deletions** | **52** | Masked | `ignore_index = -100` | **NO (Masked)** | **NO (Masked)** |
| **Sequence Padding ($T_{\text{pad}}$)** | Variable | Masked | `ignore_index = -100` | **NO (Masked)** | **NO (Masked)** |
| **Total Canonical Tokens** | **1,054** | — | — | **Denominator = 124** | **Denominator = 1,000** |

### Exact Interval Breakdown for Masked Uncertain Substitutions:
The two substitutions masked due to `err` tags (lacking definitive acoustic phone labels in TAMU gold annotations) are:
1. `BWC/arctic_a0571`: Interval label `"T, err, s"`, canonical phone `T`, realized phone `<ERR>`, `has_err_tag = True`.
2. `YDCK/arctic_a0042`: Interval label `"V, err, s"`, canonical phone `V`, realized phone `<ERR>`, `has_err_tag = True`.

Both intervals are strictly assigned `diagnosis_target = None` and `target_diagnosis = -100`.

---

## 3. Disentanglement of Objectives: Substitution Diagnosis vs. Auxiliary Classification

To eliminate scientific ambiguity, two distinct metrics are formally defined and programmatically computed:

### 3.1 Primary Metric: Substitution-Only Diagnosis Top-1 Accuracy
Evaluates the model's diagnostic capacity strictly on eligible substitution intervals:
$$\text{Acc}_{\text{diag\_sub}} = \frac{1}{|S_{\text{sub\_eligible}}|} \sum_{i \in S_{\text{sub\_eligible}}} \mathbb{I}\left( \hat{y}_i = y_i^* \right)$$
where:
- $S_{\text{sub\_eligible}} = \{ i \mid \text{error\_type}_i = \text{"substitution"} \land \neg \text{has\_err}_i \land y_i^* \in \text{ARPAbet}_{39} \}$
- Exact denominator on the 32 training utterances: **$N = 124$**.

### 3.2 Auxiliary Metric: Realized-Phone Classification Accuracy
Measures framewise acoustic realization classification across all non-deleted canonical phonemes:
$$\text{Acc}_{\text{aux\_phone}} = \frac{1}{|S_{\text{aux\_phone}}|} \sum_{i \in S_{\text{aux\_phone}}} \mathbb{I}\left( \hat{y}_i = y_i^* \right)$$
where:
- $S_{\text{aux\_phone}} = \{ i \mid \text{error\_type}_i \in \{\text{"correct"}, \text{"substitution"}\} \land \neg \text{has\_err}_i \}$
- Exact denominator on the 32 training utterances: **$N = 876 + 124 = 1,000$**.

---

## 4. Empirical Comparison: Baseline Checkpoint vs. Reconciled Repeat Run

Evaluating the baseline checkpoint (`rq1_trn_01b0_checkpoint_step_100.pt`, which was supervised on all 1,000 mixed tokens) versus the corrected repeat experiment (`rq1_trn_01b0_r_repeat_step_100.pt`, supervised strictly on the 124 substitution targets) reveals the true scientific impact:

| Metric | Pre-Training (Step 0) | Baseline Checkpoint (Step 100, Supervised on 1,000 Mixed) | Repeat Checkpoint (Step 100, Supervised Strictly on 124 Substitutions) | Scientific Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Substitution Diagnosis Accuracy** ($N=124$) | $9.68\%$ ($12 / 124$) | **$3.23\%$ ($4 / 124$)** | **$16.13\%$ ($20 / 124$)** | Baseline collapsed because 87.6% correct tokens biased head towards canonical priors. Strict supervision quadrupled substitution accuracy ($3.2\% \to 16.1\%$). |
| **Correct Phone Classification Accuracy** ($N=876$) | $1.83\%$ ($16 / 876$) | **$8.68\%$ ($76 / 876$)** | $4.57\%$ ($40 / 876$) | Baseline primarily learned to predict canonical phones for correct productions. |
| **Auxiliary Realized Classification** ($N=1000$) | $2.80\%$ ($28 / 1000$) | **$8.00\%$ ($80 / 1000$)** | $6.00\%$ ($60 / 1000$) | The previously reported 8.0% was 95% ($76/80$) correct phones. |
| **Diagnosis Cross-Entropy Loss** | $3.6336$ | $3.3913$ | $3.0961$ | Loss on true substitutions fell by $14.79\%$ under targeted supervision. |

---

## 5. Standard 39-ARPAbet Class Label Mapping Verification

The 39 phoneme classes are 0-indexed according to `STANDARD_39_ARPABET`:
`['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']`

- **Class Coverage in 124 Substitutions:** 28 distinct ARPAbet classes appear as substitution targets in the 32-utterance subset.
- **Top Substitution Targets:** `D` (18), `Z` (15), `S` (14), `T` (11), `IH` (10), `V` (8), `B` (6).
- **Zero corrupted or non-standard indices:** Verified by `test_dataset_collation_target_diagnosis_masking` ($100\%$ valid indices in range $[0, 38]$ or $-100$).

---

## 6. Regression Testing and Verification Evidence

The metric repairs are safeguarded by dedicated tests in `tests/test_metric_reconciliation.py`:
1. `test_training_subset_exact_phone_census_and_denominators`: Proves exact denominators ($N_{\text{sub}} = 124$, $N_{\text{err}} = 2$, $N_{\text{cor}} = 876$, $N_{\text{del}} = 52$, $N_{\text{total}} = 1,054$).
2. `test_textgrid_parser_masking_policy`: Asserts that `parse_gold_textgrid_content()` sets `diagnosis_target = None` for correct productions, deletions, and `err` tags.
3. `test_dataset_collation_target_diagnosis_masking`: Asserts that batched collation produces exactly 124 active positions in `target_diagnosis` and 1,000 active positions in `target_aux_phone`.
4. `test_existing_checkpoint_binary_and_diagnosis_metrics`: Asserts that re-evaluating the historical checkpoint yields exactly $3.23\%$ substitution accuracy on $N=124$, disproving the 1,000-token claim.

---

## 7. Audit Conclusion

$$\mathbf{DIAGNOSIS\_METRIC\_AUDIT: PASS}$$

**Formal Policy:** Moving forward to Fold 1 training, substitution diagnosis accuracy must be reported exclusively over eligible substitution intervals ($N = 124$ in the train subset). General realized-phone classification over all prompt positions must be labeled as an auxiliary acoustic objective. Presenting mixed accuracy as substitution diagnosis accuracy is strictly prohibited.
