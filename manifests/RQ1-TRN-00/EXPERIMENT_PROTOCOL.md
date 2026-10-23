# LANGFENS — RQ1-TRN-00: Experimental Protocol & Benchmarking Freeze

**Execution Gate:** RQ1-TRN-00 — Modeling Readiness & Reproducibility Preflight  
**Focus Area:** Task F — Protocol Freeze across 4-Fold, LOSO, Zero-Shot, Metrics, Calibration & Schema  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Reproducibility Lead  
**Deliverable Associated Path:** `manifests/RQ1-TRN-00/EXPERIMENT_PROTOCOL.md`  
**Synced Remote Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-00/`  
**Gate Evaluation Status:** `PROTOCOL_FROZEN: PASS` (Specification Complete; No Model Execution Authorized)  

---

## 1. Executive Summary & Protocol Freeze Mandate

To maintain strict scientific reproducibility and eliminate post-hoc metric hacking, this document formally freezes all experimental protocols, evaluation partitions, metric policies, threshold calibration rules, and configuration schemas for Project Langfens.

**Binding Boundary:** No acoustic model training, fine-tuning, inference, GOP scoring, or evaluation may commence until this protocol is ratified.

---

## 2. Experimental Regimes & Partition Protocols

All experiments must strictly consume the frozen split manifests located in `manifests/RQ1-SPLIT-01/`. Dynamic re-partitioning or cross-fold blending is strictly forbidden.

```text
Project Langfens Evaluation Partitions
├── 1. Primary Benchmark: 4-Fold Balanced Speaker-Independent Cross-Validation
│   ├── Target: L2-ARCTIC v5.0 (24 speakers across 6 L1s)
│   ├── Folds: 4 mutually exclusive folds (6 speakers per test fold, balanced by sex and L1)
│   └── Metric Aggregation: 4-fold Micro- and Macro-averaged PR-AUC, F1, Recall, Precision
│
├── 2. Vietnamese Specialization: LOSO Regime A (In-Domain Adaptation)
│   ├── Target: L2-ARCTIC Vietnamese cohort (HQTV, PNV, THV, TLV)
│   ├── Partition: 4 leave-one-Vietnamese-speaker-out rounds
│   └── Training: 23 speakers (including 3 Vietnamese); Test: Held-out Vietnamese speaker
│
├── 3. Vietnamese Specialization: Zero-Shot Regime B (Cross-Lingual Generalization)
│   ├── Target: L2-ARCTIC Vietnamese cohort (HQTV, PNV, THV, TLV)
│   ├── Training: Strictly 20 non-Vietnamese speakers (Arabic, Chinese, Hindi, Korean, Spanish)
│   └── Test: Zero-shot transfer evaluated on all 4 Vietnamese speakers
│
└── 4. Multi-Aspect Pronunciation Scoring Benchmark
    ├── Target: SpeechOcean762 (5,000 manifested utterances)
    └── Partition: Official Kaldi splits (Train: 2,000, Val: 500, Test: 2,500)
```

---

## 3. Metric Policies & Error Taxonomy Accounting

### 3.1 Primary Metric Policy: Strict Binary MDD
- **Target Invariant Denominator:**
  $$N_{\text{canonical}} = C + S + D = 101,633 + 14,098 + 3,420 = \mathbf{119,151} \quad (\text{Strict Invariance})$$
- **Positive Class (Errors):**
  $$E_{\text{strict}} = S + D = 14,098 + 3,420 = \mathbf{17,518} \quad (14.702\% \text{ error rate})$$
- **Negative Class (Correct):**
  $$C_{\text{strict}} = \mathbf{101,633} \quad (85.298\%)$$
- **Primary Metric:** **Precision-Recall Area Under Curve (PR-AUC)**.
  Because of the $5.80 : 1$ class imbalance, PR-AUC evaluates ranking discrimination across all thresholds without arbitrary operating cutoffs.
- **Secondary Metrics:** Precision, Recall, and F1-Score at operating threshold $\theta^*$.
- **Strictly Prohibited Metric:** Unweighted Accuracy ($\frac{TP+TN}{N}$). Predicting all-correct achieves $85.30\%$ accuracy while detecting 0 errors.

### 3.2 Sensitivity Analysis Policy: Lenient MDD
- In Second Language Acquisition (SLA), base-preserving allophonic variations (e.g. slight aspiration differences tagged `CPL,CPL*,s`; $1,112$ tokens) are often considered pedagogical non-errors.
- **Lenient Reclassification:**
  $$C_{\text{lenient}} = 101,633 + 1,112 = \mathbf{102,745} \quad (86.231\%)$$
  $$E_{\text{lenient}} = (14,098 - 1,112) + 3,420 = \mathbf{16,406} \quad (13.769\%)$$
- **Reporting Requirement:** Lenient results must be reported as a sensitivity delta ($\Delta \text{PR-AUC}$, $\Delta \text{F1}$) alongside strict metrics.

### 3.3 Disaggregated Error-Type Evaluation
Every published experiment must report disaggregated performance across the three phonological error categories:
1. **Phoneme Substitution Diagnosis Accuracy ($\text{Diag-Acc}$):**
   Evaluated strictly over the $S_{\text{eval}} = 13,829$ clean substitutions (masking the $258$ unalignable `'err'` tokens):
   $$\text{Diag-Acc} = \frac{\sum_{i=1}^{S_{\text{eval}}} \mathbb{I}(\hat{p}_i = p_i)}{S_{\text{eval}}}$$
2. **Corpus Deletion Detection Recall ($\text{Rec}_{\text{del}}$):**
   $$\text{Rec}_{\text{del}} = \frac{TP_{\text{del}}}{3,420}$$
3. **Mandatory Vietnamese Deletion Recall ($\text{Rec}_{\text{del, Viet}}$):**
   Evaluated specifically on the Vietnamese speakers (`HQTV`, `PNV`, `THV`, `TLV`):
   $$\text{Rec}_{\text{del, Viet}} = \frac{TP_{\text{del, Viet}}}{1,566} \quad (\mathbf{45.79\%} \text{ of all corpus deletions})$$
4. **Epenthetic Insertion Event Matching:**
   Evaluated via the $50\text{ ms}$ tolerance window matching protocol on the $1,092$ insertion events:
   $$\text{FAR}_{\text{ins}} = \frac{FP_{\text{ins}}}{\text{Total Audio Hours}} \quad (\approx 3.8 \text{ hours})$$

---

## 4. Validation-Only Operating Threshold Calibration

To guarantee that test results remain unbiased and free of data snooping:

1. **Threshold Optimization Rule:** The operating threshold $\theta^* \in [0, 1]$ used to convert model posterior probabilities $P(\text{Error} \mid x)$ into binary decisions $\hat{b}_i \in \{0, 1\}$ must be calibrated **strictly on the validation split**:
   $$\theta^* = \arg\max_{\theta \in [0.01, 0.99]} \text{F1}_{\text{val}}(\theta)$$
2. **Zero Test-Label Contamination:** Test split ground truth labels must remain completely blinded until after $\theta^*$ is frozen and test predictions are submitted.
3. **Threshold Calibration Reporting:** The chosen $\theta^*$ and its validation F1 must be logged alongside the resulting test F1.

---

## 5. Statistical Significance & Speaker-Clustered Confidence Intervals

To ensure reported metric differences represent genuine scientific progress rather than sampling artifacts:

1. **Speaker-Clustered Bootstrap Resampling:**
   - Resampling unit: Entire speech utterance (preserving intra-utterance phonetic correlation).
   - Resampling iterations: $B = 1,000$ bootstrap replicates.
   - Clustering: Stratified resampling clustered by speaker.
2. **$95\%$ Non-Parametric Confidence Intervals:**
   $$\text{CI}_{95\%} = \left[ q_{0.025}, \, q_{0.975} \right]$$
3. **Dual Aggregation Reporting:**
   - **Micro-Average:** Pooled across all phone instances in the test set.
   - **Macro-Average:** Computed independently per speaker and per native language (L1), then averaged with equal weight.

---

## 6. Experiment Configuration Schema (YAML / JSON)

All training runs in Project Langfens must be instantiated via a validated declarative configuration file conforming to the schema below:

```yaml
# config/experiments/exp_template.yaml
experiment_id: "EXP_RQ1_W2V2_BASE_01"
description: "Wav2Vec2-Base 4-Fold Phoneme MDD with Multi-Task Loss"
random_seed: 42
cudnn_deterministic: true

dataset:
  manifest_path: "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
  target_sr: 16000
  staging_dir: "/tmp/langfens_audio_stage"
  policy: "strict" # "strict" or "lenient"

model:
  backbone_name: "facebook/wav2vec2-base-960h"
  freeze_feature_encoder: true
  gradient_checkpointing: true
  hidden_dropout: 0.1
  attention_dropout: 0.1
  num_diagnosis_classes: 39

training:
  protocol: "4fold_cross_validation" # or "loso_vietnamese", "zero_shot_vietnamese"
  current_fold: 1
  epochs: 25
  batch_size_per_device: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  lr_scheduler: "cosine_with_warmup"
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  mixed_precision: "fp16"

loss_weights:
  ctc_loss_weight: 1.0
  binary_loss_weight: 1.0
  diagnosis_loss_weight: 0.5

evaluation:
  eval_interval_epochs: 1
  calibration_metric: "f1"
  bootstrap_replicates: 1000
```

---

## 7. Checkpoint Naming & Artifact Persistence Convention

Model checkpoints and evaluation results must adhere to the standardized directory structure:

```text
artifacts/
└── {experiment_id}/
    ├── config.yaml                          # Frozen snapshot of execution config
    ├── metrics.jsonl                        # Epoch-by-epoch structured metrics stream
    ├── fold_1/
    │   ├── checkpoint_best_val_prauc.pt     # Checkpoint achieving highest validation PR-AUC
    │   ├── checkpoint_best_val_f1.pt        # Checkpoint achieving highest validation F1
    │   ├── checkpoint_epoch_025.pt          # Final epoch checkpoint
    │   └── test_evaluation_summary.json     # Blinded test split evaluation outputs & CIs
    ├── fold_2/
    ├── fold_3/
    └── fold_4/
```

### Checkpoint Payload Schema:
Every `.pt` checkpoint file must contain:
1. `model_state_dict`: Model weights,
2. `optimizer_state_dict`: Optimizer momentum and variance states,
3. `scaler_state_dict`: GradScaler scaling state (for AMP),
4. `epoch`: Integer epoch index,
5. `val_prauc`: Validation PR-AUC at checkpoint creation,
6. `calibrated_threshold`: Optimal threshold $\theta^*$ derived on validation split,
7. `git_commit_sha`: Git commit hash at run initialization (`47b2955`).

---

## 8. Licensing Compliance & Legal Governance

1. **L2-ARCTIC Release v5.0:**
   - License: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
   - Compliance: All models, embeddings, and experimental findings generated in Project Langfens are strictly reserved for non-commercial academic research. Commercial redistribution or model monetization without rights-holder authorization is prohibited.
2. **SpeechOcean762:**
   - License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
   - Compliance: Appropriate attribution to Zhang et al. (2021) [P05] is maintained across all documentation and reports.

**Phase Gate Evaluation:** **`PROTOCOL_FROZEN: PASS`**
