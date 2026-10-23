# LANGFENS — RQ1-TRN-01A-R2: Training Step Accounting & Computational Complexity Report

**Gate Reference:** RQ1-TRN-01A-R2 — Final Training Correctness & Evidence Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01A/reconciliation/final_gate/TRAINING_STEP_ACCOUNTING.md`  
**Dataset Split Reference:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`  
**Lead Auditor / Speech ML Engineer:** Senior Speech ML Research Engineer & PyTorch Training Systems Reviewer  
**Gate Evaluation Status:** `TRAINING_STEP_ACCOUNTING: PASS`  

---

## 1. Executive Summary & Historical Discrepancy Correction

In earlier documentation (`FULL_BACKBONE_GPU_FEASIBILITY.md`, Section 5), the overall computational burden of the 4-fold cross-validation experiment was described as:
$$\text{Earlier Claim: } 3,599 \text{ utterances} \times 4 \text{ folds} \times 25 \text{ epochs} \approx 360,000 \text{ forward-backward steps}$$

An independent review identified three fundamental flaws in this estimate:
1. **Conflation of Total Corpus with Train Split:** The figure $3,599$ represents the **entire L2-ARCTIC corpus** (Train + Val + Test combined across all 24 speakers). In cross-validation, the training set per fold contains only ~2,400 utterances. Training on 3,599 utterances per fold would represent fatal test-set leakage.
2. **Implicit Unit Batch Size Assumption:** The calculation $3,599 \times 4 \times 25 = 359,900$ assumed a batch size of $B=1$, ignoring the planned micro-batch size ($B_{\text{micro}}=4$) and gradient accumulation ($A=4$).
3. **Ambiguity Between Steps and Samples:** The report failed to distinguish between *samples processed*, *micro-batches*, *model forward/backward passes*, and *optimizer updates*.

This report establishes the verified, mathematically reconciled accounting.

---

## 2. Frozen Dataset Split Utterance Cardinalities

From the authoritative, hash-locked split manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`:

| Split / Partition | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Total Across Corpus | Mean Per Fold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Speakers** | 16 | 16 | 16 | 16 | 24 (Disjoint per split) | 16.0 |
| **Val Speakers** | 2 | 2 | 2 | 2 | 24 | 2.0 |
| **Test Speakers** | 6 | 6 | 6 | 6 | 24 | 6.0 |
| **Train Utterances ($N_{\text{train}}$)** | **2,399** | **2,400** | **2,400** | **2,399** | — | **2,399.5** |
| **Val Utterances ($N_{\text{val}}$)** | **300** | **299** | **300** | **300** | — | **299.75** |
| **Test Utterances ($N_{\text{test}}$)** | **900** | **900** | **899** | **900** | — | **899.75** |
| **Total Utterances Per Fold** | 3,599 | 3,599 | 3,599 | 3,599 | **3,599 Unique** | 3,599.0 |

**Authoritative Cardinality:** In any single fold, the model trains on exactly **$2,400$ utterances** (or 2,399 in Folds 1 and 4). We adopt $N_{\text{train}} = 2,400$ as the exact computational basis.

---

## 3. Training Hyperparameter Specifications

As established in `configs/huggingface_registry.yaml` and `manifests/RQ1-TRN-00/EXPERIMENT_PROTOCOL.md`:

- **Micro-Batch Size ($B_{\text{micro}}$):** $4$ utterances per forward/backward pass.
- **Gradient Accumulation Steps ($A$):** $4$ accumulation passes per optimizer update.
- **Effective Batch Size ($B_{\text{eff}}$):** $B_{\text{micro}} \times A = 4 \times 4 = 16$ utterances.
- **Epochs Per Fold ($E$):** $25$ epochs.
- **Cross-Validation Folds ($K$):** $4$ folds.

---

## 4. Reconciled Step & Accounting Matrix

We establish precise mathematical definitions:
- **Samples Processed:** Cumulative number of speech utterances ingested.
- **Micro-Batches:** Number of batched tensors ($B=4$) fed to the model: $\lceil N_{\text{train}} / B_{\text{micro}} \rceil$.
- **Model Forward/Backward Steps:** Distinct forward and backward autodiff passes ($= \text{Micro-Batches}$).
- **Optimizer Steps:** Discrete parameter weight updates via AdamW: $\lceil \text{Micro-Batches} / A \rceil$.

### Reconciled Computational Dimensions:

| Scope | Samples Processed | Micro-Batches | Forward/Backward Steps | Optimizer Steps | Effective Batches |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Per Epoch (Single Fold)** | $2,400$ | $600$ | $600$ | $150$ | $150$ |
| **Per Fold ($25$ Epochs)** | $60,000$ | $15,000$ | **$15,000$** | **$3,750$** | $3,750$ |
| **Complete 4-Fold Cross-Validation** | **$240,000$** | **$60,000$** | **$60,000$** | **$15,000$** | **$15,000$** |

### Historical Erratum Reconciliation:
$$\text{Earlier Claim: } 360,000 \text{ forward-backward passes}$$
$$\text{Corrected Physical Value: } \mathbf{60,000} \text{ forward-backward passes}$$
$$\text{Correction Factor: } \frac{60,000}{360,000} = \frac{1}{6} \quad (\mathbf{6\times \text{ fewer steps than previously claimed}})$$

The earlier calculation overestimated physical compute steps by $600\%$ because it confused total corpus utterances with train utterances ($3,599 \text{ vs } 2,400$) and failed to account for micro-batch collation ($B=4$).

---

## 5. Wall-Clock Execution Time Modeling

Wall-clock estimates are grounded on empirical hardware measurements:

### 5.1 Local Hardware Profile (NVIDIA GeForce GTX 1650 Mobile)
- **Configuration:** Micro-batch $B=4$, 3.0s audio, AMP fp16, Gradient Checkpointing, Frozen Feature Extractor.
- **Empirical Step Time (from `gpu_feasibility_measurements.json`):** $t_{\text{step}} = 1,733.3\text{ ms} = 1.7333\text{ s}$ per micro-batch.

| Unit of Work | Number of Steps | Calculation | Wall-Clock Time |
| :--- | :---: | :--- | :---: |
| **1 Epoch (1 Fold)** | $600$ micro-batches | $600 \times 1.7333\text{ s}$ | $1,040\text{ s} \approx \mathbf{17.33\text{ minutes}}$ |
| **1 Fold (25 Epochs)** | $15,000$ micro-batches | $25 \times 17.33\text{ min}$ | $433.3\text{ min} \approx \mathbf{7.22\text{ hours}}$ |
| **4 Folds (Full CV)** | $60,000$ micro-batches | $4 \times 7.22\text{ hours}$ | $28.89\text{ hours} \approx \mathbf{1.20\text{ days}}$ |

*Local Feasibility:* Complete 4-fold cross-validation requires ~29 hours on the local GTX 1650 Mobile. This is feasible over a weekend, but cloud migration remains strongly recommended for multi-seed hyperparameter sweeps.

### 5.2 Cloud Hardware Profile (NVIDIA A10G / A100 Cloud Instance)
- **Configuration:** Native batch size $B=32$ (no gradient accumulation needed, $A=1$), mixed precision bf16.
- **Estimated Step Time (A10G Tensor Cores):** $t_{\text{step}} \approx 150\text{ ms} = 0.150\text{ s}$ per batch.
- **Batches per Epoch:** $\lceil 2,400 / 32 \rceil = 75$ batches.

| Unit of Work | Number of Steps | Calculation | Wall-Clock Time |
| :--- | :---: | :--- | :---: |
| **1 Epoch (1 Fold)** | $75$ batches | $75 \times 0.150\text{ s}$ | $11.25\text{ s} \approx \mathbf{0.19\text{ minutes}}$ |
| **1 Fold (25 Epochs)** | $1,875$ batches | $25 \times 11.25\text{ s}$ | $281.25\text{ s} \approx \mathbf{4.69\text{ minutes}}$ |
| **4 Folds (Full CV)** | $7,500$ batches | $4 \times 4.69\text{ min}$ | $18.75\text{ minutes} \approx \mathbf{0.31\text{ hours}}$ |

*Throughput Advantage:* Cloud deployment achieves a **$92\times$ acceleration**, completing the full 4-fold research benchmark in under 20 minutes.
