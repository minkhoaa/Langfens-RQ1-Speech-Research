# RQ1-TRN-02B DESIGN-R1 Corrections

**Document Type:** Additive Design Correction Specification
**Design Revision:** DESIGN-R1
**Scope:** Methodological rectification of experimental controls and compute budget.
**Execution Boundary:** STRICTLY NO implementation, training, optimizer updates, GPU allocation, or official-test partition access has occurred or is authorized by this document. Any operational execution requires separate, explicit operator authorization.

---

## 1. Original Design Defects

The initial experimental design for RQ1-TRN-02B (commit `68363673f04719eb976a5e690e0845c0164adc3c`) suffered from two critical methodological defects:

1. **Confounded Experimental Contrast (Absence of a No-Dropout Control):**
   - The prior design compared **B0** (the frozen production baseline checkpoint `best.pt` with 0 optimizer updates) against **B1** (fine-tuned from `best.pt` with prompt embedding dropout $p=0.25$).
   - This contrast confounded the causal effect of prompt embedding dropout with the effect of additional optimizer updates, extended training exposure, and continued adaptation to the Fold 1 training partition.
   - Any observed performance change in B1 could not be attributed to the dropout mechanism, as fine-tuning alone could induce identical shifts.
2. **Mathematically Inconsistent Compute Budget Statement:**
   - The prior document specified a training horizon of "11 epochs / 3,600 successful updates".
   - Under the frozen Fold 1 training specification (2,399 utterances, microbatch size 2, gradient accumulation 2), each epoch yields approximately 600 optimizer steps.
   - Consequently, 3,600 updates corresponds to **6 epochs**, whereas 11 epochs corresponds to **6,600 updates**. The statement conflated two incompatible horizons.

DESIGN-R1 fully resolves both defects while strictly preserving the underlying architectural hypothesis, frozen baseline evidence, evaluation contracts, and safety boundaries.

---

## 2. Revised Controlled Comparison

### 2.1 Three-Arm Experimental Architecture

DESIGN-R1 replaces the invalid B0/B1 contrast with a rigorously controlled three-arm architecture:

1. **Reference R (Immutable Historical Baseline):**
   - Initialized from the frozen RQ1-TRN-02A production checkpoint `checkpoints/RQ1-TRN-02A/recovered/best.pt` (SHA-256: `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`).
   - Receives zero optimizer updates and zero architectural modifications.
   - Serves as the fixed historical research anchor for reproducing Protocol O AP (`0.317021`), Protocol Z AP (`0.313530`), and greedy CTC PER (`23.64%`).
2. **Control C0 (No-Dropout Fine-Tuning Control):**
   - Model weights initialized from the exact same production checkpoint `best.pt`.
   - Optimizer (AdamW), scheduler (cosine annealing), and AMP scaler re-initialized from step 0 under the shared reset policy.
   - Prompt embedding dropout probability set to **$p = 0.00$** (exact baseline forward pass).
   - Continues fine-tuning on the frozen Fold 1 TRAIN partition (2,399 utterances) under identical hyperparameters, seed, dataloader ordering, and update budget as C1.
   - Measures performance drift, training adaptation, and calibration changes attributable to additional fine-tuning in the absence of dropout.
3. **Treatment C1 (Binary-Only Prompt Dropout Fine-Tuning):**
   - Model weights initialized from the exact same production checkpoint `best.pt`.
   - Optimizer, scheduler, and AMP scaler re-initialized identically to C0.
   - Prompt embedding dropout probability set to **$p = 0.25$**.
   - Continues fine-tuning on the identical Fold 1 TRAIN partition under the exact same update budget and exposure as C0.
   - Evaluates the effect of prompt embedding dropout under controlled fine-tuning.

### 2.2 Formal Estimands and Scientific Attribution

To guarantee correct causal attribution, three distinct performance contrasts must be reported:

- **Incremental Dropout Effect ($\text{C1} - \text{C0}$):**
  $$\Delta_{\text{dropout}} = \text{Metric}(\text{C1}) - \text{Metric}(\text{C0})$$
  This is the **primary causal estimand**. Because C0 and C1 receive identical training exposure, optimizer steps, and data sequences, this contrast isolates the specific effect of prompt embedding dropout.
- **Fine-Tuning Drift ($\text{C0} - \text{R}$):**
  $$\Delta_{\text{tuning}} = \text{Metric}(\text{C0}) - \text{Metric}(\text{R})$$
  Measures the baseline shift caused purely by additional fine-tuning on the training partition.
- **Compound Re-training Contrast ($\text{C1} - \text{R}$):**
  $$\Delta_{\text{compound}} = \text{Metric}(\text{C1}) - \text{Metric}(\text{R})$$
  Measures the overall net shift from the historical baseline. **Methodological Prohibition:** $\text{C1} - \text{R}$ must never be reported or interpreted as the causal effect of prompt dropout.

### 2.3 Minimal Architectural Intervention Preserved

The minimal intervention contract defined in the original design is strictly preserved:
- Masking affects **only** the direct canonical phone embedding entering the Binary MLP concatenation.
- Canonical phone embeddings used as cross-attention queries are **not** masked.
- Aligned acoustic vectors $\mathbf{c}_i$ are untouched at the concatenation point.
- Applied as full-vector zero masking at rate $p=0.25$ without inverted rescaling ($1/(1-p)$).
- Applied strictly at valid non-padding phoneme positions ($i < L_{\text{valid}}$).
- Active exclusively during training (`model.train()`); strictly disabled during evaluation (`model.eval()`).
- At $p=0.00$, exact mathematical equivalence to the original baseline architecture is preserved.
- No learnable gates, contrastive losses, additional heads, or modifications to the frozen Wav2Vec2 backbone or disabled diagnosis head are permitted.

---

## 3. Corrected Optimizer-Update Arithmetic

### 3.1 Derivation from Frozen Partition and Batch Parameters

The compute budget arithmetic is formally derived from the frozen Fold 1 specifications:

$$\begin{aligned}
N_{\text{train}} &= 2,399 \text{ utterances} \\
B_{\text{micro}} &= 2 \text{ utterances per microbatch} \\
N_{\text{microbatches/epoch}} &= \left\lceil \frac{2,399}{2} \right\rceil = 1,200 \text{ microbatches} \\
K_{\text{accum}} &= 2 \text{ microbatches per optimizer step} \\
N_{\text{updates/epoch}} &= \frac{1,200}{2} = 600 \text{ optimizer updates per epoch}
\end{aligned}$$

Projecting this update rate across training horizons reveals the mathematical reality:
- **6 Epochs:** $6 \times 600 = 3,600 \text{ successful optimizer updates}$.
- **11 Epochs:** $11 \times 600 = 6,600 \text{ successful optimizer updates}$.

The prior statement of "11 epochs / 3,600 updates" was mathematically impossible under these hyperparameters. 3,600 updates corresponds precisely to 6 training epochs.

### 3.2 Update Accounting: Step Attempts vs. Successful Parameter Updates

Under PyTorch Automatic Mixed Precision (AMP), gradient unscaling via `torch.cuda.amp.GradScaler` can identify non-finite gradients (Inf/NaN). When detected, the optimizer step is skipped, the parameter weights remain unchanged, and the scaler decreases its loss scale factor.

To ensure transparent accounting:
- **`step_attempts`:** The number of accumulation boundaries where `scaler.step(optimizer)` was called.
- **`successful_updates`:** The number of boundaries where gradients were finite and parameter weights were successfully updated.
- **`skipped_steps`:** The number of boundaries where updates were skipped due to gradient overflow/underflow.

The hard cap of 3,600 updates applies strictly to **`successful_updates`**.

---

## 4. Revised Compute Budget

### 4.1 Per-Arm and Total Experiment Update Caps

DESIGN-R1 establishes strict compute boundaries:

- **Control Arm C0:** Maximum 6 epochs; hard cap of **3,600 successful optimizer updates**.
- **Treatment Arm C1:** Maximum 6 epochs; hard cap of **3,600 successful optimizer updates**.
- **Aggregate Project Budget:** Maximum 12 arm-epochs; hard cap of **7,200 successful optimizer updates combined**.
- **Initial Authorizations:** The sensitivity arm ($p=0.50$), secondary random seeds, and multi-fold evaluations are explicitly excluded from the initial budget.

### 4.2 Colab Resource Estimation and Enforcement Rules

- **Compute Platform:** Single Google Colab T4 GPU (16 GB VRAM).
- **Runtime Estimate:** 3 to 5 T4 GPU-hours per arm, yielding **6 to 10 T4 GPU-hours total**.
- **Pre-Allocation Quota Check:** Prior to starting C0, remaining compute units must be verified via non-interactive CLI (`colab usage`). If available quota cannot support both runs, allocation must not occur.
- **Mutual Dependency Rule:** Control C0 and Treatment C1 form an inseparable experimental pair. Executing C1 alone without C0 is strictly forbidden.
- **No Local Fallback:** Never fall back to local CPU or GPU execution if Colab quota is exhausted.

### 4.3 Matched-Budget Comparison and Early Stopping Interaction

- **Predeclared Primary Endpoint:** The primary controlled comparison $\text{C1} - \text{C0}$ is evaluated at the predeclared common endpoint of **Epoch 6** (or 3,600 successful updates, or the last common completed epoch in case of safety abortion).
- **Exploratory Best Checkpoint:** Validation AP checkpoint selection (`best.pt`) is retained strictly as an exploratory secondary analysis.
- **Patience=5 Interaction:** The monitored validation policy (`patience = 5`) cannot trigger before Epoch 6 under normal conditions. Early stopping heuristics must not terminate one arm early while allowing the other to continue. Both arms must achieve matched data exposure.

---

## 5. Experimental Validity Improvements

DESIGN-R1 introduces substantial improvements to scientific validity and metric rigor:

1. **Isolation of Causal Mechanism:** C0 provides the missing counterfactual, enabling rigorous separation of fine-tuning dynamics from prompt embedding dropout.
2. **Directional Attribution Rule:** Higher matched AP alone does not prove acoustic reliance. Increased acoustic grounding requires widening the matched-minus-mismatched AP gap ($\Delta_{\text{gap}}$) relative to C0.
3. **Paired Token Evaluation:** All primary metrics are computed using paired evaluations on identical utterances and phoneme positions.
4. **Utterance-Cluster Bootstrap:** Statistical uncertainty is quantified via 1,000 speaker-stratified utterance-cluster bootstrap resamples over the 300 validation utterances. Token-i.i.d. intervals are prohibited.
5. **Secondary Zero-Oracle Evaluation:** Protocol Z evaluates zero-oracle inference using the frozen CMUdict frontend (`99.63%` coverage), with gold annotations reserved strictly for offline metric computation. Protocol Z is never used for checkpoint or threshold tuning.
6. **Explicit Disclosure of Scientific Limitations:**
   - Both arms initialize from a validation-selected checkpoint.
   - Initial evaluation employs a single seed (`1337`).
   - Validation contains only two speakers (`NJS`, `RRBI`).
   - Utterance bootstrap quantifies sample variance, not population generalizability.
   - Validation AP checkpoint selection introduces optimism bias.
   - Zero-oracle inference does not imply readiness for unconstrained diagnosis.

---

## 6. Outstanding Execution Prerequisites

**DESIGN-R1 is a design specification only. No operational implementation or execution is authorized.**

Before any separately authorized execution may occur, the following prerequisites must be fulfilled:

1. **Design Approval Gate:** Explicit operator sign-off on DESIGN-R1.
2. **Implementation Preflight Authorization:** Separate authorization to implement the $p=0.25$ masking flag in `src/langfens_prep/model.py` and run local CPU unit tests verifying bitwise parity at $p=0.00$.
3. **Execution Script Isolation:** Implementation of dedicated configuration files (`configs/rq1_trn_02b_c0.yaml`, `configs/rq1_trn_02b_c1.yaml`) and dedicated training entrypoints completely isolated from RQ1-TRN-02A.
4. **Data Staging & Colab Quota Verification:** Verification of local VM disk staging procedures and non-interactive confirmation of sufficient Colab compute units for both arms.
5. **Sealed Partition Integrity:** Re-verification that official test audio and TextGrids remain completely sealed and unstaged.
