# LANGFENS — RQ1-TRN-01A-R2: Loss & CTC Contract Audit Report

**Gate Reference:** RQ1-TRN-01A-R2 — Final Training Correctness & Evidence Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01A/reconciliation/final_gate/LOSS_AND_CTC_CONTRACT_AUDIT.md`  
**Inspected Components:** `src/langfens_prep/losses.py`, `src/langfens_prep/model.py`  
**Lead Auditor / Speech ML Engineer:** Senior Speech ML Research Engineer & PyTorch Training Systems Reviewer  
**Gate Evaluation Status:** `LOSS_AND_CTC_CONTRACT_AUDIT: PASS`  

---

## 1. Executive Summary & Defect Remediation Scope

This audit addresses two critical training vulnerabilities identified during the scientific reproducibility review of phase `RQ1-TRN-01A`:

1. **Defect B (Silent Label Truncation):** Binary and diagnosis loss calculations employed `min(target.shape[1], logits.shape[1])` to slice sequences, silently discarding valid phoneme positions whenever dimension mismatches occurred.
2. **Defect C (Unenforced Runtime CTC Contract & Silent `zero_infinity` Swallowing):** `validate_ctc_contract()` existed in the codebase but was never executed at runtime inside `LangfensWav2Vec2MDD.forward()`. Furthermore, `self.ctc_loss_fn` was initialized with `zero_infinity=True`, allowing mathematical sequence incompatibilities, invalid alignments, and data defects to be silently concealed as zero loss.

Both vulnerabilities have been resolved. The architecture now enforces explicit tensor shape validation and runtime CTC contract verification with strict error policies.

---

## 2. Task B: Removal of Silent Label Truncation

### 2.1 The Silent Truncation Hazard
In earlier revisions:
```python
if target_binary is not None and binary_logits is not None:
    min_len = min(target_binary.shape[1], binary_logits.shape[1])
    t_bin = target_binary[:, :min_len]
    b_logits = binary_logits[:, :min_len]
    b_loss = self.binary_loss_fn(b_logits, t_bin)
```
If a downstream collator bug, corrupted phoneme list, or mismatched query sequence produced a mismatch (e.g. `binary_logits` had 15 frames while `target_binary` had 20 labels), the trailing 5 phonemes were silently dropped. Legitimate speech errors and substitutions disappeared from loss gradients without raising any warning.

### 2.2 Corrective Implementation: Explicit Tensor Shape Contracts
We replaced slicing with explicit tensor shape assertions and padding boundary checks in `LangfensWav2Vec2MDD.forward()`:

```python
if target_binary is not None and binary_logits is not None:
    if target_binary.shape[0] != binary_logits.shape[0]:
        raise ValueError(
            f"target_binary batch size ({target_binary.shape[0]}) does not match "
            f"binary_logits batch size ({binary_logits.shape[0]})."
        )
    if target_binary.shape[1] != binary_logits.shape[1]:
        raise ValueError(
            f"target_binary sequence length ({target_binary.shape[1]}) does not match "
            f"binary_logits sequence length ({binary_logits.shape[1]}). "
            f"Silent label truncation is strictly forbidden."
        )
    if canonical_phone_lengths is not None:
        if (canonical_phone_lengths > target_binary.shape[1]).any():
            raise ValueError(
                f"canonical_phone_lengths contains length exceeding target_binary sequence length: "
                f"{canonical_phone_lengths.tolist()} vs {target_binary.shape[1]}"
            )
        for b_idx in range(target_binary.shape[0]):
            valid_len = canonical_phone_lengths[b_idx].item()
            trailing_labels = target_binary[b_idx, valid_len:]
            if trailing_labels.numel() > 0 and (trailing_labels != self.binary_loss_fn.ignore_index).any():
                invalid_pos = (trailing_labels != self.binary_loss_fn.ignore_index).nonzero(as_tuple=True)[0]
                raise ValueError(
                    f"target_binary at batch index {b_idx} contains legitimate labels beyond declared "
                    f"canonical_phone_length ({valid_len}) at offset {invalid_pos[0].item()}. "
                    f"Padding dimensions must contain strictly ignore_index={self.binary_loss_fn.ignore_index}."
                )
    b_loss = self.binary_loss_fn(binary_logits, target_binary)
    loss_dict["binary_loss"] = b_loss
```
An identical validation block is enforced for `target_diagnosis` against `diagnosis_logits`.

### 2.3 Guarantees Enforced:
1. **Zero Discard Policy:** Batch size and sequence length must match exactly between logits and targets.
2. **Padding Integrity Check:** If `canonical_phone_lengths` declares that sample $i$ has length $L_i$, all positions beyond $L_i$ in `target_binary` and `target_diagnosis` must strictly equal `ignore_index = -100`. If any active label ($0, 1$, or phoneme class) appears in padding space, an explicit `ValueError` is raised immediately.
3. **Fully Masked Loss Differentiability:** When all target labels are masked ($-100$), both `MaskedBCEWithLogitsLoss` and `MaskedCrossEntropyLoss` return `(logits * 0.0).sum()`. The output is finite ($0.0$), differentiable, and backward propagation generates exact zero gradients without NaN.

---

## 3. Task C: Runtime CTC Contract Enforcement

### 3.1 Precondition Verification in `validate_ctc_contract`
`src/langfens_prep/losses.py` defines `validate_ctc_contract()` and custom exception `CTCContractError`. The function verifies 7 mathematical preconditions before CTCLoss execution:

1. **Dimensionality:** `log_probs` is 3D ($(T, N, C)$ or $(N, T, C)$), `targets` is 2D ($(N, S)$) or 1D unpadded, `input_lengths` and `target_lengths` are 1D with length $N$.
2. **Finiteness:** `log_probs` contains strictly finite values (no NaNs or infinities).
3. **Temporal Bounds:** $0 \le \text{input\_lengths}[i] \le T$.
4. **Target Non-Negativity:** $\text{target\_lengths}[i] \ge 0$. Empty sequences ($L_i = 0$) are permitted.
5. **Repeated-Token Frame Feasibility:** In CTC, consecutive identical phonemes (e.g. $[B, B]$) require an intervening blank emission to avoid collapsing into a single token. Hence:
   $$\text{min\_required\_frames}_i = L_i + \sum_{j=1}^{L_i-1} \mathbb{I}(y_{i, j} == y_{i, j+1})$$
   If $\text{input\_lengths}[i] < \text{min\_required\_frames}_i$, the alignment is mathematically impossible.
6. **Blank Integrity:** Ground-truth realized targets $y_{i, :L_i}$ must never contain the CTC blank index ($0$).
7. **Vocabulary Bounds:** All targets must fall strictly within valid class range $[1, C-1]$.

### 3.2 Hardening `nn.CTCLoss`: Disabling Concealed Infinities
`self.ctc_loss_fn` is initialized with:
```python
self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=False)
```
Setting `zero_infinity=False` guarantees that PyTorch will never replace unhandled mathematical infinities with $0.0$ silently.

### 3.3 Runtime Policy & Counter Architecture
In `LangfensWav2Vec2MDD.forward()`:
```python
is_valid, error_msg = validate_ctc_contract(
    log_probs=log_probs,
    targets=target_ctc,
    input_lengths=frame_output_lengths,
    target_lengths=target_lengths,
    blank_id=0
)

if not is_valid:
    if self.ctc_zero_infinity_policy == "raise":
        raise CTCContractError(f"Runtime CTC Contract Violation: {error_msg}")
    elif self.ctc_zero_infinity_policy == "warn_and_zero":
        self.ctc_contract_violations += 1
        self.last_ctc_violation_details = error_msg
        warnings.warn(f"[CTC_CONTRACT_VIOLATION] Counter={self.ctc_contract_violations}: {error_msg}")
        ctc_loss = (log_probs * 0.0).sum()
        loss_dict["ctc_loss"] = ctc_loss
```

- **Policy `"raise"` (Default):** Halts execution immediately upon any annotation defect, model shape defect, or impossible alignment, reporting the exact batch index, sequence length, and token sequence.
- **Policy `"warn_and_zero"`:** Monitored fallback for fault-tolerant data ingestion; increments `self.ctc_contract_violations`, stores `self.last_ctc_violation_details`, logs a warning, and returns `(log_probs * 0.0).sum()` to preserve backward graph connectivity.
- **Empty-Target Handling:** If all targets in a batch have length 0 (`(target_lengths == 0).all()`), the model returns differentiable zero loss `(log_probs * 0.0).sum()`.

---

## 4. Summary of Positive and Negative Tests

| Test Case | Scenario / Condition | Expected Behavior | Verification Status |
| :--- | :--- | :--- | :---: |
| **Normal CTC Step** | $T_{\text{in}} = 49$, $L_{\text{tgt}} = 4$, valid ARPAbet | Contract passes; finite positive loss | **PASS** |
| **Feasible Repeats** | Target $[5, 5]$ with $T_{\text{in}} = 49 \ge 3$ frames | Contract passes; finite loss | **PASS** |
| **Infeasible Repeats** | Target $[5, 5]$ ($M=3$) with $T_{\text{in}} = 2$ | Contract fails; `CTC length violation` | **PASS** |
| **Infeasible Length** | $T_{\text{in}} = 2$, $L_{\text{tgt}} = 4$ | Raises `CTCContractError` | **PASS** |
| **Blank ID in Target** | Active sequence contains $0$ | Raises `CTCContractError` | **PASS** |
| **OOV Target ID** | Target token index $999 \ge C$ | Raises `CTCContractError` | **PASS** |
| **Warn & Zero Policy** | Infeasible alignment under `warn_and_zero` | Counter increments; finite $0.0$ grad | **PASS** |
| **Empty Target Batch** | $\text{target\_lengths} = [0, 0]$ | Finite $0.0$ loss; autograd intact | **PASS** |
| **Binary SeqLen Mismatch** | Target seq len $4$ vs logits $3$ | Raises `ValueError` (no truncation) | **PASS** |
| **Diagnosis SeqLen Mismatch** | Target seq len $2$ vs logits $3$ | Raises `ValueError` (no truncation) | **PASS** |
| **Batch Size Mismatch** | Target batch $3$ vs logits $2$ | Raises `ValueError` | **PASS** |
| **Label in Padding Space** | Active label at offset beyond $L_{\text{declared}}$ | Raises `ValueError` | **PASS** |
| **All-Masked BCE/CE** | All labels $-100$ | Loss $0.0$, finite zero gradients | **PASS** |

All positive and negative test cases pass without regression in both CPU and CUDA environments.
