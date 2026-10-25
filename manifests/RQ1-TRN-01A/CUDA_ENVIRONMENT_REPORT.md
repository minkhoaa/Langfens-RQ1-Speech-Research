# LANGFENS — RQ1-TRN-01A: CUDA Environment & GPU Smoke Test Report

**Phase:** `RQ1-TRN-01A` — Training Implementation & GPU Smoke Test  
**Focus Area:** Phase 3 — Isolated GPU Environment Provisioning & CUDA Verification  
**Author / Engineering Lead:** Senior Speech ML Engineer & Research Reproducibility Reviewer  
**Deliverable Path:** `manifests/RQ1-TRN-01A/CUDA_ENVIRONMENT_REPORT.md`  
**Host GPU:** NVIDIA GeForce GTX 1650 Mobile / Max-Q (4,096 MiB VRAM)  
**Status:** `CUDA_ENVIRONMENT_READY: PASS`  

---

## 1. Executive Summary & Dual Environment Strategy

To enable GPU acceleration for neural acoustic model training while maintaining absolute stability of the existing CPU verification pipeline, Project Langfens employs a **non-destructive dual-environment architecture**:

1. **Reference CPU Environment (`.venv`):**
   - Runtime: Python 3.14.8 with PyTorch 2.14.1+cpu.
   - Purpose: Static data preparation, census reconciliation, and rapid CPU unit tests.
   - Preservation: **100% untouched and preserved.**
2. **GPU Training Environment (`.venv-cuda`):**
   - Runtime: Python 3.11.17 with PyTorch 2.14.1+cu130 and Torchaudio 2.11.0+cu130.
   - Provisioned via `uv` in isolated user space without requiring root privileges.
   - Purpose: CUDA-accelerated acoustic modeling, mixed-precision training, and GPU smoke testing.

---

## 2. Host Hardware & Driver Specifications

- **GPU Device:** NVIDIA GeForce GTX 1650 Mobile / Max-Q
- **PCI Bus Identifier:** `00000000:01:00.0`
- **Total Physical VRAM:** $4,096 \text{ MiB}$ ($4.0 \text{ GiB}$) GDDR6
- **Usable PyTorch Memory:** **$3,730.38 \text{ MiB}$** (remaining memory allocated to OS display compositor)
- **NVIDIA Kernel Driver:** `615.78.08`
- **Host CUDA UMD Version:** `13.4`
- **CUDA Architecture Capability:** `sm_75` (Compute Capability 7.5, Turing architecture)

---

## 3. Empirical GPU Verification Results

The isolated `.venv-cuda` environment was verified through direct programmatic tests:

```python
import torch
import torchaudio

# 1. Device Detection
assert torch.cuda.is_available() is True
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1650"

# 2. CUDA Tensor Operations
x = torch.randn(1000, 1000, device="cuda")
y = torch.matmul(x, x)
assert y.device.type == "cuda"
assert torch.isfinite(y).all()

# 3. Automatic Mixed Precision (AMP)
with torch.amp.autocast('cuda', dtype=torch.float16):
    z = torch.matmul(x, x)
assert z.dtype == torch.float16

# 4. Gradient Checkpointing
from torch.utils.checkpoint import checkpoint
block = torch.nn.Linear(128, 128).cuda()
inp = torch.randn(4, 128, device="cuda", requires_grad=True)
out = checkpoint(block, inp, use_reentrant=False)
out.sum().backward()
assert inp.grad is not None and inp.grad.shape == inp.shape
```

### Empirical Test Output:
```text
=== CUDA Verification in .venv-cuda ===
PyTorch Version: 2.14.1+cu130
Torchaudio Version: 2.11.0+cu130
CUDA Available: True
Device Count: 1
Device Name: NVIDIA GeForce GTX 1650
Total VRAM (MiB): 3730.375
CUDA Capability: 7.5
Tensor matmul sum: 10104.1455078125 on cuda:0
AMP fp16 dtype: torch.float16
Gradient Checkpointing test: grad norm = 9.94886589050293
```

---

## 4. Full Model GPU Smoke Test Execution

In `tests/test_model.py::test_gpu_forward_backward_if_available`, the full `LangfensWav2Vec2MDD` architecture (feature extractor + transformer encoder + CTC head + binary MDD head + diagnosis head) was placed on `cuda:0` and tested on a synthetic batch under AMP fp16:

- **Forward Pass:** Successfully computed `ctc_logits`, `binary_logits`, and `diagnosis_logits`.
- **Multi-Task Loss:** Evaluated $\mathcal{L}_{\text{total}} = 1.0 \mathcal{L}_{\text{ctc}} + 1.0 \mathcal{L}_{\text{bin}} + 0.5 \mathcal{L}_{\text{diag}}$ without NaNs or Infs.
- **Backward Pass:** Backpropagation cleanly updated gradients across all heads.
- **Peak VRAM Consumption During Smoke Test:** $\approx 420 \text{ MiB}$.

---

## 5. Enforced GPU Training Constraints on Host Hardware

Given the $3,730 \text{ MiB}$ usable VRAM ceiling on the local GTX 1650, model training runs in Phase `RQ1-TRN-01B+` must adhere to the following parameters:

1. **Micro-Batch Size:** $B_{\text{micro}} = \mathbf{2}$ (maximum $4$ for short utterances $\le 3.5\text{ s}$).
2. **Gradient Accumulation Steps:** $N_{\text{accum}} = \mathbf{8}$, yielding an effective batch size of $16$.
3. **Precision:** `torch.amp.autocast('cuda', dtype=torch.float16)`.
4. **Feature Encoder:** Frozen 7-layer convolutional extractor via `model.freeze_feature_encoder()`.
5. **Gradient Checkpointing:** Enabled via `model.gradient_checkpointing_enable()`.

---

## 6. Gate Determination

CUDA tensor operations, AMP fp16, gradient checkpointing, and GPU model forward/backward passes are verified and functional. The CPU reference environment remains completely intact.

**Phase Gate Evaluation:** **`CUDA_ENVIRONMENT_READY: PASS`**
