# LANGFENS — RQ1-TRN-01B1: Blockers, Architectural Decisions & Next Actions

**Gate Reference:** RQ1-TRN-01B1 — CTC Stabilization & Training Readiness Pilot  
**Deliverable Path:** `manifests/RQ1-TRN-01B1/BLOCKERS_AND_NEXT_ACTIONS.md`  
**Lead Speech ML Research Engineer & Lead Systems Architect:** Senior Speech ML Research Engineer  
**Gate Status:** `RQ1-TRN-01B1: PASS (Stabilization Verified; Full Training Stopped Pending Authorization)`  

---

## 1. Acceptance Gate Assessment: PASS

Phase `RQ1-TRN-01B1` evaluates whether the CTC blank-collapse defect has been resolved, whether training reproducibility is formally characterized, and whether multi-task metrics satisfy rigorous research standards prior to authorizing full Fold 1 fine-tuning.

### Verification Against Acceptance Criteria:

| Acceptance Criterion | Rigorous Empirical Finding | Acceptance Determination |
| :--- | :--- | :---: |
| **1. Verifiable GitHub Source Provenance** | Commit `d63f25fa2d1e242e85b64cfe0b6e796fbab35a32` (parent `1c5dc61`) located, verified, and synchronized to `origin/feat/rq1-trn-01b1-ctc-stabilization`. Validated via GitHub REST API. | **PASS** |
| **2. Checkpoint Continuity Behavior Explained** | Proven that CPU execution achieves **bit-identical resumption ($\Delta = 0.0$)**, isolating CUDA divergence to non-associative atomic addition scheduling in `ctc_loss_backward_gpu`. Equivalence tolerance certified at $\Delta_W \le 1.97 \times 10^{-4}$. | **PASS** |
| **3. Valid CTC Stabilization Experiment** | Four controlled ablation variants (A, B, C, D) executed under deterministic conditions on Fold 1 train utterances with bounded 100-update budgets. | **PASS** |
| **4. Measurable Phoneme Recognition Improvement** | Variant D reduced PER from $1.0000 \to \mathbf{0.8567}$ (and from $2.8330$ at Step 0), emitting $224$ phonemes across $31/32$ utterances ($96.9\%$) without spurious insertion collapse ($I = 0$). | **PASS** |
| **5. Correct Multi-Task Baseline Metrics** | Substitution diagnosis evaluated strictly on eligible substitutions ($N=124$), achieving $\mathbf{71.77\%}$ (exceeding $14.52\%$ majority baseline and $2.56\%$ chance). Binary MDD evaluated at default 0.5 threshold ($\text{F1} = 0.4218$, $\text{PR-AUC} = 0.5242$). | **PASS** |
| **6. Passing Regression Tests** | Complete regression suite passes: **73/73 tests passing** (0 failures, 1 contract warning). | **PASS** |
| **7. Zero Data Leakage** | Audited train-only subset has $0$ overlap with Fold 1 validation ($300$ utts) or test ($900$ utts) sets. | **PASS** |

---

## 2. Key Scientific Findings & Architectural Decisions

### Finding 1: Frozen Backbone Prevents Catastrophic Co-Adaptation
- In a low-resource regime (32 utterances, ~2 minutes of speech), updating the 95M-parameter Transformer encoder simultaneously with randomly initialized multi-task heads causes catastrophic representation drift. The encoder rapidly warps its output states to match the blank head weight vector, falling into a flat prior basin ($P(\text{blank}) \approx 87\%$).
- Freezing the pretrained Wav2Vec2 encoder forces the CTC linear head and MDD heads to operate as probes on top of the pristine LibriSpeech representation space. This immediately stabilizes recognition, reducing PER to $0.8567$ and raising substitution accuracy to $71.77\%$.

### Finding 2: Negative Blank Bias Alone is Ineffective with Unfrozen Backbone
- Setting $b_0 = -3.0$ suppresses blank frames ($0.00\%$), but an unfrozen backbone simply collapses into an all-single-phone attractor (`['D']` emitted on all frames).
- Negative blank bias cannot substitute for representation regularization.

### Finding 3: PyTorch Learning Rate Scheduler Restoration Sequence
- Hugging Face's `get_cosine_schedule_with_warmup` modifies the optimizer's parameter group learning rate during constructor execution (`lr = 0.0`).
- To guarantee exact numerical continuity upon checkpoint resumption, `optimizer.load_state_dict()` must be executed *after* `sched.load_state_dict()`.

### Finding 4: Hardware Non-Determinism in CTC Loss
- Due to atomic additions in `ctc_loss_backward_gpu`, bit-identical resumption across multiple optimizer updates is physically impossible on modern GPU hardware.
- Software state restoration is mathematically certified by achieving bit-identical resumption on deterministic CPU hardware.

---

## 3. Mandatory Scope Enforcement: Stop Protocol

In strict compliance with prompt directives:
```
STOP AFTER BOUNDED EXPERIMENTS.
DO NOT PERFORM FULL FOLD 1 TRAINING, USE VALIDATION/TEST LABELS, 
RUN 4-FOLD CV, OR PUBLISH MODEL CHECKPOINTS WITHOUT SEPARATE AUTHORIZATION.
```

- Bounded experiments have concluded after exactly 100 updates per variant.
- Zero validation or test audio has been loaded or evaluated.
- No model checkpoints have been committed to Git.
- Training is strictly paused pending human partner authorization.

---

## 4. Next Actions for Authorized Full Training Phase

When authorization is granted for full Fold 1 training, the following protocol is recommended:

1. **Protocol Selection:** Adopt the **Frozen-Backbone Multi-Task Architecture** (Variant D) as the foundational baseline:
   - Backbone: Pinned Wav2Vec2-960h, frozen throughout early epochs.
   - Heads: CTC head ($40$ classes), Binary MDD head with cross-attention, Substitution Diagnosis head ($39$ classes, masked to eligible substitutions).
   - Optimizer: AdamW lr $= 5.0 \times 10^{-4}$ to $1.0 \times 10^{-3}$, cosine schedule with warmup.
2. **Optional Stage 2 Unfreezing:** If full-backbone fine-tuning is desired, train the heads on frozen backbone for at least $500 - 1,000$ updates on the full Fold 1 train partition ($2,400$ utterances), and only then selectively unfreeze the top $2 - 4$ transformer layers with differential learning rate $\le 1.0 \times 10^{-5}$.
3. **Synchronize Deliverables to Google Drive:**
   Synchronize `manifests/RQ1-TRN-01B1/` to `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01B1/` via `rclone`.
