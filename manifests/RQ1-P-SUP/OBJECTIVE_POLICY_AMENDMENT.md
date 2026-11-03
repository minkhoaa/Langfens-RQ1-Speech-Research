# P-SUP — Objective Policy Amendment & Arm Specification

Status: **ARM-SCOPED OBJECTIVE POLICY AMENDED & AUTHORIZED FOR P-SUP WORKSPACE.**
Applies strictly to workspace `/home/khoa/Projects/KLTN/langfens-rq1-p-sup`, branch `prep/rq1-p-sup`.
Historical baseline/E0/E1 contracts outside this arm remain unaltered.

## 1. Rationale and Minimum Experimental Treatment

The core scientific question of RQ1 P-SUP is:
> *Does preserving natural deletion identity in compact acoustic supervision improve omission ranking beyond contemporaneous binary supervision, without learning phone/position priors or sacrificing substitution and overall detection performance?*

The AGENTS.md default two-head CTC+binary contract is hereby amended strictly within the scope of P-SUP to authorize a compact 3-way operation supervision head alongside realized CTC loss, paired directly against an exact capacity-matched contemporaneous binary control.

## 2. Experimental Arms and Objectives

### Arm 1: T-OP (Subtype Treatment)
- **Downstream Head:** `CompactOperationHead` (266,667 trainable parameters).
- **Task Objective:** `objective="operation_ce"` (3-way categorical Cross-Entropy).
  - Targets: `correct = 0`, `substitution = 1`, `deletion = 2`.
  - Loss formulation:
    $$\mathcal{L}_{\text{task}} = -\frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} \log \frac{\exp(z_{i, y_i})}{\sum_{c=0}^2 \exp(z_{i, c})}$$
- **Acoustic CTC Objective:** Realized chronological CTC loss $\mathcal{L}_{\text{ctc}}$ over valid observed acoustic phones (correct, substitutions, additions; excluding deletions and acoustic silence).
- **Combined Loss:**
  $$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{ctc}} + 1.0 \cdot \mathcal{L}_{\text{task}}$$
- **Score Semantics:**
  - $p = \text{softmax}(z)$
  - Error score: $s_{\text{error}} = p_{\text{sub}} + p_{\text{del}} = 1 - p_{\text{corr}}$
  - Omission score: $s_{\text{omit}} = p_{\text{del}}$
  - Substitution score: $s_{\text{sub}} = p_{\text{sub}}$

### Arm 2: B-CONTROL (Capacity-Matched Contemporaneous Binary Control)
- **Downstream Head:** Identical `CompactOperationHead` (exact 266,667 parameters; identical trunk, projection, embeddings, and 3 output logits).
- **Task Objective:** `objective="binary_collapsed"` (Collapsed binary logit with Binary Cross-Entropy).
  - Collapsed logit:
    $$\tilde{z}_i = \text{logsumexp}(z_{i, 1}, z_{i, 2}) - z_{i, 0} = \log(\exp(z_{i, 1}) + \exp(z_{i, 2})) - z_{i, 0}$$
  - Binary target: $\tilde{y}_i = \mathbb{I}(y_i \neq 0) \in \{0, 1\}$.
  - Loss formulation:
    $$\mathcal{L}_{\text{task}} = \text{BCEWithLogits}(\tilde{z}_i, \tilde{y}_i) = \log(1 + e^{\tilde{z}_i}) - \tilde{y}_i \tilde{z}_i$$
  - Subtype logits in control: Under binary supervision, individual substitution ($z_1$) and deletion ($z_2$) logits are mathematically non-identifiable as distinct subtype probabilities; their sum is supervised strictly by binary error labels. In runtime outputs, $s_{\text{omit}}$ and $s_{\text{sub}}$ remain null/unidentified.
- **Acoustic CTC Objective:** Identical chronological realized CTC loss $\mathcal{L}_{\text{ctc}}$.
- **Combined Loss:** $\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{ctc}} + 1.0 \cdot \mathcal{L}_{\text{task}}$.
- **Score Semantics:**
  - Error score: $s_{\text{error}} = \sigma(\tilde{z}_i) = \frac{e^{z_1} + e^{z_2}}{e^{z_0} + e^{z_1} + e^{z_2}} = 1 - p_0$.
  - Subtype scores: $s_{\text{omit}} = \text{null}$, $s_{\text{sub}} = \text{null}$.

## 3. Parameter Matching and Capacity Inventory

| Module Component | Specification | Trainable Parameters |
|---|---|---:|
| Acoustic Projection | Linear(768, 128) | 98,432 |
| Canonical Phone Embedding | Embedding(40, 128, padding_idx=0) | 5,120 |
| Positional Embedding | Embedding(256, 128) | 32,768 |
| Cross-Attention Module | MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1) | 66,048 |
| Acoustic LayerNorm | LayerNorm(128) | 256 |
| Operation MLP | Linear(256, 128) $\to$ GELU $\to$ Dropout(0.1) $\to$ Linear(128, 3) | 33,283 |
| Realized CTC Head | Linear(768, 40) | 30,760 |
| **Total Trainable Parameters** | | **266,667** |

The pretrained backbone `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, 94.4M parameters) is strictly frozen in `eval` mode (`requires_grad=False`) for both arms.
Both arms are initialized independently with seed `1337` via `matched_initial_pair()`.

## 4. Execution and Checkpointing Contract

- **Seed:** 1337
- **Batching:** `batch_size: 2`, `gradient_accumulation_steps: 2` (effective batch size: 4).
- **Tail Divisor:** `flush_tail_divisor: "actual_group_size"`.
- **Optimizer:** AdamW (`lr: 0.001`, `weight_decay: 0.01`, `eps: 1e-8`).
- **Gradient Clipping:** `max_norm: 1.0`.
- **Schedule:** Warmup 100 successful updates, cosine decay over fixed 15,000 updates horizon.
- **Budget Endpoint:** Exactly 3,600 successful optimizer updates per arm.
- **AMP Precision:** CUDA FP16 AMP (`GradScaler` initial scale 4096).
- **Dual Checkpointing:**
  - `fixed_budget_3600.pt`: Saved at exactly update 3,600.
  - `best.pt`: Saved at validation epoch with highest strict binary AP on $s_{\text{error}}$ where `blank_fraction < 0.99`.
  - `last.pt`: Saved at the end of each epoch.
- **State Preservation:** Every checkpoint atomically contains model state dict, optimizer state dict, scheduler state dict, scaler state dict, RNG states (Python, NumPy, PyTorch, CUDA), counters (`epoch`, `successful_updates`, `skipped_amp`, `micro_steps`), metric history, and configuration metadata. SHA-256 is computed upon writing.

## 5. Partitioning and Data Sealing Invariant

- **Fold 1 TRAIN (2,399 utterances across 16 speakers):**
  - FIT: BWC, ERMS, HKK, HQTV, SKA, SVBI, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA (1,799 utterances)
  - SELECT: MBMPS, TLV (300 utterances)
  - CALIBRATION: NCC, THV (300 utterances)
- **Fold 1 VAL (300 utterances across 2 speakers):**
  - NJS, RRBI
- **Fold 1 TEST (900 utterances across 6 speakers) — SEALED:**
  - ABA, ASI, EBVS, HJK, LXC, PNV
  - Zero test audio, TextGrid, labels, or test IDs are staged, accessed, or evaluated. Physical absence is guaranteed in `authorized_RQ1-TRN-02A_fold1_train_val_only.zip` (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`).
