#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-01B1-R2: Task B / Canonical-Only Baseline Comparison
Trains and evaluates canonical-only reference baselines against the multi-task neural acoustic model.

Baselines evaluated:
1. Uniform Chance (1/39 = 2.5641%)
2. Dynamic Empirical Majority Class
3. Canonical-Only Empirical Prior P(realized | canonical) Lookup
4. Canonical-Only Learned MLP Substitution Classifier (trained with PyTorch AdamW on train probe substitutions)
5. Canonical-Only Learned MLP Binary MDD Classifier (trained with PyTorch AdamW on train probe binary labels)

Evaluated strictly on:
- Train Probe: 256 utterances (14 speakers, 936 substitutions, 7470 binary tokens)
- Holdout Probe: 64 utterances (2 speakers SVBI & ZHAA, 252 substitutions, 2081 binary tokens)
"""

import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audit_scripts.run_rq1_trn_01b1_reconciliation_campaign import ProtocolDataset
from audit_scripts.run_training_sanity import compute_pr_auc
from langfens_prep.phoneme_normalizer import (
    STANDARD_39_ARPABET,
    PHONE_TO_INDEX,
    INDEX_TO_PHONE,
    PHONE_TO_CTC_INDEX
)

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CanonicalSubstitutionMLP(nn.Module):
    """Learned neural classifier taking ONLY canonical phone tokens as input."""
    def __init__(self, num_phones: int = 40, embed_dim: int = 128, hidden_dim: int = 256, num_classes: int = 39, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_phones, embed_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, canonical_indices: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(canonical_indices)
        return self.classifier(embeds)

class CanonicalBinaryMLP(nn.Module):
    """Learned neural binary detector taking ONLY canonical phone tokens as input."""
    def __init__(self, num_phones: int = 40, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_phones, embed_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, canonical_indices: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(canonical_indices)
        return self.classifier(embeds).squeeze(-1)

def extract_tokens(dataset: ProtocolDataset) -> Dict[str, Any]:
    sub_canonical = []
    sub_realized = []
    bin_canonical = []
    bin_labels = []

    for item in dataset:
        can_idx = item["canonical_phone_indices"].tolist()
        diag_idx = item["diagnosis_target_indices"].tolist()
        b_lbl = item["binary_labels"].tolist()

        for c, d in zip(can_idx, diag_idx):
            if d != -100:
                sub_canonical.append(c)
                sub_realized.append(d)

        for c, b in zip(can_idx, b_lbl):
            if b != -100:
                bin_canonical.append(c)
                bin_labels.append(b)

    return {
        "sub_canonical": np.array(sub_canonical, dtype=np.int64),
        "sub_realized": np.array(sub_realized, dtype=np.int64),
        "bin_canonical": np.array(bin_canonical, dtype=np.int64),
        "bin_labels": np.array(bin_labels, dtype=np.float32)
    }

def main():
    set_seed(1337)
    protocol_path = "manifests/RQ1-TRN-01B1/reconciliation/INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json"
    train_ds = ProtocolDataset("internal_train_probe_256", protocol_path=protocol_path)
    holdout_ds = ProtocolDataset("internal_holdout_probe_64", protocol_path=protocol_path)

    print("Extracting tokens from train probe (256 utts) and holdout probe (64 utts)...")
    train_data = extract_tokens(train_ds)
    holdout_data = extract_tokens(holdout_ds)

    print(f"Train: substitutions={len(train_data['sub_canonical'])}, binary={len(train_data['bin_canonical'])}")
    print(f"Holdout: substitutions={len(holdout_data['sub_canonical'])}, binary={len(holdout_data['bin_canonical'])}")

    # 1. Baseline: Uniform Chance
    uniform_sub_acc = 1.0 / 39.0

    # 2. Baseline: Majority Class
    train_sub_counter = Counter(train_data["sub_realized"])
    maj_sub_class, maj_sub_count = train_sub_counter.most_common(1)[0]
    maj_sub_acc_train = maj_sub_count / len(train_data["sub_canonical"])
    maj_sub_acc_holdout = (holdout_data["sub_realized"] == maj_sub_class).sum() / len(holdout_data["sub_canonical"])

    # 3. Baseline: Canonical Empirical Prior Matrix P(realized | canonical)
    prior_matrix = np.zeros((40, 39), dtype=np.float64)  # 40 canonical indices, 39 substitution classes
    for c, r in zip(train_data["sub_canonical"], train_data["sub_realized"]):
        prior_matrix[c, r] += 1.0

    # Laplace smoothing alpha = 0.01 for log-loss computation
    smooth_alpha = 0.01
    prior_probs = (prior_matrix + smooth_alpha) / (prior_matrix.sum(axis=-1, keepdims=True) + smooth_alpha * 39.0)
    prior_preds = np.argmax(prior_matrix, axis=-1)

    # If canonical phone never had substitutions in train, fallback to majority class
    unseen_mask = (prior_matrix.sum(axis=-1) == 0)
    prior_preds[unseen_mask] = maj_sub_class

    train_prior_preds = prior_preds[train_data["sub_canonical"]]
    holdout_prior_preds = prior_preds[holdout_data["sub_canonical"]]

    train_prior_acc = float((train_prior_preds == train_data["sub_realized"]).mean())
    holdout_prior_acc = float((holdout_prior_preds == holdout_data["sub_realized"]).mean())

    # Top-3 accuracy for prior
    top3_prior_indices = np.argsort(prior_probs, axis=-1)[:, -3:]
    holdout_prior_top3 = [
        holdout_data["sub_realized"][i] in top3_prior_indices[holdout_data["sub_canonical"][i]]
        for i in range(len(holdout_data["sub_canonical"]))
    ]
    holdout_prior_top3_acc = float(np.mean(holdout_prior_top3))

    # Cross-entropy / NLL for empirical prior on holdout
    holdout_prior_probs_assigned = prior_probs[holdout_data["sub_canonical"], holdout_data["sub_realized"]]
    holdout_prior_nll = float(-np.mean(np.log(np.maximum(holdout_prior_probs_assigned, 1e-12))))

    # 4. Baseline: Learned Canonical-Only Substitution MLP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp_sub = CanonicalSubstitutionMLP(num_phones=40, embed_dim=128, hidden_dim=256, num_classes=39, dropout=0.1).to(device)
    optimizer_sub = torch.optim.AdamW(mlp_sub.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion_sub = nn.CrossEntropyLoss()

    X_train_sub = torch.tensor(train_data["sub_canonical"], dtype=torch.long, device=device)
    y_train_sub = torch.tensor(train_data["sub_realized"], dtype=torch.long, device=device)
    X_holdout_sub = torch.tensor(holdout_data["sub_canonical"], dtype=torch.long, device=device)
    y_holdout_sub = torch.tensor(holdout_data["sub_realized"], dtype=torch.long, device=device)

    for epoch in range(120):
        mlp_sub.train()
        logits = mlp_sub(X_train_sub)
        loss = criterion_sub(logits, y_train_sub)
        optimizer_sub.zero_grad()
        loss.backward()
        optimizer_sub.step()

    mlp_sub.eval()
    with torch.no_grad():
        train_sub_logits = mlp_sub(X_train_sub)
        holdout_sub_logits = mlp_sub(X_holdout_sub)
        train_learned_sub_acc = float((train_sub_logits.argmax(-1) == y_train_sub).float().mean().item())
        holdout_learned_sub_acc = float((holdout_sub_logits.argmax(-1) == y_holdout_sub).float().mean().item())
        holdout_learned_sub_loss = float(criterion_sub(holdout_sub_logits, y_holdout_sub).item())
        top3_learned = holdout_sub_logits.topk(3, dim=-1).indices
        holdout_learned_top3_acc = float((top3_learned == y_holdout_sub.unsqueeze(-1)).any(dim=-1).float().mean().item())

    # 5. Baseline: Learned Canonical-Only Binary MDD MLP
    mlp_bin = CanonicalBinaryMLP(num_phones=40, embed_dim=128, hidden_dim=256, dropout=0.1).to(device)
    optimizer_bin = torch.optim.AdamW(mlp_bin.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion_bin = nn.BCEWithLogitsLoss()

    X_train_bin = torch.tensor(train_data["bin_canonical"], dtype=torch.long, device=device)
    y_train_bin = torch.tensor(train_data["bin_labels"], dtype=torch.float32, device=device)
    X_holdout_bin = torch.tensor(holdout_data["bin_canonical"], dtype=torch.long, device=device)
    y_holdout_bin = torch.tensor(holdout_data["bin_labels"], dtype=torch.float32, device=device)

    for epoch in range(120):
        mlp_bin.train()
        logits = mlp_bin(X_train_bin)
        loss = criterion_bin(logits, y_train_bin)
        optimizer_bin.zero_grad()
        loss.backward()
        optimizer_bin.step()

    mlp_bin.eval()
    with torch.no_grad():
        holdout_bin_logits = mlp_bin(X_holdout_bin)
        holdout_bin_probs = torch.sigmoid(holdout_bin_logits).cpu().numpy()
        holdout_bin_y = y_holdout_bin.cpu().numpy()
        holdout_learned_bin_pr_auc = compute_pr_auc(holdout_bin_y, holdout_bin_probs)
        holdout_learned_bin_loss = float(criterion_bin(holdout_bin_logits, y_holdout_bin).item())

        # Evaluate at threshold 0.5
        preds_bin = (holdout_bin_probs >= 0.5).astype(int)
        tp = int(((preds_bin == 1) & (holdout_bin_y == 1)).sum())
        fp = int(((preds_bin == 1) & (holdout_bin_y == 0)).sum())
        fn = int(((preds_bin == 0) & (holdout_bin_y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    results = {
        "metadata": {
            "evaluation_date": "2026-10-25",
            "protocol": "RQ1-TRN-01B1-R2 Canonical-Only Baselines",
            "train_probe_substitutions": int(len(train_data["sub_canonical"])),
            "holdout_probe_substitutions": int(len(holdout_data["sub_canonical"])),
            "train_probe_binary_tokens": int(len(train_data["bin_canonical"])),
            "holdout_probe_binary_tokens": int(len(holdout_data["bin_canonical"]))
        },
        "substitution_baselines": {
            "uniform_chance": {
                "description": "Uniform random choice over 39 ARPAbet classes (1/39)",
                "holdout_accuracy": round(uniform_sub_acc, 4)
            },
            "majority_class": {
                "description": f"Constant majority class ({INDEX_TO_PHONE[maj_sub_class]}) from train probe",
                "class_index": int(maj_sub_class),
                "class_phone": INDEX_TO_PHONE[maj_sub_class],
                "train_accuracy": round(maj_sub_acc_train, 4),
                "holdout_accuracy": round(maj_sub_acc_holdout, 4)
            },
            "canonical_empirical_prior": {
                "description": "Bayesian lookup P(realized | canonical) computed directly from train probe",
                "train_accuracy": round(train_prior_acc, 4),
                "holdout_accuracy": round(holdout_prior_acc, 4),
                "holdout_top3_accuracy": round(holdout_prior_top3_acc, 4),
                "holdout_nll_loss": round(holdout_prior_nll, 4)
            },
            "canonical_learned_mlp": {
                "description": "Learned MLP (Embedding -> 2x Linear GELU) taking purely canonical tokens",
                "train_accuracy": round(train_learned_sub_acc, 4),
                "holdout_accuracy": round(holdout_learned_sub_acc, 4),
                "holdout_top3_accuracy": round(holdout_learned_top3_acc, 4),
                "holdout_ce_loss": round(holdout_learned_sub_loss, 4)
            }
        },
        "binary_mdd_baselines": {
            "empirical_prevalence": {
                "description": "Constant prediction of empirical positive rate",
                "train_rate": round(float(train_data["bin_labels"].mean()), 4),
                "holdout_rate": round(float(holdout_data["bin_labels"].mean()), 4)
            },
            "canonical_learned_mlp": {
                "description": "Learned MLP taking purely canonical tokens to predict error probability",
                "holdout_pr_auc": round(holdout_learned_bin_pr_auc, 4),
                "holdout_bce_loss": round(holdout_learned_bin_loss, 4),
                "holdout_precision_at_05": round(prec, 4),
                "holdout_recall_at_05": round(rec, 4),
                "holdout_f1_at_05": round(f1, 4)
            }
        }
    }

    out_dir = Path("manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "canonical_only_baselines.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved canonical-only baselines to {out_file}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
