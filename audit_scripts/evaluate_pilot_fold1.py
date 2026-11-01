#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-03B: Step 17 Forensic Evaluation for Fold 1 Pilot Checkpoints.
Executes Protocol O and Protocol Z on:
1. Arm 2A Focal Fixed-Budget (3,600 updates, Epoch 6)
2. Arm 2A Focal Validation Best (1,200 updates, Epoch 2)
3. Arm 2B Weighted Exploratory Near-Budget (3,598 updates, Epoch 6)
4. Arm 2B Weighted Validation Best (600 updates, Epoch 1)
and pairs them with Baseline R Fold 1.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

from audit_scripts.evaluate_cross_fold_validation import evaluate_single_fold_checkpoint
from langfens_prep.g2p import CMUDictG2P

def main():
    repo = "minkhoaa/langfens-rq1-cross-fold-checkpoints"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Evaluation device: {device}")

    g2p = CMUDictG2P()
    cache_dir = Path("checkpoints/pilot_evaluation_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    eval_targets = [
        {
            "arm": "Arm2A_Focal",
            "role": "fixed_budget_registered_primary",
            "filename": "pilot/pilot_arm2a_focal/fixed_budget_checkpoint.pt",
            "updates": 3600,
            "epoch": 6,
        },
        {
            "arm": "Arm2A_Focal",
            "role": "validation_best_secondary",
            "filename": "pilot/pilot_arm2a_focal/best.pt",
            "updates": 1200,
            "epoch": 2,
        },
        {
            "arm": "Arm2B_Weighted",
            "role": "exploratory_near_budget_last",
            "filename": "pilot/pilot_arm2b_weighted/last.pt",
            "updates": 3598,
            "epoch": 6,
        },
        {
            "arm": "Arm2B_Weighted",
            "role": "validation_best_secondary",
            "filename": "pilot/pilot_arm2b_weighted/best.pt",
            "updates": 600,
            "epoch": 1,
        },
    ]

    results = {}
    for t in eval_targets:
        arm = t["arm"]
        role = t["role"]
        updates = t["updates"]
        fn = t["filename"]

        base_name = Path(fn).stem
        local_p = cache_dir / f"{base_name}_{arm}_{updates}up.pt"
        if not local_p.is_file():
            print(f"[DOWNLOAD] Downloading {fn} from Hugging Face...")
            downloaded = hf_hub_download(repo_id=repo, filename=fn, repo_type="model", token=token)
            shutil.copyfile(downloaded, local_p)

        print(f"\n[EVALUATION] Evaluating {arm} ({role})...")
        res = evaluate_single_fold_checkpoint(
            fold_idx=1,
            checkpoint_role=role,
            checkpoint_path=str(local_p),
            manifest_path="manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json",
            authorized_zip="datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip",
            g2p=g2p,
            prompts_path="resources/L2-ARCTIC/prompts.json",
            device=device
        )
        clean_res = {k: v for k, v in res.items() if k not in ("raw_predictions", "per_utterance", "per_speaker")}
        key = f"{arm}_{role}"
        results[key] = clean_res
        print(f"  Matched AP: {clean_res['matched_ap']:.6f}")
        print(f"  Mismatched AP: {clean_res['mismatched_ap']:.6f}")
        print(f"  Acoustic Gap: {clean_res['acoustic_gap']:.6f}")
        print(f"  CTC PER: {clean_res['ctc_per']*100:.2f}%")
        print(f"  FNR @ 0.50: {clean_res['fnr_at_050']*100:.2f}%, FPR @ 0.50: {clean_res['fpr_at_050']*100:.2f}%")
        print(f"  FNR @ 0.20: {clean_res['fnr_at_020']*100:.2f}%, FPR @ 0.20: {clean_res['fpr_at_020']*100:.2f}%")
        print(f"  Proto Z AP: {clean_res['proto_z_aligned_ap']:.6f}")

    out_path = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Comparative report written to: {out_path}")

if __name__ == "__main__":
    main()
