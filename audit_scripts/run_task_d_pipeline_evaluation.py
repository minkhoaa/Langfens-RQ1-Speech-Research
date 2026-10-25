#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-01B1-R2: Task D / Detection and Diagnosis Pipeline Evaluation
Separates oracle-conditioned evaluation from deployable cascaded pipeline performance.

Evaluates:
1. Oracle-Conditioned Substitution Diagnosis Accuracy (only on gold substitution events).
2. Binary MDD Evaluation (AP, Precision, Recall, F1 across threshold sweep).
3. Pipeline Substitution Diagnosis (conditioned strictly on binary detector triggering).
4. End-to-End Correctly Detected & Diagnosed Substitutions (E2E Precision, Recall, F1).
5. Error Propagation Analysis:
   - Missed substitutions (False Negatives of binary detector).
   - Spurious diagnosis triggers (False Positives of binary detector).
6. Comparative Evaluation across Matched, Mismatched, and Silence Audio.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

from langfens_prep.model import LangfensWav2Vec2MDD
from langfens_prep.phoneme_normalizer import INDEX_TO_PHONE
from audit_scripts.run_rq1_trn_01b1_reconciliation_campaign import ProtocolDataset
from audit_scripts.run_training_sanity import compute_pr_auc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Task D Pipeline Evaluation on {device}...")

    ckpt_path = "checkpoints/RQ1-TRN-01B1/reconciliation/rq1_trn_01b1_r_frozen_probe_seed1337_step200.pt"
    model = LangfensWav2Vec2MDD(alignment_mode="cross_attention")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    protocol_path = "manifests/RQ1-TRN-01B1/reconciliation/INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json"
    ds = ProtocolDataset("internal_holdout_probe_64", protocol_path=protocol_path)
    items = list(ds)
    num_items = len(items)

    waveforms = [item["audio_waveform_16k"] for item in items]
    can_indices_list = [item["canonical_phone_indices"] for item in items]
    bin_labels_list = [item["binary_labels"] for item in items]
    diag_indices_list = [item["diagnosis_target_indices"] for item in items]
    can_norm_list = [item["canonical_phones_norm39"] for item in items]

    eval_conditions = {
        "matched": waveforms,
        "mismatched": [waveforms[(i + 1) % num_items] for i in range(num_items)],
        "silence": [torch.zeros_like(w) for w in waveforms]
    }

    pipeline_results = {}

    for cond, audio_list in eval_conditions.items():
        print(f"Evaluating pipeline under condition: {cond}...")
        all_bin_scores = []
        all_bin_targets = []
        all_diag_preds = []
        all_diag_targets = []
        all_can_indices = []

        with torch.no_grad():
            for i in range(num_items):
                wav = audio_list[i].unsqueeze(0).to(device)
                c_idx = can_indices_list[i].unsqueeze(0).to(device)
                L = len(can_indices_list[i])

                net_in = {
                    "source": wav,
                    "audio_lengths": torch.tensor([wav.shape[1]], device=device),
                    "canonical_phone_indices": c_idx,
                    "canonical_phone_lengths": torch.tensor([L], device=device)
                }

                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    out = model(**net_in)

                b_probs = torch.sigmoid(out.binary_logits[0, :L].float()).cpu().numpy()
                d_preds = out.diagnosis_logits[0, :L].float().argmax(dim=-1).cpu().numpy()
                b_targets = bin_labels_list[i][:L].numpy()
                d_targets = diag_indices_list[i][:L].numpy()

                valid = (b_targets != -100)
                all_bin_scores.extend(b_probs[valid].tolist())
                all_bin_targets.extend(b_targets[valid].tolist())
                all_diag_preds.extend(d_preds[valid].tolist())
                all_diag_targets.extend(d_targets[valid].tolist())
                all_can_indices.extend(can_indices_list[i][:L].numpy()[valid].tolist())

        y_bin = np.array(all_bin_targets)
        p_bin = np.array(all_bin_scores)
        y_diag = np.array(all_diag_targets)
        p_diag = np.array(all_diag_preds)
        c_idx_arr = np.array(all_can_indices)

        total_tokens = len(y_bin)
        total_errors = int((y_bin == 1).sum())
        sub_mask = (y_diag != -100)
        total_subs = int(sub_mask.sum())

        # 1. Oracle Diagnosis Performance (evaluated strictly on gold substitutions)
        oracle_correct = int((p_diag[sub_mask] == y_diag[sub_mask]).sum())
        oracle_acc = oracle_correct / total_subs if total_subs > 0 else 0.0

        # 2. Binary AP
        ap = compute_pr_auc(y_bin, p_bin)

        # 3. Threshold Sweep & Pipeline Metrics
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
        sweep_metrics = []
        best_f1 = -1.0
        best_tau = 0.5

        for tau in thresholds:
            pred_bin = (p_bin >= tau).astype(int)
            tp = int(((pred_bin == 1) & (y_bin == 1)).sum())
            fp = int(((pred_bin == 1) & (y_bin == 0)).sum())
            tn = int(((pred_bin == 0) & (y_bin == 0)).sum())
            fn = int(((pred_bin == 0) & (y_bin == 1)).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau

            # Pipeline & End-to-end metrics
            pred_error_mask = (pred_bin == 1)
            num_pred_errors = int(pred_error_mask.sum())

            # Gold substitutions correctly detected as errors and correctly diagnosed
            e2e_correct = int((pred_error_mask & sub_mask & (p_diag == y_diag)).sum())
            e2e_recall = e2e_correct / total_subs if total_subs > 0 else 0.0
            e2e_precision = e2e_correct / num_pred_errors if num_pred_errors > 0 else 0.0
            e2e_f1 = 2 * e2e_precision * e2e_recall / (e2e_precision + e2e_recall) if (e2e_precision + e2e_recall) > 0 else 0.0

            # Error propagation
            missed_subs = int(((pred_bin == 0) & sub_mask).sum())
            spurious_diag_invocations = fp  # Binary detector triggered error on clean phone

            # Diagnosis accuracy on detected substitutions
            detected_subs = int((pred_error_mask & sub_mask).sum())
            diag_acc_on_detected = e2e_correct / detected_subs if detected_subs > 0 else 0.0

            sweep_metrics.append({
                "threshold": tau,
                "binary_tp": tp,
                "binary_fp": fp,
                "binary_tn": tn,
                "binary_fn": fn,
                "binary_precision": round(prec, 4),
                "binary_recall": round(rec, 4),
                "binary_f1": round(f1, 4),
                "predicted_errors_count": num_pred_errors,
                "missed_substitutions_fn": missed_subs,
                "spurious_triggers_fp": spurious_diag_invocations,
                "detected_substitutions_count": detected_subs,
                "e2e_correct_substitutions": e2e_correct,
                "e2e_recall": round(e2e_recall, 4),
                "e2e_precision": round(e2e_precision, 4),
                "e2e_f1": round(e2e_f1, 4),
                "diagnosis_acc_on_detected_subs": round(diag_acc_on_detected, 4)
            })

        # Find metrics at standard 0.50 and optimal tau
        metric_at_05 = next(m for m in sweep_metrics if m["threshold"] == 0.50)
        metric_at_opt = next(m for m in sweep_metrics if m["threshold"] == best_tau)

        pipeline_results[cond] = {
            "total_tokens": total_tokens,
            "total_gold_errors": total_errors,
            "total_gold_substitutions": total_subs,
            "binary_average_precision": round(ap, 4),
            "oracle_diagnosis": {
                "correct": oracle_correct,
                "total": total_subs,
                "accuracy": round(oracle_acc, 4)
            },
            "operating_point_standard_05": metric_at_05,
            "operating_point_optimal_f1": metric_at_opt,
            "threshold_sweep": sweep_metrics
        }

    report = {
        "metadata": {
            "evaluation_date": "2026-10-25",
            "protocol_gate": "RQ1-TRN-01B1-R2 Task D",
            "holdout_speakers": ["SVBI", "ZHAA"],
            "model_checkpoint": ckpt_path
        },
        "conditions": pipeline_results,
        "key_pipeline_comparisons": {
            "oracle_diagnosis_accuracy": {
                "matched": pipeline_results["matched"]["oracle_diagnosis"]["accuracy"],
                "mismatched": pipeline_results["mismatched"]["oracle_diagnosis"]["accuracy"],
                "silence": pipeline_results["silence"]["oracle_diagnosis"]["accuracy"]
            },
            "binary_average_precision": {
                "matched": pipeline_results["matched"]["binary_average_precision"],
                "mismatched": pipeline_results["mismatched"]["binary_average_precision"],
                "silence": pipeline_results["silence"]["binary_average_precision"]
            },
            "end_to_end_f1_at_optimal_tau": {
                "matched": pipeline_results["matched"]["operating_point_optimal_f1"]["e2e_f1"],
                "mismatched": pipeline_results["mismatched"]["operating_point_optimal_f1"]["e2e_f1"],
                "silence": pipeline_results["silence"]["operating_point_optimal_f1"]["e2e_f1"]
            },
            "end_to_end_recall_at_standard_05": {
                "matched": pipeline_results["matched"]["operating_point_standard_05"]["e2e_recall"],
                "mismatched": pipeline_results["mismatched"]["operating_point_standard_05"]["e2e_recall"],
                "silence": pipeline_results["silence"]["operating_point_standard_05"]["e2e_recall"]
            }
        },
        "error_propagation_summary": {
            "matched_at_05": {
                "gold_substitutions": 252,
                "missed_by_binary_detector": pipeline_results["matched"]["operating_point_standard_05"]["missed_substitutions_fn"],
                "missed_fraction": round(pipeline_results["matched"]["operating_point_standard_05"]["missed_substitutions_fn"] / 252.0, 4),
                "spurious_triggers_on_clean_phones": pipeline_results["matched"]["operating_point_standard_05"]["spurious_triggers_fp"],
                "e2e_correct": pipeline_results["matched"]["operating_point_standard_05"]["e2e_correct_substitutions"]
            },
            "matched_at_opt_02": {
                "threshold": pipeline_results["matched"]["operating_point_optimal_f1"]["threshold"],
                "gold_substitutions": 252,
                "missed_by_binary_detector": pipeline_results["matched"]["operating_point_optimal_f1"]["missed_substitutions_fn"],
                "missed_fraction": round(pipeline_results["matched"]["operating_point_optimal_f1"]["missed_substitutions_fn"] / 252.0, 4),
                "spurious_triggers_on_clean_phones": pipeline_results["matched"]["operating_point_optimal_f1"]["spurious_triggers_fp"],
                "e2e_correct": pipeline_results["matched"]["operating_point_optimal_f1"]["e2e_correct_substitutions"]
            }
        }
    }

    out_dir = Path("manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "task_d_pipeline_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nTask D Pipeline Evaluation complete. Saved to: {out_file}")
    print(f"Matched Oracle Diag Acc: {report['key_pipeline_comparisons']['oracle_diagnosis_accuracy']['matched']:.2%}")
    print(f"Mismatched Oracle Diag Acc: {report['key_pipeline_comparisons']['oracle_diagnosis_accuracy']['mismatched']:.2%}")
    print(f"Matched Binary AP: {report['key_pipeline_comparisons']['binary_average_precision']['matched']:.4f}")
    print(f"Mismatched Binary AP: {report['key_pipeline_comparisons']['binary_average_precision']['mismatched']:.4f}")
    print(f"Matched E2E F1 @ opt tau: {report['key_pipeline_comparisons']['end_to_end_f1_at_optimal_tau']['matched']:.4f}")
    print(f"Mismatched E2E F1 @ opt tau: {report['key_pipeline_comparisons']['end_to_end_f1_at_optimal_tau']['mismatched']:.4f}")

if __name__ == "__main__":
    main()
