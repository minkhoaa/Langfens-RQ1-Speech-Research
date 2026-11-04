#!/usr/bin/env python3
"""
LANGFENS RQ1 — Agent A: Decision Policy & Precision Benchmark Harness
Evaluates frozen T-OP predictions across multiple CAL-derived decision policies:
1. Global CAL FPR ceilings (0.1%, 0.5%, 1%, 2%, 5%, 10%, 15%)
2. Per-phoneme CAL FPR thresholds (with fallback for sparse phones)
3. Phoneme-identity suppression (/d/, /t/, /r/ suppression at various strengths)
4. Margin-based conservative reporting (s_omit - s_sub >= delta, s_omit / s_error >= rho)
5. Selective prediction with abstention
6. Matched false-alarm budget points (FA/1k = 5, 10, 15, 20)
7. Matched recall points (Recall = 10%, 20%, 30%, 40%, 50%)
Outputs:
- POLICY_COMPARISON.csv
- PRECISION_RECALL_FRONTIER.json
- policy_summary.json (for report synthesis)
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BENCH_CACHE = '/home/khoa/.cache/langfens/psup-benchmark-v1/recovered-final-v2-20261104'

CAL_VAL_DIR = os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600', 'protocol_z')
CAL_PATH = os.path.join(CAL_VAL_DIR, 'calibration', 'offline_token_evaluation.jsonl')
VAL_PATH = os.path.join(CAL_VAL_DIR, 'offline_token_evaluation.jsonl')

def cal_threshold(scores, ceiling):
    descending = sorted(scores, reverse=True)
    if not descending:
        return 1.0
    allowed = math.floor(ceiling * len(descending))
    allowed = min(allowed, len(descending) - 1)
    return math.nextafter(descending[allowed], math.inf)

def evaluate_policy(val_records, decision_fn, policy_name, threshold_desc, policy_type):
    """
    decision_fn(r) -> bool (is omission alarm) or 'abstain'
    val_records contains all 9,993 VAL records (9,919 aligned + 37 unmapped abstentions + 37 prompt extras).
    Evaluation contract:
    - Gold canonical positions = 9,956 (aligned + unmapped gold abstentions). Excludes prompt_extra.
    - Total gold omissions = 145.
    - Total gold substitutions = 1,012.
    - Total gold correct = 8,799.
    """
    gold_records = [r for r in val_records if r.get('mapping_status') != 'prompt_extra']
    total_gold = len(gold_records)  # 9,956
    assert total_gold == 9956, f"Expected 9956 gold records, got {total_gold}"
    
    tp = 0
    fp_cor = 0
    fp_sub = 0
    fn = 0
    tn_cor = 0
    tn_sub = 0
    abstentions = 0
    covered = 0
    
    tp_njs = 0
    tp_rrbi = 0
    fp_njs = 0
    fp_rrbi = 0
    fn_njs = 0
    fn_rrbi = 0
    
    alarms_by_phone = defaultdict(int)
    tp_by_phone = defaultdict(int)
    fp_by_phone = defaultdict(int)
    
    for r in gold_records:
        op = r.get('operation_target')  # 0=correct, 1=substitution, 2=omission
        phone = r.get('canonical_phone', 'UNK')
        spk = r.get('speaker_id', 'UNK')
        
        # Check mapping status for unmapped gold abstention
        if r.get('mapping_status') == 'unmapped_gold_abstention':
            decision = False
            abstentions += 1
        else:
            decision = decision_fn(r)
            if decision == 'abstain':
                decision = False
                abstentions += 1
            else:
                covered += 1
                
        is_alarm = bool(decision)
        if is_alarm:
            alarms_by_phone[phone] += 1
            
        if op == 2:  # Omission positive
            if is_alarm:
                tp += 1
                tp_by_phone[phone] += 1
                if spk == 'NJS': tp_njs += 1
                else: tp_rrbi += 1
            else:
                fn += 1
                if spk == 'NJS': fn_njs += 1
                else: fn_rrbi += 1
        elif op == 0:  # Correct negative
            if is_alarm:
                fp_cor += 1
                fp_by_phone[phone] += 1
                if spk == 'NJS': fp_njs += 1
                else: fp_rrbi += 1
            else:
                tn_cor += 1
        elif op == 1:  # Substitution negative
            if is_alarm:
                fp_sub += 1
                fp_by_phone[phone] += 1
                if spk == 'NJS': fp_njs += 1
                else: fp_rrbi += 1
            else:
                tn_sub += 1

    total_alarms = tp + fp_cor + fp_sub
    deployed_prec = tp / total_alarms if total_alarms > 0 else 0.0
    subset_prec = tp / (tp + fp_cor) if (tp + fp_cor) > 0 else 0.0
    recall = tp / 145.0
    recall_aligned = tp / 137.0
    deployed_f1 = (2 * deployed_prec * recall / (deployed_prec + recall)) if (deployed_prec + recall) > 0 else 0.0
    fa_per_1000 = (fp_cor + fp_sub) / total_gold * 1000.0
    cor_fpr = fp_cor / 8799.0
    
    # Speaker metrics
    prec_njs = tp_njs / (tp_njs + fp_njs) if (tp_njs + fp_njs) > 0 else 0.0
    prec_rrbi = tp_rrbi / (tp_rrbi + fp_rrbi) if (tp_rrbi + fp_rrbi) > 0 else 0.0
    recall_njs = tp_njs / (tp_njs + fn_njs) if (tp_njs + fn_njs) > 0 else 0.0
    recall_rrbi = tp_rrbi / (tp_rrbi + fn_rrbi) if (tp_rrbi + fn_rrbi) > 0 else 0.0

    return {
        'policy_name': policy_name,
        'policy_type': policy_type,
        'threshold_desc': threshold_desc,
        'TP': tp,
        'FP_cor': fp_cor,
        'FP_sub': fp_sub,
        'FN': fn,
        'total_alarms': total_alarms,
        'abstentions': abstentions,
        'covered': covered,
        'deployed_precision': round(deployed_prec, 6),
        'subset_precision': round(subset_prec, 6),
        'all_gold_recall': round(recall, 6),
        'aligned_recall': round(recall_aligned, 6),
        'deployed_f1': round(deployed_f1, 6),
        'false_alarms_per_1000': round(fa_per_1000, 2),
        'correct_token_fpr': round(cor_fpr, 6),
        'NJS_precision': round(prec_njs, 4),
        'NJS_recall': round(recall_njs, 4),
        'RRBI_precision': round(prec_rrbi, 4),
        'RRBI_recall': round(recall_rrbi, 4),
        'd_alarms': alarms_by_phone.get('D', 0),
        't_alarms': alarms_by_phone.get('T', 0),
        'r_alarms': alarms_by_phone.get('R', 0)
    }

def main():
    print("Agent A: Running Precision & Decision-Policy Benchmark...")
    
    with open(CAL_PATH) as f:
        cal_records = [json.loads(line) for line in f]
    with open(VAL_PATH) as f:
        val_records = [json.loads(line) for line in f]
        
    print(f"Loaded {len(cal_records)} CAL records and {len(val_records)} VAL records.")
    
    # 1. CAL-only analysis
    cal_gold = [r for r in cal_records if r.get('mapping_status') != 'prompt_extra']
    cal_cor_scores = [r['s_omit'] for r in cal_gold if r.get('operation_target') == 0 and r.get('s_omit') is not None]
    
    # Per-phone CAL scores
    cal_phone_cor_scores = defaultdict(list)
    for r in cal_gold:
        if r.get('operation_target') == 0 and r.get('s_omit') is not None:
            cal_phone_cor_scores[r['canonical_phone']].append(r['s_omit'])
            
    policies_results = []
    
    # -------------------------------------------------------------------------
    # Policy 1: Global CAL Ceilings
    # -------------------------------------------------------------------------
    ceilings = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
    for c in ceilings:
        tau = cal_threshold(cal_cor_scores, c)
        res = evaluate_policy(
            val_records,
            lambda r, t=tau: r.get('s_omit') is not None and r['s_omit'] >= t,
            f"Global CAL {c*100:4.1f}% Ceiling",
            f"tau={tau:.6f}",
            "global_threshold"
        )
        policies_results.append(res)
        
    # Baseline reference at 15% ceiling
    base_tau = cal_threshold(cal_cor_scores, 0.15)
    
    # -------------------------------------------------------------------------
    # Policy 2: Phoneme-Identity Suppression on Baseline
    # -------------------------------------------------------------------------
    # 2a: Mute /d/, /t/, /r/ completely
    res_mute_dtr = evaluate_policy(
        val_records,
        lambda r, t=base_tau: r.get('s_omit') is not None and r['s_omit'] >= t and r.get('canonical_phone') not in ('D', 'T', 'R'),
        "Baseline + Mute {D, T, R}",
        f"tau={base_tau:.6f}, D/T/R muted",
        "phoneme_suppression"
    )
    policies_results.append(res_mute_dtr)
    
    # 2b: Mute /d/ and /t/ only (coronal stops)
    res_mute_dt = evaluate_policy(
        val_records,
        lambda r, t=base_tau: r.get('s_omit') is not None and r['s_omit'] >= t and r.get('canonical_phone') not in ('D', 'T'),
        "Baseline + Mute {D, T}",
        f"tau={base_tau:.6f}, D/T muted",
        "phoneme_suppression"
    )
    policies_results.append(res_mute_dt)
    
    # 2c: Mute /r/ only (rhotic)
    res_mute_r = evaluate_policy(
        val_records,
        lambda r, t=base_tau: r.get('s_omit') is not None and r['s_omit'] >= t and r.get('canonical_phone') != 'R',
        "Baseline + Mute {R}",
        f"tau={base_tau:.6f}, R muted",
        "phoneme_suppression"
    )
    policies_results.append(res_mute_r)

    # -------------------------------------------------------------------------
    # Policy 3: Per-Phoneme CAL FPR Thresholds
    # -------------------------------------------------------------------------
    # For each phone with >= 30 CAL correct samples, derive phone-specific threshold at 15% FPR
    # For other phones, use global 15% threshold
    phone_tau_15 = {}
    phone_tau_5 = {}
    phone_tau_1 = {}
    for p, scores in cal_phone_cor_scores.items():
        if len(scores) >= 30:
            phone_tau_15[p] = cal_threshold(scores, 0.15)
            phone_tau_5[p] = cal_threshold(scores, 0.05)
            phone_tau_1[p] = cal_threshold(scores, 0.01)
        else:
            phone_tau_15[p] = base_tau
            phone_tau_5[p] = cal_threshold(cal_cor_scores, 0.05)
            phone_tau_1[p] = cal_threshold(cal_cor_scores, 0.01)
            
    res_per_phone_15 = evaluate_policy(
        val_records,
        lambda r: r.get('s_omit') is not None and r['s_omit'] >= phone_tau_15.get(r.get('canonical_phone'), base_tau),
        "Per-Phoneme CAL 15% Thresholds",
        "Phone-specific tau at 15% CAL FPR",
        "per_phoneme_threshold"
    )
    policies_results.append(res_per_phone_15)
    
    res_per_phone_5 = evaluate_policy(
        val_records,
        lambda r: r.get('s_omit') is not None and r['s_omit'] >= phone_tau_5.get(r.get('canonical_phone'), cal_threshold(cal_cor_scores, 0.05)),
        "Per-Phoneme CAL 5% Thresholds",
        "Phone-specific tau at 5% CAL FPR",
        "per_phoneme_threshold"
    )
    policies_results.append(res_per_phone_5)

    # -------------------------------------------------------------------------
    # Policy 4: Asymmetric Thresholding (Conservative D/T/R)
    # -------------------------------------------------------------------------
    # Keep standard 15% ceiling for clean phones, but apply conservative 1% or 2% ceiling to D, T, R
    tau_1pct = cal_threshold(cal_cor_scores, 0.01)
    tau_2pct = cal_threshold(cal_cor_scores, 0.02)
    tau_5pct = cal_threshold(cal_cor_scores, 0.05)
    
    res_asym_1 = evaluate_policy(
        val_records,
        lambda r: r.get('s_omit') is not None and (
            r['s_omit'] >= tau_1pct if r.get('canonical_phone') in ('D', 'T', 'R') else r['s_omit'] >= base_tau
        ),
        "Asymmetric: 1% Ceiling on {D,T,R}, 15% on Others",
        f"tau_DTR={tau_1pct:.6f}, tau_other={base_tau:.6f}",
        "asymmetric_threshold"
    )
    policies_results.append(res_asym_1)
    
    res_asym_2 = evaluate_policy(
        val_records,
        lambda r: r.get('s_omit') is not None and (
            r['s_omit'] >= tau_2pct if r.get('canonical_phone') in ('D', 'T', 'R') else r['s_omit'] >= base_tau
        ),
        "Asymmetric: 2% Ceiling on {D,T,R}, 15% on Others",
        f"tau_DTR={tau_2pct:.6f}, tau_other={base_tau:.6f}",
        "asymmetric_threshold"
    )
    policies_results.append(res_asym_2)

    res_asym_5 = evaluate_policy(
        val_records,
        lambda r: r.get('s_omit') is not None and (
            r['s_omit'] >= tau_5pct if r.get('canonical_phone') in ('D', 'T', 'R') else r['s_omit'] >= base_tau
        ),
        "Asymmetric: 5% Ceiling on {D,T,R}, 15% on Others",
        f"tau_DTR={tau_5pct:.6f}, tau_other={base_tau:.6f}",
        "asymmetric_threshold"
    )
    policies_results.append(res_asym_5)

    # -------------------------------------------------------------------------
    # Policy 5: Margin-Based Conservative Reporting (CAL-Selected Delta)
    # -------------------------------------------------------------------------
    # On CAL, find delta such that s_omit - s_sub >= delta
    # Test delta in [0.0, 0.05, 0.10, 0.15]
    for delta in [0.02, 0.05, 0.10]:
        res_margin = evaluate_policy(
            val_records,
            lambda r, d=delta: r.get('s_omit') is not None and r['s_omit'] >= base_tau and (r['s_omit'] - r.get('s_sub', 0.0)) >= d,
            f"Margin Gating: Delta >= {delta:.2f}",
            f"tau={base_tau:.6f}, (s_omit - s_sub) >= {delta:.2f}",
            "margin_gating"
        )
        policies_results.append(res_margin)

    # Ratio gating: s_omit / s_error >= rho
    for rho in [0.6, 0.8]:
        res_ratio = evaluate_policy(
            val_records,
            lambda r, rh=rho: r.get('s_omit') is not None and r['s_omit'] >= base_tau and (
                (r['s_omit'] / max(r.get('s_error', 1e-6), 1e-6)) >= rh
            ),
            f"Ratio Gating: s_omit / s_error >= {rho:.1f}",
            f"tau={base_tau:.6f}, ratio >= {rho:.1f}",
            "ratio_gating"
        )
        policies_results.append(res_ratio)

    # -------------------------------------------------------------------------
    # Policy 6: Selective Prediction with Abstention
    # -------------------------------------------------------------------------
    # If s_omit >= tau, but |s_omit - s_sub| < 0.03, abstain from omission feedback
    res_abstain = evaluate_policy(
        val_records,
        lambda r: 'abstain' if (r.get('s_omit') is not None and r['s_omit'] >= base_tau and abs(r['s_omit'] - r.get('s_sub', 0.0)) < 0.03) else (r.get('s_omit') is not None and r['s_omit'] >= base_tau),
        "Selective Abstention (|s_omit - s_sub| < 0.03)",
        f"tau={base_tau:.6f}, abstain if margin < 0.03",
        "selective_abstention"
    )
    policies_results.append(res_abstain)

    # -------------------------------------------------------------------------
    # Matched False-Alarm Budgets: 5, 10, 15, 20 per 1,000 canonical phones
    # -------------------------------------------------------------------------
    # Determine the required threshold on CAL to hit exact FA budget
    # Target FA counts on CAL (8054 correct + 1462 sub = 9516 negatives):
    # FA / 9977 * 1000 = budget  => FA = budget * 9977 / 1000
    for target_fa in [5, 10, 15, 20]:
        target_cal_count = math.floor(target_fa * len(cal_gold) / 1000.0)
        # Sort all non-omission CAL scores descending
        cal_neg_scores = sorted([r['s_omit'] for r in cal_gold if r.get('operation_target') in (0, 1) and r.get('s_omit') is not None], reverse=True)
        tau_budget = math.nextafter(cal_neg_scores[target_cal_count], math.inf)
        res_budget = evaluate_policy(
            val_records,
            lambda r, t=tau_budget: r.get('s_omit') is not None and r['s_omit'] >= t,
            f"Matched Budget: FA <= {target_fa}/1k (CAL-Derived)",
            f"tau={tau_budget:.6f}",
            "matched_budget"
        )
        policies_results.append(res_budget)

    # Write POLICY_COMPARISON.csv
    csv_path = os.path.join(os.path.dirname(__file__), 'POLICY_COMPARISON.csv')
    fieldnames = [
        'policy_name', 'policy_type', 'threshold_desc', 'TP', 'FP_cor', 'FP_sub', 'FN',
        'total_alarms', 'abstentions', 'covered', 'deployed_precision', 'subset_precision',
        'all_gold_recall', 'aligned_recall', 'deployed_f1', 'false_alarms_per_1000',
        'correct_token_fpr', 'NJS_precision', 'NJS_recall', 'RRBI_precision', 'RRBI_recall',
        'd_alarms', 't_alarms', 'r_alarms'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in policies_results:
            writer.writerow({k: r[k] for k in fieldnames})
            
    # Also write to root directory if requested
    root_csv = os.path.join(ROOT_DIR, 'POLICY_COMPARISON.csv')
    with open(root_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in policies_results:
            writer.writerow({k: r[k] for k in fieldnames})

    # Write summary JSON
    summary_path = os.path.join(os.path.dirname(__file__), 'policy_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(policies_results, f, indent=2)
        
    print(f"Generated {len(policies_results)} policy evaluations. Saved to {csv_path} and {summary_path}.")
    
    # Print formatted comparison table
    print("\n" + "="*120)
    print(f"{'Policy Name':<42} | {'TP':<4} | {'FP_cor':<6} | {'FP_sub':<6} | {'Alarms':<6} | {'Prec(dep)':<9} | {'Recall':<7} | {'FA/1k':<6} | {'D/T/R Alarms':<14}")
    print("="*120)
    for p in policies_results:
        dtr_str = f"{p['d_alarms']}/{p['t_alarms']}/{p['r_alarms']}"
        print(f"{p['policy_name']:<42} | {p['TP']:<4} | {p['FP_cor']:<6} | {p['FP_sub']:<6} | {p['total_alarms']:<6} | {p['deployed_precision']:<9.4f} | {p['all_gold_recall']:<7.4f} | {p['false_alarms_per_1000']:<6.1f} | {dtr_str:<14}")
    print("="*120)

if __name__ == '__main__':
    main()
