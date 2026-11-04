#!/usr/bin/env python3
"""
LANGFENS — RQ1: Agent A — Precision Benchmark & PR Frontier Engine
Computes full precision-recall metrics, CAL-derived operating points across 7 FPR ceilings,
retrospective PR frontiers, false alarm rates per 1,000 phones, separate negative pools,
and full-gold vs aligned accounting for all 4 configurations under Protocol Z and O.
Outputs:
  - PRECISION_RECALL_FRONTIER.json
  - Generates comprehensive tables for PRECISION_BENCHMARK.md
"""

import json
import math
import os
import sys
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BENCH_CACHE = '/home/khoa/.cache/langfens/psup-benchmark-v1/recovered-final-v2-20261104'

CEILINGS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]

def cal_threshold(correct_scores, ceiling):
    descending = sorted(correct_scores, reverse=True)
    allowed = math.floor(ceiling * len(descending))
    return math.nextafter(descending[allowed], math.inf)

def evaluate_operating_point(val_gold, score_key, threshold, positive_target=2):
    """
    Evaluates an operating point against all gold tokens (accounting for abstentions).
    positive_target: 2 for omission, 1 for substitution, None for binary error.
    """
    # Negative pool 1: Correct tokens (target == 0)
    # Negative pool 2: Substitution tokens (target == 1) for omission task
    # Positives: target == positive_target
    
    positives = [r for r in val_gold if (r['operation_target'] == positive_target if positive_target is not None else r['binary_target'] == 1)]
    correct_negs = [r for r in val_gold if r['operation_target'] == 0]
    sub_negs = [r for r in val_gold if r['operation_target'] == 1] if positive_target == 2 else []
    
    total_pos = len(positives)
    total_cor_neg = len(correct_negs)
    total_sub_neg = len(sub_negs)
    
    # Alarms: token has finite score >= threshold
    def is_alarm(r):
        s = r.get(score_key)
        return s is not None and s >= threshold
    
    tp = sum(1 for r in positives if is_alarm(r))
    fn = total_pos - tp
    
    fp_cor = sum(1 for r in correct_negs if is_alarm(r))
    tn_cor = total_cor_neg - fp_cor
    
    fp_sub = sum(1 for r in sub_negs if is_alarm(r))
    tn_sub = total_sub_neg - fp_sub
    
    total_alarms = tp + fp_cor + fp_sub
    
    # Subset precision excludes substitutions (only considers correct negatives)
    subset_prec = tp / (tp + fp_cor) if (tp + fp_cor) > 0 else 0.0
    # Deployed precision includes all false alarms across both correct and substitutions
    deployed_prec = tp / total_alarms if total_alarms > 0 else 0.0
    
    recall = tp / total_pos if total_pos > 0 else 0.0
    subset_f1 = 2 * tp / (2 * tp + fp_cor + fn) if (2 * tp + fp_cor + fn) > 0 else 0.0
    deployed_f1 = 2 * tp / (2 * tp + fp_cor + fp_sub + fn) if (2 * tp + fp_cor + fp_sub + fn) > 0 else 0.0
    
    val_fpr_cor = fp_cor / total_cor_neg if total_cor_neg > 0 else 0.0
    fa_per_1000 = (fp_cor + fp_sub) / len(val_gold) * 1000.0 if len(val_gold) > 0 else 0.0
    
    unanswered_pos = sum(1 for r in positives if r.get(score_key) is None)
    unanswered_cor = sum(1 for r in correct_negs if r.get(score_key) is None)
    
    return {
        'threshold': threshold,
        'TP': tp,
        'FN': fn,
        'FP_cor': fp_cor,
        'TN_cor': tn_cor,
        'FP_sub': fp_sub,
        'TN_sub': tn_sub,
        'total_alarms': total_alarms,
        'recall': recall,
        'subset_precision': subset_prec,
        'deployed_precision': deployed_prec,
        'subset_f1': subset_f1,
        'deployed_f1': deployed_f1,
        'val_fpr_correct': val_fpr_cor,
        'false_alarms_per_1000': fa_per_1000,
        'unanswered_positives': unanswered_pos,
        'unanswered_correct': unanswered_cor
    }

def compute_pr_curve(val_gold, score_key, positive_target=2):
    """
    Computes dense precision-recall curve across all aligned positive and negative scores.
    """
    eligible = [r for r in val_gold if r.get(score_key) is not None and (
        (r['operation_target'] in (0, positive_target)) if positive_target is not None else True
    )]
    eligible.sort(key=lambda r: r[score_key], reverse=True)
    
    total_pos = sum(1 for r in val_gold if (r['operation_target'] == positive_target if positive_target is not None else r['binary_target'] == 1))
    
    curve = []
    tp = 0
    fp = 0
    for idx, r in enumerate(eligible):
        is_pos = (r['operation_target'] == positive_target) if positive_target is not None else (r['binary_target'] == 1)
        if is_pos:
            tp += 1
        else:
            fp += 1
        
        # Check if next score is identical (handle ties)
        if idx < len(eligible) - 1 and eligible[idx + 1][score_key] == r[score_key]:
            continue
            
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_val = tp / total_pos if total_pos > 0 else 0.0
        f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0.0
        
        curve.append({
            'threshold': r[score_key],
            'precision': p,
            'recall': r_val,
            'f1': f1,
            'tp': tp,
            'fp': fp
        })
    return curve

def main():
    print("Agent A: Computing precision benchmarks and PR frontiers...")
    
    configurations = [
        ('t_op_fixed', 't_op_fixed_budget_3600', True),
        ('b_control_fixed', 'b_control_fixed_budget_3600', False),
        ('t_op_best', 't_op_best', True),
        ('b_control_best', 'b_control_best', False)
    ]
    
    protocols = ['protocol_z', 'protocol_o']
    
    frontier_output = {
        'metadata': {
            'benchmark_cache': BENCH_CACHE,
            'description': 'Full Precision-Recall Frontier and Calibrated Operating Points for LANGFENS RQ1',
            'ceilings_evaluated': CEILINGS,
            'precision_targets_evaluated': [0.30, 0.40, 0.50, 0.60, 0.70]
        },
        'configurations': {}
    }
    
    for cfg_key, folder, is_treatment in configurations:
        frontier_output['configurations'][cfg_key] = {}
        for prot in protocols:
            cal_path = os.path.join(BENCH_CACHE, folder, prot, 'calibration', 'offline_token_evaluation.jsonl')
            val_path = os.path.join(BENCH_CACHE, folder, prot, 'offline_token_evaluation.jsonl')
            
            with open(cal_path) as f:
                cal_rows = [json.loads(line) for line in f]
            with open(val_path) as f:
                val_rows = [json.loads(line) for line in f]
                
            cal_correct = [r for r in cal_rows if r.get('mapping_status') == 'aligned' and r.get('operation_target') == 0]
            val_gold = [r for r in val_rows if r.get('gold_canonical_index') is not None]
            
            families = [('omission_legacy', 's_error')]
            if is_treatment:
                families.append(('omission_dedicated', 's_omit'))
                
            cfg_prot_data = {}
            for fam_name, score_key in families:
                correct_scores = [r[score_key] for r in cal_correct if r.get(score_key) is not None]
                
                # 1. Operating points at 7 CAL ceilings
                ceiling_points = {}
                for c in CEILINGS:
                    tau = cal_threshold(correct_scores, c)
                    pt = evaluate_operating_point(val_gold, score_key, tau, positive_target=2)
                    pt['nominal_CAL_ceiling'] = c
                    pt['cal_correct_count'] = len(correct_scores)
                    pt['cal_achieved_fpr'] = sum(s >= tau for s in correct_scores) / len(correct_scores)
                    ceiling_points[str(c)] = pt
                    
                # 2. Retrospective PR curve and precision targets
                curve = compute_pr_curve(val_gold, score_key, positive_target=2)
                
                max_prec_pt = max(curve, key=lambda x: x['precision']) if curve else None
                
                target_milestones = {}
                for target_p in [0.30, 0.40, 0.50, 0.60, 0.70]:
                    eligible_pts = [pt for pt in curve if pt['precision'] >= target_p]
                    if eligible_pts:
                        best_at_target = max(eligible_pts, key=lambda x: x['recall'])
                        target_milestones[str(target_p)] = {
                            'attainable': True,
                            'max_recall': best_at_target['recall'],
                            'achieved_precision': best_at_target['precision'],
                            'threshold': best_at_target['threshold'],
                            'tp': best_at_target['tp'],
                            'fp': best_at_target['fp']
                        }
                    else:
                        target_milestones[str(target_p)] = {
                            'attainable': False,
                            'max_recall': 0.0,
                            'achieved_precision': 0.0,
                            'threshold': None,
                            'tp': 0,
                            'fp': 0
                        }
                        
                cfg_prot_data[fam_name] = {
                    'score_key': score_key,
                    'calibrated_ceilings': ceiling_points,
                    'retrospective_targets': target_milestones,
                    'max_retrospective_precision': max_prec_pt,
                    'pr_curve_sample_points': curve[::max(1, len(curve)//50)] # sample ~50 points for storage
                }
                
            frontier_output['configurations'][cfg_key][prot] = cfg_prot_data
            
    out_json = os.path.join(ROOT_DIR, 'PRECISION_RECALL_FRONTIER.json')
    with open(out_json, 'w') as f:
        json.dump(frontier_output, f, indent=2)
    print(f"Saved {out_json} ({os.path.getsize(out_json)} bytes).")

if __name__ == '__main__':
    main()
