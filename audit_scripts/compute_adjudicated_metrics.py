#!/usr/bin/env python3
"""
LANGFENS — RQ1: Coordinator — Adjudicated Metrics Engine
Computes Horvitz-Thompson stratified weighted estimates for precision, recall, and false alarms
using the versioned review ledger ANNOTATION_REVIEW_LEDGER.jsonl.
Strictly separates:
  1. Confirmed model false positives
  2. Confirmed original annotation disagreements
  3. Cases with insufficient evidence / ambiguity
Produces conservative sensitivity bounds and standard errors.
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LEDGER_PATH = os.path.join(ROOT_DIR, 'ANNOTATION_REVIEW_LEDGER.jsonl')
MANIFEST_PATH = os.path.join(ROOT_DIR, 'CASE_MANIFEST.json')

def main():
    print("Coordinator: Computing Horvitz-Thompson Adjudicated Metrics...")
    
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    sampling_spec = manifest['sampling_strata_spec']
    
    with open(LEDGER_PATH) as f:
        ledger = [json.loads(line) for line in f]
        
    by_stratum = defaultdict(list)
    for r in ledger:
        by_stratum[r['stratum']].append(r)
        
    print(f"Loaded {len(ledger)} reviewed cases across {len(by_stratum)} strata.")
    
    # Stratum population sizes and weights
    stratum_stats = {}
    for st_name, spec in sampling_spec.items():
        sample_cases = by_stratum[st_name]
        n_sample = len(sample_cases)
        n_pop = spec['population_count']
        weight = spec['selection_weight']
        
        # Count verdicts in sample
        adj_verdicts = Counter(c['audit_results']['adjudicated_verdict'] for c in sample_cases)
        disp_cats = Counter(c['audit_results']['disposition_category'] for c in sample_cases)
        
        stratum_stats[st_name] = {
            'N_pop': n_pop,
            'n_sample': n_sample,
            'weight': weight,
            'adjudicated_verdicts': dict(adj_verdicts),
            'disposition_categories': dict(disp_cats)
        }
        
    # Population alarm strata: S1, S2, S3, S4, S5
    alarm_strata = ['S1_FP_high_conf', 'S2_FP_mid_conf', 'S3_FP_low_conf', 'S4_FP_substitution', 'S5_TP']
    total_deployed_alarms = sum(stratum_stats[st]['N_pop'] for st in alarm_strata) # 1,209
    
    # Scenario 1: Original Gold Baseline
    orig_tp = stratum_stats['S5_TP']['N_pop'] # 82
    orig_deployed_prec = orig_tp / total_deployed_alarms # 82 / 1209 = 0.0678
    orig_subset_prec = orig_tp / (orig_tp + sum(stratum_stats[st]['N_pop'] for st in ['S1_FP_high_conf', 'S2_FP_mid_conf', 'S3_FP_low_conf'])) # 82 / 1150 = 0.0713
    
    # Scenario 2: Strict Adjudication (Only adjudicated 'omission' counts as true positive)
    strict_tp_est = 0.0
    strict_var_tp = 0.0
    
    # Scenario 3: Adjusted Pronunciation Issue (Both 'omission' and 'reduction_ambiguity' count as genuine acoustic events)
    lenient_tp_est = 0.0
    lenient_var_tp = 0.0
    
    # Disagreement count: Gold said correct/sub, but adjudication said omission or reduction
    annot_disagree_est = 0.0
    annot_disagree_var = 0.0
    
    # Genuine model FP: Gold said correct/sub, and adjudication confirmed correct
    model_fp_est = 0.0
    model_fp_var = 0.0
    
    for st in alarm_strata:
        cases = by_stratum[st]
        n_h = len(cases)
        N_h = stratum_stats[st]['N_pop']
        
        # Strict TP indicators: adjudicated == 'omission'
        y_strict = [1.0 if c['audit_results']['adjudicated_verdict'] == 'omission' else 0.0 for c in cases]
        p_strict = np.mean(y_strict)
        strict_tp_est += N_h * p_strict
        if n_h > 1:
            s2_strict = np.var(y_strict, ddof=1)
            strict_var_tp += (N_h ** 2) * (1 - n_h / N_h) * (s2_strict / n_h)
            
        # Lenient TP indicators: adjudicated in ('omission', 'reduction_ambiguity')
        y_lenient = [1.0 if c['audit_results']['adjudicated_verdict'] in ('omission', 'reduction_ambiguity') else 0.0 for c in cases]
        p_lenient = np.mean(y_lenient)
        lenient_tp_est += N_h * p_lenient
        if n_h > 1:
            s2_lenient = np.var(y_lenient, ddof=1)
            lenient_var_tp += (N_h ** 2) * (1 - n_h / N_h) * (s2_lenient / n_h)
            
        # Annotation disagreement indicators: disposition_category == 'confirmed_annotation_disagreement'
        y_disagree = [1.0 if c['audit_results']['disposition_category'] == 'confirmed_annotation_disagreement' else 0.0 for c in cases]
        p_disagree = np.mean(y_disagree)
        annot_disagree_est += N_h * p_disagree
        if n_h > 1:
            s2_disagree = np.var(y_disagree, ddof=1)
            annot_disagree_var += (N_h ** 2) * (1 - n_h / N_h) * (s2_disagree / n_h)
            
        # Model FP indicators: disposition_category in ('confirmed_model_fp', 'confirmed_model_fp_sub')
        y_mod_fp = [1.0 if c['audit_results']['disposition_category'] in ('confirmed_model_fp', 'confirmed_model_fp_sub') else 0.0 for c in cases]
        p_mod_fp = np.mean(y_mod_fp)
        model_fp_est += N_h * p_mod_fp
        if n_h > 1:
            s2_mod_fp = np.var(y_mod_fp, ddof=1)
            model_fp_var += (N_h ** 2) * (1 - n_h / N_h) * (s2_mod_fp / n_h)

    strict_se = math.sqrt(strict_var_tp)
    lenient_se = math.sqrt(lenient_var_tp)
    
    strict_prec_est = strict_tp_est / total_deployed_alarms
    strict_prec_ci_low = max(0.0, (strict_tp_est - 1.96 * strict_se) / total_deployed_alarms)
    strict_prec_ci_high = min(1.0, (strict_tp_est + 1.96 * strict_se) / total_deployed_alarms)
    
    lenient_prec_est = lenient_tp_est / total_deployed_alarms
    lenient_prec_ci_low = max(0.0, (lenient_tp_est - 1.96 * lenient_se) / total_deployed_alarms)
    lenient_prec_ci_high = min(1.0, (lenient_tp_est + 1.96 * lenient_se) / total_deployed_alarms)

    print("\n--- COORDINATOR ADJUDICATED ESTIMATES ---")
    print(f"Total Deployed Alarms Evaluated: {total_deployed_alarms}")
    print(f"Original Gold Precision: {orig_deployed_prec*100:.2f}% (82 / 1209)")
    print(f"\n1. Strict Adjudication (Full Omission Consensus Only):")
    print(f"   Estimated True Omissions: {strict_tp_est:.1f} ± {1.96*strict_se:.1f}")
    print(f"   Adjudicated Deployed Precision: {strict_prec_est*100:.2f}% [95% CI: {strict_prec_ci_low*100:.2f}%, {strict_prec_ci_high*100:.2f}%]")
    print(f"\n2. Broad Pronunciation Event (Omission + Severe Unreleased Reduction):")
    print(f"   Estimated Genuine Pronunciation Events: {lenient_tp_est:.1f} ± {1.96*lenient_se:.1f}")
    print(f"   Adjudicated Deployed Precision: {lenient_prec_est*100:.2f}% [95% CI: {lenient_prec_ci_low*100:.2f}%, {lenient_prec_ci_high*100:.2f}%]")
    print(f"\n3. Breakdown of 1,209 Deployed Alarms:")
    print(f"   - Confirmed Model False Alarms: {model_fp_est:.1f} ({model_fp_est/total_deployed_alarms*100:.1f}%)")
    print(f"   - Confirmed Annotation Disagreements (Unreleased/Reduced/Omitted in gold): {annot_disagree_est:.1f} ({annot_disagree_est/total_deployed_alarms*100:.1f}%)")
    print(f"   - Confirmed True Positives (Gold Omissions): {orig_tp} ({orig_tp/total_deployed_alarms*100:.1f}%)")
    print(f"   - Ambiguous / Insufficient Evidence: {total_deployed_alarms - model_fp_est - annot_disagree_est - orig_tp:.1f}")
    
    out_adjudicated = os.path.join(ROOT_DIR, 'audit_scripts', 'adjudicated_metrics_summary.json')
    with open(out_adjudicated, 'w') as f:
        json.dump({
            'total_deployed_alarms': total_deployed_alarms,
            'original_gold_deployed_precision': round(orig_deployed_prec, 6),
            'strict_adjudication': {
                'estimated_tp': round(strict_tp_est, 2),
                'se': round(strict_se, 2),
                'precision_point_estimate': round(strict_prec_est, 6),
                'ci_95': [round(strict_prec_ci_low, 6), round(strict_prec_ci_high, 6)]
            },
            'lenient_adjudication': {
                'estimated_tp': round(lenient_tp_est, 2),
                'se': round(lenient_se, 2),
                'precision_point_estimate': round(lenient_prec_est, 6),
                'ci_95': [round(lenient_prec_ci_low, 6), round(lenient_prec_ci_high, 6)]
            },
            'alarm_decomposition': {
                'confirmed_model_fp': round(model_fp_est, 1),
                'confirmed_annotation_disagreement': round(annot_disagree_est, 1),
                'original_true_positives': orig_tp,
                'ambiguous_or_unresolved': round(total_deployed_alarms - model_fp_est - annot_disagree_est - orig_tp, 1)
            },
            'stratum_details': stratum_stats
        }, f, indent=2)
    print(f"Saved adjudicated summary to {out_adjudicated}.")

if __name__ == '__main__':
    main()
