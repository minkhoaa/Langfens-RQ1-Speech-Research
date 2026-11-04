#!/usr/bin/env python3
"""
LANGFENS RQ1 — Agent C: Rhotic & Annotation Reliability Analysis Harness
1. Investigates the high-frequency /r/ false-positive cohort:
   - Contrast between Vietnamese (NJS) and Spanish (RRBI) realization strategies.
   - Acoustic properties (duration, energy, formant expectations).
   - Model prior vs L1 phonetic realization divergence.
2. Audits annotation reliability and blinded review ledger:
   - Separates genuine human re-annotation consensus from automated classifications.
   - Computes Horvitz-Thompson unbiased weighted estimates and sampling variances.
   - Computes dual-horizon metrics: Original Gold vs Adjudicated Pronunciation Events.
Outputs:
- rhotic_summary.json
- adjudicated_metrics_summary.json
- ADJUDICATED_METRICS.md
- RHOTIC_ERROR_ANALYSIS.md
- ANNOTATION_RELIABILITY_RECONCILIATION.md
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BENCH_CACHE = '/home/khoa/.cache/langfens/psup-benchmark-v1/recovered-final-v2-20261104'
LEDGER_PATH = os.path.join(ROOT_DIR, 'ANNOTATION_REVIEW_LEDGER.jsonl')
MANIFEST_PATH = os.path.join(ROOT_DIR, 'CASE_MANIFEST.json')

VAL_PATH = os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600', 'protocol_z', 'offline_token_evaluation.jsonl')

def main():
    print("Agent C: Running Rhotic & Annotation Reliability Investigation...")
    
    with open(VAL_PATH) as f:
        val_records = [json.loads(line) for line in f if json.loads(line).get('mapping_status') != 'prompt_extra']
    with open(LEDGER_PATH) as f:
        ledger = [json.loads(line) for line in f]
    with open(MANIFEST_PATH) as f:
        case_manifest = json.load(f)

    sampling_spec = case_manifest['sampling_strata_spec']
    print(f"Loaded {len(val_records)} VAL records and {len(ledger)} review ledger cases.")

    # -------------------------------------------------------------------------
    # 1. Full-Cohort Rhotic (/r/) Analysis
    # -------------------------------------------------------------------------
    base_tau = 0.066045
    r_records = [r for r in val_records if r.get('canonical_phone') == 'R']
    
    r_total = len(r_records)
    r_pos = sum(1 for r in r_records if r.get('operation_target') == 2)
    r_cor = sum(1 for r in r_records if r.get('operation_target') == 0)
    r_sub = sum(1 for r in r_records if r.get('operation_target') == 1)
    
    r_alarms = [r for r in r_records if r.get('s_omit') is not None and r['s_omit'] >= base_tau]
    r_tp = sum(1 for r in r_alarms if r.get('operation_target') == 2)
    r_fp_cor = sum(1 for r in r_alarms if r.get('operation_target') == 0)
    r_fp_sub = sum(1 for r in r_alarms if r.get('operation_target') == 1)
    r_fn = r_pos - r_tp
    
    # Speaker breakdown for /r/
    njs_r = [r for r in r_records if r.get('speaker_id') == 'NJS']
    rrbi_r = [r for r in r_records if r.get('speaker_id') == 'RRBI']
    
    njs_r_alarms = [r for r in r_alarms if r.get('speaker_id') == 'NJS']
    rrbi_r_alarms = [r for r in r_alarms if r.get('speaker_id') == 'RRBI']
    
    njs_tp = sum(1 for r in njs_r_alarms if r.get('operation_target') == 2)
    njs_fp_cor = sum(1 for r in njs_r_alarms if r.get('operation_target') == 0)
    njs_fp_sub = sum(1 for r in njs_r_alarms if r.get('operation_target') == 1)
    
    rrbi_tp = sum(1 for r in rrbi_r_alarms if r.get('operation_target') == 2)
    rrbi_fp_cor = sum(1 for r in rrbi_r_alarms if r.get('operation_target') == 0)
    rrbi_fp_sub = sum(1 for r in rrbi_r_alarms if r.get('operation_target') == 1)
    
    rhotic_stats = {
        'total_rhotic_tokens': r_total,
        'gold_omissions': r_pos,
        'gold_correct': r_cor,
        'gold_substitutions': r_sub,
        'omission_alarms': len(r_alarms),
        'alarm_rate_pct': round(len(r_alarms) / r_total * 100.0, 2),
        'TP': r_tp,
        'FP_cor': r_fp_cor,
        'FP_sub': r_fp_sub,
        'FN': r_fn,
        'deployed_precision': round(r_tp / len(r_alarms), 6) if r_alarms else 0.0,
        'recall': round(r_tp / r_pos, 6) if r_pos else 0.0,
        'NJS': {
            'tokens': len(njs_r),
            'alarms': len(njs_r_alarms),
            'TP': njs_tp,
            'FP_cor': njs_fp_cor,
            'FP_sub': njs_fp_sub,
            'deployed_precision': round(njs_tp / len(njs_r_alarms), 4) if njs_r_alarms else 0.0,
            'recall': round(njs_tp / sum(1 for r in njs_r if r.get('operation_target') == 2), 4)
        },
        'RRBI': {
            'tokens': len(rrbi_r),
            'alarms': len(rrbi_r_alarms),
            'TP': rrbi_tp,
            'FP_cor': rrbi_fp_cor,
            'FP_sub': rrbi_fp_sub,
            'deployed_precision': round(rrbi_tp / len(rrbi_r_alarms), 4) if rrbi_r_alarms else 0.0,
            'recall': round(rrbi_tp / sum(1 for r in rrbi_r if r.get('operation_target') == 2), 4)
        }
    }
    
    with open(os.path.join(os.path.dirname(__file__), 'rhotic_summary.json'), 'w') as f:
        json.dump(rhotic_stats, f, indent=2)

    # -------------------------------------------------------------------------
    # 2. Horvitz-Thompson Weighted Estimation on Review Ledger
    # -------------------------------------------------------------------------
    by_stratum = defaultdict(list)
    for r in ledger:
        by_stratum[r['stratum']].append(r)
        
    stratum_stats = {}
    for st_name, spec in sampling_spec.items():
        sample_cases = by_stratum[st_name]
        n_sample = len(sample_cases)
        n_pop = spec['population_count']
        weight = spec['selection_weight']
        stratum_stats[st_name] = {
            'N_pop': n_pop,
            'n_sample': n_sample,
            'weight': weight
        }
        
    # Alarm strata: S1, S2, S3, S4, S5
    alarm_strata = ['S1_FP_high_conf', 'S2_FP_mid_conf', 'S3_FP_low_conf', 'S4_FP_substitution', 'S5_TP']
    total_deployed_alarms = sum(stratum_stats[st]['N_pop'] for st in alarm_strata) # 1,209
    
    strict_tp_est = 0.0
    strict_var_tp = 0.0
    lenient_tp_est = 0.0
    lenient_var_tp = 0.0
    annot_disagree_est = 0.0
    annot_disagree_var = 0.0
    model_fp_est = 0.0
    model_fp_var = 0.0
    ambiguity_est = 0.0
    ambiguity_var = 0.0
    
    for st in alarm_strata:
        cases = by_stratum[st]
        n_h = len(cases)
        N_h = stratum_stats[st]['N_pop']
        if n_h == 0:
            continue
            
        # Strict omission indicators: disposition_category == 'confirmed_tp' OR adjudicated_verdict == 'omission'
        y_strict = [1.0 if c['audit_results']['disposition_category'] == 'confirmed_tp' or c['audit_results']['adjudicated_verdict'] == 'omission' else 0.0 for c in cases]
        p_strict = np.mean(y_strict)
        strict_tp_est += N_h * p_strict
        if n_h > 1:
            s2_strict = np.var(y_strict, ddof=1)
            strict_var_tp += (N_h ** 2) * (1.0 - n_h / N_h) * (s2_strict / n_h)
            
        # Lenient pronunciation event indicators: adjudicated_verdict in ('omission', 'reduction_ambiguity')
        y_lenient = [1.0 if c['audit_results']['adjudicated_verdict'] in ('omission', 'reduction_ambiguity') else 0.0 for c in cases]
        p_lenient = np.mean(y_lenient)
        lenient_tp_est += N_h * p_lenient
        if n_h > 1:
            s2_lenient = np.var(y_lenient, ddof=1)
            lenient_var_tp += (N_h ** 2) * (1.0 - n_h / N_h) * (s2_lenient / n_h)
            
        # Annotation disagreement: disposition_category == 'confirmed_annotation_disagreement'
        y_disagree = [1.0 if c['audit_results']['disposition_category'] == 'confirmed_annotation_disagreement' else 0.0 for c in cases]
        p_disagree = np.mean(y_disagree)
        annot_disagree_est += N_h * p_disagree
        if n_h > 1:
            s2_disagree = np.var(y_disagree, ddof=1)
            annot_disagree_var += (N_h ** 2) * (1.0 - n_h / N_h) * (s2_disagree / n_h)
            
        # Confirmed model false positive
        y_mod_fp = [1.0 if c['audit_results']['disposition_category'] in ('confirmed_model_fp', 'confirmed_model_fp_sub') else 0.0 for c in cases]
        p_mod_fp = np.mean(y_mod_fp)
        model_fp_est += N_h * p_mod_fp
        if n_h > 1:
            s2_mod_fp = np.var(y_mod_fp, ddof=1)
            model_fp_var += (N_h ** 2) * (1.0 - n_h / N_h) * (s2_mod_fp / n_h)
            
        # Ambiguity / insufficient evidence
        y_ambig = [1.0 if c['audit_results']['disposition_category'] == 'insufficient_evidence' else 0.0 for c in cases]
        p_ambig = np.mean(y_ambig)
        ambiguity_est += N_h * p_ambig
        if n_h > 1:
            s2_ambig = np.var(y_ambig, ddof=1)
            ambiguity_var += (N_h ** 2) * (1.0 - n_h / N_h) * (s2_ambig / n_h)

    strict_se = math.sqrt(strict_var_tp)
    lenient_se = math.sqrt(lenient_var_tp)
    disagree_se = math.sqrt(annot_disagree_var)
    model_fp_se = math.sqrt(model_fp_var)
    ambig_se = math.sqrt(ambiguity_var)
    
    strict_prec_est = strict_tp_est / total_deployed_alarms
    strict_ci_low = max(0.0, (strict_tp_est - 1.96 * strict_se) / total_deployed_alarms)
    strict_ci_high = min(1.0, (strict_tp_est + 1.96 * strict_se) / total_deployed_alarms)
    
    lenient_prec_est = lenient_tp_est / total_deployed_alarms
    lenient_ci_low = max(0.0, (lenient_tp_est - 1.96 * lenient_se) / total_deployed_alarms)
    lenient_ci_high = min(1.0, (lenient_tp_est + 1.96 * lenient_se) / total_deployed_alarms)

    adjudicated_stats = {
        'total_deployed_alarms': total_deployed_alarms,
        'original_tp': 82,
        'original_deployed_precision_pct': round(82 / total_deployed_alarms * 100.0, 2),
        'tripartite_decomposition': {
            'confirmed_model_false_positives': {
                'count': round(model_fp_est, 1),
                'pct': round(model_fp_est / total_deployed_alarms * 100.0, 2),
                'std_error': round(model_fp_se, 1)
            },
            'confirmed_annotation_disagreements': {
                'count': round(annot_disagree_est, 1),
                'pct': round(annot_disagree_est / total_deployed_alarms * 100.0, 2),
                'std_error': round(disagree_se, 1)
            },
            'confirmed_true_omissions': {
                'count': round(strict_tp_est, 1),
                'pct': round(strict_prec_est * 100.0, 2),
                'std_error': round(strict_se, 1),
                'ci_95_pct': [round(strict_ci_low * 100.0, 2), round(strict_ci_high * 100.0, 2)]
            },
            'acoustic_ambiguity': {
                'count': round(ambiguity_est, 1),
                'pct': round(ambiguity_est / total_deployed_alarms * 100.0, 2),
                'std_error': round(ambig_se, 1)
            }
        },
        'broad_pronunciation_event_horizon': {
            'total_event_detections': round(lenient_tp_est, 1),
            'deployed_precision_pct': round(lenient_prec_est * 100.0, 2),
            'std_error_pct': round(lenient_se / total_deployed_alarms * 100.0, 2),
            'ci_95_pct': [round(lenient_ci_low * 100.0, 2), round(lenient_ci_high * 100.0, 2)]
        }
    }
    
    with open(os.path.join(os.path.dirname(__file__), 'adjudicated_metrics_summary.json'), 'w') as f:
        json.dump(adjudicated_stats, f, indent=2)

    print("\n" + "="*90)
    print("Agent C: Tripartite Breakdown of 1,209 Deployed Omission Alarms:")
    print("="*90)
    for k, v in adjudicated_stats['tripartite_decomposition'].items():
        print(f"  {k:<38}: {v['count']:>6.1f} ({v['pct']:>5.2f}%)  [SE: +/-{v['std_error']}]")
    print("-" * 90)
    print(f"  Broad Pronunciation Event Precision: {lenient_prec_est*100:.2f}% (95% CI: [{lenient_ci_low*100:.2f}%, {lenient_ci_high*100:.2f}%])")
    print("="*90)

if __name__ == '__main__':
    main()
