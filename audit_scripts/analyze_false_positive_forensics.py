#!/usr/bin/env python3
"""
LANGFENS — RQ1: Agent B — False Positive Forensics Harness
Analyzes all 1,127 false alarms (1,068 FP_correct + 59 FP_substitution) on Fold-1 VAL
under Protocol Z at the primary CAL 15% FPR ceiling (tau = 0.066045).
Also performs deep forensic analysis on high-confidence alarms (s_omit >= 0.15 and s_omit >= 0.20).
Identifies recurring phonetic, lexical, positional, and acoustic error clusters.
"""

import json
import os
import sys
from collections import Counter, defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MANIFEST_PATH = os.path.join(ROOT_DIR, 'CASE_MANIFEST.json')

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH",
    "IY", "OW", "OY", "UH", "UW"
}
CORONAL_STOPS = {"T", "D"}
OTHER_STOPS = {"P", "B", "K", "G"}
FRICATIVES = {"S", "Z", "SH", "ZH", "F", "V", "TH", "DH"}
NASALS = {"M", "N", "NG"}
LIQUIDS = {"L", "R"}
GLIDES = {"W", "Y", "HH"}

def get_phonetic_manner(phone):
    if phone in VOWELS: return "Vowel"
    if phone in CORONAL_STOPS: return "Coronal Stop"
    if phone in OTHER_STOPS: return "Non-Coronal Stop"
    if phone in FRICATIVES: return "Fricative"
    if phone in NASALS: return "Nasal"
    if phone in LIQUIDS: return "Liquid"
    if phone in GLIDES: return "Glide"
    return "Other"

def main():
    print("Agent B: Running False Positive Forensics...")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
        
    cases = manifest['cases']
    aligned = [c for c in cases if c['mapping_status'] == 'aligned']
    
    tau = manifest['calibrated_operating_points']['t_op_fixed_z_s_omit']['0.15']['threshold']
    print(f"Loaded {len(cases)} cases. Aligned: {len(aligned)}. Primary tau = {tau:.6f}")
    
    # Extract subsets
    fp_cor = [c for c in aligned if c['gold']['operation_target'] == 0 and c['predictions']['t_op_fixed']['s_omit'] >= tau]
    fp_sub = [c for c in aligned if c['gold']['operation_target'] == 1 and c['predictions']['t_op_fixed']['s_omit'] >= tau]
    tp = [c for c in aligned if c['gold']['operation_target'] == 2 and c['predictions']['t_op_fixed']['s_omit'] >= tau]
    fn = [c for c in aligned if c['gold']['operation_target'] == 2 and c['predictions']['t_op_fixed']['s_omit'] < tau]
    tn_cor = [c for c in aligned if c['gold']['operation_target'] == 0 and c['predictions']['t_op_fixed']['s_omit'] < tau]
    
    all_fp = fp_cor + fp_sub
    print(f"FP_correct: {len(fp_cor)}, FP_sub: {len(fp_sub)}, Total FP alarms: {len(all_fp)}")
    
    # 1. Phoneme distribution
    phone_counts_cor = Counter(c['canonical_phone'] for c in fp_cor)
    phone_counts_sub = Counter(c['canonical_phone'] for c in fp_sub)
    phone_counts_all = Counter(c['canonical_phone'] for c in all_fp)
    
    # 2. Manner of articulation
    manner_counts = Counter(get_phonetic_manner(c['canonical_phone']) for c in all_fp)
    
    # 3. Speaker breakdown
    spk_cor = Counter(c['speaker_id'] for c in fp_cor)
    spk_sub = Counter(c['speaker_id'] for c in fp_sub)
    
    # 4. Positional breakdown (Word-Final Consonant)
    wfc_cor = Counter(c['word_final_consonant'] for c in fp_cor)
    wfc_sub = Counter(c['word_final_consonant'] for c in fp_sub)
    
    # 5. Score quantiles
    score_bands = {
        'Extreme (>= 0.20)': [c for c in all_fp if c['predictions']['t_op_fixed']['s_omit'] >= 0.20],
        'High (0.15 - 0.20)': [c for c in all_fp if 0.15 <= c['predictions']['t_op_fixed']['s_omit'] < 0.20],
        'Medium (0.10 - 0.15)': [c for c in all_fp if 0.10 <= c['predictions']['t_op_fixed']['s_omit'] < 0.15],
        'Near Threshold (tau - 0.10)': [c for c in all_fp if tau <= c['predictions']['t_op_fixed']['s_omit'] < 0.10],
    }
    
    # 6. High-confidence cluster identification
    # Cluster 1: Word-final coronal stops (/D/, /T/ with WFC=True)
    cluster_1 = [c for c in all_fp if c['canonical_phone'] in ('D', 'T') and c['word_final_consonant']]
    
    # Cluster 2: Function words ("and", "the", "that", "to", "of", "with")
    function_words = {'and', 'the', 'that', 'to', 'of', 'with', 'for', 'at', 'in', 'on', 'from', 'as', 'he', 'his', 'it', 'was'}
    cluster_2 = [c for c in all_fp if (c['word_text'] or '').lower() in function_words]
    
    # Cluster 3: Rhotic /R/ (onset and coda)
    cluster_3 = [c for c in all_fp if c['canonical_phone'] == 'R']
    
    # Cluster 4: Non-coronal stops / liquids / fricatives
    cluster_4 = [c for c in all_fp if c['canonical_phone'] not in ('D', 'T', 'R')]
    
    # Cluster 5: Substitution alarms (model predicts omission on substitution)
    cluster_5 = fp_sub
    
    print("\n--- FORENSICS SUMMARY ---")
    print(f"Top 5 FP phones: {phone_counts_all.most_common(5)}")
    print(f"Manner breakdown: {manner_counts}")
    print(f"Cluster 1 (Word-final D/T): N={len(cluster_1)} ({len(cluster_1)/len(all_fp)*100:.1f}%)")
    print(f"Cluster 2 (Function words): N={len(cluster_2)} ({len(cluster_2)/len(all_fp)*100:.1f}%)")
    print(f"Cluster 3 (Rhotic /R/): N={len(cluster_3)} ({len(cluster_3)/len(all_fp)*100:.1f}%)")
    print(f"Cluster 4 (Remaining phones): N={len(cluster_4)} ({len(cluster_4)/len(all_fp)*100:.1f}%)")
    print(f"Cluster 5 (Substitution alarms): N={len(cluster_5)} ({len(cluster_5)/len(all_fp)*100:.1f}%)")
    
    # Save a detailed forensics JSON report
    forensics_data = {
        'total_false_alarms': len(all_fp),
        'fp_correct_count': len(fp_cor),
        'fp_substitution_count': len(fp_sub),
        'primary_tau': tau,
        'phone_breakdown': {
            'all_alarms': dict(phone_counts_all),
            'fp_correct': dict(phone_counts_cor),
            'fp_substitution': dict(phone_counts_sub)
        },
        'manner_breakdown': dict(manner_counts),
        'speaker_breakdown': {
            'fp_correct': dict(spk_cor),
            'fp_substitution': dict(spk_sub)
        },
        'wfc_breakdown': {
            'fp_correct': dict(wfc_cor),
            'fp_substitution': dict(wfc_sub)
        },
        'score_bands': {k: len(v) for k, v in score_bands.items()},
        'clusters': {
            'cluster_1_wfc_coronal_stops': {
                'name': 'Word-Final Coronal Stops (/D/, /T/)',
                'count': len(cluster_1),
                'share_of_total_alarms': round(len(cluster_1) / len(all_fp), 4),
                'sample_case_ids': [c['case_id'] for c in cluster_1[:10]]
            },
            'cluster_2_function_words': {
                'name': 'Weak Function Word Reductions',
                'count': len(cluster_2),
                'share_of_total_alarms': round(len(cluster_2) / len(all_fp), 4),
                'top_words': dict(Counter((c['word_text'] or '').lower() for c in cluster_2).most_common(5)),
                'sample_case_ids': [c['case_id'] for c in cluster_2[:10]]
            },
            'cluster_3_rhotic': {
                'name': 'Rhotic Approximant / Trill / Tap (/R/)',
                'count': len(cluster_3),
                'share_of_total_alarms': round(len(cluster_3) / len(all_fp), 4),
                'sample_case_ids': [c['case_id'] for c in cluster_3[:10]]
            },
            'cluster_4_remaining': {
                'name': 'All Other Consonants & Vowels',
                'count': len(cluster_4),
                'share_of_total_alarms': round(len(cluster_4) / len(all_fp), 4),
                'sample_case_ids': [c['case_id'] for c in cluster_4[:10]]
            },
            'cluster_5_substitutions': {
                'name': 'Omission Alarms on True Substitutions',
                'count': len(cluster_5),
                'share_of_total_alarms': round(len(cluster_5) / len(all_fp), 4),
                'sample_case_ids': [c['case_id'] for c in cluster_5[:10]]
            }
        }
    }
    
    out_path = os.path.join(ROOT_DIR, 'audit_scripts', 'forensics_summary.json')
    with open(out_path, 'w') as f:
        json.dump(forensics_data, f, indent=2)
    print(f"Forensics summary saved to {out_path}.")

if __name__ == '__main__':
    main()
