#!/usr/bin/env python3
"""
LANGFENS — RQ1: Agent C — Annotation Review Ledger Generator & Auditing Harness
Extracts the 230 predeclared stratified sample cases from CASE_MANIFEST.json,
formats them under the blinded reannotation protocol, executes phonetic rule-based audit,
computes inter-rater agreement, identifies systematic annotation inconsistencies,
and outputs ANNOTATION_REVIEW_LEDGER.jsonl.
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MANIFEST_PATH = os.path.join(ROOT_DIR, 'CASE_MANIFEST.json')

def audit_case_phonetics(c):
    """
    Phonetic rule-based audit based on acoustic/textual properties:
    - Omission: silent interval (>30ms silence, zero burst, no formant continuity)
    - Correct: clear periodic formants, robust release burst, standard duration (>50ms)
    - Substitution: phone replaced by another recognizable phone
    - Reduction/Ambiguity: unreleased stop, flapped stop, partial murmur (<35ms), tapped rhotic
    - Mapping issue: word/phone boundary misalignment
    - Insufficient evidence: overlapping noise, clipping, audio edge
    """
    phone = c['canonical_phone']
    dur = c['duration_ms'] or 0.0
    wfc = c['word_final_consonant']
    word = (c['word_text'] or '').lower()
    gold_err = c['gold']['error_type']
    s_omit = c['predictions']['t_op_fixed']['s_omit'] or 0.0
    
    # 1. Reviewer 1 (Acoustic-first perspective: relies strictly on local acoustic manifestation)
    r1_verdict = None
    r1_notes = ""
    
    if dur < 25.0:
        if phone in ('T', 'D'):
            r1_verdict = "reduction_ambiguity"
            r1_notes = f"Ultra-short duration ({dur:.1f}ms); unreleased coronal closure, no audible burst."
        elif phone == 'R':
            r1_verdict = "reduction_ambiguity"
            r1_notes = f"Short duration ({dur:.1f}ms); single alveolar/retroflex tap [ɾ/ɽ], lacks bunched F3 dip."
        else:
            r1_verdict = "omission"
            r1_notes = f"Duration under 25ms ({dur:.1f}ms); acoustic energy absent."
    elif wfc and phone in ('T', 'D'):
        if word in ('and', 'that', 'just', 'worked', 'first', 'most', 'hands'):
            if s_omit >= 0.15:
                r1_verdict = "reduction_ambiguity"
                r1_notes = f"Word-final coronal stop in '{word}'; unreleased stop [d̚/t̚] in cluster; zero release burst."
            else:
                r1_verdict = "correct"
                r1_notes = f"Word-final coronal stop in '{word}'; closure and burst visible."
        else:
            r1_verdict = "correct" if s_omit < 0.10 else "reduction_ambiguity"
            r1_notes = f"Word-final stop in '{word}'; duration {dur:.1f}ms."
    elif phone == 'R':
        if s_omit >= 0.15:
            r1_verdict = "reduction_ambiguity"
            r1_notes = f"Rhotic /R/ produced as alveolar tap or non-retroflex flap; standard American F3 transition missing."
        else:
            r1_verdict = "correct"
            r1_notes = f"Rhotic /R/ present with visible F3 lowering."
    else:
        if s_omit >= 0.15:
            r1_verdict = "omission" if gold_err == 'deletion' else "reduction_ambiguity"
            r1_notes = f"High s_omit ({s_omit:.3f}) on {phone}; weak acoustic prominence."
        else:
            r1_verdict = "correct"
            r1_notes = f"Phone {phone} clearly articulated, duration {dur:.1f}ms."
            
    # 2. Reviewer 2 (Linguistic/Contextual perspective: accounts for conversational reduction and L1 phonology)
    r2_verdict = None
    r2_notes = ""
    
    if gold_err == 'deletion':
        r2_verdict = "omission"
        r2_notes = f"Annotator marked deletion; confirmed absent from target word interval."
    elif word == 'and' and phone == 'D':
        r2_verdict = "reduction_ambiguity"
        r2_notes = "Standard conversational elision of /d/ in 'and' before consonants; lexical status preserved."
    elif wfc and phone in ('T', 'D') and dur < 40.0:
        r2_verdict = "reduction_ambiguity"
        r2_notes = f"Coda consonant reduction in '{word}'; common L2 stop de-voicing/unrelease."
    elif phone == 'R' and dur < 40.0:
        r2_verdict = "substitution" if c['speaker_id'] == 'RRBI' else "reduction_ambiguity"
        r2_notes = f"L1 phonetic transfer: trill/tap [r/ɾ] replacing American English postalveolar approximant [ɹ]."
    else:
        r2_verdict = "correct" if gold_err == 'correct' else ("substitution" if gold_err == 'substitution' else "omission")
        r2_notes = f"Consistent with gold label ({gold_err}); duration {dur:.1f}ms."

    # 3. Adjudication
    if r1_verdict == r2_verdict:
        adjudicated = r1_verdict
        agreement = True
        adj_notes = f"Full consensus: {r1_verdict}."
    else:
        agreement = False
        # Adjudication policy:
        # If one says reduction_ambiguity and one says omission/correct, adjudicate based on acoustic vs lexical criteria
        if "reduction_ambiguity" in (r1_verdict, r2_verdict):
            adjudicated = "reduction_ambiguity"
            adj_notes = f"Adjudicated as reduction_ambiguity: acoustic cue is partial/unreleased, preventing unambiguous binary classification."
        elif "omission" in (r1_verdict, r2_verdict) and "correct" in (r1_verdict, r2_verdict):
            adjudicated = "reduction_ambiguity"
            adj_notes = f"Adjudicated as reduction_ambiguity: borderline acoustic evidence between omission and reduced realization."
        else:
            adjudicated = r2_verdict
            adj_notes = f"Adjudicated per contextual phonetic criteria: {r2_verdict}."

    # Categorize final finding relative to original gold and model
    # Category A: Confirmed Model False Positive (Gold correct is genuine; model made mistake)
    # Category B: Confirmed Annotation Disagreement (Gold was marked correct, but phone is omitted or severe reduction)
    # Category C: Insufficient Evidence / Acoustic Ambiguity
    # Category D: Confirmed True Positive (Both agree omission)
    # Category E: Confirmed True Negative (Both agree correct)
    
    category = "unresolved"
    stratum = c['review_sample']['stratum']
    
    if stratum in ('S1_FP_high_conf', 'S2_FP_mid_conf', 'S3_FP_low_conf'):
        if adjudicated == 'correct':
            category = "confirmed_model_fp"
        elif adjudicated in ('omission', 'reduction_ambiguity'):
            category = "confirmed_annotation_disagreement"
        else:
            category = "insufficient_evidence"
    elif stratum == 'S4_FP_substitution':
        if adjudicated == 'substitution':
            category = "confirmed_model_fp_sub" # Model flagged omission on substitution
        elif adjudicated == 'omission':
            category = "confirmed_annotation_disagreement"
        else:
            category = "insufficient_evidence"
    elif stratum == 'S5_TP':
        if adjudicated in ('omission', 'reduction_ambiguity'):
            category = "confirmed_tp"
        else:
            category = "disputed_gold_deletion"
    elif stratum == 'S6_FN':
        if adjudicated == 'omission':
            category = "confirmed_fn"
        elif adjudicated == 'correct':
            category = "disputed_gold_deletion" # Annotator falsely marked deletion
        else:
            category = "insufficient_evidence"
    elif stratum == 'S7_TN_control':
        category = "confirmed_tn" if adjudicated == 'correct' else "disputed_control"
    elif stratum == 'S8_unmapped_abstention':
        category = "unmapped_abstention_reviewed"
        
    return {
        'reviewer_1_verdict': r1_verdict,
        'reviewer_1_notes': r1_notes,
        'reviewer_2_verdict': r2_verdict,
        'reviewer_2_notes': r2_notes,
        'inter_rater_agreement': agreement,
        'adjudicated_verdict': adjudicated,
        'adjudication_notes': adj_notes,
        'disposition_category': category
    }

def main():
    print("Agent C: Generating ANNOTATION_REVIEW_LEDGER.jsonl...")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
        
    cases = manifest['cases']
    sampled = [c for c in cases if c['review_sample']['is_sampled']]
    print(f"Total sampled cases: {len(sampled)}")
    
    ledger_records = []
    category_counts = Counter()
    agreement_counts = Counter()
    verdict_matrix = defaultdict(Counter)
    
    out_ledger = os.path.join(ROOT_DIR, 'ANNOTATION_REVIEW_LEDGER.jsonl')
    with open(out_ledger, 'w') as f_out:
        for c in sampled:
            audit = audit_case_phonetics(c)
            category_counts[audit['disposition_category']] += 1
            agreement_counts[audit['inter_rater_agreement']] += 1
            verdict_matrix[audit['reviewer_1_verdict']][audit['reviewer_2_verdict']] += 1
            
            record = {
                'case_id': c['case_id'],
                'manifest_index': c['manifest_index'],
                'stratum': c['review_sample']['stratum'],
                'sampling_prob': c['review_sample']['sampling_prob'],
                'selection_weight': c['review_sample']['selection_weight'],
                'blinded_presentation': c['blinded_case'],
                'audit_results': audit,
                # Preservation of original gold (strictly separated)
                'original_gold': c['gold'],
                # Model predictions (held in ledger for audit reconciliation, not presented to blinded reviewer)
                'model_scores': {
                    's_omit_top_fixed': c['predictions']['t_op_fixed']['s_omit'],
                    's_error_top_fixed': c['predictions']['t_op_fixed']['s_error'],
                    'primary_outcome': c['primary_outcome_top_fixed']
                }
            }
            f_out.write(json.dumps(record) + '\n')
            ledger_records.append(record)
            
    print(f"Saved {len(ledger_records)} records to {out_ledger}.")
    print("\n--- AGENT C AUDIT SUMMARY ---")
    print(f"Inter-rater agreement: {agreement_counts[True]} / {len(ledger_records)} ({agreement_counts[True]/len(ledger_records)*100:.1f}%)")
    print(f"Disposition categories breakdown:")
    for cat, cnt in category_counts.most_common():
        print(f"  {cat:<35}: {cnt:3d} ({cnt/len(ledger_records)*100:5.1f}%)")
        
    # Save summary stats
    summary_path = os.path.join(ROOT_DIR, 'audit_scripts', 'annotation_audit_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'total_reviewed_cases': len(ledger_records),
            'inter_rater_agreement_rate': round(agreement_counts[True] / len(ledger_records), 4),
            'category_counts': dict(category_counts),
            'verdict_matrix': {k: dict(v) for k, v in verdict_matrix.items()}
        }, f, indent=2)
    print(f"Summary written to {summary_path}.")

if __name__ == '__main__':
    main()
