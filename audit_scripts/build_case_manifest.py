#!/usr/bin/env python3
"""
LANGFENS — RQ1: Coordinated Case Manifest Generator
Creates CASE_MANIFEST.json freezing all 9,993 Fold-1 VAL token positions across Protocol Z and O,
model predictions for T-OP and B-Control (fixed and best), calibrated operating points,
and the predeclared stratified sampling design for Agent C (Annotation Reliability).
"""

import hashlib
import io
import json
import math
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from langfens_prep.textgrid_parser import parse_gold_textgrid_content

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BENCH_CACHE = '/home/khoa/.cache/langfens/psup-benchmark-v1/recovered-final-v2-20261104'
ZIP_PATH = os.path.join(ROOT_DIR, 'datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip')

RANDOM_SEED = 1337
random.seed(RANDOM_SEED)

CEILINGS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]

def cal_threshold(correct_scores, ceiling):
    descending = sorted(correct_scores, reverse=True)
    allowed = math.floor(ceiling * len(descending))
    return math.nextafter(descending[allowed], math.inf)

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Step 1: Loading TextGrid gold annotations from staging zip...")
    annotations = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for spk in ['NJS', 'RRBI']:
            with zipfile.ZipFile(io.BytesIO(z.read(f'{spk}.zip'))) as rz:
                for name in rz.namelist():
                    if name.endswith('.TextGrid'):
                        uid = name.split('/')[-1].replace('.TextGrid', '')
                        tg_text = rz.read(name).decode('utf-8')
                        annotations[(spk, uid)] = parse_gold_textgrid_content(tg_text, name)
    print(f"Loaded TextGrids for {len(annotations)} utterances.")

    print("Step 2: Loading CAL evaluation files to derive exact operating point thresholds...")
    cal_files = {
        't_op_fixed_z': os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600/protocol_z/calibration/offline_token_evaluation.jsonl'),
        't_op_fixed_o': os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600/protocol_o/calibration/offline_token_evaluation.jsonl'),
        'b_control_fixed_z': os.path.join(BENCH_CACHE, 'b_control_fixed_budget_3600/protocol_z/calibration/offline_token_evaluation.jsonl'),
        'b_control_fixed_o': os.path.join(BENCH_CACHE, 'b_control_fixed_budget_3600/protocol_o/calibration/offline_token_evaluation.jsonl'),
        't_op_best_z': os.path.join(BENCH_CACHE, 't_op_best/protocol_z/calibration/offline_token_evaluation.jsonl'),
        'b_control_best_z': os.path.join(BENCH_CACHE, 'b_control_best/protocol_z/calibration/offline_token_evaluation.jsonl'),
    }

    cal_tokens = {}
    for k, p in cal_files.items():
        with open(p) as f:
            cal_tokens[k] = [json.loads(line) for line in f]

    cal_correct_scores = {
        't_op_fixed_z_s_omit': [r['s_omit'] for r in cal_tokens['t_op_fixed_z'] if r.get('mapping_status') == 'aligned' and r.get('operation_target') == 0 and r.get('s_omit') is not None],
        't_op_fixed_z_s_error': [r['s_error'] for r in cal_tokens['t_op_fixed_z'] if r.get('mapping_status') == 'aligned' and r.get('operation_target') == 0 and r.get('s_error') is not None],
        'b_control_fixed_z_s_error': [r['s_error'] for r in cal_tokens['b_control_fixed_z'] if r.get('mapping_status') == 'aligned' and r.get('operation_target') == 0 and r.get('s_error') is not None],
        't_op_fixed_o_s_omit': [r['s_omit'] for r in cal_tokens['t_op_fixed_o'] if r.get('mapping_status') == 'aligned' and r.get('operation_target') == 0 and r.get('s_omit') is not None],
        't_op_fixed_o_s_error': [r['s_error'] for r in cal_tokens['t_op_fixed_o'] if r.get('mapping_status') == 'aligned' and r.get('operation_target') == 0 and r.get('s_error') is not None],
        'b_control_fixed_o_s_error': [r['s_error'] for r in cal_tokens['b_control_fixed_o'] if r.get('mapping_status') == 'aligned' and r.get('operation_target') == 0 and r.get('s_error') is not None],
        't_op_best_z_s_omit': [r['s_omit'] for r in cal_tokens['t_op_best_z'] if r.get('mapping_status') == 'aligned' and r.get('operation_target') == 0 and r.get('s_omit') is not None],
        'b_control_best_z_s_error': [r['s_error'] for r in cal_tokens['b_control_best_z'] if r.get('mapping_status') == 'aligned' and r.get('operation_target') == 0 and r.get('s_error') is not None],
    }

    operating_points = {}
    for name, scores in cal_correct_scores.items():
        operating_points[name] = {}
        for c in CEILINGS:
            tau = cal_threshold(scores, c)
            achieved = sum(s >= tau for s in scores) / len(scores)
            operating_points[name][str(c)] = {
                'ceiling': c,
                'threshold': tau,
                'cal_n': len(scores),
                'cal_achieved_fpr': achieved,
                'cal_false_alarms': sum(s >= tau for s in scores)
            }

    print("Step 3: Loading VAL prediction tokens from all 4 configurations...")
    val_files = {
        't_op_fixed_z': os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600/protocol_z/offline_token_evaluation.jsonl'),
        't_op_fixed_o': os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600/protocol_o/offline_token_evaluation.jsonl'),
        'b_control_fixed_z': os.path.join(BENCH_CACHE, 'b_control_fixed_budget_3600/protocol_z/offline_token_evaluation.jsonl'),
        'b_control_fixed_o': os.path.join(BENCH_CACHE, 'b_control_fixed_budget_3600/protocol_o/offline_token_evaluation.jsonl'),
        't_op_best_z': os.path.join(BENCH_CACHE, 't_op_best/protocol_z/offline_token_evaluation.jsonl'),
        'b_control_best_z': os.path.join(BENCH_CACHE, 'b_control_best/protocol_z/offline_token_evaluation.jsonl'),
    }

    val_tokens = {}
    for k, p in val_files.items():
        with open(p) as f:
            val_tokens[k] = [json.loads(line) for line in f]

    # Also load runtime predictions for t_op_fixed_z and b_control_fixed_z for raw probabilities
    rt_t_op_path = os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600/protocol_z/runtime_predictions.jsonl')
    rt_b_ctrl_path = os.path.join(BENCH_CACHE, 'b_control_fixed_budget_3600/protocol_z/runtime_predictions.jsonl')
    
    rt_t_op = {}
    with open(rt_t_op_path) as f:
        for line in f:
            d = json.loads(line)
            rt_t_op[(d['speaker_id'], d['utterance_id'], d['prompt_index'])] = d
            
    rt_b_ctrl = {}
    with open(rt_b_ctrl_path) as f:
        for line in f:
            d = json.loads(line)
            rt_b_ctrl[(d['speaker_id'], d['utterance_id'], d['prompt_index'])] = d

    # Index other VAL runs by (speaker_id, utterance_id, prompt_index, gold_canonical_index)
    index_b_fixed = {}
    for r in val_tokens['b_control_fixed_z']:
        index_b_fixed[(r['speaker_id'], r['utterance_id'], r.get('prompt_index'), r.get('gold_canonical_index'))] = r

    index_t_best = {}
    for r in val_tokens['t_op_best_z']:
        index_t_best[(r['speaker_id'], r['utterance_id'], r.get('prompt_index'), r.get('gold_canonical_index'))] = r

    index_b_best = {}
    for r in val_tokens['b_control_best_z']:
        index_b_best[(r['speaker_id'], r['utterance_id'], r.get('prompt_index'), r.get('gold_canonical_index'))] = r

    # Primary reference rows: t_op_fixed_z (9,993 rows)
    primary_rows = val_tokens['t_op_fixed_z']
    print(f"Primary rows in Protocol Z: {len(primary_rows)}")

    tau_primary = operating_points['t_op_fixed_z_s_omit']['0.15']['threshold']
    print(f"Primary operating point threshold (CAL 15% ceiling): {tau_primary:.6f}")

    print("Step 4: Assembling full case records and assigning strata...")
    cases = []
    strata_buckets = defaultdict(list)

    for idx, r in enumerate(primary_rows):
        spk = r['speaker_id']
        uid = r['utterance_id']
        p_idx = r.get('prompt_index')
        g_idx = r.get('gold_canonical_index')
        c_phone = r.get('canonical_phone')
        map_st = r.get('mapping_status')
        target = r.get('operation_target')
        bin_tgt = r.get('binary_target')
        wfc = r.get('word_final_consonant', False)
        
        # Case ID
        p_str = f"{p_idx:03d}" if p_idx is not None else "NA"
        g_str = f"{g_idx:03d}" if g_idx is not None else "NA"
        phone_str = c_phone if c_phone else "NONE"
        case_id = f"CASE-Z-{spk}-{uid}-P{p_str}-G{g_str}-{phone_str}"

        # TextGrid interval info
        xmin, xmax, dur_ms = None, None, None
        err_type, real_phone = None, None
        word_text = None
        prompt_sentence = None

        if (spk, uid) in annotations:
            annot = annotations[(spk, uid)]
            words = [w['text'] for w in annot.words_intervals if w['text']]
            prompt_sentence = " ".join(words)
            if g_idx is not None and g_idx < len(annot.canonical_intervals):
                intv = annot.canonical_intervals[g_idx]
                xmin = round(intv.xmin, 4)
                xmax = round(intv.xmax, 4)
                dur_ms = round((intv.xmax - intv.xmin) * 1000.0, 2)
                err_type = intv.error_type
                real_phone = intv.realized_phone
                # Word match
                mid = (intv.xmin + intv.xmax) / 2.0
                for w in annot.words_intervals:
                    if w['xmin'] <= mid <= w['xmax']:
                        word_text = w['text']
                        break

        # Scores across models
        s_omit_top_fixed = r.get('s_omit')
        s_error_top_fixed = r.get('s_error')
        s_sub_top_fixed = r.get('s_sub')

        key_tuple = (spk, uid, p_idx, g_idx)
        b_fix_row = index_b_fixed.get(key_tuple, {})
        s_error_b_fixed = b_fix_row.get('s_error')

        t_best_row = index_t_best.get(key_tuple, {})
        s_omit_t_best = t_best_row.get('s_omit')
        s_error_t_best = t_best_row.get('s_error')

        b_best_row = index_b_best.get(key_tuple, {})
        s_error_b_best = b_best_row.get('s_error')

        # Runtime probabilities
        rt_top_data = rt_t_op.get((spk, uid, p_idx), {}) if p_idx is not None else {}
        rt_b_data = rt_b_ctrl.get((spk, uid, p_idx), {}) if p_idx is not None else {}

        prob_top = rt_top_data.get('probabilities')
        prob_b = rt_b_data.get('probabilities')

        # Compute outcome at primary operating point (T-OP fixed s_omit >= tau_primary)
        outcome_primary = "unmapped"
        if map_st == 'aligned':
            alarm = s_omit_top_fixed is not None and s_omit_top_fixed >= tau_primary
            if target == 2:
                outcome_primary = "TP" if alarm else "FN"
            elif target == 0:
                outcome_primary = "FP_correct" if alarm else "TN_correct"
            elif target == 1:
                outcome_primary = "FP_substitution" if alarm else "TN_substitution"
        elif map_st == 'unmapped_gold_abstention':
            outcome_primary = "unmapped_gold_abstention"
        elif map_st == 'prompt_extra':
            outcome_primary = "extra_prompt"

        # Outcomes across all 7 ceilings
        ceilings_outcomes = {}
        for c in CEILINGS:
            tau_c = operating_points['t_op_fixed_z_s_omit'][str(c)]['threshold']
            if map_st == 'aligned':
                al = s_omit_top_fixed is not None and s_omit_top_fixed >= tau_c
                if target == 2:
                    out_c = "TP" if al else "FN"
                elif target == 0:
                    out_c = "FP_cor" if al else "TN_cor"
                elif target == 1:
                    out_c = "FP_sub" if al else "TN_sub"
            else:
                out_c = map_st
            ceilings_outcomes[str(c)] = out_c

        # Assign stratum for sampling
        stratum = None
        if map_st == 'unmapped_gold_abstention':
            stratum = 'S8_unmapped_abstention'
        elif map_st == 'aligned':
            if target == 2:
                if s_omit_top_fixed is not None and s_omit_top_fixed >= tau_primary:
                    stratum = 'S5_TP'
                else:
                    stratum = 'S6_FN'
            elif target == 1:
                if s_omit_top_fixed is not None and s_omit_top_fixed >= tau_primary:
                    stratum = 'S4_FP_substitution'
                else:
                    stratum = 'OTHER_TN_sub'
            elif target == 0:
                if s_omit_top_fixed is not None and s_omit_top_fixed >= 0.20:
                    stratum = 'S1_FP_high_conf'
                elif s_omit_top_fixed is not None and s_omit_top_fixed >= 0.12:
                    stratum = 'S2_FP_mid_conf'
                elif s_omit_top_fixed is not None and s_omit_top_fixed >= tau_primary:
                    stratum = 'S3_FP_low_conf'
                elif s_omit_top_fixed is not None and s_omit_top_fixed < 0.01:
                    stratum = 'S7_TN_control'
                else:
                    stratum = 'OTHER_TN_mid'
        else:
            stratum = 'OTHER_prompt_extra'

        strata_buckets[stratum].append(idx)

        # Build blinded case payload
        audio_start = max(0.0, (xmin or 0.0) - 0.10) if xmin is not None else 0.0
        audio_end = (xmax or 0.0) + 0.10 if xmax is not None else 0.0
        blinded = {
            'case_id': case_id,
            'speaker_id': spk,
            'utterance_id': uid,
            'canonical_phone': c_phone,
            'word_text': word_text,
            'prompt_sentence': prompt_sentence,
            'audio_window_start_sec': round(audio_start, 4),
            'audio_window_end_sec': round(audio_end, 4),
            'target_xmin_sec': xmin,
            'target_xmax_sec': xmax,
            'target_duration_ms': dur_ms
        }

        case_obj = {
            'case_id': case_id,
            'manifest_index': idx,
            'speaker_id': spk,
            'utterance_id': uid,
            'prompt_index': p_idx,
            'gold_canonical_index': g_idx,
            'canonical_phone': c_phone,
            'mapping_status': map_st,
            'word_text': word_text,
            'word_final_consonant': wfc,
            'time_start_sec': xmin,
            'time_end_sec': xmax,
            'duration_ms': dur_ms,
            'gold': {
                'operation_target': target,
                'binary_target': bin_tgt,
                'error_type': err_type,
                'realized_phone': real_phone
            },
            'predictions': {
                't_op_fixed': {
                    's_omit': s_omit_top_fixed,
                    's_error': s_error_top_fixed,
                    's_sub': s_sub_top_fixed,
                    'probabilities': prob_top
                },
                'b_control_fixed': {
                    's_error': s_error_b_fixed,
                    'probabilities': prob_b
                },
                't_op_best': {
                    's_omit': s_omit_t_best,
                    's_error': s_error_t_best
                },
                'b_control_best': {
                    's_error': s_error_b_best
                }
            },
            'primary_outcome_top_fixed': outcome_primary,
            'ceilings_outcomes_top_fixed': ceilings_outcomes,
            'stratum': stratum,
            'review_sample': {
                'is_sampled': False,
                'stratum': stratum,
                'sampling_prob': 0.0,
                'selection_weight': 0.0
            },
            'blinded_case': blinded
        }
        cases.append(case_obj)

    print("Step 5: Executing predeclared stratified sampling for Agent C...")
    # Target sample sizes per stratum
    sample_targets = {
        'S1_FP_high_conf': 8,      # 100% of 8
        'S2_FP_mid_conf': 50,     # ~17.1% of 293
        'S3_FP_low_conf': 40,     # ~5.2% of 767
        'S4_FP_substitution': 20, # ~33.9% of 59
        'S5_TP': 40,              # ~48.8% of 82
        'S6_FN': 30,              # ~54.5% of 55
        'S7_TN_control': 30,      # ~0.67% of 4482
        'S8_unmapped_abstention': 12 # ~32.4% of 37
    }

    sampling_spec = {}
    sampled_indices = set()

    for st_name, target_n in sample_targets.items():
        indices = strata_buckets[st_name]
        pop_n = len(indices)
        actual_k = min(target_n, pop_n)
        prob = actual_k / pop_n if pop_n > 0 else 0.0
        weight = 1.0 / prob if prob > 0 else 0.0

        chosen = random.sample(indices, actual_k)
        sampled_indices.update(chosen)

        sampling_spec[st_name] = {
            'population_count': pop_n,
            'sample_count': actual_k,
            'sampling_prob': round(prob, 6),
            'selection_weight': round(weight, 6)
        }

        for idx in chosen:
            cases[idx]['review_sample'] = {
                'is_sampled': True,
                'stratum': st_name,
                'sampling_prob': round(prob, 6),
                'selection_weight': round(weight, 6)
            }

    print(f"Total cases sampled for independent review: {len(sampled_indices)}")
    for k, v in sampling_spec.items():
        print(f"  {k:<24}: Pop={v['population_count']:4d} | Sample={v['sample_count']:2d} | Prob={v['sampling_prob']:.4f} | Weight={v['selection_weight']:.2f}")

    print("Step 6: Writing CASE_MANIFEST.json...")
    manifest = {
        'manifest_version': '1.0.0',
        'created_at': '2026-11-04T10:00:00+07:00',
        'random_seed': RANDOM_SEED,
        'protocol_description': 'LANGFENS RQ1 Parallel Precision & Dataset Reliability Investigation Frozen Case Manifest',
        'provenance': {
            'evaluation_contract': 'Evaluation Contract V1 (Fold-1 VAL)',
            'split_manifest': 'manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json',
            'split_sha256': '7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b',
            'staged_dataset_zip': 'datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip',
            'staged_dataset_sha256': 'b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9',
            'benchmark_cache_root': BENCH_CACHE,
            'benchmark_git_revision': '7a9eae3d7e7ae4709f03e3766c2310991e1f314a'
        },
        'cohort_summary': {
            'total_positions': len(cases),
            'aligned_positions': sum(1 for c in cases if c['mapping_status'] == 'aligned'),
            'unmapped_gold_abstentions': sum(1 for c in cases if c['mapping_status'] == 'unmapped_gold_abstention'),
            'extra_prompt_positions': sum(1 for c in cases if c['mapping_status'] == 'prompt_extra'),
            'gold_positions_total': sum(1 for c in cases if c['gold_canonical_index'] is not None),
            'gold_correct': sum(1 for c in cases if c['gold']['operation_target'] == 0),
            'gold_substitution': sum(1 for c in cases if c['gold']['operation_target'] == 1),
            'gold_omission': sum(1 for c in cases if c['gold']['operation_target'] == 2),
            'speakers': ['NJS', 'RRBI'],
            'utterance_count': 300
        },
        'calibrated_operating_points': operating_points,
        'sampling_strata_spec': sampling_spec,
        'cases': cases
    }

    out_path = os.path.join(ROOT_DIR, 'CASE_MANIFEST.json')
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"CASE_MANIFEST.json created successfully at {out_path} ({os.path.getsize(out_path)} bytes).")

if __name__ == '__main__':
    main()
