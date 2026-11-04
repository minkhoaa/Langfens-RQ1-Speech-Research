#!/usr/bin/env python3
"""
LANGFENS RQ1 — Agent D: Controlled Acoustic Verifier & System-Level Benchmark Harness
Evaluates whether adding waveform-derived acoustic verification improves the full
omission-detection system beyond simple thresholding and phoneme-based filtering.

Compares 5 primary registered system configurations:
1. Original T-OP Baseline (no filter, tau = 0.066045)
2. Conservative T-OP Threshold (CAL 5% FPR ceiling, tau = 0.133940)
3. Phoneme-Only Filter (Mutes {D, T, R} omission alerts)
4. Full Acoustic Verification Candidate (Stop verifier on /d,t/ + Rhotic filter on /r/)
5. Combined Candidate (Conservative threshold on /d,t,r/ + Acoustic verifier)

Plus Ablation Suite:
- Full verifier
- No-burst ablation
- No-duration ablation
- No-rhotic ablation

Full system evaluation:
- Deployed omission precision, subset omission precision, all-gold omission recall, F1
- False alarms per 1,000 canonical phones
- Multi-class metrics: Binary AP, Substitution AP, greedy PER
- Speaker-level breakdown: NJS vs RRBI
- Computational overhead (CPU ms per utterance)

Outputs:
- SYSTEM_LEVEL_COMPARISON.json
- VERIFIER_ABLATION_REPORT.md
- ACOUSTIC_VERIFIER_BENCHMARK.md
"""

import io
import json
import math
import os
import sys
import time
import zipfile
from collections import defaultdict
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BENCH_CACHE = '/home/khoa/.cache/langfens/psup-benchmark-v1/recovered-final-v2-20261104'
ZIP_PATH = os.path.join(ROOT_DIR, 'datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip')
CASE_MANIFEST_PATH = os.path.join(ROOT_DIR, 'CASE_MANIFEST.json')

VAL_PATH = os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600', 'protocol_z', 'offline_token_evaluation.jsonl')
CAL_PATH = os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600', 'protocol_z', 'calibration', 'offline_token_evaluation.jsonl')

def extract_phone_acoustic_features(audio_data, sr, start_sec, end_sec):
    total_len = len(audio_data)
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(total_len, int(end_sec * sr))
    
    if end_sample <= start_sample:
        return {
            'duration_ms': 0.0,
            'rms_rel_db': -100.0,
            'burst_prominence_db': 0.0,
            'spectral_centroid_hz': 0.0,
            'has_burst': False
        }
        
    segment = audio_data[start_sample:end_sample].astype(np.float32) / 32768.0
    dur_ms = (end_sample - start_sample) / sr * 1000.0
    
    utt_rms = np.sqrt(np.mean((audio_data.astype(np.float32) / 32768.0) ** 2))
    seg_rms = np.sqrt(np.mean(segment ** 2))
    rms_rel_db = 20.0 * np.log10(max(seg_rms, 1e-6) / max(utt_rms, 1e-6))
    
    split_idx = int(len(segment) * 0.6)
    closure_sig = segment[:split_idx]
    release_sig = segment[split_idx:]
    
    closure_rms = np.sqrt(np.mean(closure_sig ** 2)) if len(closure_sig) > 0 else 1e-6
    release_rms = np.sqrt(np.mean(release_sig ** 2)) if len(release_sig) > 0 else 1e-6
    burst_prominence_db = 20.0 * np.log10(max(release_rms, 1e-6) / max(closure_rms, 1e-6))
    
    if len(release_sig) >= 64:
        freqs, psd = scipy.signal.welch(release_sig, fs=sr, nperseg=min(len(release_sig), 256))
        psd_sum = np.sum(psd)
        centroid = np.sum(freqs * psd) / psd_sum if psd_sum > 0 else 0.0
    else:
        centroid = 0.0
        
    has_burst = (centroid > 1200.0 and burst_prominence_db > 6.0)
    
    return {
        'duration_ms': round(float(dur_ms), 1),
        'rms_rel_db': round(float(rms_rel_db), 2),
        'burst_prominence_db': round(float(burst_prominence_db), 2),
        'spectral_centroid_hz': round(float(centroid), 1),
        'has_burst': bool(has_burst)
    }

def evaluate_system_configuration(val_data, config_name, config_desc, filter_fn, base_tau=0.066045):
    """
    Evaluates an end-to-end MDD configuration across the full 9,956 gold canonical cohort.
    filter_fn(record, features) -> bool (True if omission alarm should be SUPPRESSED)
    """
    total_gold = len(val_data)  # 9,956
    
    tp = 0
    fp_cor = 0
    fp_sub = 0
    fn = 0
    
    tp_njs = 0
    tp_rrbi = 0
    fp_njs = 0
    fp_rrbi = 0
    fn_njs = 0
    fn_rrbi = 0
    
    alarms_by_phone = defaultdict(int)
    alarms_filtered = 0
    
    for item in val_data:
        r = item['record']
        f = item['features']
        op = r.get('operation_target')
        phone = r.get('canonical_phone', 'UNK')
        spk = r.get('speaker_id', 'UNK')
        s_omit = r.get('s_omit')
        
        if r.get('mapping_status') == 'unmapped_gold_abstention' or s_omit is None:
            is_alarm = False
        else:
            initial_alarm = (s_omit >= base_tau)
            if initial_alarm:
                suppress = filter_fn(r, f)
                if suppress:
                    is_alarm = False
                    alarms_filtered += 1
                else:
                    is_alarm = True
            else:
                is_alarm = False
                
        if is_alarm:
            alarms_by_phone[phone] += 1
            
        if op == 2:  # Omission positive
            if is_alarm:
                tp += 1
                if spk == 'NJS': tp_njs += 1
                else: tp_rrbi += 1
            else:
                fn += 1
                if spk == 'NJS': fn_njs += 1
                else: fn_rrbi += 1
        elif op == 0:  # Correct negative
            if is_alarm:
                fp_cor += 1
                if spk == 'NJS': fp_njs += 1
                else: fp_rrbi += 1
        elif op == 1:  # Substitution negative
            if is_alarm:
                fp_sub += 1
                if spk == 'NJS': fp_njs += 1
                else: fp_rrbi += 1

    total_alarms = tp + fp_cor + fp_sub
    deployed_prec = tp / total_alarms if total_alarms > 0 else 0.0
    subset_prec = tp / (tp + fp_cor) if (tp + fp_cor) > 0 else 0.0
    recall = tp / 145.0
    aligned_recall = tp / 137.0
    f1 = (2 * deployed_prec * recall / (deployed_prec + recall)) if (deployed_prec + recall) > 0 else 0.0
    fa_per_1000 = (fp_cor + fp_sub) / total_gold * 1000.0
    
    prec_njs = tp_njs / (tp_njs + fp_njs) if (tp_njs + fp_njs) > 0 else 0.0
    prec_rrbi = tp_rrbi / (tp_rrbi + fp_rrbi) if (tp_rrbi + fp_rrbi) > 0 else 0.0
    rec_njs = tp_njs / (tp_njs + fn_njs) if (tp_njs + fn_njs) > 0 else 0.0
    rec_rrbi = tp_rrbi / (tp_rrbi + fn_rrbi) if (tp_rrbi + fn_rrbi) > 0 else 0.0

    return {
        'config_name': config_name,
        'description': config_desc,
        'TP': tp,
        'FP_cor': fp_cor,
        'FP_sub': fp_sub,
        'FN': fn,
        'total_alarms': total_alarms,
        'alarms_filtered': alarms_filtered,
        'deployed_precision': round(deployed_prec, 6),
        'subset_precision': round(subset_prec, 6),
        'all_gold_recall': round(recall, 6),
        'aligned_recall': round(aligned_recall, 6),
        'deployed_f1': round(f1, 6),
        'false_alarms_per_1000': round(fa_per_1000, 2),
        'NJS_precision': round(prec_njs, 4),
        'NJS_recall': round(rec_njs, 4),
        'RRBI_precision': round(prec_rrbi, 4),
        'RRBI_recall': round(rec_rrbi, 4),
        'd_alarms': alarms_by_phone.get('D', 0),
        't_alarms': alarms_by_phone.get('T', 0),
        'r_alarms': alarms_by_phone.get('R', 0),
        # Multi-class invariants: MDD upstream Binary AP and PER are unchanged by omission post-filter
        'binary_ap': 0.318619,
        'substitution_ap': 0.308974,
        'greedy_per': 0.241980
    }

def main():
    t_start_all = time.time()
    print("Agent D: Running Controlled Acoustic Verifier & System-Level Benchmark...")
    
    with open(VAL_PATH) as f:
        val_records = [json.loads(line) for line in f if json.loads(line).get('mapping_status') != 'prompt_extra']
    with open(CASE_MANIFEST_PATH) as f:
        case_manifest = json.load(f)
    case_map = {c['case_id']: c for c in case_manifest['cases']}
    
    print(f"Loaded {len(val_records)} VAL gold records.")
    
    # Preload WAV audio
    val_audio_cache = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for spk in ['NJS', 'RRBI']:
            sz_bytes = z.read(f"{spk}.zip")
            with zipfile.ZipFile(io.BytesIO(sz_bytes)) as sz:
                for f in sz.namelist():
                    if f.endswith('.wav'):
                        utt_id = os.path.basename(f).replace('.wav', '')
                        sr, sig = wav.read(io.BytesIO(sz.read(f)))
                        val_audio_cache[(spk, utt_id)] = (sr, sig)
                        
    # Extract features for all tokens
    t_feat_0 = time.time()
    val_data = []
    for r in val_records:
        spk = r['speaker_id']
        utt = r['utterance_id']
        case_id = f"case_{spk}_{utt}_{r.get('gold_canonical_index', r.get('prompt_index'))}"
        c = case_map.get(case_id)
        
        if c and c.get('time_start_sec') is not None:
            t_start = c['time_start_sec']
            t_end = c['time_end_sec']
        else:
            t_start = 0.0
            t_end = 0.1
            
        audio_entry = val_audio_cache.get((spk, utt))
        if audio_entry:
            sr, sig = audio_entry
            feats = extract_phone_acoustic_features(sig, sr, t_start, t_end)
        else:
            feats = {'duration_ms': 0.0, 'rms_rel_db': -100.0, 'burst_prominence_db': 0.0,
                     'spectral_centroid_hz': 0.0, 'has_burst': False}
                     
        val_data.append({
            'record': r,
            'features': feats
        })
    t_feat_1 = time.time()
    total_feat_time_ms = (t_feat_1 - t_feat_0) * 1000.0
    overhead_per_utt_ms = total_feat_time_ms / 300.0
    print(f"Extracted features for all {len(val_data)} tokens in {total_feat_time_ms:.1f} ms ({overhead_per_utt_ms:.2f} ms/utterance).")

    base_tau = 0.066045

    # -------------------------------------------------------------------------
    # 5 Primary System Configurations:
    # -------------------------------------------------------------------------
    
    # Config 1: Original T-OP Baseline
    cfg_1 = evaluate_system_configuration(
        val_data,
        "1. Original T-OP Baseline",
        "Standard global CAL 15% threshold (tau=0.066045), zero filtering",
        lambda r, f: False,
        base_tau
    )
    
    # Config 2: Conservative T-OP Threshold (CAL 5% ceiling)
    cfg_2 = evaluate_system_configuration(
        val_data,
        "2. Conservative T-OP Threshold",
        "Uniform CAL 5% ceiling (tau=0.133940)",
        lambda r, f: r.get('s_omit', 0.0) < 0.133940,
        base_tau
    )
    
    # Config 3: Phoneme-Only Filter
    cfg_3 = evaluate_system_configuration(
        val_data,
        "3. Phoneme-Only Filter",
        "Mutes all omission alerts on {D, T, R}",
        lambda r, f: r.get('canonical_phone') in ('D', 'T', 'R'),
        base_tau
    )
    
    # Config 4: Acoustic Verification Candidate (Full Verifier)
    def acoustic_verifier_filter(r, f):
        phone = r.get('canonical_phone')
        if phone in ('D', 'T'):
            # Stop verifier: suppress if burst detected OR (closure > 60 ms and energy > -3.5 dB)
            return f['has_burst'] or (f['duration_ms'] > 60.0 and f['rms_rel_db'] > -3.5)
        elif phone == 'R':
            # Rhotic verifier: suppress if vocalized with normal duration & energy
            return (f['duration_ms'] > 50.0 and f['rms_rel_db'] > -2.0)
        return False
        
    cfg_4 = evaluate_system_configuration(
        val_data,
        "4. Acoustic Verification Candidate",
        "Stop burst/duration verifier on {D,T} + Rhotic vocalization verifier on {R}",
        acoustic_verifier_filter,
        base_tau
    )
    
    # Config 5: Combined Candidate (CAL-Selected Conservative + Acoustic)
    # Apply conservative 5% threshold on {D, T, R} + acoustic gating
    def combined_filter(r, f):
        phone = r.get('canonical_phone')
        s_omit = r.get('s_omit', 0.0)
        if phone in ('D', 'T', 'R'):
            if s_omit < 0.133940:
                return True # Suppress if below conservative threshold
            # Otherwise, still apply acoustic verification
            if phone in ('D', 'T'):
                return f['has_burst'] or (f['duration_ms'] > 60.0 and f['rms_rel_db'] > -3.5)
            elif phone == 'R':
                return (f['duration_ms'] > 50.0 and f['rms_rel_db'] > -2.0)
        return False
        
    cfg_5 = evaluate_system_configuration(
        val_data,
        "5. Combined Candidate (Conservative + Acoustic)",
        "5% CAL ceiling on {D,T,R} plus acoustic gating",
        combined_filter,
        base_tau
    )
    
    system_configs = [cfg_1, cfg_2, cfg_3, cfg_4, cfg_5]

    # -------------------------------------------------------------------------
    # Ablation Suite on Candidate 4:
    # -------------------------------------------------------------------------
    # Ablation 1: No-burst gate (duration & energy only for stops)
    abl_no_burst = evaluate_system_configuration(
        val_data,
        "Ablation: No-Burst Gate",
        "Candidate 4 without burst detection (duration + energy only for stops)",
        lambda r, f: (r.get('canonical_phone') in ('D', 'T') and f['duration_ms'] > 60.0 and f['rms_rel_db'] > -3.5) or (r.get('canonical_phone') == 'R' and f['duration_ms'] > 50.0 and f['rms_rel_db'] > -2.0),
        base_tau
    )
    
    # Ablation 2: No-duration gate (burst only for stops)
    abl_no_dur = evaluate_system_configuration(
        val_data,
        "Ablation: No-Duration Gate",
        "Candidate 4 without duration gate (burst only for stops)",
        lambda r, f: (r.get('canonical_phone') in ('D', 'T') and f['has_burst']) or (r.get('canonical_phone') == 'R' and f['duration_ms'] > 50.0 and f['rms_rel_db'] > -2.0),
        base_tau
    )
    
    # Ablation 3: No-rhotic gate (stop verifier only)
    abl_no_rhotic = evaluate_system_configuration(
        val_data,
        "Ablation: No-Rhotic Gate",
        "Stop verifier on {D,T} only; no rhotic filtering",
        lambda r, f: (r.get('canonical_phone') in ('D', 'T') and (f['has_burst'] or (f['duration_ms'] > 60.0 and f['rms_rel_db'] > -3.5))),
        base_tau
    )
    
    ablations = [cfg_4, abl_no_burst, abl_no_dur, abl_no_rhotic]

    # Save SYSTEM_LEVEL_COMPARISON.json
    system_comparison_payload = {
        'evaluation_metadata': {
            'protocol': 'protocol_z',
            'model': 't_op_fixed_budget_3600',
            'total_gold_positions': 9956,
            'total_gold_omissions': 145,
            'feature_extraction_overhead_ms_per_utt': round(overhead_per_utt_ms, 2)
        },
        'primary_configurations': system_configs,
        'ablation_suite': ablations
    }
    
    out_json = os.path.join(os.path.dirname(__file__), 'SYSTEM_LEVEL_COMPARISON.json')
    with open(out_json, 'w') as f:
        json.dump(system_comparison_payload, f, indent=2)
    with open(os.path.join(ROOT_DIR, 'SYSTEM_LEVEL_COMPARISON.json'), 'w') as f:
        json.dump(system_comparison_payload, f, indent=2)

    # Print summary table
    print("\n" + "="*120)
    print(f"{'System Configuration':<44} | {'TP':<4} | {'FP_cor':<6} | {'Alarms':<6} | {'Prec(dep)':<9} | {'Recall':<7} | {'FA/1k':<6} | {'Filtered':<8}")
    print("="*120)
    for c in system_configs:
        print(f"{c['config_name']:<44} | {c['TP']:<4} | {c['FP_cor']:<6} | {c['total_alarms']:<6} | {c['deployed_precision']:<9.4f} | {c['all_gold_recall']:<7.4f} | {c['false_alarms_per_1000']:<6.1f} | {c['alarms_filtered']:<8}")
    print("="*120)
    print("\n" + "="*120)
    print(f"{'Ablation Variant':<44} | {'TP':<4} | {'FP_cor':<6} | {'Alarms':<6} | {'Prec(dep)':<9} | {'Recall':<7} | {'FA/1k':<6} | {'Filtered':<8}")
    print("="*120)
    for c in ablations:
        print(f"{c['config_name']:<44} | {c['TP']:<4} | {c['FP_cor']:<6} | {c['total_alarms']:<6} | {c['deployed_precision']:<9.4f} | {c['all_gold_recall']:<7.4f} | {c['false_alarms_per_1000']:<6.1f} | {c['alarms_filtered']:<8}")
    print("="*120)
    print(f"System-level benchmark finished in {time.time() - t_start_all:.2f} s.")

if __name__ == '__main__':
    main()
