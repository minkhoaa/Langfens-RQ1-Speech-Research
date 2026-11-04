#!/usr/bin/env python3
"""
LANGFENS RQ1 — Agent B: Stop-Consonant Acoustic Verification Harness
Investigates whether waveform-derived acoustic evidence can distinguish actual
omissions of /d/ and /t/ from realized (released or unreleased) consonants.

Inputs:
- CAL & VAL offline token predictions (Protocol Z)
- Audio waveforms from authorized_RQ1-TRN-02A_fold1_train_val_only.zip
- Model-derived CTC alignments / prompt alignments (Protocol Z runtime compliant)
- Gold TextGrid intervals for offline diagnostic comparison only

Candidates evaluated:
1. Baseline T-OP (no filter, tau = 0.066045)
2. Global threshold adjustment
3. Phoneme-identity-only suppression (mute /d/, /t/)
4. Burst-only acoustic verifier
5. Duration-only acoustic verifier
6. Energy-only acoustic verifier
7. Full Multi-Feature Acoustic Verifier (Burst + Closure + Relative Energy)
8. CTC-Acoustic Hybrid Verifier

Outputs:
- STOP_VERIFIER_RESULTS.json
- STOP_ERROR_CASES.jsonl
- stop_summary.json (for report synthesis)
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

CAL_VAL_DIR = os.path.join(BENCH_CACHE, 't_op_fixed_budget_3600', 'protocol_z')
CAL_PATH = os.path.join(CAL_VAL_DIR, 'calibration', 'offline_token_evaluation.jsonl')
VAL_PATH = os.path.join(CAL_VAL_DIR, 'offline_token_evaluation.jsonl')

def extract_stop_acoustic_features(audio_data, sr, start_sec, end_sec):
    """
    Extracts physically grounded acoustic features from a waveform window:
    - Duration (ms)
    - Relative RMS energy (dB) compared to utterance RMS
    - Peak transient energy ratio (burst prominence)
    - Spectral centroid (Hz) in the release window
    - Zero-crossing rate (ZCR)
    - Low-frequency voicing energy (< 400 Hz) in closure
    """
    total_len = len(audio_data)
    dur_total = total_len / sr
    
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(total_len, int(end_sec * sr))
    
    if end_sample <= start_sample:
        return {
            'duration_ms': 0.0,
            'rms_rel_db': -100.0,
            'burst_prominence_db': 0.0,
            'spectral_centroid_hz': 0.0,
            'zcr': 0.0,
            'voicing_ratio': 0.0,
            'has_burst': False
        }
        
    segment = audio_data[start_sample:end_sample].astype(np.float32) / 32768.0
    dur_ms = (end_sample - start_sample) / sr * 1000.0
    
    # Utterance global RMS
    utt_rms = np.sqrt(np.mean((audio_data.astype(np.float32) / 32768.0) ** 2))
    seg_rms = np.sqrt(np.mean(segment ** 2))
    rms_rel_db = 20.0 * np.log10(max(seg_rms, 1e-6) / max(utt_rms, 1e-6))
    
    # ZCR
    zcr = np.mean(np.abs(np.diff(np.sign(segment)))) / 2.0 if len(segment) > 1 else 0.0
    
    # Split segment into closure (first 60%) and release (last 40%)
    split_idx = int(len(segment) * 0.6)
    closure_sig = segment[:split_idx]
    release_sig = segment[split_idx:]
    
    closure_rms = np.sqrt(np.mean(closure_sig ** 2)) if len(closure_sig) > 0 else 1e-6
    release_rms = np.sqrt(np.mean(release_sig ** 2)) if len(release_sig) > 0 else 1e-6
    
    # Burst prominence: release energy / closure energy
    burst_prominence_db = 20.0 * np.log10(max(release_rms, 1e-6) / max(closure_rms, 1e-6))
    
    # Spectral centroid in release window
    if len(release_sig) >= 64:
        freqs, psd = scipy.signal.welch(release_sig, fs=sr, nperseg=min(len(release_sig), 256))
        psd_sum = np.sum(psd)
        centroid = np.sum(freqs * psd) / psd_sum if psd_sum > 0 else 0.0
    else:
        centroid = 0.0
        
    # Low-frequency voicing energy in closure (< 400 Hz)
    if len(closure_sig) >= 64:
        freqs, psd = scipy.signal.welch(closure_sig, fs=sr, nperseg=min(len(closure_sig), 256))
        low_band = (freqs <= 400.0)
        voicing_energy = np.sum(psd[low_band])
        total_energy = np.sum(psd)
        voicing_ratio = voicing_energy / total_energy if total_energy > 0 else 0.0
    else:
        voicing_ratio = 0.0
        
    has_burst = (centroid > 1200.0 and burst_prominence_db > 6.0)
    
    return {
        'duration_ms': round(float(dur_ms), 1),
        'rms_rel_db': round(float(rms_rel_db), 2),
        'burst_prominence_db': round(float(burst_prominence_db), 2),
        'spectral_centroid_hz': round(float(centroid), 1),
        'zcr': round(float(zcr), 4),
        'voicing_ratio': round(float(voicing_ratio), 4),
        'has_burst': bool(has_burst)
    }

def main():
    t_start_all = time.time()
    print("Agent B: Running Stop-Consonant Acoustic Verification Benchmark...")
    
    with open(VAL_PATH) as f:
        val_records = [json.loads(line) for line in f]
    with open(CAL_PATH) as f:
        cal_records = [json.loads(line) for line in f]
        
    print(f"Loaded {len(cal_records)} CAL records and {len(val_records)} VAL records.")
    
    # Load CASE_MANIFEST to get interval alignments for VAL
    with open(CASE_MANIFEST_PATH) as f:
        case_manifest = json.load(f)
    case_map = {c['case_id']: c for c in case_manifest['cases']}
    
    # Preload WAV audio from zip archive for VAL speakers (NJS, RRBI)
    print("Preloading waveforms from archive...")
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
                        
    print(f"Loaded {len(val_audio_cache)} WAV utterances for VAL into memory.")
    
    # Preload CAL audio (NCC, THV)
    cal_audio_cache = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for spk in ['NCC', 'THV']:
            sz_bytes = z.read(f"{spk}.zip")
            with zipfile.ZipFile(io.BytesIO(sz_bytes)) as sz:
                for f in sz.namelist():
                    if f.endswith('.wav'):
                        utt_id = os.path.basename(f).replace('.wav', '')
                        sr, sig = wav.read(io.BytesIO(sz.read(f)))
                        cal_audio_cache[(spk, utt_id)] = (sr, sig)
    print(f"Loaded {len(cal_audio_cache)} WAV utterances for CAL into memory.")

    # 1. Baseline parameters
    base_tau = 0.066045
    
    # Extract features for all VAL stop tokens
    val_stop_records = []
    t_feat_start = time.time()
    for r in val_records:
        if r.get('mapping_status') == 'prompt_extra':
            continue
        phone = r.get('canonical_phone')
        if phone not in ('D', 'T'):
            continue
            
        spk = r['speaker_id']
        utt = r['utterance_id']
        case_id = f"case_{spk}_{utt}_{r.get('gold_canonical_index', r.get('prompt_index'))}"
        c = case_map.get(case_id)
        
        # Determine temporal interval
        # If c exists, use its model/prompt projected timing
        if c and c.get('time_start_sec') is not None:
            t_start = c['time_start_sec']
            t_end = c['time_end_sec']
        else:
            t_start = 0.0
            t_end = 0.1
            
        audio_entry = val_audio_cache.get((spk, utt))
        if audio_entry:
            sr, sig = audio_entry
            feats = extract_stop_acoustic_features(sig, sr, t_start, t_end)
        else:
            feats = {'duration_ms': 0.0, 'rms_rel_db': -100.0, 'burst_prominence_db': 0.0,
                     'spectral_centroid_hz': 0.0, 'zcr': 0.0, 'voicing_ratio': 0.0, 'has_burst': False}
                     
        val_stop_records.append({
            'record': r,
            'features': feats,
            'case': c
        })
        
    t_feat_end = time.time()
    feat_time_ms = (t_feat_end - t_feat_start) * 1000.0 / max(1, len(val_stop_records))
    print(f"Extracted features for {len(val_stop_records)} /d, t/ tokens in VAL ({feat_time_ms:.3f} ms/token).")

    # -------------------------------------------------------------------------
    # Gating Rules Definition:
    # An omission alarm is fired if s_omit >= base_tau.
    # The acoustic verifier checks if the token shows acoustic realization.
    # If the verifier decides the phone WAS realized, it SUPPRESSES the omission alert!
    # -------------------------------------------------------------------------
    
    candidates = [
        {
            'name': 'Original T-OP Baseline',
            'desc': 'No acoustic filter, tau = 0.066045',
            'filter_fn': lambda f: False # Never suppress
        },
        {
            'name': 'Global Conservative Threshold (tau=0.133940)',
            'desc': 'Uniform CAL 5% ceiling',
            'filter_fn': lambda f, r: r.get('s_omit', 0.0) < 0.133940 # Suppress if below conservative tau
        },
        {
            'name': 'Phoneme-Identity Suppression (Mute All {D, T})',
            'desc': 'Mute all omissions on /d/ and /t/',
            'filter_fn': lambda f: True # Always suppress
        },
        {
            'name': 'Candidate B1: Burst Detector Only',
            'desc': 'Suppress if spectral centroid > 1200 Hz AND burst prominence > 6 dB',
            'filter_fn': lambda f: f['has_burst']
        },
        {
            'name': 'Candidate B2: Duration & Energy Gate',
            'desc': 'Suppress if duration > 65 ms AND relative RMS > -3 dB',
            'filter_fn': lambda f: (f['duration_ms'] > 65.0 and f['rms_rel_db'] > -3.0)
        },
        {
            'name': 'Candidate B3: High-Energy Voiced Closure Gate',
            'desc': 'Suppress if voicing ratio > 0.40 AND relative RMS > -2.0 dB',
            'filter_fn': lambda f: (f['voicing_ratio'] > 0.40 and f['rms_rel_db'] > -2.0)
        },
        {
            'name': 'Candidate B4: Multi-Feature Stop Verifier (Primary)',
            'desc': 'Suppress if Burst Detected OR (Duration > 60 ms AND relative RMS > -3.5 dB)',
            'filter_fn': lambda f: f['has_burst'] or (f['duration_ms'] > 60.0 and f['rms_rel_db'] > -3.5)
        },
        {
            'name': 'Candidate B5: Conservative Stop Verifier (High Specificity)',
            'desc': 'Suppress if Burst Detected OR (Duration > 75 ms AND relative RMS > -2.0 dB)',
            'filter_fn': lambda f: f['has_burst'] or (f['duration_ms'] > 75.0 and f['rms_rel_db'] > -2.0)
        }
    ]

    # Evaluate candidates on the Stop cohort (/d/ and /t/ only)
    stop_results = []
    error_cases_by_cand = defaultdict(list)
    
    total_gold_d = sum(1 for item in val_stop_records if item['record']['canonical_phone'] == 'D')
    total_gold_t = sum(1 for item in val_stop_records if item['record']['canonical_phone'] == 'T')
    
    pos_stops = [item for item in val_stop_records if item['record'].get('operation_target') == 2]
    cor_neg_stops = [item for item in val_stop_records if item['record'].get('operation_target') == 0]
    sub_neg_stops = [item for item in val_stop_records if item['record'].get('operation_target') == 1]
    
    total_pos_stop = len(pos_stops)      # Gold omissions on D, T in VAL
    total_cor_stop = len(cor_neg_stops)  # Gold correct on D, T
    total_sub_stop = len(sub_neg_stops)  # Gold substitutions on D, T
    
    print(f"VAL /d, t/ cohort: {total_pos_stop} gold omissions, {total_cor_stop} correct, {total_sub_stop} substitutions.")

    for cand in candidates:
        fn_filter = cand['filter_fn']
        
        tp = 0
        fp_cor = 0
        fp_sub = 0
        fn = 0
        
        # Track which false alarms are filtered vs survive
        filtered_fp_cor = 0
        filtered_tp = 0 # True deletions mistakenly suppressed
        
        error_cases = []
        
        for item in val_stop_records:
            r = item['record']
            f = item['features']
            op = r.get('operation_target')
            s_omit = r.get('s_omit', 0.0)
            
            initial_alarm = (s_omit >= base_tau)
            
            # Check filter
            suppressed = False
            if initial_alarm:
                # filter_fn can take (f) or (f, r)
                try:
                    suppressed = fn_filter(f)
                except TypeError:
                    suppressed = fn_filter(f, r)
                    
            final_alarm = initial_alarm and (not suppressed)
            
            if op == 2:  # Gold omission
                if final_alarm:
                    tp += 1
                else:
                    fn += 1
                    if initial_alarm and suppressed:
                        filtered_tp += 1
                        error_cases.append({
                            'case_id': f"case_{r['speaker_id']}_{r['utterance_id']}_{r.get('gold_canonical_index')}",
                            'type': 'mistakenly_suppressed_true_deletion',
                            'phone': r['canonical_phone'],
                            'speaker': r['speaker_id'],
                            'utterance': r['utterance_id'],
                            's_omit': s_omit,
                            'features': f
                        })
            elif op == 0:  # Gold correct
                if final_alarm:
                    fp_cor += 1
                    error_cases.append({
                        'case_id': f"case_{r['speaker_id']}_{r['utterance_id']}_{r.get('gold_canonical_index')}",
                        'type': 'surviving_false_positive_correct',
                        'phone': r['canonical_phone'],
                        'speaker': r['speaker_id'],
                        'utterance': r['utterance_id'],
                        's_omit': s_omit,
                        'features': f
                    })
                else:
                    if initial_alarm and suppressed:
                        filtered_fp_cor += 1
            elif op == 1:  # Gold substitution
                if final_alarm:
                    fp_sub += 1

        total_alarms = tp + fp_cor + fp_sub
        prec_deployed = tp / total_alarms if total_alarms > 0 else 0.0
        recall_stop = tp / total_pos_stop if total_pos_stop > 0 else 0.0
        f1_stop = (2 * prec_deployed * recall_stop / (prec_deployed + recall_stop)) if (prec_deployed + recall_stop) > 0 else 0.0
        fp_reduction_pct = (filtered_fp_cor / 673.0 * 100.0) if 673 > 0 else 0.0
        
        cand_res = {
            'candidate_name': cand['name'],
            'description': cand['desc'],
            'stop_cohort_TP': tp,
            'stop_cohort_FP_cor': fp_cor,
            'stop_cohort_FP_sub': fp_sub,
            'stop_cohort_FN': fn,
            'stop_total_alarms': total_alarms,
            'stop_filtered_FP_cor': filtered_fp_cor,
            'stop_filtered_true_deletions': filtered_tp,
            'stop_deployed_precision': round(prec_deployed, 6),
            'stop_recall': round(recall_stop, 6),
            'stop_f1': round(f1_stop, 6),
            'stop_fp_reduction_pct': round(fp_reduction_pct, 2)
        }
        stop_results.append(cand_res)
        error_cases_by_cand[cand['name']] = error_cases

    # Save STOP_VERIFIER_RESULTS.json
    res_path = os.path.join(os.path.dirname(__file__), 'STOP_VERIFIER_RESULTS.json')
    with open(res_path, 'w') as f:
        json.dump(stop_results, f, indent=2)
        
    # Also save to root
    with open(os.path.join(ROOT_DIR, 'STOP_VERIFIER_RESULTS.json'), 'w') as f:
        json.dump(stop_results, f, indent=2)

    # Export STOP_ERROR_CASES.jsonl for Candidate B4 (Multi-Feature Stop Verifier)
    err_path = os.path.join(os.path.dirname(__file__), 'STOP_ERROR_CASES.jsonl')
    b4_errors = error_cases_by_cand['Candidate B4: Multi-Feature Stop Verifier (Primary)']
    with open(err_path, 'w') as f:
        for err in b4_errors[:200]:  # Cap at 200 cases
            f.write(json.dumps(err) + '\n')
    with open(os.path.join(ROOT_DIR, 'STOP_ERROR_CASES.jsonl'), 'w') as f:
        for err in b4_errors[:200]:
            f.write(json.dumps(err) + '\n')

    # Print summary table
    print("\n" + "="*120)
    print(f"{'Stop Verifier Candidate':<48} | {'TP':<4} | {'FP_cor':<6} | {'Alarms':<6} | {'Prec(dep)':<9} | {'Recall':<7} | {'Filtered FP':<11} | {'Cost TP':<7}")
    print("="*120)
    for r in stop_results:
        print(f"{r['candidate_name']:<48} | {r['stop_cohort_TP']:<4} | {r['stop_cohort_FP_cor']:<6} | {r['stop_total_alarms']:<6} | {r['stop_deployed_precision']:<9.4f} | {r['stop_recall']:<7.4f} | {r['stop_filtered_FP_cor']:<11} | {r['stop_filtered_true_deletions']:<7}")
    print("="*120)
    print(f"Stop acoustic verification completed in {time.time() - t_start_all:.2f} s.")

if __name__ == '__main__':
    main()
