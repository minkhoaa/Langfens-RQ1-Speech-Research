#!/usr/bin/env python3
"""
LANGFENS — RQ1: Agent D — Acoustic Evidence Analysis Harness
Extracts acoustic metrics from Fold-1 VAL waveforms for sampled cases in ANNOTATION_REVIEW_LEDGER.jsonl.
Computes:
  - Duration (ms)
  - RMS energy (dB) of target phone vs utterance mean
  - Spectral centroid (Hz)
  - Zero-crossing rate (ZCR)
  - Contrast between Matched and Mismatched audio (from AUDIO_MISMATCH_MAPPING.json)
  - Separation of measured acoustic evidence from mechanistic hypotheses.
Outputs summary statistics to audit_scripts/acoustic_evidence_summary.json.
"""

import io
import json
import math
import os
import sys
import zipfile
from collections import defaultdict
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ZIP_PATH = os.path.join(ROOT_DIR, 'datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip')
LEDGER_PATH = os.path.join(ROOT_DIR, 'ANNOTATION_REVIEW_LEDGER.jsonl')
MISMATCH_PATH = '/home/khoa/.cache/langfens/psup-benchmark-v1/recovered-final-v2-20261104/AUDIO_MISMATCH_MAPPING.json'

def compute_acoustic_features(audio_segment, sr):
    if len(audio_segment) == 0:
        return {'rms_db': -100.0, 'spectral_centroid_hz': 0.0, 'zcr': 0.0}
    
    # Float normalization
    sig = audio_segment.astype(np.float32) / 32768.0
    
    # RMS Energy in dB
    rms = np.sqrt(np.mean(sig ** 2))
    rms_db = 20.0 * np.log10(max(rms, 1e-6))
    
    # Zero Crossing Rate
    zcr = np.mean(np.abs(np.diff(np.sign(sig)))) / 2.0
    
    # Spectral Centroid
    if len(sig) >= 64:
        freqs, psd = scipy.signal.welch(sig, fs=sr, nperseg=min(len(sig), 256))
        psd_sum = np.sum(psd)
        centroid = np.sum(freqs * psd) / psd_sum if psd_sum > 0 else 0.0
    else:
        centroid = 0.0
        
    return {
        'rms_db': round(float(rms_db), 2),
        'spectral_centroid_hz': round(float(centroid), 1),
        'zcr': round(float(zcr), 4)
    }

def main():
    print("Agent D: Computing acoustic evidence for sampled review cases...")
    
    with open(LEDGER_PATH) as f:
        ledger = [json.loads(line) for line in f]
    print(f"Loaded {len(ledger)} cases from ledger.")
    
    with open(MISMATCH_PATH) as f:
        mismatch_map = json.load(f)
        
    # Preload WAV audio from staging zip
    wav_cache = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for spk in ['NJS', 'RRBI']:
            with zipfile.ZipFile(io.BytesIO(z.read(f'{spk}.zip'))) as rz:
                for name in rz.namelist():
                    if name.endswith('.wav'):
                        uid = name.split('/')[-1].replace('.wav', '')
                        data = rz.read(name)
                        sr, audio = wav.read(io.BytesIO(data))
                        wav_cache[(spk, uid)] = (sr, audio)
    print(f"Loaded {len(wav_cache)} audio files into memory.")
    
    acoustic_results = []
    by_category_features = defaultdict(lambda: defaultdict(list))
    
    for rec in ledger:
        b = rec['blinded_presentation']
        spk = b['speaker_id']
        uid = b['utterance_id']
        t_start = b['target_xmin_sec']
        t_end = b['target_xmax_sec']
        cat = rec['audit_results']['disposition_category']
        phone = b['canonical_phone']
        
        feat = None
        mismatched_feat = None
        
        if (spk, uid) in wav_cache and t_start is not None and t_end is not None:
            sr, full_audio = wav_cache[(spk, uid)]
            i_start = int(t_start * sr)
            i_end = int(t_end * sr)
            seg = full_audio[i_start:i_end]
            feat = compute_acoustic_features(seg, sr)
            
            # Utterance mean RMS
            utt_rms = compute_acoustic_features(full_audio, sr)['rms_db']
            relative_energy_db = round(feat['rms_db'] - utt_rms, 2)
            feat['relative_energy_db'] = relative_energy_db
            
            # Mismatched audio comparison
            # In mismatch mapping, key is e.g. "NJS/arctic_a0003" -> "NJS/arctic_a0005"
            map_key = f"{spk}/{uid}"
            mismatched_target = mismatch_map.get(map_key)
            if mismatched_target:
                m_spk, m_uid = mismatched_target.split('/')
                if (m_spk, m_uid) in wav_cache:
                    m_sr, m_audio = wav_cache[(m_spk, m_uid)]
                    # Extract same time window from mismatched audio
                    m_seg = m_audio[i_start:min(len(m_audio), i_end)]
                    mismatched_feat = compute_acoustic_features(m_seg, m_sr)
                    m_utt_rms = compute_acoustic_features(m_audio, m_sr)['rms_db']
                    mismatched_feat['relative_energy_db'] = round(mismatched_feat['rms_db'] - m_utt_rms, 2)
                    
            dur_ms = round((t_end - t_start) * 1000.0, 2)
            by_category_features[cat]['duration_ms'].append(dur_ms)
            by_category_features[cat]['relative_energy_db'].append(relative_energy_db)
            by_category_features[cat]['spectral_centroid_hz'].append(feat['spectral_centroid_hz'])
            by_category_features[cat]['zcr'].append(feat['zcr'])
            
        acoustic_results.append({
            'case_id': rec['case_id'],
            'category': cat,
            'canonical_phone': phone,
            'duration_ms': dur_ms if t_start is not None else None,
            'matched_acoustic': feat,
            'mismatched_acoustic': mismatched_feat,
            'model_s_omit': rec['model_scores']['s_omit_top_fixed']
        })
        
    print("\n--- ACOUSTIC EVIDENCE SUMMARY BY CATEGORY ---")
    summary_by_cat = {}
    for cat, metrics in by_category_features.items():
        summary_by_cat[cat] = {
            'count': len(metrics['duration_ms']),
            'mean_duration_ms': round(float(np.mean(metrics['duration_ms'])), 1),
            'median_duration_ms': round(float(np.median(metrics['duration_ms'])), 1),
            'mean_rel_energy_db': round(float(np.mean(metrics['relative_energy_db'])), 1),
            'mean_spectral_centroid_hz': round(float(np.mean(metrics['spectral_centroid_hz'])), 1),
            'mean_zcr': round(float(np.mean(metrics['zcr'])), 4)
        }
        print(f"Category: {cat:<35} (N={summary_by_cat[cat]['count']:2d})")
        print(f"  Duration: Mean={summary_by_cat[cat]['mean_duration_ms']}ms, Median={summary_by_cat[cat]['median_duration_ms']}ms")
        print(f"  Rel Energy: Mean={summary_by_cat[cat]['mean_rel_energy_db']} dB | Centroid: {summary_by_cat[cat]['mean_spectral_centroid_hz']} Hz | ZCR: {summary_by_cat[cat]['mean_zcr']}")
        
    out_summary = os.path.join(ROOT_DIR, 'audit_scripts', 'acoustic_evidence_summary.json')
    with open(out_summary, 'w') as f:
        json.dump({
            'summary_by_category': summary_by_cat,
            'case_measurements': acoustic_results
        }, f, indent=2)
    print(f"Saved acoustic summary to {out_summary}.")

if __name__ == '__main__':
    main()
