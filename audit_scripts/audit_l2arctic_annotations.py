import zipfile
import io
import re
import csv
import json
import wave
from collections import defaultdict, Counter

# Import our textgrid parser
from textgrid_parser import parse_textgrid

SPEAKERS_INFO = {
    'ABA': {'gender': 'M', 'l1': 'Arabic'},
    'SKA': {'gender': 'F', 'l1': 'Arabic'},
    'YBAA': {'gender': 'M', 'l1': 'Arabic'},
    'ZHAA': {'gender': 'F', 'l1': 'Arabic'},
    'BWC': {'gender': 'M', 'l1': 'Chinese'},
    'LXC': {'gender': 'F', 'l1': 'Chinese'},
    'NCC': {'gender': 'F', 'l1': 'Chinese'},
    'TXHC': {'gender': 'M', 'l1': 'Chinese'},
    'ASI': {'gender': 'M', 'l1': 'Hindi'},
    'RRBI': {'gender': 'M', 'l1': 'Hindi'},
    'SVBI': {'gender': 'F', 'l1': 'Hindi'},
    'TNI': {'gender': 'F', 'l1': 'Hindi'},
    'HJK': {'gender': 'F', 'l1': 'Korean'},
    'HKK': {'gender': 'M', 'l1': 'Korean'},
    'YDCK': {'gender': 'F', 'l1': 'Korean'},
    'YKWK': {'gender': 'M', 'l1': 'Korean'},
    'EBVS': {'gender': 'M', 'l1': 'Spanish'},
    'ERMS': {'gender': 'M', 'l1': 'Spanish'},
    'MBMPS': {'gender': 'F', 'l1': 'Spanish'},
    'NJS': {'gender': 'F', 'l1': 'Spanish'},
    'HQTV': {'gender': 'M', 'l1': 'Vietnamese'},
    'PNV': {'gender': 'F', 'l1': 'Vietnamese'},
    'THV': {'gender': 'F', 'l1': 'Vietnamese'},
    'TLV': {'gender': 'M', 'l1': 'Vietnamese'},
}

# Standard ARPAbet phoneme base symbols (without stress digits)
STANDARD_ARPABET = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'B', 'CH', 'D', 'DH', 
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 
    'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 
    'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
}
SILENCE_TAGS = {'', 'sil', 'sp', 'spn', 'silence'}

def strip_stress(phone):
    return re.sub(r'[0-9]+$', '', phone).strip().upper()

def run_l2arctic_audit(zip_path):
    zf = zipfile.ZipFile(zip_path, 'r')
    
    total_annotations_inspected = 0
    speaker_stats = {}
    
    # Aggregators
    error_events = [] # list of dicts
    label_issues = [] # list of dicts
    
    # Global counters
    global_counts = {
        'total_phones_all': 0,
        'total_phones_nonsil': 0,
        'correct_phones': 0,
        'substitution_count': 0,
        'deletion_count': 0,
        'addition_count': 0,
        'err_tags': 0,
        'asterisk_tags': 0,
        'unknown_symbols': Counter(),
        'whitespace_issue_count': 0,
    }

    # Per L1 counters
    l1_stats = defaultdict(lambda: {
        'utterances': 0,
        'phones_nonsil': 0,
        'correct': 0,
        'sub': 0,
        'del': 0,
        'add': 0,
        'err': 0,
        'asterisk': 0,
    })

    print("Beginning detailed inspection of 24 speaker archives...")

    issue_id_counter = 1

    for spk, meta in SPEAKERS_INFO.items():
        print(f"Auditing speaker: {spk} ({meta['l1']}, {meta['gender']})...")
        spk_zip_bytes = zf.read(f"{spk}.zip")
        szf = zipfile.ZipFile(io.BytesIO(spk_zip_bytes))
        
        all_names = szf.namelist()
        wav_files = {n.split('/')[-1].replace('.wav', ''): n for n in all_names if n.startswith(f"{spk}/wav/") and n.endswith('.wav')}
        transcript_files = {n.split('/')[-1].replace('.txt', ''): n for n in all_names if n.startswith(f"{spk}/transcript/") and n.endswith('.txt')}
        textgrid_files = {n.split('/')[-1].replace('.TextGrid', ''): n for n in all_names if n.startswith(f"{spk}/textgrid/") and n.endswith('.TextGrid')}
        annotation_files = {n.split('/')[-1].replace('.TextGrid', ''): n for n in all_names if n.startswith(f"{spk}/annotation/") and n.endswith('.TextGrid')}
        
        spk_total_wavs = len(wav_files)
        spk_total_ann = len(annotation_files)
        total_annotations_inspected += spk_total_ann
        
        spk_counts = {
            'phones_all': 0,
            'phones_nonsil': 0,
            'correct': 0,
            'sub': 0,
            'del': 0,
            'add': 0,
            'err': 0,
            'asterisk': 0,
        }
        
        for utt_id, ann_path in annotation_files.items():
            # Check consistency C6:
            # 1. Matching wav exists?
            if utt_id not in wav_files:
                label_issues.append({
                    'issue_id': f"ISSUE_{issue_id_counter:06d}",
                    'speaker_id': spk,
                    'utterance_id': utt_id,
                    'file_path': ann_path,
                    'tier_name': 'N/A',
                    'interval_index': -1,
                    'start_time': -1,
                    'end_time': -1,
                    'raw_label': 'N/A',
                    'issue_category': 'MISSING_AUDIO',
                    'description': f"Annotation {ann_path} has no corresponding WAV file in {spk}/wav/"
                })
                issue_id_counter += 1
            
            # 2. Matching transcript exists?
            if utt_id not in transcript_files:
                label_issues.append({
                    'issue_id': f"ISSUE_{issue_id_counter:06d}",
                    'speaker_id': spk,
                    'utterance_id': utt_id,
                    'file_path': ann_path,
                    'tier_name': 'N/A',
                    'interval_index': -1,
                    'start_time': -1,
                    'end_time': -1,
                    'raw_label': 'N/A',
                    'issue_category': 'MISSING_TRANSCRIPT',
                    'description': f"Annotation {ann_path} has no corresponding transcript in {spk}/transcript/"
                })
                issue_id_counter += 1
            
            # Read TextGrid content
            tg_raw = szf.read(ann_path).decode('utf-8', errors='replace')
            tiers = parse_textgrid(tg_raw)
            
            if 'phones' not in tiers:
                label_issues.append({
                    'issue_id': f"ISSUE_{issue_id_counter:06d}",
                    'speaker_id': spk,
                    'utterance_id': utt_id,
                    'file_path': ann_path,
                    'tier_name': 'N/A',
                    'interval_index': -1,
                    'start_time': -1,
                    'end_time': -1,
                    'raw_label': 'N/A',
                    'issue_category': 'CORRUPTED_TEXTGRID',
                    'description': f"TextGrid {ann_path} missing 'phones' tier"
                })
                issue_id_counter += 1
                continue
            
            # Verify audio duration vs textgrid xmax
            audio_duration = None
            if utt_id in wav_files:
                try:
                    wav_bytes = szf.read(wav_files[utt_id])
                    with wave.open(io.BytesIO(wav_bytes)) as w:
                        audio_duration = w.getnframes() / float(w.getframerate())
                except Exception as e:
                    label_issues.append({
                        'issue_id': f"ISSUE_{issue_id_counter:06d}",
                        'speaker_id': spk,
                        'utterance_id': utt_id,
                        'file_path': wav_files[utt_id],
                        'tier_name': 'N/A',
                        'interval_index': -1,
                        'start_time': -1,
                        'end_time': -1,
                        'raw_label': 'N/A',
                        'issue_category': 'CORRUPTED_AUDIO',
                        'description': f"Could not read WAV header: {str(e)}"
                    })
                    issue_id_counter += 1

            tg_xmax = tiers['phones']['xmax']
            if audio_duration is not None and abs(tg_xmax - audio_duration) > 0.05: # > 50ms discrepancy
                label_issues.append({
                    'issue_id': f"ISSUE_{issue_id_counter:06d}",
                    'speaker_id': spk,
                    'utterance_id': utt_id,
                    'file_path': ann_path,
                    'tier_name': 'phones',
                    'interval_index': -1,
                    'start_time': 0,
                    'end_time': tg_xmax,
                    'raw_label': f"tg_xmax={tg_xmax}, wav_dur={audio_duration}",
                    'issue_category': 'DURATION_MISMATCH',
                    'description': f"TextGrid duration ({tg_xmax:.3f}s) differs from audio duration ({audio_duration:.3f}s) by > 50ms"
                })
                issue_id_counter += 1

            # Parse intervals in phones tier
            intervals = tiers['phones']['intervals']
            prev_end = 0.0
            for idx, (xmin, xmax, raw_label) in enumerate(intervals):
                spk_counts['phones_all'] += 1
                global_counts['total_phones_all'] += 1
                
                # Check boundary sanity
                if xmin >= xmax:
                    label_issues.append({
                        'issue_id': f"ISSUE_{issue_id_counter:06d}",
                        'speaker_id': spk,
                        'utterance_id': utt_id,
                        'file_path': ann_path,
                        'tier_name': 'phones',
                        'interval_index': idx + 1,
                        'start_time': xmin,
                        'end_time': xmax,
                        'raw_label': raw_label,
                        'issue_category': 'INVALID_INTERVAL_BOUNDARY',
                        'description': f"Interval start ({xmin}) >= end ({xmax})"
                    })
                    issue_id_counter += 1
                
                if xmin < prev_end - 0.001: # negative overlap > 1ms
                    label_issues.append({
                        'issue_id': f"ISSUE_{issue_id_counter:06d}",
                        'speaker_id': spk,
                        'utterance_id': utt_id,
                        'file_path': ann_path,
                        'tier_name': 'phones',
                        'interval_index': idx + 1,
                        'start_time': xmin,
                        'end_time': xmax,
                        'raw_label': raw_label,
                        'issue_category': 'OVERLAPPING_INTERVALS',
                        'description': f"Interval start ({xmin}) precedes previous interval end ({prev_end})"
                    })
                    issue_id_counter += 1
                prev_end = xmax
                
                # Check whitespace issues
                trimmed_label = raw_label.strip()
                if raw_label != trimmed_label:
                    global_counts['whitespace_issue_count'] += 1
                    # Record representative sample
                    if global_counts['whitespace_issue_count'] <= 100:
                        label_issues.append({
                            'issue_id': f"ISSUE_{issue_id_counter:06d}",
                            'speaker_id': spk,
                            'utterance_id': utt_id,
                            'file_path': ann_path,
                            'tier_name': 'phones',
                            'interval_index': idx + 1,
                            'start_time': xmin,
                            'end_time': xmax,
                            'raw_label': raw_label,
                            'issue_category': 'TRAILING_OR_LEADING_WHITESPACE',
                            'description': f"Raw label contains unstripped whitespace: {repr(raw_label)}"
                        })
                        issue_id_counter += 1

                # Empty or silence
                if trimmed_label.lower() in SILENCE_TAGS:
                    continue
                
                # Check if it has comma (error tag)
                if ',' in trimmed_label:
                    parts = [p.strip() for p in trimmed_label.split(',')]
                    
                    if len(parts) == 3:
                        cpl, ppl, err_type = parts[0], parts[1], parts[2]
                        err_type = err_type.lower()
                        
                        has_err = ('err' in cpl.lower() or 'err' in ppl.lower())
                        has_star = ('*' in cpl or '*' in ppl)
                        if has_err:
                            spk_counts['err'] += 1
                            global_counts['err_tags'] += 1
                            l1_stats[meta['l1']]['err'] += 1
                        if has_star:
                            spk_counts['asterisk'] += 1
                            global_counts['asterisk_tags'] += 1
                            l1_stats[meta['l1']]['asterisk'] += 1

                        # Clean phone symbols for inventory check
                        clean_cpl = strip_stress(cpl.replace('*', ''))
                        clean_ppl = strip_stress(ppl.replace('*', ''))

                        if err_type == 's':
                            spk_counts['sub'] += 1
                            spk_counts['phones_nonsil'] += 1
                            global_counts['substitution_count'] += 1
                            global_counts['total_phones_nonsil'] += 1
                            l1_stats[meta['l1']]['sub'] += 1
                            l1_stats[meta['l1']]['phones_nonsil'] += 1
                            
                            error_events.append({
                                'speaker_id': spk,
                                'gender': meta['gender'],
                                'l1': meta['l1'],
                                'utterance_id': utt_id,
                                'error_type': 'substitution',
                                'canonical_phone': cpl,
                                'realized_phone': ppl,
                                'clean_canonical': clean_cpl,
                                'clean_realized': clean_ppl,
                                'has_err_tag': has_err,
                                'has_deviation_star': has_star,
                                'duration': xmax - xmin
                            })
                            
                        elif err_type == 'd':
                            spk_counts['del'] += 1
                            spk_counts['phones_nonsil'] += 1
                            global_counts['deletion_count'] += 1
                            global_counts['total_phones_nonsil'] += 1
                            l1_stats[meta['l1']]['del'] += 1
                            l1_stats[meta['l1']]['phones_nonsil'] += 1
                            
                            error_events.append({
                                'speaker_id': spk,
                                'gender': meta['gender'],
                                'l1': meta['l1'],
                                'utterance_id': utt_id,
                                'error_type': 'deletion',
                                'canonical_phone': cpl,
                                'realized_phone': ppl,
                                'clean_canonical': clean_cpl,
                                'clean_realized': clean_ppl,
                                'has_err_tag': has_err,
                                'has_deviation_star': has_star,
                                'duration': xmax - xmin
                            })
                            
                        elif err_type == 'a':
                            spk_counts['add'] += 1
                            # Addition is an inserted phone segment
                            global_counts['addition_count'] += 1
                            l1_stats[meta['l1']]['add'] += 1
                            
                            error_events.append({
                                'speaker_id': spk,
                                'gender': meta['gender'],
                                'l1': meta['l1'],
                                'utterance_id': utt_id,
                                'error_type': 'addition',
                                'canonical_phone': cpl,
                                'realized_phone': ppl,
                                'clean_canonical': clean_cpl,
                                'clean_realized': clean_ppl,
                                'has_err_tag': has_err,
                                'has_deviation_star': has_star,
                                'duration': xmax - xmin
                            })
                        else:
                            # Malformed error type
                            label_issues.append({
                                'issue_id': f"ISSUE_{issue_id_counter:06d}",
                                'speaker_id': spk,
                                'utterance_id': utt_id,
                                'file_path': ann_path,
                                'tier_name': 'phones',
                                'interval_index': idx + 1,
                                'start_time': xmin,
                                'end_time': xmax,
                                'raw_label': raw_label,
                                'issue_category': 'MALFORMED_ERROR_TAG',
                                'description': f"Unknown error type '{err_type}' in 3-part tag: {raw_label}"
                            })
                            issue_id_counter += 1
                    else:
                        label_issues.append({
                            'issue_id': f"ISSUE_{issue_id_counter:06d}",
                            'speaker_id': spk,
                            'utterance_id': utt_id,
                            'file_path': ann_path,
                            'tier_name': 'phones',
                            'interval_index': idx + 1,
                            'start_time': xmin,
                            'end_time': xmax,
                            'raw_label': raw_label,
                            'issue_category': 'MALFORMED_COMMA_TAG',
                            'description': f"Tag contains comma but not 3 parts: {raw_label}"
                        })
                        issue_id_counter += 1
                else:
                    # Single phone segment -> Correctly pronounced
                    clean_phone = strip_stress(trimmed_label)
                    spk_counts['correct'] += 1
                    spk_counts['phones_nonsil'] += 1
                    global_counts['correct_phones'] += 1
                    global_counts['total_phones_nonsil'] += 1
                    l1_stats[meta['l1']]['correct'] += 1
                    l1_stats[meta['l1']]['phones_nonsil'] += 1
                    
                    if clean_phone not in STANDARD_ARPABET and clean_phone not in {'<UNK>', 'UNK'}:
                        global_counts['unknown_symbols'][clean_phone] += 1
                        label_issues.append({
                            'issue_id': f"ISSUE_{issue_id_counter:06d}",
                            'speaker_id': spk,
                            'utterance_id': utt_id,
                            'file_path': ann_path,
                            'tier_name': 'phones',
                            'interval_index': idx + 1,
                            'start_time': xmin,
                            'end_time': xmax,
                            'raw_label': raw_label,
                            'issue_category': 'NONSTANDARD_PHONE_SYMBOL',
                            'description': f"Symbol '{clean_phone}' is not in standard 39-ARPAbet phone set"
                        })
                        issue_id_counter += 1

        l1_stats[meta['l1']]['utterances'] += spk_total_ann
        
        tot_err = spk_counts['sub'] + spk_counts['del'] + spk_counts['add']
        tot_eval_targets = spk_counts['correct'] + spk_counts['sub'] + spk_counts['del'] # canonical targets
        err_rate = (tot_err / tot_eval_targets * 100.0) if tot_eval_targets > 0 else 0.0
        
        speaker_stats[spk] = {
            'speaker_id': spk,
            'gender': meta['gender'],
            'l1': meta['l1'],
            'total_wavs': spk_total_wavs,
            'total_annotations': spk_total_ann,
            'annotated_phones_nonsil': spk_counts['phones_nonsil'],
            'correct_phones_count': spk_counts['correct'],
            'substitution_count': spk_counts['sub'],
            'deletion_count': spk_counts['del'],
            'addition_count': spk_counts['add'],
            'total_errors': tot_err,
            'error_rate_pct': round(err_rate, 2),
            'err_tag_count': spk_counts['err'],
            'asterisk_deviation_count': spk_counts['asterisk']
        }

    zf.close()
    print("Inspection complete.")
    return {
        'total_inspected': total_annotations_inspected,
        'global_counts': global_counts,
        'speaker_stats': speaker_stats,
        'l1_stats': dict(l1_stats),
        'error_events': error_events,
        'label_issues': label_issues
    }

if __name__ == "__main__":
    res = run_l2arctic_audit("datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip")
    
    # Save outputs
    print("\nWriting deliverables...")
    # 1. Speaker statistics CSV
    with open("manifests/RQ1-AUD-01/L2_ARCTIC_SPEAKER_STATISTICS.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'speaker_id', 'gender', 'l1', 'total_wavs', 'total_annotations',
            'annotated_phones_nonsil', 'correct_phones_count', 'substitution_count',
            'deletion_count', 'addition_count', 'total_errors', 'error_rate_pct',
            'err_tag_count', 'asterisk_deviation_count'
        ])
        writer.writeheader()
        for row in res['speaker_stats'].values():
            writer.writerow(row)
    print("Saved manifests/RQ1-AUD-01/L2_ARCTIC_SPEAKER_STATISTICS.csv")

    # 2. Error statistics CSV
    # Aggregate error patterns: (canonical, realized, error_type, l1) -> count
    pattern_counts = Counter()
    for ev in res['error_events']:
        key = (ev['clean_canonical'], ev['clean_realized'], ev['error_type'], ev['l1'])
        pattern_counts[key] += 1
    
    with open("manifests/RQ1-AUD-01/L2_ARCTIC_ERROR_STATISTICS.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['canonical_phone', 'realized_phone', 'error_type', 'l1', 'frequency'])
        for (c, r, et, l1), freq in pattern_counts.most_common():
            writer.writerow([c, r, et, l1, freq])
    print("Saved manifests/RQ1-AUD-01/L2_ARCTIC_ERROR_STATISTICS.csv")

    # 3. Label issues CSV
    with open("manifests/RQ1-AUD-01/L2_ARCTIC_LABEL_ISSUES.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'issue_id', 'speaker_id', 'utterance_id', 'file_path', 'tier_name',
            'interval_index', 'start_time', 'end_time', 'raw_label', 'issue_category',
            'description'
        ])
        writer.writeheader()
        for row in res['label_issues']:
            writer.writerow(row)
    print("Saved manifests/RQ1-AUD-01/L2_ARCTIC_LABEL_ISSUES.csv")

    # 4. Save JSON summary for reporting
    summary = {
        'total_annotations_inspected': res['total_inspected'],
        'global_counts': {
            k: (v if not isinstance(v, Counter) else dict(v)) for k, v in res['global_counts'].items()
        },
        'l1_stats': res['l1_stats']
    }
    with open("audit_scripts/l2arctic_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved audit_scripts/l2arctic_audit_summary.json")

    print("\n=== AUDIT RESULTS PREVIEW ===")
    print(f"Total annotations inspected: {res['total_inspected']}")
    gc = res['global_counts']
    print(f"Total phones (including sil): {gc['total_phones_all']}")
    print(f"Total non-silence phone tokens: {gc['total_phones_nonsil']}")
    print(f"Correctly pronounced tokens: {gc['correct_phones']}")
    print(f"Substitutions: {gc['substitution_count']}")
    print(f"Deletions: {gc['deletion_count']}")
    print(f"Additions: {gc['addition_count']}")
    print(f"Err tags: {gc['err_tags']}")
    print(f"Asterisk deviation tags: {gc['asterisk_tags']}")
    print(f"Total label issues flagged: {len(res['label_issues'])}")
