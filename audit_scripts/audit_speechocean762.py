import tarfile
import json
import csv
import math
from collections import defaultdict, Counter

def run_speechocean_audit(tar_path):
    print("=== Auditing SpeechOcean762 ===")
    tf = tarfile.open(tar_path, 'r:gz')

    scores_raw = tf.extractfile(tf.getmember('speechocean762/resource/scores.json')).read().decode('utf-8')
    scores = json.loads(scores_raw)

    detail_raw = tf.extractfile(tf.getmember('speechocean762/resource/scores-detail.json')).read().decode('utf-8')
    details = json.loads(detail_raw)

    # Metadata splits
    train_utts = set(l.split()[0] for l in tf.extractfile(tf.getmember('speechocean762/train/utt2spk')).read().decode('utf-8').splitlines() if l.strip())
    test_utts = set(l.split()[0] for l in tf.extractfile(tf.getmember('speechocean762/test/utt2spk')).read().decode('utf-8').splitlines() if l.strip())

    spk2gender = {}
    for split in ['train', 'test']:
        for l in tf.extractfile(tf.getmember(f'speechocean762/{split}/spk2gender')).read().decode('utf-8').splitlines():
            if l.strip():
                p = l.split()
                spk2gender[p[0]] = p[1]

    spk2age = {}
    for split in ['train', 'test']:
        for l in tf.extractfile(tf.getmember(f'speechocean762/{split}/spk2age')).read().decode('utf-8').splitlines():
            if l.strip():
                p = l.split()
                spk2age[p[0]] = p[1]

    utt2spk = {}
    for split in ['train', 'test']:
        for l in tf.extractfile(tf.getmember(f'speechocean762/{split}/utt2spk')).read().decode('utf-8').splitlines():
            if l.strip():
                p = l.split()
                utt2spk[p[0]] = p[1]

    # Metrics collectors
    dimension_values = defaultdict(list)
    rater_variances = defaultdict(list)
    
    total_words = 0
    total_phones = 0
    phone_score_histogram = Counter() # bin 0.0 to 2.0
    phone_tokens_by_symbol = Counter()
    
    label_issues = []
    issue_counter = 1

    # Check each utterance
    for utt_id, utt in scores.items():
        split = "train" if utt_id in train_utts else ("test" if utt_id in test_utts else "unknown")
        spk = utt2spk.get(utt_id, "unknown")
        
        # Check utterance dimensions
        for dim in ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']:
            val = utt.get(dim)
            if val is None:
                label_issues.append({
                    'issue_id': f"SO_ISSUE_{issue_counter:06d}",
                    'utterance_id': utt_id,
                    'speaker_id': spk,
                    'split': split,
                    'dimension': dim,
                    'level': 'utterance',
                    'observed_value': 'None',
                    'issue_category': 'MISSING_DIMENSION',
                    'description': f"Utterance {utt_id} missing dimension {dim}"
                })
                issue_counter += 1
            else:
                dimension_values[f"utt_{dim}"].append(float(val))
                # Check range (accuracy, fluency, prosodic, total: 0-10; completeness: 0-10 or 0-1)
                max_expected = 10.0
                if val < 0 or val > max_expected:
                    label_issues.append({
                        'issue_id': f"SO_ISSUE_{issue_counter:06d}",
                        'utterance_id': utt_id,
                        'speaker_id': spk,
                        'split': split,
                        'dimension': dim,
                        'level': 'utterance',
                        'observed_value': str(val),
                        'issue_category': 'OUT_OF_RANGE_VALUE',
                        'description': f"Value {val} outside [0, {max_expected}]"
                    })
                    issue_counter += 1

        # Check detail rater variability if available
        det = details.get(utt_id)
        if det:
            for dim in ['accuracy', 'completeness', 'fluency', 'prosodic']:
                r_scores = det.get(dim, [])
                if len(r_scores) >= 2:
                    mean_r = sum(r_scores) / len(r_scores)
                    var_r = sum((x - mean_r) ** 2 for x in r_scores) / len(r_scores)
                    std_r = math.sqrt(var_r)
                    rater_variances[f"utt_{dim}"].append(std_r)

        # Inspect words and phones
        words = utt.get('words', [])
        total_words += len(words)
        
        for w_idx, w in enumerate(words):
            w_text = w.get('text', '')
            w_acc = w.get('accuracy')
            w_stress = w.get('stress')
            w_tot = w.get('total')
            
            if w_acc is not None:
                dimension_values['word_accuracy'].append(float(w_acc))
            if w_stress is not None:
                dimension_values['word_stress'].append(float(w_stress))
            if w_tot is not None:
                dimension_values['word_total'].append(float(w_tot))

            phones = w.get('phones', [])
            phones_acc = w.get('phones-accuracy', [])
            
            if len(phones) != len(phones_acc):
                label_issues.append({
                    'issue_id': f"SO_ISSUE_{issue_counter:06d}",
                    'utterance_id': utt_id,
                    'speaker_id': spk,
                    'split': split,
                    'dimension': 'phones-accuracy',
                    'level': 'word',
                    'observed_value': f"len(phones)={len(phones)}, len(acc)={len(phones_acc)}",
                    'issue_category': 'PHONE_SCORE_LENGTH_MISMATCH',
                    'description': f"In word '{w_text}', phones list length != phones-accuracy list length"
                })
                issue_counter += 1

            for p_sym, p_score in zip(phones, phones_acc):
                total_phones += 1
                dimension_values['phone_accuracy'].append(float(p_score))
                rounded_bin = round(p_score, 1)
                phone_score_histogram[rounded_bin] += 1
                phone_tokens_by_symbol[p_sym] += 1
                
                if p_score < 0.0 or p_score > 2.0:
                    label_issues.append({
                        'issue_id': f"SO_ISSUE_{issue_counter:06d}",
                        'utterance_id': utt_id,
                        'speaker_id': spk,
                        'split': split,
                        'dimension': 'phone_accuracy',
                        'level': 'phone',
                        'observed_value': str(p_score),
                        'issue_category': 'OUT_OF_RANGE_PHONE_SCORE',
                        'description': f"Phone score {p_score} outside [0.0, 2.0]"
                    })
                    issue_counter += 1

    tf.close()
    print("SpeechOcean762 inspection finished.")
    return {
        'total_utterances': len(scores),
        'total_words': total_words,
        'total_phones': total_phones,
        'train_count': len(train_utts),
        'test_count': len(test_utts),
        'dimension_values': dimension_values,
        'rater_variances': rater_variances,
        'phone_score_histogram': phone_score_histogram,
        'phone_tokens_by_symbol': phone_tokens_by_symbol,
        'label_issues': label_issues
    }

if __name__ == "__main__":
    res = run_speechocean_audit("datasets/SpeechOcean762/original/speechocean762.tar.gz")

    # Generate SPEECHOCEAN762_SCORE_STATISTICS.csv
    with open("manifests/RQ1-AUD-01/SPEECHOCEAN762_SCORE_STATISTICS.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            'dimension', 'level', 'sample_count', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max', 'mean_inter_rater_std'
        ])
        
        for dim_name, vals in sorted(res['dimension_values'].items()):
            n = len(vals)
            if n == 0: continue
            s_vals = sorted(vals)
            mean_val = sum(s_vals) / n
            var_val = sum((x - mean_val) ** 2 for x in s_vals) / n
            std_val = math.sqrt(var_val)
            min_val = s_vals[0]
            max_val = s_vals[-1]
            q25 = s_vals[int(n * 0.25)]
            median = s_vals[int(n * 0.50)]
            q75 = s_vals[int(n * 0.75)]
            
            inter_rater_std = "N/A"
            if dim_name in res['rater_variances']:
                r_stds = res['rater_variances'][dim_name]
                if r_stds:
                    inter_rater_std = f"{sum(r_stds) / len(r_stds):.3f}"
            
            level = "sentence" if dim_name.startswith("utt_") else ("word" if dim_name.startswith("word_") else "phone")
            writer.writerow([
                dim_name, level, n, f"{mean_val:.3f}", f"{std_val:.3f}", f"{min_val:.1f}",
                f"{q25:.1f}", f"{median:.1f}", f"{q75:.1f}", f"{max_val:.1f}", inter_rater_std
            ])
    print("Saved manifests/RQ1-AUD-01/SPEECHOCEAN762_SCORE_STATISTICS.csv")

    # Generate SPEECHOCEAN762_LABEL_ISSUES.csv
    with open("manifests/RQ1-AUD-01/SPEECHOCEAN762_LABEL_ISSUES.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'issue_id', 'utterance_id', 'speaker_id', 'split', 'dimension', 'level', 'observed_value', 'issue_category', 'description'
        ])
        writer.writeheader()
        for row in res['label_issues']:
            writer.writerow(row)
    print("Saved manifests/RQ1-AUD-01/SPEECHOCEAN762_LABEL_ISSUES.csv")

    # Save summary json
    summary = {
        'total_utterances': res['total_utterances'],
        'train_utterances': res['train_count'],
        'test_utterances': res['test_count'],
        'total_words': res['total_words'],
        'total_phones': res['total_phones'],
        'phone_score_histogram': dict(res['phone_score_histogram']),
        'total_issues': len(res['label_issues'])
    }
    with open("audit_scripts/speechocean_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved audit_scripts/speechocean_audit_summary.json")

    print("\n=== SPEECHOCEAN762 AUDIT PREVIEW ===")
    print(f"Total utterances: {res['total_utterances']}")
    print(f"Total words: {res['total_words']}")
    print(f"Total phones: {res['total_phones']}")
    print(f"Phone score distribution (0.0 to 2.0):")
    for score_bin, count in sorted(res['phone_score_histogram'].items()):
        pct = (count / res['total_phones']) * 100
        print(f"  Score {score_bin:3.1f}: {count:6d} ({pct:5.2f}%)")
    print(f"Total issues flagged: {len(res['label_issues'])}")
