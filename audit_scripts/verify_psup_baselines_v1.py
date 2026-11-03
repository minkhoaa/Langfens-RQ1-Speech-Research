"""Recompute archival E0 and P-SCORE metrics on their identical Z aligned cohort.

No model inference/training; no comparison to the 9956-position O population.
Gold operations are independently cross-checked against archived P-SUP gold rows.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('archival', Path(__file__).with_name('verify_psup_archival_v1.py'))
archival = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archival)


def digest_keys(keys):
    data = '\n'.join('/'.join(map(str, key)) for key in sorted(keys))+'\n'
    return hashlib.sha256(data.encode()).hexdigest()


def summarize(rows):
    result = dict(evidence_status='archival-only', protocol='Z-PROMPT-V1',
        n=len(rows), gold_key_sha256=digest_keys(rows),
        label_counts=dict(Counter(str(row['operation']) for row in rows.values())), metrics={})
    for name, ops, positive in (('binary',(0,1,2),None), ('omission_legacy',(0,2),2), ('substitution_legacy',(0,1),1)):
        selected = [row for row in rows.values() if row['operation'] in ops]
        y = [int(row['operation'] != 0) if positive is None else int(row['operation'] == positive) for row in selected]
        s = [row['score'] for row in selected]
        result['metrics'][name] = dict(n=len(y), positives=sum(y), ap=archival.average_precision(y,s),
            retrospective_fpr_points=archival.operating_points(y,s), score='s_error')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--p-score-evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gold_path = args.dataset_root/'manifests/RQ1-P-SUP/evaluation/t_op_fixed_budget_3600/offline_token_evaluation.jsonl'
    gold = {(r['speaker_id'],r['utterance_id'],r['gold_canonical_index']):r['operation_target']
        for r in map(json.loads, gold_path.read_text().splitlines())}
    source_hashes = {str(gold_path): archival.sha256(gold_path)}
    populations, results = {}, {}
    for scorer in ('sf_s','sf_sd','gop_sa','greedy_dp'):
        path = args.p_score_evaluation/f'offline_token_evaluation_{scorer}.jsonl.gz'
        source_hashes[str(path)] = archival.sha256(path)
        rows = {}
        with gzip.open(path,'rt') as f:
            for r in map(json.loads,f):
                key = r['speaker'],r['utterance_id'],r['gold_canonical_index']
                if key in rows:
                    raise ValueError('duplicate P-SCORE key')
                if key not in gold:
                    raise ValueError('unauthorized or unknown P-SCORE key')
                op = {'correct':0,'substitution':1,'deletion':2}[r['error_type']]
                if op != gold[key]:
                    raise ValueError('gold operation mismatch')
                if r['status'] == 'ok' and r['s_error'] is not None:
                    rows[key] = dict(operation=op,score=r['s_error'])
        populations[scorer] = rows
        results['P-SCORE_'+scorer] = summarize(rows)
    e0_path = args.dataset_root/'manifests/RQ1-TRN-02A/evaluation_reconciliation/zero_oracle_verification/PROTOCOL_Z_PREDICTION_RECORDS.json'
    source_hashes[str(e0_path)] = archival.sha256(e0_path)
    record = json.loads(e0_path.read_text())
    rows = {}
    for utterance in record['utterances']:
        for gold_index,prompt_index,_ in utterance['aligned_pairs']:
            key = utterance['speaker'],utterance['utterance_id'],gold_index
            if key in rows or int(gold[key] != 0) != utterance['gold_labels'][gold_index]:
                raise ValueError('E0 duplicate or gold mismatch')
            rows[key] = dict(operation=gold[key],score=utterance['g2p_scores'][prompt_index])
    if set(rows) != set(populations['sf_sd']) or set(rows) != set(populations['sf_s']):
        raise ValueError('E0 and P-SCORE aligned cohorts differ; no direct comparison')
    results['E0'] = summarize(rows)
    result = dict(status='ARCHIVAL_METRIC_RECOMPUTATION_ONLY', source_sha256=archival.sha256(__file__),
        source_hashes=source_hashes, results=results,
        precision_qualification='Historical E0 CPU FP32 and P-SCORE frozen FP32 emissions; not a new contemporaneous E0 replay.',
        forbidden_comparison='Do not rank these 9919-token Z scores directly against 9956-token P-SUP O scores.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name, summary in results.items():
        print(name,summary['n'],{k:v['ap'] for k,v in summary['metrics'].items()})


if __name__ == '__main__':
    main()
