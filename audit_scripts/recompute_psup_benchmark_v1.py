"""Independent, model-free P-SUP benchmark recomputation from saved token records.

Paths supply actual protocol; serialized protocol disagreements are reported, not
silently accepted. Full O and aligned Z populations are never pooled. Runtime and
offline scores must agree exactly. FPR points are explicitly retrospective VAL.
"""
import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('archival',Path(__file__).with_name('verify_psup_archival_v1.py'))
metrics=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)
CONFIGS=('t_op_fixed_budget_3600','b_control_fixed_budget_3600','t_op_best','b_control_best')


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows, treatment):
    result={}
    for name,operations,positive,score_name in (
        ('binary',(0,1,2),None,'s_error'),
        ('omission_legacy',(0,2),2,'s_error'),
        ('omission_dedicated',(0,2),2,'s_omit'),
        ('substitution_legacy',(0,1),1,'s_error'),
        ('substitution_dedicated',(0,1),1,'s_sub')):
        selected=[r for r in rows if r['operation_target'] in operations]
        if not treatment and score_name in ('s_omit','s_sub'):
            result[name]=dict(evidence_status='unavailable',ap=None,reason='binary_collapsed subtype semantics unidentified')
            continue
        if any(r.get(score_name) is None for r in selected):
            result[name]=dict(evidence_status='unavailable',ap=None,reason='Missing score; do not invent probabilities')
            continue
        y=[r['binary_target'] if positive is None else int(r['operation_target']==positive) for r in selected]
        scores=[r[score_name] for r in selected]
        ops=metrics.operating_points(y,scores)
        for point in ops.values():
            threshold=point['threshold']
            deployed=[r for r in rows if r.get(score_name) is not None and r[score_name]>=threshold]
            event_tp=sum(r['operation_target']==positive if positive is not None else r['binary_target']==1 for r in deployed)
            point['deployed_event_TP']=event_tp
            point['deployed_predicted_count']=len(deployed)
            point['deployed_event_precision']=event_tp/len(deployed) if deployed else None
            point['deployed_wrong_event_count']=len(deployed)-event_tp
        result[name]=dict(evidence_status='reproduced',score=score_name,n=len(y),positive=sum(y),negative=len(y)-sum(y),
            ap=metrics.average_precision(y,scores),retrospective_val_fpr_points=ops)
    return result


def audit(path, protocol, treatment, expected_ids):
    rp=path/'runtime_predictions.jsonl'; op=path/'offline_token_evaluation.jsonl'
    runtime=records(rp); offline=records(op)
    key=lambda r:(r['speaker_id'],r['utterance_id'],r['prompt_index'])
    by_runtime={key(r):r for r in runtime}
    assert len(by_runtime)==len(runtime),'duplicate runtime key'
    assert list(dict.fromkeys(f'{r["speaker_id"]}/{r["utterance_id"]}' for r in runtime))==expected_ids,'cohort/order mismatch'
    scored=[]; gold_keys=set(); unanswered=[]; extras=[]
    for row in offline:
        gold_index=row.get('gold_canonical_index')
        if gold_index is None:
            extras.append(row)
            continue
        gold_key=row['speaker_id'],row['utterance_id'],gold_index
        assert gold_key not in gold_keys,'duplicate gold position'
        gold_keys.add(gold_key)
        assert row['operation_target'] in (0,1,2)
        assert row['binary_target']==int(row['operation_target']!=0)
        if row.get('s_error') is None or row.get('prompt_index') is None:
            unanswered.append(row)
            continue
        rr=by_runtime[key(row)]
        for score in ('s_error','s_omit','s_sub'):
            assert row.get(score)==rr['scores'].get(score),'runtime/offline score mismatch'
        if not treatment:
            assert row.get('s_omit') is None and row.get('s_sub') is None
        scored.append(row)
    issues=[]
    actual_protocol='O-ARCHIVE' if protocol=='protocol_o' else 'Z-PROMPT-V1'
    serialized=sorted(set(r['protocol'] for r in runtime))
    if serialized!=[actual_protocol]:
        issues.append(f'Runtime protocol field {serialized} differs from actual {actual_protocol}')
    if len(gold_keys)!=9956:
        issues.append(f'Offline gold coverage incomplete: {len(gold_keys)}/9956; all unmapped positions must be explicit')
    if protocol=='protocol_z' and len(unanswered)!=37:
        issues.append(f'Expected37 explicit unmapped gold abstentions, got{len(unanswered)}')
    result=dict(evidence_status='reproduced',actual_protocol=actual_protocol,serialized_protocols=serialized,
        utterances=len(expected_ids),runtime_tokens=len(runtime),offline_rows=len(offline),
        gold_positions=len(gold_keys),answered_positions=len(scored),unanswered_gold_positions=len(unanswered),
        extra_prompt_rows=len(extras),label_counts=dict(Counter(str(r['operation_target']) for r in scored)),
        eligible_gold_key_sha256=hashlib.sha256(('\n'.join('/'.join(map(str,(r['speaker_id'],r['utterance_id'],r['gold_canonical_index']))) for r in sorted(scored,key=lambda r:(r['speaker_id'],r['utterance_id'],r['gold_canonical_index'])))+'\n').encode()).hexdigest(),
        validation_issues=issues,source_hashes={p.name:metrics.sha256(p) for p in [rp,op]},
        metrics=summarize(scored,treatment),by_speaker={})
    for speaker in ('NJS','RRBI'):
        selected=[r for r in scored if r['speaker_id']==speaker]
        result['by_speaker'][speaker]=dict(n=len(selected),metrics=summarize(selected,treatment))
    # All-gold recall accounting: unmapped true errors are undetected.
    for name,data in result['metrics'].items():
        if 'retrospective_val_fpr_points' not in data:
            continue
        positive=None if name=='binary' else (2 if name.startswith('omission') else 1)
        unresolved_positive=sum(r['binary_target'] if positive is None else r['operation_target']==positive for r in unanswered)
        unresolved_correct=sum(r['operation_target']==0 for r in unanswered)
        for point in data['retrospective_val_fpr_points'].values():
            denominator=data['positive']+unresolved_positive
            point['all_gold_recall']=point['TP']/denominator if denominator else None
            point['unanswered_true_errors']=unresolved_positive
            negatives=data['negative']+unresolved_correct
            point['all_gold_correct_fpr_lower']=point['FP']/negatives if negatives else None
            point['all_gold_correct_fpr_upper']=(point['FP']+unresolved_correct)/negatives if negatives else None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--split-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier audit evidence')
    manifest=json.loads(args.split_manifest.read_text())
    assert metrics.sha256(args.split_manifest)=='7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b'
    ids=next(f for f in manifest['folds'] if f['fold_index']==1)['val']['utterances']
    results={}
    for config in CONFIGS:
        results[config]={}
        for protocol in ('protocol_o','protocol_z'):
            path=args.raw_dir/config/protocol
            result=audit(path,protocol,config.startswith('t_op'),ids)
            results[config][protocol]=result
            print(config,protocol,result['answered_positions'],{name:data['ap'] for name,data in result['metrics'].items()},result['validation_issues'])
    output=dict(status='INDEPENDENT_METRIC_RECOMPUTATION',source_sha256=metrics.sha256(__file__),
        helper_source_sha256=metrics.sha256(Path(__file__).with_name('verify_psup_archival_v1.py')),
        raw_directory=str(args.raw_dir),results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
