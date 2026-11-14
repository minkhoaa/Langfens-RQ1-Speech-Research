from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/khoa/Projects/New-Method-Research')
HERE=ROOT/'experiments/sm13_three_directions_20261113/data_expansion'
sys.path.insert(0,str(ROOT/'experiments/sm13_three_directions_20261113/shared'))
sys.path.insert(0,str(ROOT/'experiments/rq1_encoder_bench_01/common'))
from dataset_util import DatasetManager
from protocol import SplitProtocol

dm=DatasetManager(); proto=SplitProtocol(dm)
known=set(dm.order); report={'status':'PASS','speakers':{},'folds':{}}
all_extra_ids=[]
for spk in sorted({spk for g in proto.groups for spk in g} if hasattr(proto,'groups') else set(dm.speakers.values())):
    cp=HERE/'features'/f'speaker_{spk}_unlabeled_features.npz'
    ep=HERE/'features'/f'speaker_{spk}_event_manifest.jsonl'
    rp=HERE/'features'/f'speaker_{spk}_extraction_receipt.json'
    assert cp.exists() and ep.exists() and rp.exists(),spk
    events=[json.loads(line) for line in ep.read_text().splitlines() if line.strip()]
    ids=sorted({r['utterance_id'] for r in events})
    all_extra_ids.extend(ids)
    assert len(ids)==50,(spk,len(ids))
    assert not (known & set(ids)),f'{spk}: overlap with labeled TRAIN'
    assert all(r['speaker_id']==spk and r['status']=='feature_extracted_unlabeled' for r in events)
    assert all(r.get('audio_sha256') and r.get('source_archive_sha256') for r in events)
    with np.load(cp) as z:
        x=z['target_features']; refs=z['target_refs']
    assert x.shape==(len(events),13,2304),(spk,x.shape,len(events))
    assert refs.shape==(len(events),39)
    assert np.allclose(refs.sum(1),1.0)
    receipt=json.loads(rp.read_text())
    phone_map=json.loads((HERE/'features'/f'fold{receipt["first_fit_fold"]}_extraction_receipt.json').read_text())['phone_reference_columns']
    assert all(int(np.argmax(row))==phone_map[event['intended_phone']] for row,event in zip(refs,events)),spk
    first=receipt['first_fit_fold']
    # A speaker cache is first extracted only when that speaker is part of fit.
    assert spk in proto.split(first).fit_speakers,(spk,first)
    report['speakers'][spk]={'first_fit_fold':first,'utterances':len(ids),'target_phone_rows':len(events),'features_shape':list(x.shape),'unlabeled_ids_sha256':__import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest()}
assert len(all_extra_ids)==600 and len(set(all_extra_ids))==600
report['unique_extra_utterances_total']=len(set(all_extra_ids))
for fold in range(4):
    split=proto.split(fold)
    rows=[json.loads(line) for line in (HERE/'features'/f'fold{fold}_event_manifest.jsonl').read_text().splitlines() if line.strip()]
    speakers={r['speaker_id'] for r in rows}
    assert speakers==set(split.fit_speakers),(fold,speakers,split.fit_speakers)
    assert not (speakers & set(split.calibration_speakers))
    assert not (speakers & set(split.held_speakers))
    assert all(r['utterance_id'] not in known for r in rows)
    receipt=json.loads((HERE/'features'/f'fold{fold}_extraction_receipt.json').read_text())
    assert receipt['phone_reference_columns_verified_against_train'] is True
    report['folds'][str(fold)]={'fit_speakers':list(split.fit_speakers),'calibration_excluded':list(split.calibration_speakers),'held_excluded':list(split.held_speakers),'event_rows':len(rows),'distinct_extra_utterances':len({r['utterance_id'] for r in rows})}
out=HERE/'CACHE_LEAKAGE_AUDIT.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'speaker_count':len(report['speakers']),'fold_counts':{f:v['event_rows'] for f,v in report['folds'].items()}},indent=2))
