"""Keyed event-level detection, NOT paper hierarchical phone-alignment F1.

Inference exports probabilities only. Gold is joined by exact event keys solely
inside this evaluator; missing/duplicate/extra events are hard failures.
"""
from __future__ import annotations
import numpy as np

from .epoch50 import ap

SPEC = 'SM13-COMMON-EVENT-DETECTION-v1'
GRID = tuple(i/20 for i in range(21))


def population(rows):
    result=[];seen=set()
    label_map={'CORRECT':0,'SUBSTITUTION':1,'OMISSION':2}
    for row in rows:
        uid=row['utterance_id'];speaker=row['speaker_id']
        if uid in seen or not uid.startswith(speaker+'_'):
            raise ValueError('duplicate utterance or speaker mismatch')
        seen.add(uid)
        n=len(row['target_phones'])
        if len(row['gaps'])!=n+1:
            raise ValueError('N targets must have N+1 exact gaps')
        for kind,events in [('target',row['target_phones']),('gap',row['gaps'])]:
            for position,event in enumerate(events):
                if kind=='target':
                    gold=label_map[event['label']]
                else:
                    if event['has_insertion'] not in [True,False,0,1]:
                        raise ValueError('invalid insertion label')
                    gold=int(event['has_insertion'])
                result.append({'key':f'L2_ARCTIC|{speaker}|{uid}|{kind}|{position}',
                               'speaker':speaker,'uid':uid,'kind':kind,'gold':gold})
    return result


def scores(probs,kind):
    probs=np.asarray(probs)
    classes=3 if kind=='target' else 2 if kind=='gap' else None
    if (classes is None or probs.ndim!=2 or probs.shape[1]!=classes or
            not np.isfinite(probs).all() or np.any(probs<0) or np.any(probs>1) or
            not np.allclose(probs.sum(1),1.,rtol=0,atol=1e-5)):
        raise ValueError('finite normalized class probabilities required')
    return probs[:,1:].sum(1) if kind=='target' else probs[:,1]


def counts(gold,pred):
    gold=np.asarray(gold,dtype=bool);pred=np.asarray(pred,dtype=bool)
    if gold.ndim!=1 or gold.shape!=pred.shape:
        raise ValueError('binary populations differ')
    return {'TP':int((gold&pred).sum()),'FP':int((~gold&pred).sum()),
            'FN':int((gold&~pred).sum()),'TN':int((~gold&~pred).sum()),
            'support':int(gold.sum()),'events':len(gold)}


def ratios(c):
    tp,fp,fn=c['TP'],c['FP'],c['FN']
    return {**c,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def joined(gold_rows,prediction_rows):
    expected={r['key']:r for r in gold_rows}
    if len(expected)!=len(gold_rows):
        raise ValueError('duplicate gold event key')
    actual={}
    for row in prediction_rows:
        if set(row)!={'key','score'} or row['key'] in actual:
            raise ValueError('duplicate prediction or forbidden export fields')
        score=row['score']
        if not np.isfinite(score) or not 0<=score<=1:
            raise ValueError('invalid detection score')
        actual[row['key']]=float(score)
    if set(expected)!=set(actual):
        raise ValueError('missing/extra event IDs; never drop failed cases')
    y=np.array([r['gold']!=0 for r in gold_rows],bool)
    s=np.array([actual[r['key']] for r in gold_rows],float)
    kinds=np.array([r['kind'] for r in gold_rows])
    if any(kind not in ['target','gap'] for kind in kinds):
        raise ValueError('unregistered event kind')
    return y,s,kinds


def evaluate(gold_rows,prediction_rows,*,target_threshold,gap_threshold):
    if not all(np.isfinite(t) and 0<=t<=1 for t in [target_threshold,gap_threshold]):
        raise ValueError('registered finite thresholds required')
    y,s,kinds=joined(gold_rows,prediction_rows)
    pred=s>=np.where(kinds=='target',target_threshold,gap_threshold)
    result={'spec':SPEC,'thresholds':{'target':target_threshold,'gap':gap_threshold},
            'micro':ratios(counts(y,pred)),'subgroups':{},'per_speaker':{},
            'coverage':{'expected_events':len(y),'predicted_events':len(s),'missing':0,'extra':0}}
    for kind in ['target','gap']:
        mask=kinds==kind
        result['subgroups'][kind]=ratios(counts(y[mask],pred[mask]))
    speakers=np.array([r['speaker'] for r in gold_rows])
    uids={r['uid'] for r in gold_rows}
    for speaker in sorted(set(speakers)):
        mask=speakers==speaker
        result['per_speaker'][speaker]=ratios(counts(y[mask],pred[mask]))
    result['false_warnings_per_utterance']=int((~y&pred).sum())/len(uids) if uids else None
    result['binary_AP']={kind:ap(y[kinds==kind],s[kinds==kind]) for kind in ['target','gap']}
    return result


def calibrate(gold_rows,prediction_rows,*,scope):
    if scope!='TRAIN_SPEAKER_DISJOINT_CALIBRATION':
        raise ValueError('threshold search only on registered TRAIN calibration, not final evaluation labels')
    y,s,kinds=joined(gold_rows,prediction_rows)
    best=None
    for tt in GRID:
        for gt in GRID:
            result=ratios(counts(y,s>=np.where(kinds=='target',tt,gt)))
            # Same bounded441 choices for all arms. Undefined F1 cannot win.
            rank=(result['F1'] if result['F1'] is not None else -1.,
                  result['precision'] if result['precision'] is not None else -1.,tt,gt)
            if best is None or rank>best[0]:
                best=(rank,tt,gt,result)
    return {'spec':SPEC,'scope':scope,'target_threshold':best[1],'gap_threshold':best[2],
            'calibration_micro':best[3],'grid':list(GRID),'candidates':len(GRID)**2,
            'tie_break':'higher micro F1, higher precision, higher target threshold, higher gap threshold',
            'not_independent_confirmation':True}
