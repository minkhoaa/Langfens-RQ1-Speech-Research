#!/usr/bin/env python3
"""Nested TRAIN speaker transfer for five fixed target-phone scorers."""
from __future__ import annotations
import csv, json, math, sys, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_audit as audit
from run_temporal_probe import GROUPS, model
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/rq1_encoder_bench_01/common'))
from dataset_util import DatasetManager
PAIRS=(('IH','IY','Vowel'),('AH','AO','Vowel'),('OW','AO','Vowel'),('D','T','Stop'),('T','D','Stop'))

def threshold_rows(score,y):
    if len(score)==0:return [{'threshold':1.,'tp':0,'fp':0,'precision':None}]
    order=np.argsort(-score,kind='stable'); s=score[order]; y=y[order].astype(bool)
    tp=np.cumsum(y); fp=np.cumsum(~y); ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    rows=[{'threshold':1.,'tp':0,'fp':0,'precision':None}]
    for j in ends:
        rows.append({'threshold':float(np.nextafter(s[j],-np.inf)),'tp':int(tp[j]),'fp':int(fp[j]),'precision':float(tp[j]/(tp[j]+fp[j]))})
    return rows

def select_thresholds(inner, data, speakers):
    utterances=len({r['utterance'] for r in data['rows'] if r['speaker'] in speakers}); cap=math.floor(.08*utterances+1e-9)
    choices=[]
    for pi,(phone,realized,manner) in enumerate(PAIRS):
        mask=np.isin(data['speaker'],list(speakers))&(data['phone']==phone)
        cand=[r for r in threshold_rows(inner[pi,mask],data['y'][mask]==1) if not r['tp'] or r['precision']>=.5]
        choices.append((phone,manner,mask,cand))
    # Exhaustive dynamic programming over integer FP cost; maximize minimum manner recall, then total TP.
    states={(0,(0,0,0,0,0)):(0,0,[None]*5)}
    # Store tuples of selected options and sufficient statistics; compact by cost and TP vector.
    frontier=[(0,0,0,[])]
    for i,(phone,manner,mask,cand) in enumerate(choices):
        support=int(np.sum(mask&(data['y']==1))); expanded=[]
        for cost,tv,ts,opts in frontier:
            for r in cand:
                nc=cost+r['fp']
                if nc<=cap: expanded.append((nc,tv+(r['tp'] if manner=='Vowel' else 0),ts+(r['tp'] if manner=='Stop' else 0),opts+[r]))
        # Keep all nondominated (cost, vowel TP, stop TP), deterministic threshold tie break.
        keep={}
        for st in expanded:
            k=st[:3]
            if k not in keep:keep[k]=st
        frontier=list(keep.values())
    supports=[]
    for phone,_,mask,_ in choices: supports.append(int(np.sum(mask&np.isin(data['speaker'],list(speakers))&(data['y']==1))))
    v_support=sum(supports[i] for i,(_,m,_,_) in enumerate(choices) if m=='Vowel'); s_support=sum(supports[i] for i,(_,m,_,_) in enumerate(choices) if m=='Stop')
    best=max(frontier,key=lambda z:(min(z[1]/max(v_support,1),z[2]/max(s_support,1)),z[1]+z[2],-z[0]))
    return {'cap_fp':cap,'joint_fp':best[0],'vowel_tp':best[1],'stop_tp':best[2],'vowel_support':v_support,'stop_support':s_support,'options':best[3]}

def evaluate(scores, thresholds, data, speakers):
    target=np.isin(data['speaker'],list(speakers)); gap=np.isin(data['gap_speaker'],list(speakers))
    alert=np.zeros(len(data['y']),bool); pair_idx=np.full(len(data['y']),-1,int)
    for pi,(phone,_,_) in enumerate(PAIRS):
        mask=target&(data['phone']==phone); th=thresholds['options'][pi]['threshold']; a=mask&(scores[pi]>=th)
        alert|=a; pair_idx[a]=pi
    tau_o=audit.TAU_O; trigger_o=target&(data['o']>=tau_o)
    # Use the pair-specific threshold margin for arbitration against frozen O.
    margins=np.zeros(len(data['y']))
    for pi,(_,_,_) in enumerate(PAIRS): margins[pair_idx==pi]=scores[pi,pair_idx==pi]-thresholds['options'][pi]['threshold']
    flag_s=alert&(~trigger_o|(margins>data['o']-tau_o)); flag_o=trigger_o&~flag_s
    flag_i=gap&(data['gap_s']>=audit.TAU_I)
    fp=int(np.sum(flag_s&(data['y']!=1))+np.sum(flag_o&(data['y']!=2))+np.sum(flag_i&(data['gap_y']==0)))
    utt=len({r['utterance'] for r in data['rows'] if r['speaker'] in speakers}); out={'utterances':utt,'total_fp':fp,'fwr':fp/utt,'pairs':{}}
    for pi,(phone,realized,manner) in enumerate(PAIRS):
        mask=target&(data['phone']==phone); tp=int(np.sum(mask&flag_s&(data['y']==1))); f=int(np.sum(mask&flag_s&(data['y']!=1))); sup=int(np.sum(mask&(data['y']==1)))
        out['pairs'][f'{phone}->{realized}']={'manner':manner,'tp':tp,'fp':f,'support':sup,'recall':tp/sup if sup else None,'precision':tp/(tp+f) if tp+f else None}
    for manner in ('Vowel','Stop'):
        ps=[p for p,_,m in PAIRS if m==manner]; mask=target&np.isin(data['phone'],ps); tp=int(np.sum(mask&flag_s&(data['y']==1))); f=int(np.sum(mask&flag_s&(data['y']!=1))); sup=int(np.sum(target&(data['manner']==manner)&(data['y']==1)))
        out[manner]={'tp':tp,'fp':f,'support':sup,'recall':tp/sup if sup else None,'precision':tp/(tp+f) if tp+f else None}
    return out

def main():
    t0=time.perf_counter(); data=audit.load_data(); dm=DatasetManager()
    with np.load(ROOT/'data/derived/l2arctic/features/train_features.npz') as f:
        X=f['target_X'].astype(np.float32); meta=f['target_meta']
    phones=np.asarray([r['phone'] for r in data['rows']]); data['phone']=phones
    if len(X)!=len(phones):raise ValueError('manifest/features mismatch')
    outer_scores=np.full((len(PAIRS),len(X)),np.nan,np.float32); inner_all=np.full((4,len(PAIRS),len(X)),np.nan,np.float32)
    for fold,held_group in enumerate(GROUPS):
        outer=dm.outer_masks(fold); held=np.asarray(outer['t_held'])
        for pi,(phone,_,manner) in enumerate(PAIRS):
            mask=(phones==phone)&(data['manner']==manner); fit=np.flatnonzero(~held&mask); ev=np.flatnonzero(held&mask)
            y=(data['y'][fit]==1).astype(int); pos=int(y.sum()); neg=len(y)-pos; w=np.where(y==1,np.sqrt(neg/pos),1.).astype(np.float32)
            with threadpool_limits(limits=4): clf=model(101+fold); clf.fit(X[fit],y,sample_weight=w); outer_scores[pi,ev]=clf.predict_proba(X[ev])[:,1]
        for inf in range(3):
            masks=dm.inner_masks(fold,inf)
            for pi,(phone,_,manner) in enumerate(PAIRS):
                mask=(phones==phone)&(data['manner']==manner); fit=np.flatnonzero(masks['t_fit']&mask); ev=np.flatnonzero(masks['t_held']&mask)
                y=(data['y'][fit]==1).astype(int); pos=int(y.sum()); neg=len(y)-pos; w=np.where(y==1,np.sqrt(neg/pos),1.).astype(np.float32)
                with threadpool_limits(limits=4): clf=model(201+fold); clf.fit(X[fit],y,sample_weight=w); inner_all[fold,pi,ev]=clf.predict_proba(X[ev])[:,1]
    
    for pi,(phone,_,_) in enumerate(PAIRS):
        mask=phones==phone
        outer_scores[pi,~mask]=0.0
        inner_all[:,pi,~mask]=0.0
        wrong_manner=(phones==phone)&(data['manner']!=PAIRS[pi][2])
        outer_scores[pi,wrong_manner]=0.0
        inner_all[:,pi,wrong_manner]=0.0
    if not np.isfinite(outer_scores).all():raise ValueError('incomplete outer OOF scores')
    for fold,group in enumerate(GROUPS):
        trainmask=~np.isin(data['speaker'],list(group))
        if not np.isfinite(inner_all[fold,:,trainmask]).all(): raise ValueError(f'incomplete inner OOF scores fold={fold}')
    fold_results=[]
    for fold,held_group in enumerate(GROUPS):
        train=set(data['speaker'])-set(held_group); sel=select_thresholds(inner_all[fold],data,train)
        obs=evaluate(outer_scores,sel,data,set(held_group)); fold_results.append({'fold':fold,'held_speakers':list(held_group),'thresholds':sel,'outer':obs})
        print(f"fold={fold} V={obs['Vowel']['recall']:.3f}/{obs['Vowel']['precision'] or 0:.3f} S={obs['Stop']['recall']:.3f}/{obs['Stop']['precision'] or 0:.3f} FWR={obs['fwr']:.3f}",flush=True)
    pooled={'utterances':sum(f['outer']['utterances'] for f in fold_results),'total_fp':sum(f['outer']['total_fp'] for f in fold_results)}; pooled['fwr']=pooled['total_fp']/pooled['utterances']
    for m in ('Vowel','Stop'):
        tp=sum(f['outer'][m]['tp'] for f in fold_results); fp=sum(f['outer'][m]['fp'] for f in fold_results); sup=sum(f['outer'][m]['support'] for f in fold_results)
        pooled[m]={'tp':tp,'fp':fp,'support':sup,'recall':tp/sup,'precision':tp/(tp+fp) if tp+fp else None}
    out={'protocol':'PHONE_CONTRAST_PREREGISTRATION.md','exploratory':True,'pairs':[list(x) for x in PAIRS],'folds':fold_results,'pooled':pooled,'seconds':time.perf_counter()-t0}
    (HERE/'PHONE_CONTRAST_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(HERE/'PHONE_CONTRAST_OOF.npz',outer_scores=outer_scores,inner_scores=inner_all)
    print(json.dumps({'pooled':pooled,'seconds':out['seconds']},indent=2))
if __name__=='__main__':main()
