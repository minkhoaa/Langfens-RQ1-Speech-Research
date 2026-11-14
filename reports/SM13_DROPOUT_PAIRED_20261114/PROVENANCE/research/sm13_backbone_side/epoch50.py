"""Bounded S0 SM13 epoch-extension; no test, E4 changes or backbone training.

Exploratory historical TRAIN outer0: fit9 speakers, hold ABA/ASI/BWC. Compare
neural-only target12/gap5 with target50/gap50 under one seed and identical fit.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch
from torch import nn

from .audit import TRAIN,sha
from .features import W2V2,extract
from .heads import ScalarMixingTargetHead,ScalarMixingGapHead

DECISION='SM13-EPOCH50-20261114-v1'
HELD={'ABA','ASI','BWC'}
HALT=False


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def ap(y,score):
    """Non-interpolated average precision, tied scores grouped like sklearn."""
    y=np.asarray(y,dtype=bool);score=np.asarray(score)
    if y.shape!=score.shape or not np.isfinite(score).all(): raise ValueError('invalid AP inputs')
    positives=int(y.sum())
    if not positives:return None
    order=np.argsort(-score,kind='stable');scores=score[order];tp=np.cumsum(y[order])
    ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1]
    return float(np.sum(np.diff(np.r_[0,tp[ends]])/positives*tp[ends]/(ends+1)))


def metrics(y,probs,classes):
    y=np.asarray(y);probs=np.asarray(probs)
    if probs.shape!=(len(y),classes) or not np.isfinite(probs).all():raise ValueError('invalid inference output')
    pred=probs.argmax(1);result={}
    for c in range(1,classes):
        tp=int(((y==c)&(pred==c)).sum());fp=int(((y!=c)&(pred==c)).sum());fn=int(((y==c)&(pred!=c)).sum())
        result[str(c)]={'AP':ap(y==c,probs[:,c]),'F1':2*tp/max(1,2*tp+fp+fn),'TP':tp,'FP':fp,'FN':fn,'support':int((y==c).sum())}
    return result


def gap_weights(y):
    counts=np.bincount(y,minlength=2).astype(np.float32)
    w=np.sqrt(len(y)/(2*np.maximum(counts,1)));return w/w.mean()


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],np.uint32),n[2],n[3],n[4]));torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda'])!=torch.cuda.device_count():raise ValueError('RNG device inventory changed')
        torch.cuda.set_rng_state_all(state['cuda'])


def assert_equal_state(actual,expected):
    if isinstance(expected,torch.Tensor):
        if not torch.equal(actual.cpu(),expected.detach().cpu()):raise ValueError('checkpoint tensor changed')
    elif isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('checkpoint keys changed')
        for k in expected:assert_equal_state(actual[k],expected[k])
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):raise ValueError('checkpoint sequence changed')
        for a,b in zip(actual,expected):assert_equal_state(a,b)
    elif actual!=expected:raise ValueError('checkpoint scalar changed')


def atomic_checkpoint(path,state):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    assert_equal_state(restored,state)
    dump(path.with_suffix('.receipt.json'),{'sha256':sha(path),'bytes':path.stat().st_size,
         'epoch':state['epoch'],'cursor':state['cursor'],'updates':state['updates'],'head':state['head'],
         'decision':state.get('decision',DECISION),'optimizer_rng_order_saved':True,'pending_gradients':0})


def validate_requests(rows):
    if len(rows)!=1800 or len({r['utterance_id'] for r in rows})!=1800 or {r['speaker_id'] for r in rows}!=TRAIN:
        raise ValueError('only authorized TRAIN1800 is allowed')
    for r in rows:
        if set(r)!={'utterance_id','speaker_id','audio_sha256','canonical_phones'}:raise ValueError('gold input in inference request')
        if not r['utterance_id'].startswith(r['speaker_id']+'_'):raise ValueError('speaker mismatch')
    return [r['utterance_id'] for r in rows]


def load_inputs(folder):
    inventory=json.loads((folder/'EPOCH50_INPUT_INVENTORY.json').read_text())
    if inventory['decision']!=DECISION:raise ValueError('unregistered side training')
    for name,meta in inventory['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or (folder/name).is_symlink():raise ValueError('unsafe training inventory')
        if sha(folder/name)!=meta['sha256'] or (folder/name).stat().st_size!=meta['bytes']:raise ValueError('side training input changed')
    rows=[json.loads(l) for l in (folder/'TRAIN_REQUESTS.jsonl').read_text().splitlines()]
    validate_requests(rows)
    z=np.load(folder/'TRAIN_HEAD_INPUTS.npz',allow_pickle=False)
    expected={'target_refs','gap_refs','target_y','gap_y','target_speaker','gap_speaker','target_uid','gap_uid'}
    if set(z.files)!=expected:raise ValueError('unexpected labels/reference inventory')
    nt=sum(len(r['canonical_phones']) for r in rows);ng=nt+len(rows)
    if (nt,ng)!=(59562,61362):raise ValueError('event population changed')
    for kind,n,dim,classes in [('target',nt,39,3),('gap',ng,78,2)]:
        if z[kind+'_refs'].shape!=(n,dim) or not np.isfinite(z[kind+'_refs']).all():raise ValueError('reference shape mismatch')
        if z[kind+'_y'].shape!=(n,) or not np.isin(z[kind+'_y'],range(classes)).all():raise ValueError('invalid labels')
        sp=np.concatenate([np.repeat(r['speaker_id'],len(r['canonical_phones'])+(kind=='gap')) for r in rows])
        uid=np.concatenate([np.repeat(r['utterance_id'],len(r['canonical_phones'])+(kind=='gap')) for r in rows])
        if not np.array_equal(z[kind+'_speaker'],sp) or not np.array_equal(z[kind+'_uid'],uid):raise ValueError('labels/features reordered')
    return rows,z,inventory


def resources(device,*,initial=False,local=False):
    import psutil
    if shutil.disk_usage('/kaggle/working' if not local else '/var/tmp').free<(14 if local else 1)*(1<<30):
        raise OSError('registered disk reserve exhausted')
    if psutil.virtual_memory().available<(8 if initial else 2)*(1<<30):raise MemoryError('side RAM reserve')
    if device.type=='cuda':
        free,total=torch.cuda.mem_get_info(device)
        if free<(512*(1<<20) if local else 2*(1<<30)):raise MemoryError('side VRAM reserve')


def create_features(folder,assets,work,rows,device,expired,local=False):
    """Generate original frozen features to runtime memmaps, never historical files."""
    from transformers import AutoModel,AutoFeatureExtractor
    import soundfile as sf
    cache=work/'features';cache.mkdir(exist_ok=True)
    receipt=cache/'FEATURE_RECEIPT.json'
    if receipt.exists():
        meta=json.loads(receipt.read_text())
        for name,h in meta['files'].items():
            if sha(cache/name)!=h:raise ValueError('resumed feature cache changed')
        return meta
    progress=cache/'PROGRESS.json';completed=json.loads(progress.read_text()).get('completed_utterances',0) if progress.exists() else 0
    maps={}
    for kind,n,width in [('target',59562,2304),('gap',61362,1536)]:
        p=cache/(kind+'.npy')
        if p.exists() and not progress.exists():raise ValueError('unregistered partial feature cache')
        maps[kind]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype=np.float32,shape=(n,13,width))
    if not progress.exists():dump(progress,{'completed_utterances':0})
    native=folder/'native_w2v2';fe=AutoFeatureExtractor.from_pretrained(native,local_files_only=True)
    model=AutoModel.from_pretrained(native,local_files_only=True).to(device).eval().requires_grad_(False)
    if (model.config.model_type,model.config.hidden_size,model.config.num_hidden_layers)!=('wav2vec2',768,12):raise ValueError('wrong S0 backbone')
    model.config.apply_spec_augment=False
    alignment=np.load(folder/'TRAIN_ALIGNMENT.npz',allow_pickle=False)
    parity=np.load(folder/'FEATURE_PARITY.npz',allow_pickle=False)
    parity_ids=json.loads((folder/'FEATURE_PARITY_IDS.json').read_text())
    parity_report=[];t=g=0;started=time.monotonic()
    for i,r in enumerate(rows):
        uid=r['utterance_id'];n=len(r['canonical_phones'])
        if i<completed:t+=n;g+=n+1;continue
        resources(device,local=local)
        wav=assets/'audio'/(uid+'.wav')
        if sha(wav)!=r['audio_sha256']:raise ValueError('allowlisted TRAIN audio changed')
        samples,rate=sf.read(wav,dtype='float32',always_2d=True)
        if rate!=16000 or not np.isfinite(samples).all():raise ValueError('invalid audio')
        values=fe(samples.mean(1),sampling_rate=16000,return_tensors='pt').input_values.to(device)
        with torch.inference_mode():hidden=model(values,output_hidden_states=True).hidden_states
        target,gap=extract(hidden,alignment['centers__'+uid],int(alignment['nframes__'+uid]),W2V2)
        if target.shape!=(n,13,2304) or gap.shape!=(n+1,13,1536):raise ValueError('feature topology mismatch')
        if uid in parity_ids:
            for kind,value in [('target',target),('gap',gap)]:
                ref=parity[kind+'__'+uid];delta=float(np.max(np.abs(value-ref)))
                if not np.allclose(value,ref,rtol=1e-3,atol=3e-4):raise ValueError(f'S0 original feature parity failed: {uid}/{kind}/{delta}')
                parity_report.append({'id':uid,'head':kind,'max_abs_error':delta})
        maps['target'][t:t+n]=target;maps['gap'][g:g+n+1]=gap;t+=n;g+=n+1
        if (i+1)%50==0 or i==len(rows)-1 or expired():
            for m in maps.values():m.flush()
            dump(progress,{'completed_utterances':i+1,'target_cursor':t,'gap_cursor':g})
            print(json.dumps({'event':'sm13_feature_progress','completed':i+1,'total':1800,'seconds':time.monotonic()-started}),flush=True)
        if expired():return None
    if (t,g)!=(59562,61362):raise ValueError('feature population incomplete')
    # A resumed cache can contain previously checked parity IDs: check all again.
    t=g=0;parity_report=[]
    for r in rows:
        uid=r['utterance_id'];n=len(r['canonical_phones'])
        if uid in parity_ids:
            for kind,value in [('target',maps['target'][t:t+n]),('gap',maps['gap'][g:g+n+1])]:
                ref=parity[kind+'__'+uid]
                if not np.allclose(value,ref,rtol=1e-3,atol=3e-4):raise ValueError('persisted S0 feature parity failed')
                parity_report.append({'id':uid,'head':kind,'max_abs_error':float(np.max(np.abs(value-ref)))})
        t+=n;g+=n+1
    if len(parity_report)!=2*len(parity_ids):raise ValueError('incomplete real S0 parity')
    for m in maps.values():m.flush()
    meta={'decision':DECISION,'status':'S0_FULL_FROZEN_FEATURES_VERIFIED','parity':parity_report,
          'normalization':'original HF feature extractor only; no SpeechBrain extra layernorm',
          'dtype':'float32','backbone_trainable_parameters':0,'targets':t,'gaps':g,
          'input_inventory_sha256':sha(folder/'EPOCH50_INPUT_INVENTORY.json'),
          'files':{kind+'.npy':sha(cache/(kind+'.npy')) for kind in maps}}
    dump(receipt,meta);del model,maps;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return meta


def evaluate(model,features,refs,y,indices,weights,device,batch):
    model.eval();probs=[];loss_sum=weight_sum=0.
    with torch.inference_mode():
        for start in range(0,len(indices),batch):
            ix=indices[start:start+batch];by=torch.as_tensor(y[ix],dtype=torch.long,device=device)
            logits=model(torch.as_tensor(np.array(features[ix]),device=device),torch.as_tensor(refs[ix],device=device))
            loss_sum+=float(nn.functional.cross_entropy(logits,by,weight=weights,reduction='sum'))
            weight_sum+=float(weights[by].sum());probs.append(logits.softmax(-1).cpu().numpy())
    probs=np.concatenate(probs);return loss_sum/weight_sum,metrics(y[indices],probs,probs.shape[1]),probs


def fit_head(kind,features,refs,y,speakers,uids,work,device,identity,expired,*,epochs=50,local=False,
             geometry=W2V2,decision=DECISION,resource_check=None,model_factory=None):
    is_t=kind=='target';seed=101 if is_t else 1101;batch=512 if is_t else 1024
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=(model_factory(kind,geometry) if model_factory is not None else
           ScalarMixingTargetHead(geometry) if is_t else ScalarMixingGapHead(geometry)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001 if is_t else .0008,weight_decay=.01 if is_t else .0001)
    held=np.isin(speakers,sorted(HELD));fit_ix=np.flatnonzero(~held);held_ix=np.flatnonzero(held)
    if set(speakers[fit_ix])&set(speakers[held_ix]) or not len(fit_ix) or not len(held_ix):raise ValueError('speaker split leak')
    weights=torch.tensor([1.,2.5,4.] if is_t else gap_weights(y[fit_ix]),dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    path=work/(kind+'_last.pt');history=[];epoch=cursor=updates=0;order=None;losses=[]
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['identity']!=identity or saved['head']!=kind or saved['decision']!=decision:raise ValueError('resume identity changed')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        epoch,cursor,updates=saved['epoch'],saved['cursor'],saved['updates']
        order=saved['order'];history=saved['history'];losses=saved['partial_losses']
        if order is not None and sorted(order)!=sorted(fit_ix.tolist()):raise ValueError('resume shuffle order differs')
    def save(destination):
        state={'decision':decision,'identity':identity,'head':kind,'epoch':epoch,'cursor':cursor,'updates':updates,
               'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'order':order,
               'history':history,'partial_losses':losses,'scheduler':None,'scaler':None,'pending_gradients':0}
        atomic_checkpoint(destination,state)
    while epoch<epochs:
        if expired():save(path);return False,history
        if order is None:order=np.random.permutation(fit_ix).tolist();cursor=0;losses=[]
        model.train();tick=time.monotonic()
        while cursor<len(order):
            if resource_check is not None and resource_check() is False:
                save(path);return False,history
            resources(device,local=local);ix=order[cursor:cursor+batch]
            bx=torch.as_tensor(np.array(features[ix]),device=device);br=torch.as_tensor(refs[ix],device=device)
            by=torch.as_tensor(y[ix],dtype=torch.long,device=device)
            opt.zero_grad(set_to_none=True);loss=criterion(model(bx,br),by)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite SM13 training loss')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite gradients')
            opt.step();opt.zero_grad(set_to_none=True)
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite model')
            losses.append(float(loss.detach()));cursor+=len(ix);updates+=1
            if expired():save(path);return False,history
        if resource_check is not None and resource_check() is False:
            save(path);return False,history
        val_loss,val_metrics,probs=evaluate(model,features,refs,y,held_ix,weights,device,batch)
        record={'event':'sm13_epoch','head':kind,'epoch':epoch+1,'updates':updates,'train_loss':float(np.mean(losses)),
                'held_loss':val_loss,'held_metrics':val_metrics,'seconds':time.monotonic()-tick,
                'held_speakers':sorted(HELD),'fit_events':len(fit_ix),'held_events':len(held_ix),
                'layer_weights':model.layer_logits.softmax(0).detach().cpu().tolist(),'scale':float(model.scale.detach())}
        history.append(record);epoch+=1;cursor=0;order=None;losses=[]
        save(work/f'{kind}_epoch{epoch:02d}.pt');save(path)
        np.savez_compressed(work/f'{kind}_held_epoch{epoch:02d}.npz',probs=probs,gold=y[held_ix],speakers=speakers[held_ix],uids=uids[held_ix])
        dump(work/(kind+'_history.json'),history);print(json.dumps(record),flush=True)
    return True,history


def comparison(target,gap):
    if len(target)<50 or len(gap)<50:return {'status':'PENDING_50_EPOCHS_BOTH_HEADS'}
    def score(t,g):
        values=[t['held_metrics']['1']['AP'],t['held_metrics']['2']['AP'],g['held_metrics']['1']['AP']]
        return {'S_AP':values[0],'O_AP':values[1],'I_AP':values[2],
                'macro3_AP':sum(values)/3 if all(v is not None for v in values) else None}
    base=score(target[11],gap[4]);extended=score(target[49],gap[49])
    delta=extended['macro3_AP']-base['macro3_AP'] if base['macro3_AP'] is not None and extended['macro3_AP'] is not None else None
    return {'status':'TERMINAL_EPOCH_COMPARISON_COMPLETE','baseline_target12_gap5':base,'extended_target50_gap50':extended,
            'delta_macro3_AP':delta,'scope':'EXPLORATORY_REUSED_TRAIN_OUTER0_NEURAL_ONLY_NOT_FULL_FUSION',
            'seed':101,'held_speakers':sorted(HELD),'test_opened':False,'one_seed_one_fold_not_confirmation':True}


def main():
    global HALT
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--state',type=Path)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--mode',choices=['side_train'],required=True)
    p.add_argument('--seconds',type=int,required=True);p.add_argument('--side-input-inventory',type=Path,required=True)
    p.add_argument('--local-admission-failure',type=Path)
    a=p.parse_args();local=not Path('/kaggle/input').is_dir()
    if not 0<a.seconds<=14400:raise ValueError('bounded4hdriver required')
    if local and (not a.local_admission_failure or not a.local_admission_failure.is_file()):
        raise RuntimeError('local production requires preserved Kaggle resource admission failure')
    if local:
        admission=json.loads(a.local_admission_failure.read_text())
        if (admission.get('decision')!=DECISION or admission.get('reason')!='RESOURCE_ADMISSION_REJECTED' or
                admission.get('remote_job_running') is not False):
            raise RuntimeError('fallback requires confirmed resource rejection and no live remote duplicate')
    a.work.mkdir(parents=True,exist_ok=True)
    lock_path=Path('/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock') if local else a.work/'RUNTIME.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True);lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('GPUtraining requested; no GPU admitted')
    device=torch.device('cuda:0');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();deadline=started+a.seconds-120
    def stop(signum,frame):
        global HALT
        HALT=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def expired():return HALT or time.monotonic()>=deadline
    status={'decision':DECISION,'status':'PREFLIGHT','local_training':local,'official_test_opened':False,
            'dev_cal_opened':False,'backbone_frozen':True,'seed':101,'target_epochs':50,'gap_epochs':50,
            'evaluation':'reusedTRAINouter0neuralonly','device':torch.cuda.get_device_name(0)}
    try:
        resources(device,initial=True,local=local)
        if local:
            import psutil
            if psutil.virtual_memory().available<16*(1<<30):raise MemoryError('local16GiBRAM admission floor')
            if shutil.disk_usage(a.work).free<28*(1<<30):raise OSError('local needs14GiBcache plus14GiBmainreserve')
        folder=a.side_input_inventory.parent;rows,z,inventory=load_inputs(folder)
        # All audio is rehashed even on feature-cache resume, never only sample-checked.
        for r in rows:
            if sha(a.assets/'audio'/(r['utterance_id']+'.wav'))!=r['audio_sha256']:raise ValueError('TRAIN audio mismatch')
        status['status']='EXTRACTING_FROZEN_FEATURES';dump(a.work/'STATUS.json',status)
        feature=create_features(folder,a.assets,a.work,rows,device,expired,local)
        if feature is None:
            status['status']='PAUSED_FEATURE_BOUNDARY';return
        identity=hashlib.sha256(json.dumps({'input':sha(a.side_input_inventory),'feature':sha(a.work/'features/FEATURE_RECEIPT.json'),
                       'source':sha(Path(__file__)),'head_source':sha(Path(__file__).with_name('heads.py')),'seed':101,'epochs':50,
                       'held':sorted(HELD),'normalization':feature['normalization']},sort_keys=True).encode()).hexdigest()
        histories={};status['status']='HEAD_TRAINING';dump(a.work/'STATUS.json',status)
        for kind in ['target','gap']:
            x=np.load(a.work/'features'/(kind+'.npy'),mmap_mode='r',allow_pickle=False)
            complete,history=fit_head(kind,x,z[kind+'_refs'],z[kind+'_y'],z[kind+'_speaker'],z[kind+'_uid'],
                                      a.work,device,identity,expired,local=local)
            histories[kind]=history;del x;gc.collect();torch.cuda.empty_cache()
            if not complete:
                status['status']='PAUSED_AT_OPTIMIZER_BOUNDARY';return
        report=comparison(histories['target'],histories['gap']);dump(a.work/'COMPARISON.json',report)
        status['status']='COMPLETE_TARGET50_GAP50_EXPLORATORY';status['comparison']=report
    except BaseException as exc:
        import traceback
        status.update(status='FAILED_NO_BATCH_SKIPPED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
    finally:
        status['seconds']=time.monotonic()-started;dump(a.work/'STATUS.json',status)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
