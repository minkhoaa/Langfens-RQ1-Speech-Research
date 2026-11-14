"""Local-only S1 heads: registered TRAIN-speaker-disjoint calibration stage.

Fit1350/hold450 precedes fullTRAIN1800 fit -> DEV_SEARCH599 comparison. This
stage is reused development, neural-only; never rank CROTTC on these450.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from . import epoch50 as engine
from .audit import ROOT, sha
from .detection import SPEC, GRID, calibrate, evaluate, population, scores
from .features import WAVLM
from .local_s1 import LOCK, DECISION, dump, check_resources

FIT_DECISION='SM13-S1-LOCAL-CALIBRATION50-20261114-v1'


def fit_source_hashes():
    return {name:sha(Path(__file__).with_name(name)) for name in
            ['local_s1_fit.py','epoch50.py','heads.py','features.py','detection.py']}


def validate_fit_registration(reg):
    if (reg['decision']!=FIT_DECISION or reg['platform']!='LOCAL_ONLY' or
            reg['arm']!='S1_NATIVE_FROZEN' or reg['stage']!='TRAIN_SPEAKER_DISJOINT_CALIBRATION' or
            reg['held_speakers']!=sorted(engine.HELD) or reg['target_epochs']!=50 or reg['gap_epochs']!=50 or
            reg['seed']!=101 or reg['metric_spec']!=SPEC or reg['grid']!=list(GRID) or
            reg['checkpoint_rule']!='fixed_terminal50_per_head' or
            reg['source_hashes']!=fit_source_hashes() or not 0<reg['seconds']<=14400):
        raise ValueError('unregistered S1 calibration recipe/source')
    if reg['floors']!={'initial_ram_bytes':16*(1<<30),'ram_bytes':4*(1<<30),
                       'disk_bytes':14*(1<<30),'vram_bytes':512*(1<<20)}:
        raise ValueError('S1 fit safety floors changed')
    return True


def keyed_predictions(gold_rows, target, gap):
    arrays={'target':scores(target,'target'),'gap':scores(gap,'gap')}
    cursor={'target':0,'gap':0};result=[]
    for row in gold_rows:
        kind=row['kind'];i=cursor[kind]
        if i>=len(arrays[kind]):
            raise ValueError('missing keyed prediction')
        result.append({'key':row['key'],'score':float(arrays[kind][i])});cursor[kind]+=1
    if any(cursor[kind]!=len(arrays[kind]) for kind in cursor):
        raise ValueError('extra keyed prediction')
    return result


def summarize(work, rows, z):
    gold_rows=[r for r in population(rows) if r['speaker'] in engine.HELD]
    curve=[]
    for epoch in range(1,51):
        pred={}
        for kind in ['target','gap']:
            with np.load(work/f'{kind}_held_epoch{epoch:02d}.npz',allow_pickle=False) as data:
                mask=np.isin(z[kind+'_speaker'],sorted(engine.HELD))
                for name,key in [('gold','y'),('speakers','speaker'),('uids','uid')]:
                    if not np.array_equal(data[name],z[kind+'_'+key][mask]):
                        raise ValueError('held prediction event order/labels changed')
                pred[kind]=data['probs']
        export=keyed_predictions(gold_rows,pred['target'],pred['gap'])
        fixed=evaluate(gold_rows,export,target_threshold=.5,gap_threshold=.5)
        curve.append({'epoch':epoch,'fixed05_diagnostic':fixed})
    dump(work/'DETECTION_CURVE.json',{'spec':SPEC,'scope':'REUSED_TRAIN_CALIBRATION_NOT_FINAL_EVALUATION',
                                    'curve':curve,'fixed05_is_not_tuned_primary':True})
    calibration=calibrate(gold_rows,export,scope='TRAIN_SPEAKER_DISJOINT_CALIBRATION')
    dump(work/'CALIBRATION_THRESHOLDS.json',calibration)
    export_path=work/'CALIBRATION_PREDICTIONS.jsonl'
    tmp=export_path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        for row in export:
            stream.write(json.dumps(row)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,export_path)
    return {'spec':SPEC,'fixed_terminal50_detection':curve[-1]['fixed05_diagnostic'],
            'calibration':calibration,'prediction_export_sha256':sha(export_path),
            'scope':'EXPLORATORY_REUSED_TRAIN_OUTER0_CALIBRATION_NEURAL_ONLY',
            'not_fullTRAIN1800_fit_or_DEV_SEARCH_comparison':True,'no_backbone_gain_claim':True,
            'next_gate':'verify states and curves; fullTRAIN matched S0/S1 and DEV_SEARCH prediction pipeline still pending'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    reg=json.loads(a.registration.read_text());validate_fit_registration(reg)
    work=Path(reg['work']);work.resolve().relative_to(Path('/var/tmp'))
    work.mkdir(parents=True,exist_ok=a.resume)
    lock=LOCK.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();halt=False
    def pause(signum,frame):
        nonlocal halt
        halt=True
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    def expired():return halt or time.monotonic()-started>=reg['seconds']-120
    report={'decision':FIT_DECISION,'platform':'LOCAL_ONLY','arm':'S1_NATIVE_FROZEN',
            'status':'PREFLIGHT','pid':os.getpid(),'official_test_opened':False,'dev_cal_opened':False,
            'backbone_trainable_parameters':0,'backbone_loaded_during_head_fit':False,
            'target_epochs':50,'gap_epochs':50,'registration_sha256':sha(a.registration),
            'scope':'TRAIN1350_FIT_TRAIN450_REUSED_CALIBRATION_NOT_DEV_SEARCH_COMPARISON'}
    def resource_guard():
        try:
            check_resources(reg,work,device)
            return True
        except (MemoryError,OSError) as exc:
            report['resource_pause_reason']=str(exc)
            return False
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('local GPU required; no remote fallback')
        device=torch.device('cuda:0')
        import subprocess
        jobs=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                            capture_output=True,text=True,check=True,timeout=10).stdout
        if any(line.strip().isdigit() and int(line.strip())!=os.getpid() for line in jobs.splitlines()):
            raise RuntimeError('other local GPU process detected')
        report['admission']=check_resources(reg,work,device,initial=True)
        cache_reg_path=Path(reg['cache_registration'])
        if sha(cache_reg_path)!=reg['cache_registration_sha256']:
            raise ValueError('cache registration changed')
        cache_reg=json.loads(cache_reg_path.read_text());cache=Path(cache_reg['work'])
        proof=json.loads((cache/'FEATURE_RECEIPT.json').read_text())
        verification_path=Path(reg['cache_verification'])
        verification=json.loads(verification_path.read_text())
        if (sha(verification_path)!=reg['cache_verification_sha256'] or
                verification['status']!='S1_NATIVE_TRAIN_CACHE_INDEPENDENTLY_VERIFIED' or
                verification['registration_sha256']!=reg['cache_registration_sha256'] or
                verification['feature_receipt_sha256']!=sha(cache/'FEATURE_RECEIPT.json')):
            raise ValueError('independent persisted cache verification required')
        if (proof['decision']!=DECISION or proof['status']!='S1_NATIVE_TRAIN_CACHE_COMPLETE_VERIFIED' or
                proof['registration_sha256']!=reg['cache_registration_sha256'] or
                proof['completed_utterances']!=1800):
            raise ValueError('complete registered S1 cache required; never train partial cache')
        for name,meta in proof['files'].items():
            if name not in ['target.npy','gap.npy'] or sha(cache/name)!=meta['sha256']:
                raise ValueError('S1 cache SHA changed')
        if set(proof['files'])!={'target.npy','gap.npy'}:
            raise ValueError('cache inventory missing head')
        inputs=Path(reg['inputs'])
        if sha(inputs/'EPOCH50_INPUT_INVENTORY.json')!=reg['input_inventory_sha256']:
            raise ValueError('frozen input inventory changed')
        requests,z,inventory=engine.load_inputs(inputs)
        manifest=ROOT/'data/derived/l2arctic/manifests/train.jsonl'
        if sha(manifest)!=reg['train_manifest_sha256'] or sha(manifest)!=cache_reg['train_manifest_sha256']:
            raise ValueError('TRAIN manifest changed')
        rows=[json.loads(line) for line in manifest.read_text().splitlines()]
        if [r['utterance_id'] for r in rows]!=[r['utterance_id'] for r in requests]:
            raise ValueError('training manifest order differs')
        audio_path=Path(cache_reg['audit_dir'])/'TRAIN_INFERENCE_REQUESTS.jsonl'
        if sha(audio_path)!=cache_reg['requests_sha256']:
            raise ValueError('frozen acoustic requests changed')
        audio_rows=[json.loads(line) for line in audio_path.read_text().splitlines()]
        for row,request in zip(audio_rows,requests):
            if row['utterance_id']!=request['utterance_id'] or sha(row['wav'])!=request['audio_sha256']:
                raise ValueError('frozen TRAIN audio/request changed')
        identity=hashlib.sha256(json.dumps({'registration':sha(a.registration),
                   'features':sha(cache/'FEATURE_RECEIPT.json'),'geometry':{'hidden_dim':1024,'layers':list(WAVLM.layer_indices)}},sort_keys=True).encode()).hexdigest()
        report.update(status='HEAD_TRAINING',identity=identity,cache_receipt_sha256=sha(cache/'FEATURE_RECEIPT.json'))
        dump(work/'STATUS.json',report)
        histories={}
        for kind,n,width in [('target',59562,3072),('gap',61362,2048)]:
            check_resources(reg,work,device)
            values=np.load(cache/(kind+'.npy'),mmap_mode='r',allow_pickle=False)
            if values.shape!=(n,13,width) or values.dtype!=np.float32:
                raise ValueError('S1 cache/head geometry mismatch')
            complete,history=engine.fit_head(kind,values,z[kind+'_refs'],z[kind+'_y'],
                z[kind+'_speaker'],z[kind+'_uid'],work,device,identity,expired,
                epochs=50,local=True,geometry=WAVLM,decision=FIT_DECISION,
                resource_check=resource_guard)
            histories[kind]=history
            del values;gc.collect();torch.cuda.empty_cache()
            if not complete:
                report.update(status='PAUSED_SAFE_OPTIMIZER_BOUNDARY',head=kind);return
        report['summary']=summarize(work,rows,z)
        dump(work/'SUMMARY.json',report['summary'])
        report.update(status='S1_CALIBRATION_TARGET50_GAP50_COMPLETE_PENDING_INDEPENDENT_STATE_VERIFICATION')
    except BaseException as exc:
        report.update(status='FAILED_NO_BATCH_SKIPPED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        report['seconds']=time.monotonic()-started;dump(work/'STATUS.json',report)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
