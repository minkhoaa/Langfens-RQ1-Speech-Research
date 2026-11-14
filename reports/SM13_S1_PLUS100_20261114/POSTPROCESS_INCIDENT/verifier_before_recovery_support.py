"""Finite CPU-only independent state/curve verifier for local S1 +100.

No training launch, restart, GPU or remote service use.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from research.sm13_backbone_side import epoch50 as engine
from research.sm13_backbone_side.audit import ROOT, sha
from research.sm13_backbone_side.local_s1 import dump
from research.sm13_backbone_side.local_s1_extend import DECISION, START, END, curve_report, plot_curves


def verify(registration,receipt):
    if receipt.exists():raise FileExistsError('preserve previous verification')
    reg=json.loads(registration.read_text());work=Path(reg['work'])
    status=json.loads((work/'STATUS.json').read_text())
    if (status['status']!='S1_TARGET150_GAP150_COMPLETE_PENDING_INDEPENDENT_VERIFICATION' or
            status['registration_sha256']!=sha(registration) or status['official_test_opened'] or
            status['dev_cal_opened'] or status['backbone_loaded'] or status['threshold_retuning_performed']):
        raise ValueError('continuation incomplete or lineage mismatch')
    for key in ['parent_registration','parent_verification','cache_verification','calibration_thresholds']:
        if sha(reg[key])!=reg[key+'_sha256']:raise ValueError('registered parent/proof changed')
    parent_reg=json.loads(Path(reg['parent_registration']).read_text());parent=Path(parent_reg['work'])
    requests,z,inventory=engine.load_inputs(Path(parent_reg['inputs']))
    manifest=ROOT/'data/derived/l2arctic/manifests/train.jsonl'
    if sha(manifest)!=parent_reg['train_manifest_sha256']:raise ValueError('TRAIN manifest changed')
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    fingerprints={};verified_states=0
    for kind,classes in [('target',3),('gap',2)]:
        source=parent/f'{kind}_epoch50.pt'
        if sha(source)!=reg['parent_head_sha256'][kind]:raise ValueError('original epoch50 checkpoint changed')
        original=torch.load(source,map_location='cpu',weights_only=True)
        seed_path=work/f'{kind}_resume_epoch50.pt'
        migration=json.loads(seed_path.with_suffix('.migration.json').read_text())
        if migration['parent_sha256']!=sha(source) or migration['destination_sha256']!=sha(seed_path):
            raise ValueError('migration SHA mismatch')
        migrated=torch.load(seed_path,map_location='cpu',weights_only=True)
        if migrated['identity']!=status['identity'] or migrated['decision']!=DECISION:
            raise ValueError('migration provenance mismatch')
        engine.assert_equal_state({k:v for k,v in migrated.items() if k not in ['decision','identity']},
                                  {k:v for k,v in original.items() if k not in ['decision','identity']})
        verified_states+=1
        history=json.loads((work/(kind+'_history.json')).read_text())
        if [h['epoch'] for h in history]!=list(range(1,END+1)) or history[:START]!=original['history']:
            raise ValueError('continuation history not contiguous or original50 rewritten')
        previous=original['updates'];mask=np.isin(z[kind+'_speaker'],sorted(engine.HELD))
        for epoch in range(START+1,END+1):
            path=work/f'{kind}_epoch{epoch:02d}.pt';proof=json.loads(path.with_suffix('.receipt.json').read_text())
            if proof['decision']!=DECISION or sha(path)!=proof['sha256']:raise ValueError('checkpoint SHA/decision mismatch')
            saved=torch.load(path,map_location='cpu',weights_only=True);h=history[epoch-1]
            if (saved['decision']!=DECISION or saved['identity']!=status['identity'] or saved['head']!=kind or
                    saved['epoch']!=epoch or saved['cursor']!=0 or saved['order'] is not None or
                    saved['partial_losses'] or saved['pending_gradients']!=0 or saved['history']!=history[:epoch] or
                    not {'python','numpy','torch','cuda'}<=set(saved['rng'])):
                raise ValueError('full state/history/boundary mismatch')
            batch=512 if kind=='target' else 1024;steps=(h['fit_events']+batch-1)//batch
            if saved['updates']-previous!=steps:raise ValueError('optimizer updates not contiguous')
            previous=saved['updates']
            if not saved['optimizer']['state'] or any(float(s['step'])!=saved['updates'] for s in saved['optimizer']['state'].values()):
                raise ValueError('Adam counter reset or lost state')
            for state in [saved['model'],*saved['optimizer']['state'].values()]:
                if any(isinstance(value,torch.Tensor) and not torch.isfinite(value).all() for value in state.values()):
                    raise ValueError('nonfinite model/Adam')
            if not np.isfinite([h['train_loss'],h['held_loss']]).all():raise ValueError('nonfinite loss')
            pred_path=work/f'{kind}_held_epoch{epoch:02d}.npz'
            with np.load(pred_path,allow_pickle=False) as pred:
                for name,key in [('gold','y'),('speakers','speaker'),('uids','uid')]:
                    if not np.array_equal(pred[name],z[kind+'_'+key][mask]):raise ValueError('prediction event order changed')
                if engine.metrics(pred['gold'],pred['probs'],classes)!=h['held_metrics']:
                    raise ValueError('head metrics do not reproduce')
            fingerprints[path.name]=proof['sha256'];fingerprints[pred_path.name]=sha(pred_path);verified_states+=1
        last=work/(kind+'_last.pt');proof=json.loads(last.with_suffix('.receipt.json').read_text())
        if sha(last)!=proof['sha256']:raise ValueError('last checkpoint SHA mismatch')
        engine.assert_equal_state(torch.load(last,map_location='cpu',weights_only=True),saved);verified_states+=1
    thresholds=json.loads(Path(reg['calibration_thresholds']).read_text())
    parent_verified=json.loads(Path(reg['parent_verification']).read_text())
    if thresholds!=parent_verified['summary']['calibration']:
        raise ValueError('thresholds differ from independently verified epoch50 calibration')
    report,histories=curve_report(work,parent,rows,z,thresholds)
    if report!=json.loads((work/'CURVE_REPORT.json').read_text()):raise ValueError('full1..150 curve report does not reproduce')
    if report!=json.loads((Path(reg['reports'])/'CURVE_REPORT.json').read_text()):raise ValueError('published curve report differs')
    cache_reg=json.loads(Path(parent_reg['cache_registration']).read_text());cache=Path(cache_reg['work'])
    cached=json.loads(Path(reg['cache_verification']).read_text())
    for name,meta in cached['files'].items():
        if sha(cache/name)!=meta['sha256']:raise ValueError('read-only cache changed')
    plot_curves(report,histories,receipt.parent/'VERIFIED_LEARNING_CURVES.png')
    result={'status':'S1_PLUS100_STATES_AND_FULL_CURVES_VERIFIED','decision':DECISION,'platform':'LOCAL_ONLY',
            'start_epoch':START,'end_epoch':END,'additional_epochs_per_head':100,'states_verified':verified_states,
            'expected_states':204,'parent_model_optimizer_rng_exactly_preserved':True,
            'registration_sha256':sha(registration),'file_sha256':fingerprints,'endpoints':report['endpoints'],
            'threshold_retuning_performed':False,'official_test_opened':False,'dev_cal_opened':False,
            'fullTRAIN_DEV_SEARCH_comparison_complete':False,'backbone_gain_claim':False}
    if verified_states!=204:raise ValueError('incomplete state verification')
    dump(receipt,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--wait-seconds',type=int,default=0)
    a=p.parse_args()
    if not 0<=a.wait_seconds<=14400:raise ValueError('finite observer max4h')
    torch.set_num_threads(2)
    if torch.cuda.is_available():raise RuntimeError('CPU verifier requires CUDA_VISIBLE_DEVICES empty')
    reg=json.loads(a.registration.read_text());work=Path(reg['work']);started=time.monotonic()
    observer=a.receipt.parent/'VERIFIER_STATUS.json'
    try:
        while True:
            path=work/'STATUS.json';status=json.loads(path.read_text()) if path.exists() else {}
            current=status.get('status','WAITING_FOR_STATUS')
            dump(observer,{'status':'WAITING_FOR_LOCAL_S1_PLUS100_TERMINAL','training_status':current,
                           'training_submission_performed':False,'elapsed_seconds':time.monotonic()-started})
            if current=='S1_TARGET150_GAP150_COMPLETE_PENDING_INDEPENDENT_VERIFICATION':break
            if current.startswith(('FAILED','PAUSED')):raise RuntimeError('continuation did not complete: '+current)
            if time.monotonic()-started>=a.wait_seconds:raise TimeoutError('finite observer budget exhausted')
            time.sleep(max(0,min(10,a.wait_seconds-(time.monotonic()-started))))
        result=verify(a.registration,a.receipt)
        dump(observer,{'status':result['status'],'states_verified':result['states_verified'],
                       'training_submission_performed':False,'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        dump(observer,{'status':'VERIFICATION_FAILED_OR_TRAINING_NOT_COMPLETE','error':str(exc),
                       'traceback':traceback.format_exc(),'training_submission_performed':False});raise


if __name__=='__main__':main()
