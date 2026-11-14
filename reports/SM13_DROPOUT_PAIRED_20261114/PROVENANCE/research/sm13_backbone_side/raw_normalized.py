"""Local paired raw-vs-fit-RMS pilot50, then conditional normalized total300.

No backbone fitting/new cache/remote/sealed-data access. Both arms start with
identical trainable weights and per-head RNG. Gate locked before pilot outcomes.
Checkpoint outputs use /home, keeping /var/tmp CROTTC export reserve untouched.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from . import epoch50 as engine
from .audit import ROOT,sha
from .conditioning import ConditionedSM13Head,fit_layer_rms
from .detection import SPEC,evaluate,population
from .features import WAVLM
from .heads import ScalarMixingTargetHead,ScalarMixingGapHead
from .local_s1 import LOCK,dump,check_resources
from .local_s1_fit import keyed_predictions

DECISION='SM13-RAW-NORMALIZED-CONDITIONAL300-20261114-v1'
PILOT=50
FINAL=300
FLOORS={'initial_ram_bytes':16*(1<<30),'ram_bytes':4*(1<<30),
        'disk_bytes':14*(1<<30),'vram_bytes':512*(1<<20)}
OUTPUT_FLOOR=4*(1<<30)
OUTPUT_BUDGET=11*(1<<29)  #5.5GiB: raw50 + normalized300, states/predictions/receipts.
GATE={'min_fixed50_microF1_delta':.01,'min_last10_mean_microF1_delta':0.,
      'max_macro3_AP_regression':.005,'max_each_S_O_I_AP_regression':.02,
      'min_speakers_with_nonnegative_microF1_delta':2,'max_false_warnings_per_utterance_increase':.5}
RUNTIME=Path('/var/tmp/langfens-crottc-full-20261113-t_car8m4')
REPORTS=ROOT/'reports/SM13_RAW_NORMALIZED_20261114'
WORK=Path('/home/khoa/.local/share/langfens/sm13_raw_normalized_20261114_v1')


def source_hashes():
    paths=[Path(__file__).with_name(n) for n in
           ['raw_normalized.py','conditioning.py','epoch50.py','heads.py','features.py','detection.py','local_s1.py','local_s1_fit.py']]
    paths.append(ROOT/'tools/verify_sm13_raw_normalized.py')
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def validate_registration(reg):
    if (reg['decision']!=DECISION or reg['platform']!='LOCAL_ONLY' or reg['arm']!='S1_NATIVE_FROZEN' or
        reg['pilot_epochs']!=PILOT or reg['conditional_total_epochs']!=FINAL or reg['gate']!=GATE or
        reg['floors']!=FLOORS or reg['output_floor_bytes']!=OUTPUT_FLOOR or reg['output_budget_bytes']!=OUTPUT_BUDGET or
        reg['seed']!=101 or reg['held_speakers']!=sorted(engine.HELD) or reg['source_hashes']!=source_hashes() or
        reg['threshold_policy']!='fixed0.5_for_both_arms_all_epochs_no_retuning' or reg['metric_spec']!=SPEC or
        not 0<reg['seconds']<=14400):
        raise ValueError('unregistered paired pilot/conditional300 contract')
    work=Path(reg['work']).resolve()
    work.relative_to(Path('/home/khoa/.local/share/langfens'))
    if work.is_relative_to(ROOT):raise ValueError('weights must be outside repository')
    return True


def prepare():
    registration=REPORTS/'REGISTRATION.json'
    if registration.exists() or WORK.exists():raise FileExistsError('preserve prior registration/runtime')
    cache_proof=ROOT/'reports/SM13_NATIVE_LOCAL_20261114/LOCAL_CACHE_VERIFICATION.json'
    base_reg=ROOT/'reports/SM13_NATIVE_LOCAL_20261114/FIT_CALIBRATION50_REGISTRATION_v1.json'
    base=json.loads(base_reg.read_text())
    reg={'decision':DECISION,'platform':'LOCAL_ONLY','arm':'S1_NATIVE_FROZEN',
         'user_instruction':'đối chứng raw vs normalized trước, nếu ổn thì train lại với300epoch',
         'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'work':str(WORK),'reports':str(REPORTS),'seconds':14400,'pilot_epochs':PILOT,'conditional_total_epochs':FINAL,
         'gate':GATE,'seed':101,'held_speakers':sorted(engine.HELD),'source_hashes':source_hashes(),
         'base_registration':str(base_reg),'base_registration_sha256':sha(base_reg),
         'cache_verification':str(cache_proof),'cache_verification_sha256':sha(cache_proof),
         'floors':FLOORS,'output_floor_bytes':OUTPUT_FLOOR,'output_budget_bytes':OUTPUT_BUDGET,
         'metric_spec':SPEC,'threshold_policy':'fixed0.5_for_both_arms_all_epochs_no_retuning',
         'normalization':'13 frozen per-layer RMS scales fit only on1350fit utterances; references unchanged',
         'recipe':'same1350fit450held/native frozen FP32 cache/layers/heads/seed101gap1101/LR/loss/weightdecay/batches/dropout; only RMS preprocessing differs',
         'ordering':['raw_target50','normalized_target50','raw_gap50','normalized_gap50'],
         'conditional300_semantics':'fresh normalized head0->50 pilot; if pilot verified+gate passes, same full state50->300,total300 not300additional and not old350checkpoint',
         'primary_gate_metric':'fixed0.5 micro event detection F1 at exactly50, plus preregistered safeguards; development diagnostic, not final test primary',
         'selection':'fixed50 for pilot, fixed300 for final; full curves and fixed50/100/150/200/250/300, no bestepochselection',
         'uncertainty':'one paired seed/reused3heldspeakers; exact27 ordered speaker-bootstrap resamples descriptive only, not independent confirmation',
         'storage_plan':'reuse existing read-only14.95GiB cache in /var/tmp; new checkpoints on /home outside repo,4GiB output-floor plus5.5GiB admission; /var/tmp14GiB CROTTC floor unchanged',
         'scope':'exploratory reused TRAIN neural-only; not fullTRAIN->DEV_SEARCH/common CROTTC comparison',
         'official_test_opened':False,'dev_cal_opened':False,'backbone_trainable_parameters':0,
         'after300':'stop; no auto extension/grid/fullbackbonefit'}
    validate_registration(reg);REPORTS.mkdir(parents=True,exist_ok=True);dump(registration,reg)
    print(json.dumps({'registration':str(registration),'work':str(WORK),'gate':GATE},indent=2));return reg


def load_data(reg,verify_cache=True):
    for key in ['base_registration','cache_verification']:
        if sha(reg[key])!=reg[key+'_sha256']:raise ValueError('base registration/cache proof changed')
    base=json.loads(Path(reg['base_registration']).read_text());proof=json.loads(Path(reg['cache_verification']).read_text())
    if proof['status']!='S1_NATIVE_TRAIN_CACHE_INDEPENDENTLY_VERIFIED':raise ValueError('verified native cache required')
    cp=Path(base['cache_registration'])
    if sha(cp)!=base['cache_registration_sha256']:raise ValueError('cache registration changed')
    cache_reg=json.loads(cp.read_text());cache=Path(cache_reg['work'])
    if sha(cache/'FEATURE_RECEIPT.json')!=proof['feature_receipt_sha256']:raise ValueError('cache receipt changed')
    if set(proof['files'])!={'target.npy','gap.npy'}:raise ValueError('cache inventory mismatch')
    if verify_cache:
        for name,meta in proof['files'].items():
            if sha(cache/name)!=meta['sha256']:raise ValueError('read-only feature cache SHA mismatch')
    inputs=Path(base['inputs'])
    if sha(inputs/'EPOCH50_INPUT_INVENTORY.json')!=base['input_inventory_sha256']:raise ValueError('head input inventory changed')
    requests,zfile,_=engine.load_inputs(inputs);z={key:np.array(zfile[key]) for key in zfile.files}
    manifest=ROOT/'data/derived/l2arctic/manifests/train.jsonl'
    if sha(manifest)!=base['train_manifest_sha256']:raise ValueError('TRAIN manifest changed')
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    if [r['utterance_id'] for r in rows]!=[r['utterance_id'] for r in requests]:raise ValueError('TRAIN order mismatch')
    audio_path=Path(cache_reg['audit_dir'])/'TRAIN_INFERENCE_REQUESTS.jsonl'
    if sha(audio_path)!=cache_reg['requests_sha256']:raise ValueError('acoustic requests changed')
    audio_rows=[json.loads(line) for line in audio_path.read_text().splitlines()]
    if len(audio_rows)!=len(requests):raise ValueError('incomplete audio population')
    for row,request in zip(audio_rows,requests):
        if row['utterance_id']!=request['utterance_id'] or sha(row['wav'])!=request['audio_sha256']:raise ValueError('TRAIN audio changed')
    features={}
    for kind,n,width in [('target',59562,3072),('gap',61362,2048)]:
        values=np.load(cache/(kind+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(n,13,width) or values.dtype!=np.float32:raise ValueError('native feature geometry/dtype changed')
        features[kind]=values
    return rows,z,features,proof


def make_head(kind,geometry,arm,stats):
    if arm=='raw':return (ScalarMixingTargetHead if kind=='target' else ScalarMixingGapHead)(geometry)
    if arm=='normalized':return ConditionedSM13Head(kind,geometry,stats[kind]['rms'])
    raise ValueError('only raw/normalized arms registered')


def initialization_pair(kind,stats):
    seed=101 if kind=='target' else 1101
    torch.manual_seed(seed);raw=make_head(kind,WAVLM,'raw',stats)
    torch.manual_seed(seed);normalized=make_head(kind,WAVLM,'normalized',stats)
    engine.assert_equal_state(raw.state_dict(),normalized.head.state_dict())
    fingerprint=hashlib.sha256()
    for key,value in raw.state_dict().items():fingerprint.update(key.encode());fingerprint.update(value.detach().numpy().tobytes())
    return {'kind':kind,'seed':seed,'trainable_initialization_exactly_equal':True,
            'common_head_state_sha256':fingerprint.hexdigest(),'normalized_has_only_extra_frozen13scale_buffer':True}


def summarize_arm(folder,rows,z,epochs):
    gold=[r for r in population(rows) if r['speaker'] in engine.HELD];curve=[]
    histories={kind:json.loads((folder/(kind+'_history.json')).read_text()) for kind in ['target','gap']}
    if any([h['epoch'] for h in history]!=list(range(1,epochs+1)) for history in histories.values()):
        raise ValueError('incomplete arm history')
    for epoch in range(1,epochs+1):
        probs={}
        for kind in ['target','gap']:
            with np.load(folder/f'{kind}_held_epoch{epoch:02d}.npz',allow_pickle=False) as pred:
                mask=np.isin(z[kind+'_speaker'],sorted(engine.HELD))
                for name,key in [('gold','y'),('speakers','speaker'),('uids','uid')]:
                    if not np.array_equal(pred[name],z[kind+'_'+key][mask]):raise ValueError('held population/order mismatch')
                probs[kind]=pred['probs']
        report=evaluate(gold,keyed_predictions(gold,probs['target'],probs['gap']),target_threshold=.5,gap_threshold=.5)
        aps=[histories['target'][epoch-1]['held_metrics'][str(c)]['AP'] for c in [1,2]]+[histories['gap'][epoch-1]['held_metrics']['1']['AP']]
        curve.append({'epoch':epoch,'detection':report,'S_O_I_AP':aps,'macro3_AP':sum(aps)/3 if all(v is not None for v in aps) else None})
    return {'curve':curve,'endpoint':curve[-1],'last10_mean_F1':float(np.mean([c['detection']['micro']['F1'] for c in curve[-10:]])),
            'epochs':epochs,'scope':'EXPLORATORY_REUSED_TRAIN_NEURAL_ONLY','threshold_retuning_performed':False}


def gate_decision(raw,normalized):
    a=raw['endpoint'];b=normalized['endpoint']
    delta=b['detection']['micro']['F1']-a['detection']['micro']['F1']
    macro=b['macro3_AP']-a['macro3_AP'];typed=[y-x for x,y in zip(a['S_O_I_AP'],b['S_O_I_AP'])]
    sp={s:b['detection']['per_speaker'][s]['F1']-a['detection']['per_speaker'][s]['F1'] for s in a['detection']['per_speaker']}
    last=normalized['last10_mean_F1']-raw['last10_mean_F1']
    fw=b['detection']['false_warnings_per_utterance']-a['detection']['false_warnings_per_utterance']
    checks={'fixed50_F1_gain_at_least1pp':delta>=GATE['min_fixed50_microF1_delta'],
            'last10_mean_F1_not_worse':last>=GATE['min_last10_mean_microF1_delta'],
            'macroAP_regression_no_more0_5pp':macro>=-GATE['max_macro3_AP_regression'],
            'each_typedAP_regression_no_more2pp':all(v>=-GATE['max_each_S_O_I_AP_regression'] for v in typed),
            'at_least2of3_speakers_not_worse':sum(v>=0 for v in sp.values())>=GATE['min_speakers_with_nonnegative_microF1_delta'],
            'falsewarnings_increase_no_more0_5_per_utterance':fw<=GATE['max_false_warnings_per_utterance_increase']}
    return {'quality_gate_passed':all(checks.values()),'checks':checks,'fixed50_microF1_delta':delta,
            'last10_mean_F1_delta':last,'macro3_AP_delta':macro,'S_O_I_AP_deltas':typed,'per_speaker_F1_deltas':sp,
            'false_warnings_per_utterance_delta':fw,'gate_registration':GATE,
            'interpretation':'internal exploratory development admission only, not independent evidence or final metric certification'}


def paired_uncertainty(raw,normalized,rows):
    counts=[];speakers=sorted(raw['endpoint']['detection']['per_speaker'])
    for chosen in itertools.product(speakers,repeat=len(speakers)):
        values=[]
        for arm in [raw,normalized]:
            c={k:sum(arm['endpoint']['detection']['per_speaker'][s][k] for s in chosen) for k in ['TP','FP','FN']}
            den=2*c['TP']+c['FP']+c['FN'];values.append(2*c['TP']/den if den else np.nan)
        if np.isfinite(values).all():counts.append(values[1]-values[0])
    return {'resamples':len(counts),'ordered_speaker_resamples':27,
            'descriptive95percent_delta_interval':np.quantile(counts,[.025,.975]).tolist() if counts else None,
            'limit':'only3reusedheldspeakers, one pairedseed, NOT strong statistical confirmation'}


def resources(reg,work,device,initial=False):
    # /var/tmp14GiB reserve remains unchanged, regardless of checkpoint placement.
    info=check_resources(reg,Path('/var/tmp'),device,initial=initial)
    free=shutil.disk_usage(work).free
    needed=reg['output_floor_bytes']+(reg['output_budget_bytes'] if initial else 0)
    if free<needed:raise OSError('dedicated /home output floor/admission budget exhausted')
    return {**info,'output_filesystem_free_bytes':free,'output_floor_bytes':reg['output_floor_bytes'],
            'output_admission_budget_bytes':reg['output_budget_bytes'] if initial else 0}


def verify_stage(registration,stage):
    reports=Path(json.loads(registration.read_text())['reports'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (reports/(stage+'_VERIFIER.log')).open('a') as log:
        subprocess.run([sys.executable,'-u','-m','tools.verify_sm13_raw_normalized','--registration',str(registration),'--stage',stage],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    return json.loads((reports/(stage.upper()+'_VERIFICATION.json')).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--registration',type=Path)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.prepare:prepare();return
    if not a.registration:p.error('--registration required')
    reg=json.loads(a.registration.read_text());validate_registration(reg);work=Path(reg['work'])
    work.mkdir(parents=True,exist_ok=a.resume)
    lock=LOCK.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();halt=False
    def stop(signum,frame):
        nonlocal halt
        halt=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def expired():return halt or time.monotonic()-started>=reg['seconds']-180
    status={'decision':DECISION,'status':'PREFLIGHT','pid':os.getpid(),'platform':'LOCAL_ONLY',
            'registration_sha256':sha(a.registration),'pilot_epochs':PILOT,'conditional_total_epochs':FINAL,
            'official_test_opened':False,'dev_cal_opened':False,'backbone_loaded':False,'threshold_retuning_performed':False}
    def guard():
        try:resources(reg,work,device);return True
        except (MemoryError,OSError) as exc:status['resource_pause_reason']=str(exc);return False
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('local GPU required, no remote/CPU fallback')
        device=torch.device('cuda:0')
        jobs=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout
        if any(line.strip().isdigit() and int(line.strip())!=os.getpid() for line in jobs.splitlines()):raise RuntimeError('other local GPU job present')
        status['admission']=resources(reg,work,device,initial=not a.resume)
        status['environment']={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                               'device':torch.cuda.get_device_name(device),'kernel':os.uname().release,'cpu_threads':2}
        dump(work/'STATUS.json',status)
        rows,z,features,cache_proof=load_data(reg)
        stats={}
        for kind in ['target','gap']:
            sp=z[kind+'_speaker'];fit=set(sp)-engine.HELD
            rms=fit_layer_rms(features[kind],sp,fit_speakers=fit,held_speakers=engine.HELD)
            stats[kind]={'rms':rms.tolist(),'fit_speakers':sorted(fit),'held_speakers':sorted(engine.HELD),
                         'fit_events':int((~np.isin(sp,sorted(engine.HELD))).sum()),
                         'feature_sha256':cache_proof['files'][kind+'.npy']['sha256']}
        stats_path=work/'NORMALIZATION_STATS.json'
        if stats_path.exists() and json.loads(stats_path.read_text())!=stats:raise ValueError('resumed fit-only normalization stats changed')
        if not stats_path.exists():dump(stats_path,stats)
        initialization={kind:initialization_pair(kind,stats) for kind in ['target','gap']}
        dump(work/'INITIALIZATION_PAIR.json',initialization)
        # Disposable one-batch resource/correctness smoke; reset seed for real fits.
        profile=[]
        for kind,refdim,batch in [('target',39,512),('gap',78,1024)]:
            ix=np.flatnonzero(~np.isin(z[kind+'_speaker'],sorted(engine.HELD)))[:batch]
            for arm in ['raw','normalized']:
                model=make_head(kind,WAVLM,arm,stats).to(device)
                x=torch.as_tensor(np.array(features[kind][ix]),device=device);refs=torch.as_tensor(z[kind+'_refs'][ix],device=device)
                opt=torch.optim.AdamW(model.parameters());torch.cuda.reset_peak_memory_stats(device)
                tick=time.monotonic();loss=torch.nn.functional.cross_entropy(model(x,refs),torch.as_tensor(z[kind+'_y'][ix],device=device))
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite disposable profile loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite profile gradient')
                opt.step();torch.cuda.synchronize(device)
                profile.append({'arm':arm,'head':kind,'batch':len(ix),'seconds':time.monotonic()-tick,
                                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'resources':resources(reg,work,device),
                                'scope':'one disposable smoke update, not fitted checkpoint'})
                del model,opt,x,refs,loss;gc.collect();torch.cuda.empty_cache()
        dump(work/'LOCAL_HEAD_PROFILE.json',profile)
        identities={arm:hashlib.sha256((sha(a.registration)+sha(stats_path)+arm).encode()).hexdigest() for arm in ['raw','normalized']}
        status.update(status='PILOT_TRAINING',identities=identities,normalization_stats_sha256=sha(stats_path))
        dump(work/'STATUS.json',status)
        pilot_receipt=Path(reg['reports'])/'PILOT_VERIFICATION.json'
        if pilot_receipt.exists():
            # Manual resume after a verified pilot must never retrain/overwrite it.
            proof=json.loads(pilot_receipt.read_text())
            if proof['registration_sha256']!=sha(a.registration):raise ValueError('verified pilot lineage changed')
            for checked in proof['arm_verifications']:
                for path,digest in checked['file_sha256'].items():
                    if sha(path)!=digest:raise ValueError('verified pilot epoch/prediction artifact changed')
            comparison=json.loads((work/'PILOT_COMPARISON.json').read_text())
            pilot=comparison['arms'];quality=gate_decision(pilot['raw'],pilot['normalized'])
            if quality!=proof['quality_gate']:raise ValueError('verified pilot quality gate changed')
        else:
            for kind in ['target','gap']:
                for arm in ['raw','normalized']:
                    folder=work/arm;folder.mkdir(exist_ok=True)
                    status.update(arm=arm,head=kind);dump(work/'STATUS.json',status)
                    complete,_=engine.fit_head(kind,features[kind],z[kind+'_refs'],z[kind+'_y'],z[kind+'_speaker'],z[kind+'_uid'],
                        folder,device,identities[arm],expired,epochs=PILOT,local=True,geometry=WAVLM,decision=DECISION,
                        resource_check=guard,model_factory=lambda k,g:make_head(k,g,arm,stats))
                    gc.collect();torch.cuda.empty_cache()
                    if not complete:status['status']='PAUSED_SAFE_OPTIMIZER_BOUNDARY';return
            pilot={arm:summarize_arm(work/arm,rows,z,PILOT) for arm in ['raw','normalized']}
            quality=gate_decision(pilot['raw'],pilot['normalized'])
            comparison={'decision':DECISION,'arms':pilot,'quality_gate':quality,
                        'uncertainty':paired_uncertainty(pilot['raw'],pilot['normalized'],rows),
                        'registration_sha256':sha(a.registration),'normalization_stats_sha256':sha(stats_path)}
            dump(work/'PILOT_COMPARISON.json',comparison);dump(Path(reg['reports'])/'PILOT_COMPARISON.json',comparison)
            status['status']='PILOT_COMPLETE_PENDING_VERIFICATION';dump(work/'STATUS.json',status)
            proof=verify_stage(a.registration,'pilot')
        if proof['status']!='PAIRED_PILOT50_STATES_PREDICTIONS_GATE_VERIFIED':raise ValueError('pilot correctness not verified')
        if not quality['quality_gate_passed']:
            status.update(status='COMPLETE_PILOT_GATE_NOT_PASSED_NO_300_TRAINING',quality_gate=quality);return
        admission_path=work/'CONDITIONAL300_ADMISSION.json'
        if admission_path.exists():
            admission=json.loads(admission_path.read_text())
            if admission['pilot_verification_sha256']!=sha(pilot_receipt) or admission['registration_sha256']!=sha(a.registration):
                raise ValueError('conditional continuation admission lineage changed')
        else:
            dump(admission_path,{'pilot_verification_sha256':sha(pilot_receipt),
                 'quality_gate':quality,'resources':resources(reg,work,device),'continuation':'fresh normalized pilot50->total300',
                 'registration_sha256':sha(a.registration)})
        # Preserve pilot50 as an immutable full-state boundary before any continuation.
        for kind in ['target','gap']:
            source=work/'normalized'/f'{kind}_epoch50.pt'
            path=work/(kind+'_normalized50_parent.json');record={'sha256':sha(source),'source':str(source)}
            if path.exists() and json.loads(path.read_text())!=record:raise ValueError('normalized50 parent changed')
            if not path.exists():dump(path,record)
        status['status']='NORMALIZED_CONDITIONAL50_TO300';dump(work/'STATUS.json',status)
        for kind in ['target','gap']:
            status.update(arm='normalized',head=kind);dump(work/'STATUS.json',status)
            complete,_=engine.fit_head(kind,features[kind],z[kind+'_refs'],z[kind+'_y'],z[kind+'_speaker'],z[kind+'_uid'],
                work/'normalized',device,identities['normalized'],expired,epochs=FINAL,local=True,geometry=WAVLM,decision=DECISION,
                resource_check=guard,model_factory=lambda k,g:make_head(k,g,'normalized',stats))
            gc.collect();torch.cuda.empty_cache()
            if not complete:status.update(status='PAUSED_SAFE_OPTIMIZER_BOUNDARY',arm='normalized',head=kind);return
        final=summarize_arm(work/'normalized',rows,z,FINAL)
        dump(work/'NORMALIZED300_REPORT.json',final);dump(Path(reg['reports'])/'NORMALIZED300_REPORT.json',final)
        status['status']='NORMALIZED300_COMPLETE_PENDING_VERIFICATION';dump(work/'STATUS.json',status)
        proof=verify_stage(a.registration,'final')
        if proof['status']!='NORMALIZED300_STATES_AND_CURVES_VERIFIED':raise ValueError('terminal300 not verified')
        status['status']='COMPLETE_NORMALIZED300_VERIFIED'
    except BaseException as exc:
        status.update(status='FAILED_NO_BATCH_SKIPPED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
    finally:
        status['seconds']=time.monotonic()-started;dump(work/'STATUS.json',status)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
