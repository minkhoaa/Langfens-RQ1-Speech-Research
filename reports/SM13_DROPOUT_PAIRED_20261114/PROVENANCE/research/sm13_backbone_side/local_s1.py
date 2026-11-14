"""Local-only, offline S1 WavLM Large profile and resumable TRAIN cache.

No Kaggle API/launcher, backbone backward, gold timing, DEV_CAL or test access.
Production head fitting remains gated on the registered common evaluator.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from .audit import ROOT, TRAIN, sha, validate_population
from .backbone import FrozenWavLM
from .features import WAVLM, cache_identity
from .heads import ScalarMixingGapHead, ScalarMixingTargetHead, parameter_counts

DECISION = 'SM13-NATIVE-LOCAL-20261114-v1'
LOCK = Path('/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock')
NORMALIZATION = 'HF_feature_extractor_then_SB_layer_norm_if_do_normalize; batch1; no_output_norm; eval_specaugment_disabled'
FILES = ['local_s1.py', 'backbone.py', 'features.py', 'heads.py']


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def source_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in FILES}


def profile_rows(rows, durations):
    if len(rows) != len(durations) or not rows or not np.isfinite(durations).all() or min(durations) <= 0:
        raise ValueError('positive finite duration inventory required')
    order = sorted(range(len(rows)), key=lambda i: (durations[i], rows[i]['utterance_id']))
    return [(tag, rows[order[min(len(rows)-1, math.ceil(q*len(rows))-1)]])
            for tag, q in [('median', .5), ('p95', .95), ('longest', 1.)]]


def cache_bytes(targets=59562, gaps=61362):
    return (targets*13*3072 + gaps*13*2048)*4 + 256


def validate_registration(reg):
    if (reg['decision'] != DECISION or reg['platform'] != 'LOCAL_ONLY' or
            reg['arm'] != 'S1_NATIVE_FROZEN' or reg['dtype'] != 'float32' or
            reg['backbone_trainable_parameters'] != 0 or reg['cpu_threads'] != 2 or
            reg['source_hashes'] != source_hashes() or reg['normalization'] != NORMALIZATION or
            reg['phase'] not in ['profile', 'cache']):
        raise ValueError('unregistered S1 local contract/source change')
    if not 0 < reg['seconds'] <= (600 if reg['phase'] == 'profile' else 14400):
        raise ValueError('bounded local budget required')
    floors = reg['floors']
    if (floors['initial_ram_bytes'] < 16*(1<<30) or floors['ram_bytes'] < 4*(1<<30) or
            floors['disk_bytes'] < 14*(1<<30) or floors['vram_bytes'] < 512*(1<<20)):
        raise ValueError('registered safety floors cannot be reduced')
    return True


def load_requests(reg):
    audit_dir = Path(reg['audit_dir'])
    audit = json.loads((audit_dir/'AUDIT.json').read_text())
    path = audit_dir/'TRAIN_INFERENCE_REQUESTS.jsonl'
    if sha(path) != reg['requests_sha256'] or sha(path) != audit['inference_requests_sha256']:
        raise ValueError('frozen requests changed')
    manifest = ROOT/'data/derived/l2arctic/manifests/train.jsonl'
    if sha(manifest) != reg['train_manifest_sha256']:
        raise ValueError('TRAIN manifest changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = validate_population(rows, 1800, TRAIN)
    if hashlib.sha256('\n'.join(ids).encode()).hexdigest() != audit['ordered_ids_sha256']:
        raise ValueError('event order changed')
    for row in rows:
        if set(row) != {'utterance_id', 'speaker_id', 'wav', 'audio_sha256', 'canonical_phones', 'inference_contract'}:
            raise ValueError('unexpected fields in inference request')
        audio = Path(row['wav']).resolve()
        audio.relative_to(ROOT/'data/derived/l2arctic/wavs_16k')
    if sum(len(row['canonical_phones']) for row in rows) != 59562:
        raise ValueError('target population changed')
    align_path = Path(reg['alignment'])
    if sha(align_path) != reg['alignment_sha256'] or sha(align_path) != audit['alignment_sha256']:
        raise ValueError('predicted alignment changed')
    alignment = np.load(align_path, allow_pickle=False)
    if set(alignment.files) != {prefix+uid for uid in ids for prefix in ['centers__', 'nframes__']}:
        raise ValueError('unexpected/protected alignment IDs')
    for row in rows:
        uid = row['utterance_id']
        if len(alignment['centers__'+uid]) != len(row['canonical_phones']):
            raise ValueError('alignment event topology changed')
    native = Path(reg['native_dir'])
    receipt = json.loads(Path(reg['native_receipt']).read_text())
    if receipt['status'] != 'VERIFIED_COMPLETE' or receipt['revision'] != reg['native_revision']:
        raise ValueError('unverified native lineage')
    for name, meta in receipt['files'].items():
        if native.joinpath(name).stat().st_size != meta['bytes'] or sha(native/name) != meta['sha256']:
            raise ValueError('native model hash mismatch')
    if reg['native_files'] != receipt['files']:
        raise ValueError('native file registration changed')
    return rows, alignment, audit


def check_resources(reg, work, device, *, initial=False, remaining_bytes=0):
    import psutil
    import shutil
    import torch
    floors = reg['floors']
    ram = psutil.virtual_memory().available
    disk = shutil.disk_usage(work).free
    free, total = torch.cuda.mem_get_info(device)
    if ram < floors['initial_ram_bytes' if initial else 'ram_bytes']:
        raise MemoryError('S1 local RAM safety floor')
    if disk < floors['disk_bytes'] + remaining_bytes:
        raise OSError('S1 disk safety floor including remaining cache; preserve CROTTC export reserve')
    if free < floors['vram_bytes']:
        raise MemoryError('S1 local VRAM safety floor')
    return {'ram_available_bytes': ram, 'disk_free_bytes': disk, 'vram_free_bytes': free,
            'vram_total_bytes': total, 'rss_bytes': psutil.Process().memory_info().rss}


def audio_features(model, row, alignment):
    import soundfile as sf
    path = Path(row['wav'])
    if sha(path) != row['audio_sha256']:
        raise ValueError('allowlisted TRAIN audio SHA changed: '+row['utterance_id'])
    waveform, rate = sf.read(path, dtype='float32')
    uid = row['utterance_id']
    target, gap = model.features(waveform, rate, alignment['centers__'+uid], int(alignment['nframes__'+uid]))
    n = len(row['canonical_phones'])
    for values, shape in [(target, (n,13,3072)), (gap, (n+1,13,2048))]:
        if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('native feature geometry/dtype/finite check failed')
    return target, gap


def profile(reg, rows, alignment, model, work, device, expired, report):
    import soundfile as sf
    import torch
    durations = [sf.info(row['wav']).duration for row in rows]
    selected = profile_rows(rows, durations)
    expected = [(item['tag'], item['utterance_id']) for item in reg['profile_selection']]
    if [(tag, row['utterance_id']) for tag, row in selected] != expected:
        raise ValueError('registered duration selection changed')
    report.update(samples=[], head_batches=[])
    # One real warm-up, then synchronize around each timed inference.
    audio_features(model, selected[0][1], alignment)
    torch.cuda.synchronize(device)
    for tag, row in selected:
        if expired():
            raise TimeoutError('bounded local profile expired')
        torch.cuda.reset_peak_memory_stats(device)
        before = check_resources(reg, work, device)
        torch.cuda.synchronize(device)
        tick = time.monotonic()
        target, gap = audio_features(model, row, alignment)
        torch.cuda.synchronize(device)
        report['samples'].append({'tag': tag, 'id': row['utterance_id'],
            'seconds': time.monotonic()-tick, 'duration': sf.info(row['wav']).duration,
            'target_shape': list(target.shape), 'gap_shape': list(gap.shape),
            'target_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
            'gap_sha256': hashlib.sha256(gap.tobytes()).hexdigest(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'before': before, 'after': check_resources(reg, work, device)})
        dump(work/'PROFILE.json', report)
    # Backbone is freed before the production-style head batch/backward checks.
    del model.model
    gc.collect()
    torch.cuda.empty_cache()
    for kind, values, cls, batch, refdim in [
            ('target', target, ScalarMixingTargetHead, 512, 39),
            ('gap', gap, ScalarMixingGapHead, 1024, 78)]:
        if expired():
            raise TimeoutError('bounded local profile expired before head batch')
        torch.cuda.reset_peak_memory_stats(device)
        head = cls(WAVLM).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=.001 if kind=='target' else .0008)
        x = torch.tensor(values[np.arange(batch) % len(values)], device=device)
        refs = torch.zeros(batch, refdim, device=device)
        labels = torch.arange(batch, device=device) % (3 if kind=='target' else 2)
        tick = time.monotonic()
        loss = torch.nn.functional.cross_entropy(head(x, refs), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite head profile loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
            raise FloatingPointError('nonfinite head profile gradient')
        opt.step()
        torch.cuda.synchronize(device)
        report['head_batches'].append({'head': kind, 'batch': batch, 'seconds': time.monotonic()-tick,
            'loss': float(loss.detach()), 'parameters': sum(p.numel() for p in head.parameters()),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'resources': check_resources(reg, work, device),
            'scope': 'one disposable resource-smoke update; NOT a fitted model'})
        del head, opt, x, refs, labels, loss
        gc.collect()
        torch.cuda.empty_cache()
    # Conservative admission estimate, not a throughput benchmark or runtime promise.
    total_audio = sum(durations)
    ratio = max(s['seconds']/s['duration'] for s in report['samples'])
    report['estimated_train_cache_seconds_2x_sample_rtf'] = 2*total_audio*ratio
    report['train_audio_seconds'] = total_audio
    report['status'] = 'PROFILE_PASSED_NATIVE_FROZEN_AND_DISPOSABLE_HEAD_BATCHES'
    dump(work/'PROFILE.json', report)


def create_cache(reg, rows, alignment, model, work, device, expired, report, audit):
    proof = json.loads(Path(reg['profile_proof']).read_text())
    if sha(reg['profile_proof']) != reg['profile_proof_sha256'] or proof['status'] != 'PROFILE_PASSED_NATIVE_FROZEN_AND_DISPOSABLE_HEAD_BATCHES':
        raise ValueError('verified local S1 profile required before production cache')
    if proof['native_files'] != reg['native_files'] or proof['source_hashes'] != reg['source_hashes']:
        raise ValueError('profile native/source identity changed')
    code = hashlib.sha256(json.dumps(reg['source_hashes'], sort_keys=True).encode()).hexdigest()
    identity, provenance = cache_identity(model_sha256=reg['native_files']['pytorch_model.bin']['sha256'],
        config_sha256=reg['native_files']['config.json']['sha256'], normalization=model.normalization,
        alignment_sha256=reg['alignment_sha256'], manifest_sha256=reg['train_manifest_sha256'],
        ordered_ids_sha256=audit['ordered_ids_sha256'], geometry=WAVLM, source_sha256=code)
    report.update(cache_identity=identity, cache_provenance=provenance)
    progress_path = work/'CACHE_PROGRESS.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'completed':0,'target_cursor':0,'gap_cursor':0,'identity':identity,'row_fingerprints':[]}
    if progress['identity'] != identity:
        raise ValueError('partial cache provenance changed')
    completed = progress['completed']
    if not 0 <= completed <= len(rows) or len(progress['row_fingerprints']) != completed:
        raise ValueError('invalid persisted cache cursor')
    t = sum(len(r['canonical_phones']) for r in rows[:completed]); g = t+completed
    if (t,g) != (progress['target_cursor'],progress['gap_cursor']):
        raise ValueError('persisted cache topology changed')
    remaining = cache_bytes(59562-t,61362-g)
    # Arrays are sparse on creation; account for not-yet-written bytes as well.
    check_resources(reg, work, device, remaining_bytes=remaining)
    maps = {}
    for kind,n,width in [('target',59562,3072),('gap',61362,2048)]:
        path = work/(kind+'.npy')
        if path.exists() and not progress_path.exists():
            raise ValueError('unregistered partial cache; preserve and inspect, do not overwrite')
        maps[kind] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float32, shape=(n,13,width))
        if maps[kind].shape != (n,13,width) or maps[kind].dtype != np.float32:
            raise ValueError('persisted cache geometry changed')
    if not progress_path.exists():
        dump(progress_path, progress)
    # Verify every committed row before resume; no unverified partial cache reuse.
    ti = gi = 0
    for row, fingerprint in zip(rows[:completed], progress['row_fingerprints']):
        n = len(row['canonical_phones'])
        actual = {'id':row['utterance_id'], 'target':hashlib.sha256(maps['target'][ti:ti+n].tobytes()).hexdigest(),
                  'gap':hashlib.sha256(maps['gap'][gi:gi+n+1].tobytes()).hexdigest()}
        if actual != fingerprint:
            raise ValueError('committed row SHA mismatch on cache resume')
        ti += n; gi += n+1
    for i,row in enumerate(rows):
        if i < completed:
            continue
        if expired():
            report['status'] = 'PAUSED_BOUNDED_WINDOW_COMMITTED_FEATURE_ROWS'
            return
        check_resources(reg, work, device, remaining_bytes=cache_bytes(59562-t,61362-g))
        target,gap = audio_features(model,row,alignment)
        n = len(row['canonical_phones'])
        maps['target'][t:t+n] = target; maps['gap'][g:g+n+1] = gap
        t += n; g += n+1
        for values in maps.values():
            values.flush()
        progress['row_fingerprints'].append({'id':row['utterance_id'],
            'target':hashlib.sha256(target.tobytes()).hexdigest(), 'gap':hashlib.sha256(gap.tobytes()).hexdigest()})
        progress.update(completed=i+1,target_cursor=t,gap_cursor=g)
        dump(progress_path,progress)
        if (i+1)%25==0 or i==len(rows)-1:
            report.update(completed_utterances=i+1,target_cursor=t,gap_cursor=g)
            dump(work/'STATUS.json',report)
            print(json.dumps({'event':'S1_local_TRAIN_cache','completed':i+1,'total':len(rows)}),flush=True)
    if (t,g)!=(59562,61362):
        raise ValueError('incomplete TRAIN feature population')
    files = {kind+'.npy':{'sha256':sha(work/(kind+'.npy')),'bytes':(work/(kind+'.npy')).stat().st_size} for kind in maps}
    report.update(status='S1_NATIVE_TRAIN_CACHE_COMPLETE_VERIFIED',completed_utterances=len(rows),files=files,
                  row_fingerprints_sha256=sha(progress_path),head_fitting_performed=False)
    dump(work/'FEATURE_RECEIPT.json',report)


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    validate_registration(reg)
    work = Path(reg['work'])
    work.resolve().relative_to(Path('/var/tmp'))
    work.mkdir(parents=True, exist_ok=args.resume)
    lock = LOCK.open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started = time.monotonic()
    halted = False
    def pause(signum, frame):
        nonlocal halted
        halted = True
    signal.signal(signal.SIGTERM,pause)
    signal.signal(signal.SIGINT,pause)
    def expired():
        return halted or time.monotonic()-started >= reg['seconds']-30
    report = {'decision':DECISION,'platform':'LOCAL_ONLY','arm':'S1_NATIVE_FROZEN','phase':reg['phase'],
        'status':'PREFLIGHT','pid':os.getpid(),'official_test_opened':False,'dev_cal_opened':False,
        'production_head_fitting_performed':False,'backbone_trainable_parameters':0,
        'registration_sha256':sha(args.registration),'source_hashes':reg['source_hashes'],'native_files':reg['native_files'],
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('local CUDA required; no automatic remote/CPU fallback')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        import subprocess
        jobs = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=10).stdout
        if any(line.strip().isdigit() and int(line.strip()) != os.getpid() for line in jobs.splitlines()):
            raise RuntimeError('other local GPU job detected')
        report['admission'] = check_resources(reg,work,device,initial=True)
        report['environment'] = {'python':os.sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                                 'device':torch.cuda.get_device_name(device),'cpu_threads':2}
        dump(work/'STATUS.json',report)
        rows,alignment,audit = load_requests(reg)
        model = FrozenWavLM(reg['native_dir'],device=device)
        if model.normalization != reg['normalization'] or model.parent is not None:
            raise ValueError('unregistered native normalization/parent')
        if sum(p.numel() for p in model.model.parameters()) != 315453120 or any(p.requires_grad for p in model.model.parameters()):
            raise ValueError('native WavLM Large size/freeze mismatch')
        report.update(status='RUNNING',normalization=model.normalization,head_capacity=parameter_counts(WAVLM))
        dump(work/'STATUS.json',report)
        if reg['phase']=='profile':
            profile(reg,rows,alignment,model,work,device,expired,report)
        else:
            create_cache(reg,rows,alignment,model,work,device,expired,report,audit)
    except BaseException as exc:
        report.update(status='FAILED_NO_SAMPLE_SKIPPED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        dump(work/'STATUS.json',report)
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


if __name__=='__main__':
    main()
