"""Bounded local normalized baseline-vs-2x-dropout reproduction/control.

Exactly50 epochs/head/arm. No LS, backbone training, new cache, remote service,
threshold tuning, or automatic continuation. Admission precedes work creation.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import hashlib
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
import psutil
import torch
from torch import nn

from . import epoch50 as engine
from . import raw_normalized as prior
from .audit import ROOT, sha
from .conditioning import fit_layer_rms
from .features import WAVLM

DECISION = 'SM13-DROPOUT-PAIRED-REPRODUCE-20261114-v1'
REPORTS = ROOT / 'reports/SM13_DROPOUT_PAIRED_20261114'
WORK = Path('/home/khoa/.local/share/langfens/sm13_dropout_verify_20261114_v1')
PROTOCOL = ROOT / 'reports/SM13_DROPOUT_VERIFY_20261114/PROTOCOL.json'
PRIOR_REGISTRATION = prior.REPORTS / 'REGISTRATION.json'
ARMS = ('baseline', 'dropout2x')
EPOCHS = 50


def source_hashes():
    paths = [Path(__file__), ROOT / 'tools/verify_sm13_dropout_paired.py']
    paths += [Path(__file__).with_name(n) for n in ['epoch50.py', 'heads.py', 'conditioning.py', 'features.py',
              'detection.py', 'local_s1.py', 'local_s1_fit.py', 'raw_normalized.py']]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def validate_registration(reg):
    if (reg['decision'] != DECISION or reg['platform'] != 'LOCAL_ONLY' or reg['epochs'] != EPOCHS or
            reg['arms'] != list(ARMS) or reg['source_hashes'] != source_hashes() or
            reg['protocol'] != str(PROTOCOL) or reg['protocol_sha256'] != sha(PROTOCOL) or
            reg['prior_registration'] != str(PRIOR_REGISTRATION) or
            reg['prior_registration_sha256'] != sha(PRIOR_REGISTRATION) or
            reg['historical_verification_sha256'] != sha(prior.REPORTS / 'FINAL_VERIFICATION.json') or
            reg['historical_stats_sha256'] != sha(prior.WORK / 'NORMALIZATION_STATS.json') or
            reg['work'] != str(WORK) or reg['reports'] != str(REPORTS) or reg['seconds'] != 3600 or
            reg['threshold_policy'] != 'fixed.5/no_retuning/no_bestepoch' or reg['seed'] != 101):
        raise ValueError('paired reproduction contract/source changed')
    protocol = json.loads(PROTOCOL.read_text())
    if (protocol['epochs_per_head_per_arm'] != 50 or protocol['quality_gates'] != prior.GATE or
            protocol['root_CROTTC_export_floor_bytes'] != 14 * (1 << 30) or
            protocol['home_floor_bytes'] != 4 * (1 << 30) or
            protocol['new_output_admission_bytes'] != 1536 * (1 << 20) or
            protocol['arms'] != {'baseline': {'target_dropout': [.25, .20], 'gap_dropout': [.15]},
                                 'candidate': {'target_dropout': [.50, .40], 'gap_dropout': [.30]}}):
        raise ValueError('original scope/resource/quality gates changed')
    return protocol


def admission(protocol, *, initial, device=None):
    root_free = shutil.disk_usage('/var/tmp').free
    home_free = shutil.disk_usage('/home/khoa').free
    ram = psutil.virtual_memory().available
    checks = {
        'root_CROTTC_reserve': root_free >= protocol['root_CROTTC_export_floor_bytes'],
        'home_floor_and_budget': home_free >= protocol['home_floor_bytes'] + (protocol['new_output_admission_bytes'] if initial else 0),
        'RAM_floor': ram >= protocol['initial_RAM_floor_bytes' if initial else 'runtime_RAM_floor_bytes'],
    }
    record = {'root_free_bytes': root_free, 'home_free_bytes': home_free, 'RAM_available_bytes': ram,
              'checks': checks, 'initial': initial, 'GPU_checked': device is not None}
    if device is not None:
        free, total = torch.cuda.mem_get_info(device)
        checks['VRAM_floor'] = free >= protocol['VRAM_floor_bytes']
        record.update(VRAM_free_bytes=free, VRAM_total_bytes=total)
    record['passed'] = all(checks.values())
    return record


def make_head(kind, geometry, arm, stats):
    if arm not in ARMS:
        raise ValueError('only baseline and dropout2x are registered')
    model = prior.make_head(kind, geometry, 'normalized', stats)
    if arm == 'dropout2x':
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p *= 2
    expected = ([.25, .20] if kind == 'target' else [.15])
    if arm == 'dropout2x':
        expected = [2 * value for value in expected]
    if [m.p for m in model.modules() if isinstance(m, nn.Dropout)] != expected:
        raise ValueError('registered dropout recipe not constructed')
    return model


def initialization_pair(kind, stats):
    seed = 101 if kind == 'target' else 1101
    torch.manual_seed(seed)
    baseline = make_head(kind, WAVLM, 'baseline', stats)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    candidate = make_head(kind, WAVLM, 'dropout2x', stats)
    engine.assert_equal_state(baseline.state_dict(), candidate.state_dict())
    if not torch.equal(rng, torch.get_rng_state()):
        raise ValueError('construction changed paired initial RNG')
    digest = hashlib.sha256()
    for key, tensor in baseline.state_dict().items():
        digest.update(key.encode()); digest.update(tensor.numpy().tobytes())
    return {'head': kind, 'seed': seed, 'state_and_initial_RNG_equal': True, 'state_sha256': digest.hexdigest()}


def load_data(reg):
    base = json.loads(Path(reg['prior_registration']).read_text())
    prior.validate_registration(base)
    return prior.load_data(base, verify_cache=True)


def normalization_stats(z, features, cache_proof):
    stats = {}
    for kind in ['target', 'gap']:
        sp = z[kind + '_speaker']; fit = set(sp) - engine.HELD
        rms = fit_layer_rms(features[kind], sp, fit_speakers=fit, held_speakers=engine.HELD)
        stats[kind] = {'rms': rms.tolist(), 'fit_speakers': sorted(fit), 'held_speakers': sorted(engine.HELD),
                       'fit_events': int((~np.isin(sp, sorted(engine.HELD))).sum()),
                       'feature_sha256': cache_proof['files'][kind + '.npy']['sha256']}
    previous = json.loads((prior.WORK / 'NORMALIZATION_STATS.json').read_text())
    if stats != previous:
        raise ValueError('fit-only scales differ from historical normalized baseline')
    return stats


def prepare():
    if REPORTS.exists() or WORK.exists():
        raise FileExistsError('preserve prior paired artifacts/registration')
    reg = {'decision': DECISION, 'platform': 'LOCAL_ONLY', 'epochs': EPOCHS, 'arms': list(ARMS),
           'source_hashes': source_hashes(), 'protocol': str(PROTOCOL), 'protocol_sha256': sha(PROTOCOL),
           'prior_registration': str(PRIOR_REGISTRATION), 'prior_registration_sha256': sha(PRIOR_REGISTRATION),
           'historical_verification_sha256': sha(prior.REPORTS / 'FINAL_VERIFICATION.json'),
           'historical_stats_sha256': sha(prior.WORK / 'NORMALIZATION_STATS.json'),
           'work': str(WORK), 'reports': str(REPORTS), 'seconds': 3600, 'seed': 101,
           'threshold_policy': 'fixed.5/no_retuning/no_bestepoch',
           'authorization': 'user demands actual reproduction/control and continue execution, not brainstorming',
           'ordering': 'baseline target50+gap50; blocking independent baseline reproduction verification; candidate target50+gap50; blocking paired verification',
           'after50': 'stop regardless of quality; no automatic extension or retry',
           'scope': 'one paired seed/reused TRAIN development, neural-only, not final DEV_SEARCH or paper/test'}
    protocol = validate_registration(reg)
    historical = prior.WORK / 'normalized'
    estimate = 2 * sum((historical / f'{kind}_epoch50.pt').stat().st_size * 51 +
                       (historical / f'{kind}_held_epoch50.npz').stat().st_size * 50 for kind in ['target', 'gap'])
    if estimate * 1.15 > protocol['new_output_admission_bytes']:
        raise OSError('measured paired outputs plus15% margin exceed registered output budget')
    reg['output_estimate_bytes'] = estimate
    reg['output_estimate_basis'] = 'historical same-capacity full states51/head +50predictions/head,2arms,15%budgetmargin'
    REPORTS.mkdir()
    engine.dump(REPORTS / 'REGISTRATION.json', reg)
    provenance = REPORTS / 'PROVENANCE'; provenance.mkdir()
    for relative in reg['source_hashes']:
        dest = provenance / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    return reg


def verify_stage(registration, stage, timeout):
    if timeout < 1:
        raise TimeoutError('no verification time remains')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (REPORTS / (stage + '_VERIFIER.log')).open('a') as log:
        subprocess.run([sys.executable, '-u', '-m', 'tools.verify_sm13_dropout_paired', '--registration', str(registration), '--stage', stage],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    return json.loads((REPORTS / (stage.upper() + '_VERIFICATION.json')).read_text())


def run(registration, *, resume=False, check_only=False):
    reg = json.loads(registration.read_text()); protocol = validate_registration(reg)
    if resume:
        if not WORK.is_dir() or not (WORK / 'STATUS.json').is_file():
            raise FileNotFoundError('cannot resume nonexistent run or bypass fresh output admission')
        previous = json.loads((WORK / 'STATUS.json').read_text())
        if previous['registration_sha256'] != sha(registration):
            raise ValueError('resume run belongs to different registration')
    # CPU-only storage/RAM precheck before work directory, CUDA allocation or GPU lock.
    initial = admission(protocol, initial=not resume)
    snapshot = REPORTS / ('ADMISSION_' + str(time.time_ns()) + '.json')
    engine.dump(snapshot, initial)
    if check_only or not initial['passed']:
        status = {'decision': DECISION, 'status': 'CPU_ADMISSION_PASSED_GPU_PENDING' if initial['passed'] else 'RESOURCE_BLOCKED_NO_TRAINING',
                  'admission_snapshot': str(snapshot), 'training_started': False, 'auto_launch_scheduled': False}
        engine.dump(REPORTS / 'LATEST_ADMISSION_STATUS.json', status)
        print(json.dumps(status), flush=True)
        return
    if WORK.exists() and not resume:
        raise FileExistsError('explicit manual --resume required; preserve original run')
    lock = prior.LOCK.open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic(); halted = False
    status = {'decision': DECISION, 'status': 'PREFLIGHT', 'pid': os.getpid(), 'registration_sha256': sha(registration),
              'training_started': False, 'official_test_opened': False, 'dev_cal_opened': False,
              'backbone_loaded': False, 'threshold_retuning_performed': False, 'platform': 'LOCAL_ONLY'}
    def persist():
        engine.dump(REPORTS / 'STATUS.json', status)
        if WORK.exists():
            engine.dump(WORK / 'STATUS.json', status)
    def stop(signum, frame):
        nonlocal halted
        halted = True
    def expired():
        return halted or time.monotonic() - started >= reg['seconds'] - 300
    def remaining():
        return max(0, int(reg['seconds'] - (time.monotonic() - started)))
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required; no CPU/remote fallback')
        device = torch.device('cuda:0')
        jobs = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True, timeout=10).stdout
        if any(s.strip().isdigit() and int(s.strip()) != os.getpid() for s in jobs.splitlines()):
            raise RuntimeError('other GPU process present')
        status['admission'] = admission(protocol, initial=not resume, device=device)
        if not status['admission']['passed']:
            status['status'] = 'RESOURCE_BLOCKED_NO_TRAINING'; return
        WORK.mkdir(exist_ok=resume)
        status['environment'] = {'python': sys.version, 'torch': torch.__version__, 'CUDA': torch.version.cuda,
                                 'device': torch.cuda.get_device_name(device), 'cpu_threads': 2}
        persist()
        rows, z, features, cache_proof = load_data(reg)
        if expired():
            status['status'] = 'BUDGET_EXPIRED_BEFORE_FIT'; return
        stats = normalization_stats(z, features, cache_proof)
        stats_path = WORK / 'NORMALIZATION_STATS.json'
        if stats_path.exists() and json.loads(stats_path.read_text()) != stats:
            raise ValueError('resume normalization changed')
        if not stats_path.exists():
            engine.dump(stats_path, stats)
        pairs = {kind: initialization_pair(kind, stats) for kind in ['target', 'gap']}
        engine.dump(WORK / 'INITIALIZATION_PAIR.json', pairs)
        status['identities'] = {arm: hashlib.sha256((sha(registration) + sha(stats_path) + arm).encode()).hexdigest() for arm in ARMS}
        status['normalization_stats_sha256'] = sha(stats_path)
        def guard():
            info = admission(protocol, initial=False, device=device)
            if not info['passed']:
                status['resource_pause'] = info
            return info['passed']
        if not guard():
            status['status'] = 'RESOURCE_BLOCKED_BEFORE_PROFILE'; return
        profile_path = WORK / 'LOCAL_PROFILE.json'
        if not profile_path.exists():
            profile = []
            for kind, batch in [('target', 512), ('gap', 1024)]:
                ix = np.flatnonzero(~np.isin(z[kind + '_speaker'], sorted(engine.HELD)))[:batch]
                for arm in ARMS:
                    if expired() or not guard():
                        status['status'] = 'PAUSED_BEFORE_PROFILE_COMPLETE'; return
                    model = make_head(kind, WAVLM, arm, stats).to(device)
                    opt = torch.optim.AdamW(model.parameters(), lr=.001 if kind == 'target' else .0008,
                                            weight_decay=.01 if kind == 'target' else .0001)
                    torch.cuda.reset_peak_memory_stats(device); tick = time.monotonic()
                    x = torch.tensor(np.array(features[kind][ix]), device=device)
                    refs = torch.tensor(z[kind + '_refs'][ix], device=device)
                    labels = torch.tensor(z[kind + '_y'][ix], device=device)
                    weights = torch.tensor([1., 2.5, 4.] if kind == 'target' else engine.gap_weights(z[kind + '_y'][~np.isin(z[kind + '_speaker'], sorted(engine.HELD))]), device=device)
                    loss = torch.nn.functional.cross_entropy(model(x, refs), labels, weight=weights)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite disposable profile loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('nonfinite profile gradient')
                    opt.step(); torch.cuda.synchronize(device)
                    profile.append({'head': kind, 'arm': arm, 'batch': batch, 'seconds': time.monotonic() - tick,
                                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                                    'scope': 'disposable resource smoke, not fitted checkpoint'})
                    del model, opt, x, refs, labels, weights, loss
                    gc.collect(); torch.cuda.empty_cache()
            engine.dump(profile_path, profile)
        for arm in ARMS:
            proof_path = REPORTS / ('BASELINE_VERIFICATION.json' if arm == 'baseline' else 'PAIRED_VERIFICATION.json')
            if proof_path.exists():
                proof = json.loads(proof_path.read_text())
                if proof['registration_sha256'] != sha(registration):
                    raise ValueError('existing verification belongs to different run')
                for verified_arm in proof['arm_verifications']:
                    for path, digest in verified_arm['file_sha256'].items():
                        if sha(path) != digest:
                            raise ValueError('verified artifact changed')
                continue
            folder = WORK / arm; folder.mkdir(exist_ok=True)
            for kind in ['target', 'gap']:
                status.update(status='TRAINING', arm=arm, head=kind, training_started=True); persist()
                done, _ = engine.fit_head(kind, features[kind], z[kind + '_refs'], z[kind + '_y'],
                    z[kind + '_speaker'], z[kind + '_uid'], folder, device, status['identities'][arm], expired,
                    epochs=EPOCHS, local=True, geometry=WAVLM, decision=DECISION, resource_check=guard,
                    model_factory=lambda k, g: make_head(k, g, arm, stats))
                gc.collect(); torch.cuda.empty_cache()
                if not done:
                    status['status'] = 'PAUSED_SAFE_OPTIMIZER_BOUNDARY'; return
            summary = prior.summarize_arm(folder, rows, z, EPOCHS)
            engine.dump(WORK / (arm + '_SUMMARY.json'), summary)
            stage = 'baseline' if arm == 'baseline' else 'paired'
            if stage == 'paired':
                baseline = json.loads((WORK / 'baseline_SUMMARY.json').read_text())
                comparison = {'decision': DECISION, 'registration_sha256': sha(registration),
                              'arms': {'baseline': baseline, 'dropout2x': summary},
                              'quality_gate': prior.gate_decision(baseline, summary),
                              'uncertainty': prior.paired_uncertainty(baseline, summary, rows)}
                engine.dump(WORK / 'COMPARISON.json', comparison)
                engine.dump(REPORTS / 'COMPARISON.json', comparison)
            status.update(status='BASELINE_PENDING_VERIFICATION' if stage == 'baseline' else 'PAIRED_PENDING_VERIFICATION'); persist()
            proof = verify_stage(registration, stage, min(900, remaining()))
            if proof['status'] != ('BASELINE_REPRODUCTION_VERIFIED' if stage == 'baseline' else 'PAIRED50_STATES_METRICS_VERIFIED'):
                raise ValueError('independent verification did not pass')
        status['status'] = 'COMPLETE_PAIRED50_VERIFIED_NO_EXTENSION'
    except BaseException as exc:
        status.update(status='FAILED_NO_BATCH_SKIPPED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status['seconds'] = time.monotonic() - started
        persist()
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-admission', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), indent=2)); return
    if args.registration is None:
        parser.error('--registration required')
    run(args.registration, resume=args.resume, check_only=args.check_admission)


if __name__ == '__main__':
    main()
