"""Independent CPU subprocess verifier: baseline must reproduce before candidate.

Checks all204states, saved predictions, sklearn curves, full-state boundaries,
fixed gate and fresh endpoint fit/held inference. Never fits or launches training.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from research.sm13_backbone_side import epoch50 as engine
from research.sm13_backbone_side import dropout_paired as run
from research.sm13_backbone_side import raw_normalized as prior
from research.sm13_backbone_side.audit import sha
from research.sm13_backbone_side.features import WAVLM
from tools.debug_sm13_s1 import forward_all, probability_diagnostics


def verify_arm(folder, arm, z, stats, identity, *, epochs=50):
    files = {}; states = 0
    for kind, classes in [('target', 3), ('gap', 2)]:
        held = np.isin(z[kind + '_speaker'], sorted(engine.HELD))
        history = json.loads((folder / (kind + '_history.json')).read_text())
        if [h['epoch'] for h in history] != list(range(1, epochs + 1)):
            raise ValueError('noncontiguous history')
        steps = (int((~held).sum()) + (512 if kind == 'target' else 1024) - 1) // (512 if kind == 'target' else 1024)
        for epoch, h in enumerate(history, 1):
            path = folder / f'{kind}_epoch{epoch:02d}.pt'
            receipt_path = path.with_suffix('.receipt.json')
            receipt = json.loads(receipt_path.read_text())
            if sha(path) != receipt['sha256'] or receipt['decision'] != run.DECISION:
                raise ValueError('checkpoint receipt SHA/decision mismatch')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if (saved['decision'] != run.DECISION or saved['identity'] != identity or saved['head'] != kind or
                    saved['epoch'] != epoch or saved['cursor'] != 0 or saved['order'] is not None or
                    saved['partial_losses'] or saved['pending_gradients'] != 0 or saved['updates'] != epoch * steps or
                    saved['history'] != history[:epoch] or saved['scheduler'] is not None or saved['scaler'] is not None or
                    not {'python', 'numpy', 'torch', 'cuda'} <= set(saved['rng'])):
                raise ValueError('full state/counter/history mismatch')
            group = saved['optimizer']['param_groups']
            if len(group) != 1 or group[0]['lr'] != (.001 if kind == 'target' else .0008) or group[0]['weight_decay'] != (.01 if kind == 'target' else .0001):
                raise ValueError('optimizer recipe changed')
            if not saved['optimizer']['state'] or any(float(v['step']) != saved['updates'] for v in saved['optimizer']['state'].values()):
                raise ValueError('Adam counters missing/reset')
            for state in [saved['model'], *saved['optimizer']['state'].values()]:
                if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in state.values()):
                    raise ValueError('nonfinite model/optimizer')
            if not torch.equal(saved['model']['normalizer.rms'], torch.tensor(stats[kind]['rms'], dtype=torch.float32)):
                raise ValueError('frozen fit-only RMS changed')
            if not np.isfinite([h['held_loss'], h['train_loss']]).all():
                raise ValueError('nonfinite history')
            pred_path = folder / f'{kind}_held_epoch{epoch:02d}.npz'
            with np.load(pred_path, allow_pickle=False) as pred:
                for name, key in [('gold', 'y'), ('speakers', 'speaker'), ('uids', 'uid')]:
                    if not np.array_equal(pred[name], z[kind + '_' + key][held]):
                        raise ValueError('prediction population/order mismatch')
                p = pred['probs']
                if p.shape != (int(held.sum()), classes) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, rtol=0, atol=2e-7):
                    raise ValueError('invalid probabilities')
                if engine.metrics(pred['gold'], p, classes) != h['held_metrics']:
                    raise ValueError('saved typed metrics do not reproduce')
            files[str(path)] = receipt['sha256']; files[str(receipt_path)] = sha(receipt_path)
            files[str(pred_path)] = sha(pred_path); states += 1
        last = folder / (kind + '_last.pt')
        proof = json.loads(last.with_suffix('.receipt.json').read_text())
        if sha(last) != proof['sha256']:
            raise ValueError('last receipt SHA mismatch')
        engine.assert_equal_state(torch.load(last, map_location='cpu', weights_only=True), saved)
        files[str(last)] = sha(last); files[str(folder / (kind + '_history.json'))] = sha(folder / (kind + '_history.json'))
        states += 1
    return {'arm': arm, 'states_verified': states, 'file_sha256': files}


def independently_replay_curve(folder, summary):
    for epoch in range(1, len(summary['curve']) + 1):
        labels = []; scores = []; aps = []
        for kind in ['target', 'gap']:
            with np.load(folder / f'{kind}_held_epoch{epoch:02d}.npz', allow_pickle=False) as pred:
                y, p = pred['gold'], pred['probs']
                labels.extend((y != 0).tolist()); scores.extend(p[:, 1:].sum(1).tolist())
                aps.extend(float(average_precision_score(y == c, p[:, c])) for c in range(1, p.shape[1]))
        y, selected = np.asarray(labels), np.asarray(scores) >= .5
        counts = {'TP': int((y & selected).sum()), 'FP': int((~y & selected).sum()),
                  'FN': int((y & ~selected).sum()), 'TN': int((~y & ~selected).sum())}
        c = summary['curve'][epoch - 1]
        if any(counts[k] != c['detection']['micro'][k] for k in counts) or not np.allclose(aps, c['S_O_I_AP'], rtol=0, atol=1e-12):
            raise ValueError('independent counts/AP mismatch')
        f1 = 2 * counts['TP'] / (2 * counts['TP'] + counts['FP'] + counts['FN'])
        if abs(f1 - c['detection']['micro']['F1']) > 1e-12:
            raise ValueError('independent micro F1 mismatch')


def fresh_endpoint(folder, arm, z, stats, features, expired):
    result = {}
    for kind in ['target', 'gap']:
        model = run.make_head(kind, WAVLM, arm, stats).eval()
        saved = torch.load(folder / f'{kind}_epoch50.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'], strict=True)
        y = z[kind + '_y']; held = np.isin(z[kind + '_speaker'], sorted(engine.HELD))
        weights = np.array([1., 2.5, 4.]) if kind == 'target' else engine.gap_weights(y[~held])
        result[kind] = {}
        for split, ix in [('fit', np.flatnonzero(~held)), ('held', np.flatnonzero(held))]:
            p, logits = forward_all(model, features[kind], z[kind + '_refs'], ix, expired, return_logits=True)
            by = y[ix]
            d = probability_diagnostics(by, p, logits)
            shifted = logits.astype(np.float64) - logits.max(1, keepdims=True)
            ce = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(by)), by]
            d['pooled_weighted_CE'] = float((ce * weights[by]).sum() / weights[by].sum())
            if split == 'held':
                with np.load(folder / f'{kind}_held_epoch50.npz', allow_pickle=False) as pred:
                    delta = float(abs(p - pred['probs']).max())
                    if delta > 1e-5 or not np.array_equal(p.argmax(1), pred['probs'].argmax(1)):
                        raise ValueError('fresh CPU held output mismatch')
                    d['saved_GPU_probability_max_abs_delta'] = delta
                if abs(d['pooled_weighted_CE'] - saved['history'][-1]['held_loss']) > 1e-5:
                    raise ValueError('fresh held CE mismatch')
            result[kind][split] = d
    return result


def plot_comparison(summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    for arm in run.ARMS:
        curve = summaries[arm]['curve']; x = [c['epoch'] for c in curve]
        axes[0].plot(x, [100 * c['detection']['micro']['F1'] for c in curve], label=arm)
        axes[1].plot(x, [100 * c['macro3_AP'] for c in curve], label=arm)
        for kind in ['target', 'gap']:
            h = json.loads((run.WORK / arm / (kind + '_history.json')).read_text())
            axes[2].plot(x, [c['held_loss'] for c in h], label=arm + '/' + kind)
    axes[0].set_ylabel('held micro F1@.5 %'); axes[1].set_ylabel('macro S/O/I AP %')
    axes[2].set_ylabel('held weighted CE')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(); ax.set_xlabel('epoch/head')
    fig.suptitle('Normalized S1 baseline vs2x dropout: one paired seed/reused TRAIN, NOT test')
    fig.tight_layout(); fig.savefig(run.REPORTS / 'PAIRED_CURVES.png', dpi=150); plt.close(fig)


def historical_parity(z):
    receipt = json.loads((prior.REPORTS / 'FINAL_VERIFICATION.json').read_text())
    if receipt['status'] != 'NORMALIZED300_STATES_AND_CURVES_VERIFIED':
        raise ValueError('historical normalized baseline not verified')
    hashes = receipt['arm_verifications'][0]['file_sha256']; result = {}
    for kind in ['target', 'gap']:
        historical_path = prior.WORK / 'normalized' / f'{kind}_epoch50.pt'
        if sha(historical_path) != hashes[str(historical_path)]:
            raise ValueError('historical state changed')
        old = torch.load(historical_path, map_location='cpu', weights_only=True)
        new = torch.load(run.WORK / 'baseline' / f'{kind}_epoch50.pt', map_location='cpu', weights_only=True)
        if set(old['model']) != set(new['model']) or not all(torch.allclose(v, new['model'][k], rtol=1e-5, atol=1e-6) for k, v in old['model'].items()):
            raise ValueError('fresh baseline does not reproduce historical normalized50 state')
        for epoch in range(1, 51):
            path = prior.WORK / 'normalized' / f'{kind}_held_epoch{epoch:02d}.npz'
            if sha(path) != hashes[str(path)]:
                raise ValueError('historical predictions changed')
            with np.load(path, allow_pickle=False) as a, np.load(run.WORK / 'baseline' / f'{kind}_held_epoch{epoch:02d}.npz', allow_pickle=False) as b:
                if not np.array_equal(a['gold'], b['gold']) or not np.array_equal(a['uids'], b['uids']) or not np.allclose(a['probs'], b['probs'], rtol=1e-5, atol=1e-6):
                    raise ValueError('fresh baseline curve differs from historical normalized50')
        result[kind] = {'max_abs_weight_delta': max(float(abs(v - new['model'][k]).max()) for k, v in old['model'].items()),
                        'historical_checkpoint_sha256': sha(historical_path), 'all50_prediction_epochs_reproduced': True}
    return result


def verify(registration, stage):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available():
        raise RuntimeError('independent verifier must be CPU only')
    reg = json.loads(registration.read_text()); run.validate_registration(reg)
    status = json.loads((run.WORK / 'STATUS.json').read_text())
    expected = 'BASELINE_PENDING_VERIFICATION' if stage == 'baseline' else 'PAIRED_PENDING_VERIFICATION'
    if status['status'] != expected or status['registration_sha256'] != sha(registration):
        raise ValueError('run not at registered verification boundary')
    if any(status[k] for k in ['official_test_opened', 'dev_cal_opened', 'backbone_loaded', 'threshold_retuning_performed']):
        raise ValueError('scope changed')
    destination = run.REPORTS / (stage.upper() + '_VERIFICATION.json')
    if destination.exists():
        raise FileExistsError('preserve previous verifier receipt')
    started = time.monotonic()
    def expired():
        return time.monotonic() - started >= 850
    rows, z, features, cache = run.load_data(reg)
    stats = run.normalization_stats(z, features, cache)
    if stats != json.loads((run.WORK / 'NORMALIZATION_STATS.json').read_text()) or sha(run.WORK / 'NORMALIZATION_STATS.json') != status['normalization_stats_sha256']:
        raise ValueError('fit-only stats changed')
    initial = {kind: run.initialization_pair(kind, stats) for kind in ['target', 'gap']}
    if initial != json.loads((run.WORK / 'INITIALIZATION_PAIR.json').read_text()):
        raise ValueError('initial state/RNG pair does not reproduce')
    arms = ['baseline'] if stage == 'baseline' else list(run.ARMS)
    proofs = []; summaries = {}; inference = {}
    for arm in arms:
        proofs.append(verify_arm(run.WORK / arm, arm, z, stats, status['identities'][arm]))
        summary = prior.summarize_arm(run.WORK / arm, rows, z, 50)
        if summary != json.loads((run.WORK / (arm + '_SUMMARY.json')).read_text()):
            raise ValueError('summary/curve changed')
        independently_replay_curve(run.WORK / arm, summary)
        summaries[arm] = summary
        inference[arm] = fresh_endpoint(run.WORK / arm, arm, z, stats, features, expired)
    proof = {'status': 'BASELINE_REPRODUCTION_VERIFIED' if stage == 'baseline' else 'PAIRED50_STATES_METRICS_VERIFIED',
             'decision': run.DECISION, 'registration_sha256': sha(registration),
             'states_verified': sum(p['states_verified'] for p in proofs), 'arm_verifications': proofs,
             'fresh_CPU_fit_held': inference, 'fit_only_stats_recomputed': True,
             'paired_initialization_exact': True, 'training_performed': False, 'GPU_used': False,
             'scope': 'one paired seed/reused TRAIN exploratory/fixed50/fixed.5'}
    if stage == 'baseline':
        proof['historical_normalized50_parity'] = historical_parity(z)
    else:
        base_path = run.REPORTS / 'BASELINE_VERIFICATION.json'
        base = json.loads(base_path.read_text())
        if base['status'] != 'BASELINE_REPRODUCTION_VERIFIED' or base['registration_sha256'] != sha(registration):
            raise ValueError('candidate lacked verified reproduction control')
        for check in base['arm_verifications']:
            for path, digest in check['file_sha256'].items():
                if sha(path) != digest:
                    raise ValueError('verified baseline overwritten')
        # Compare each epoch's NumPy shuffle state and updates: same sample order.
        for kind in ['target', 'gap']:
            for epoch in range(1, 51):
                a = torch.load(run.WORK / 'baseline' / f'{kind}_epoch{epoch:02d}.pt', map_location='cpu', weights_only=True)
                b = torch.load(run.WORK / 'dropout2x' / f'{kind}_epoch{epoch:02d}.pt', map_location='cpu', weights_only=True)
                engine.assert_equal_state(a['rng']['numpy'], b['rng']['numpy'])
                engine.assert_equal_state(a['rng']['python'], b['rng']['python'])
                if a['updates'] != b['updates']:
                    raise ValueError('paired update budget differs')
        comparison = {'decision': run.DECISION, 'registration_sha256': sha(registration), 'arms': summaries,
                      'quality_gate': prior.gate_decision(summaries['baseline'], summaries['dropout2x']),
                      'uncertainty': prior.paired_uncertainty(summaries['baseline'], summaries['dropout2x'], rows)}
        for path in [run.WORK / 'COMPARISON.json', run.REPORTS / 'COMPARISON.json']:
            if json.loads(path.read_text()) != comparison:
                raise ValueError('comparison/gate does not reproduce')
        proof.update(baseline_verification_sha256=sha(base_path), quality_gate=comparison['quality_gate'],
                     endpoints={arm: summaries[arm]['endpoint'] for arm in run.ARMS},
                     last10_mean_F1={arm: summaries[arm]['last10_mean_F1'] for arm in run.ARMS},
                     matched_epoch_numpy_shuffle_RNG=True, uncertainty=comparison['uncertainty'])
        plot_comparison(summaries)
    for check in proofs:
        for path, digest in check['file_sha256'].items():
            if sha(path) != digest:
                raise ValueError('artifact changed during verification')
    proof['seconds'] = time.monotonic() - started
    engine.dump(destination, proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--stage', choices=['baseline', 'paired'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    try:
        result = verify(args.registration, args.stage)
        print(json.dumps({'status': result['status'], 'states_verified': result['states_verified']}))
    except BaseException as exc:
        engine.dump(run.REPORTS / (args.stage.upper() + '_VERIFICATION_FAILED.json'), {'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
