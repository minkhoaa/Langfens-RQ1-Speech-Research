#!/usr/bin/env python3
"""RQ1-DIAG-02B: bounded CPU inference; no training, no TEST member reads.

Gold intervals are OFFLINE diagnostic/intervention windows, never model inputs.
Run from repository root: PYTHONPATH=.:src HF_HUB_OFFLINE=1 .venv/bin/python
  audit_scripts/diagnose_counterfactual_validity.py --outdir <new-directory>
Existing output directories are refused to preserve evidence.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.stats import rankdata
import torch

from audit_scripts.diagnose_temporal_alignment_validation import (
    FOLD_CHECKPOINTS, FOLD_ZIPS, average_precision_binary as ap,
    compute_sha256 as sha256, find_expected_omission_span, student_t_ci,
)
from audit_scripts.diagnose_omissions_and_false_negatives import extract_word_positions
from langfens_prep.audio_loader import get_l2arctic_reader, load_l2arctic_audio_from_zip
from langfens_prep.model import LangfensWav2Vec2MDD
from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
from langfens_prep.textgrid_parser import parse_gold_textgrid_content

PRED = 'manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz'
SPLIT = 'manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json'
SPLIT_SHA = '7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b'
PRED_SHA = '355b6e23bcb172d20b99ba3f9f4e07dc15453846ac312eca847471ad476e5892'


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save_json(path, obj):
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def allowed_record(u, folds):
    """Fail closed before any annotation/audio read. No TEST partition opened."""
    fold = folds[u['fold']]
    key = f"{u['speaker']}/{u['utterance_id']}"
    if key not in fold['val']['utterances'] or u['speaker'] not in fold['val']['speakers']:
        raise ValueError(f'Non-validation record refused: fold {u["fold"]} {key}')
    if u['speaker'] in fold['test']['speakers']:
        raise ValueError('Active fold TEST speaker refused')


def frame_window(start, end, length):
    a = min(length - 1, max(0, int(start / .02)))
    b = min(length, max(a + 1, int(np.ceil(end / .02))))
    return a, b


def shifted_window(a, b, length):
    """Adjacent same-width window, clipped without changing width."""
    k = b - a
    start = b if b + k <= length else max(0, a - k)
    return start, start + k


def random_window(a, b, length, rng):
    """Same width; uniform among NONOVERLAPPING placements when possible."""
    k = b - a
    starts = [i for i in range(length - k + 1) if i + k <= a or i >= b]
    if not starts:
        starts = list(range(length - k + 1))
    start = int(rng.choice(starts))
    return start, start + k


def head_forward(model, features, indices, padding, windows=None):
    """Use ORIGINAL MHA: learned per-head Q/K/V, softmax, concat and out_proj.

    windows is one allowed [a,b) per query. Padding is always preserved.
    No averaged-weight reconstruction or forced uniform attention is used.
    Returns logits for ALL tokens; callers apply original active-token mask.
    """
    if model.training:
        raise ValueError('Diagnostics require eval mode')
    ca = model.cross_attention_module
    embeds = ca.canonical_phone_embedding(indices)
    pos = torch.arange(indices.shape[1], device=indices.device).clamp(max=ca.max_canonical_len - 1)
    q = ca.canonical_pos_embedding(pos).unsqueeze(0)
    if not ca.audio_only_queries:
        q = q + ca.phone_projection(embeds)
    q = q.expand(indices.shape[0], -1, -1)
    mask = None
    if windows is not None:
        mask = torch.ones(indices.shape[1], features.shape[1], dtype=torch.bool, device=features.device)
        for i, (a, b) in enumerate(windows):
            mask[i, a:b] = False
        if (mask.unsqueeze(0) | padding.unsqueeze(1)).all(-1).any():
            raise ValueError('Restriction would mask all active frames')
    out, weights = ca.cross_attention(q, features, features, key_padding_mask=padding,
                                     attn_mask=mask, average_attn_weights=False)
    aligned = ca.layer_norm(ca.dropout(out))
    logits = model.binary_head(torch.cat([aligned, embeds], -1)).squeeze(-1)
    return logits, weights


def metric(y, scores):
    y, s = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    auc = ((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
           if n1 and n0 else float('nan'))
    return {'n': len(y), 'positives': n1, 'prevalence': float(y.mean()) if len(y) else None,
            'ap': ap(y, s) if len(y) else None, 'roc_auc': auc}


def diagnostic_threshold(negative, ceiling=.15):
    """Conservative tie-aware in-sample diagnostic threshold, NOT calibration."""
    s = np.asarray(negative)
    if len(s) == 0:
        return float('nan')
    candidates = np.r_[np.unique(s), np.nextafter(s.max(), np.inf)]
    return float(next(t for t in candidates if np.mean(s >= t) <= ceiling))


def operating(y, s, threshold):
    y, s = np.asarray(y), np.asarray(s)
    tpr = float(np.mean(s[y == 1] >= threshold)) if (y == 1).any() else float('nan')
    return {'threshold': float(threshold), 'tpr': tpr, 'fnr': 1 - tpr,
            'fpr': float(np.mean(s[y == 0] >= threshold)) if (y == 0).any() else None}


def cluster_ci(rows, statistic, reps, seed=1337):
    groups = defaultdict(list)
    for r in rows:
        groups[r['speaker']].append(r)
    speakers = sorted(groups)
    if len(speakers) < 2:
        return {'clusters': len(speakers), 'ci95': None}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        sample = [r for spk in rng.choice(speakers, len(speakers)) for r in groups[spk]]
        vals.append(statistic(sample))
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    return {'clusters': len(speakers), 'resamples': reps, 'valid_resamples': len(vals),
            'ci95': np.quantile(vals, [.025, .975]).tolist() if len(vals) else None}


def matched_pairs(rows, caliper=.020):
    """1:1 nearest-duration WITHOUT replacement; exact speaker/phone/position.

    Unknown/unmapped word positions excluded; deterministic lexical tie break.
    No fallback silently changes the estimand. Selection ignores predictions.
    """
    correct = [r for r in rows if r['error_type'] == 'correct']
    used, pairs, unmatched = set(), [], []
    for d in sorted((r for r in rows if r['error_type'] == 'deletion'), key=lambda r: r['token_id']):
        options = [c for c in correct if c['token_id'] not in used
                   and d['word_position'] != 'unknown'
                   and all(c[k] == d[k] for k in ('speaker', 'phone', 'word_position'))
                   and abs(c['duration_sec'] - d['duration_sec']) <= caliper + 1e-12]
        if not options:
            unmatched.append(d['token_id'])
            continue
        c = min(options, key=lambda c: (abs(c['duration_sec'] - d['duration_sec']), c['token_id']))
        used.add(c['token_id'])
        pairs.append({'pair_id': len(pairs), 'speaker': d['speaker'], 'omission': d['token_id'],
                      'control': c['token_id'], 'phone': d['phone'], 'word_position': d['word_position'],
                      'omission_duration': d['duration_sec'], 'control_duration': c['duration_sec'],
                      'duration_difference': abs(c['duration_sec'] - d['duration_sec'])})
    return pairs, unmatched


def ctc_summary(rows, pairs, reps):
    by_id = {r['token_id']: r for r in rows}
    matched = [by_id[p[k]] for p in pairs for k in ('omission', 'control')]
    result = {}
    scores = ['blank_mean', 'missing_canonical_mean', 'missing_canonical_peak',
              'blank_canonical_margin', 'other_canonical_margin', 'missing_emission_fraction',
              'base_prob', 'negative_duration']
    for field in scores:
        if not matched:
            result[field] = {'status': 'NO_MATCHED_CONTROLS'}
            continue
        y = [r['error_type'] == 'deletion' for r in matched]
        s = [r[field] for r in matched]
        threshold = diagnostic_threshold([r[field] for r in matched if r['error_type'] == 'correct'])
        def stat(rs, key):
            return metric([r['error_type'] == 'deletion' for r in rs], [r[field] for r in rs])[key]
        result[field] = {**metric(y, s), 'diagnostic_fpr15': operating(y, s, threshold),
                         'ap_cluster_bootstrap': cluster_ci(matched, lambda rs: stat(rs, 'ap'), reps),
                         'auc_cluster_bootstrap': cluster_ci(matched, lambda rs: stat(rs, 'roc_auc'), reps),
                         'tpr_cluster_bootstrap': cluster_ci(matched, lambda rs: operating(
                             [r['error_type'] == 'deletion' for r in rs], [r[field] for r in rs], threshold)['tpr'], reps),
                         'per_speaker': {spk: metric([r['error_type'] == 'deletion' for r in matched if r['speaker'] == spk],
                                                   [r[field] for r in matched if r['speaker'] == spk])
                                         for spk in sorted({r['speaker'] for r in matched})}}
    return result


def intervention_summary(rows, reps):
    result, speaker_rows = {}, []
    for mode in sorted({r['mode'] for r in rows}):
        rs = [r for r in rows if r['mode'] == mode]
        for subset in ('all_errors', 'omission_vs_correct'):
            sub = rs if subset == 'all_errors' else [r for r in rs if r['error_type'] != 'substitution']
            y = [r['error_type'] != 'correct' for r in sub]
            base = metric(y, [r['base_prob'] for r in sub])
            changed = metric(y, [r['prob'] for r in sub])
            def delta_ap(items):
                yy = [r['error_type'] != 'correct' for r in items]
                return metric(yy, [r['prob'] for r in items])['ap'] - metric(yy, [r['base_prob'] for r in items])['ap']
            def recall_delta(items):
                ds = [r for r in items if r['error_type'] == 'deletion']
                return np.mean([int(r['prob'] >= .2) - int(r['base_prob'] >= .2) for r in ds]) if ds else float('nan')
            per_spk = []
            for spk in sorted({r['speaker'] for r in sub}):
                ss = [r for r in sub if r['speaker'] == spk]
                per_spk.append({'mode': mode, 'subset': subset, 'speaker': spk, 'n': len(ss),
                                'delta_ap': delta_ap(ss), 'mean_logit_delta': np.mean([r['logit'] - r['base_logit'] for r in ss]),
                                'mean_prob_delta': np.mean([r['prob'] - r['base_prob'] for r in ss]),
                                'mean_abs_prob_delta': np.mean([abs(r['prob'] - r['base_prob']) for r in ss]),
                                'omission_recall_delta_at020': recall_delta(ss)})
            speaker_rows.extend(per_spk)
            result[f'{mode}/{subset}'] = {
                'baseline': base, 'intervened': changed, 'paired_delta_ap': delta_ap(sub),
                'delta_ap_cluster_bootstrap': cluster_ci(sub, delta_ap, reps),
                'mean_logit_delta': np.mean([r['logit'] - r['base_logit'] for r in sub]),
                'mean_prob_delta': np.mean([r['prob'] - r['base_prob'] for r in sub]),
                'mean_abs_prob_delta': np.mean([abs(r['prob'] - r['base_prob']) for r in sub]),
                'omission_recall_delta_at020': recall_delta(sub),
                'recall_delta_cluster_bootstrap': cluster_ci(sub, recall_delta, reps),
                'speaker_macro_probability_delta': student_t_ci([r['mean_prob_delta'] for r in per_spk]),
                'speaker_macro_abs_probability_delta': student_t_ci([r['mean_abs_prob_delta'] for r in per_spk]),
                'omission_operating_points': {str(t): {'base': operating(
                    [r['error_type'] == 'deletion' for r in sub if r['error_type'] != 'substitution'],
                    [r['base_prob'] for r in sub if r['error_type'] != 'substitution'], t), 'intervened': operating(
                    [r['error_type'] == 'deletion' for r in sub if r['error_type'] != 'substitution'],
                    [r['prob'] for r in sub if r['error_type'] != 'substitution'], t)} for t in (.2, .5)}}
    return result, speaker_rows


def mass_summary(rows):
    result = {}
    for et in ('correct', 'substitution', 'deletion'):
        for head in sorted({r['head'] for r in rows}):
            sub = [r for r in rows if r['error_type'] == et and r['head'] == head]
            if not sub:
                continue
            obs, uni, rand = [np.array([r[k] for r in sub]) for k in ('mass', 'uniform_mass', 'random_mass')]
            result[f'{et}/head{head}'] = {
                'n': len(sub), 'mass_mean': obs.mean(), 'uniform_mean': uni.mean(), 'random_mean': rand.mean(),
                'mean_token_uniform_enrichment': np.mean(obs / uni),
                'ratio_of_sums_uniform': obs.sum() / uni.sum(),
                'ratio_of_sums_random': obs.sum() / rand.sum() if rand.sum() > 1e-8 else None,
                'mass_minus_random_mean': (obs - rand).mean(),
                'entropy_mean': np.mean([r['entropy'] for r in sub]),
                'offset_mean_sec': np.mean([r['offset_sec'] for r in sub]),
                'max_weight_mean': np.mean([r['max_weight'] for r in sub]),
                'speaker_macro_mass_difference': student_t_ci([
                    np.mean([r['mass'] - r['random_mass'] for r in sub if r['speaker'] == spk])
                    for spk in sorted({r['speaker'] for r in sub})])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--samples-per-speaker', type=int, default=10)
    parser.add_argument('--waveform-pairs-per-speaker', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--continue-on-archive-drift', action='store_true',
                        help='Explicitly collect non-confirmatory diagnostics after archive parity failures; never report PASS.')
    args = parser.parse_args()
    if not 1 <= args.samples_per_speaker <= 10 or not 0 <= args.waveform_pairs_per_speaker <= 2:
        parser.error('Bounded inference: 1..10 utterances and 0..2 waveform pairs per speaker')
    if not 1 <= args.threads <= 8 or not 100 <= args.bootstrap <= 2000:
        parser.error('Use 1..8 CPU threads and 100..2000 bootstrap resamples')
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    save_json(outdir / 'RUN_STATE.json', {'status': 'RUNNING', 'args': vars(args),
                                         'restart': 'Use a NEW output directory; never overwrite evidence.'})
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    assert sha256(SPLIT) == SPLIT_SHA
    assert sha256(PRED) == PRED_SHA
    folds = {f['fold_index']: f for f in json.loads(Path(SPLIT).read_text())['folds']}
    with gzip.open(PRED, 'rt') as f:
        pred = json.load(f)
    # Validate entire prediction index BEFORE opening any annotation archive member.
    for u in pred['utterances']:
        allowed_record(u, folds)
    all_rows, annotations = [], {}
    for u in pred['utterances']:
        fold, spk, uid = u['fold'], u['speaker'], u['utterance_id']
        ann = parse_gold_textgrid_content(get_l2arctic_reader(FOLD_ZIPS[fold]).read_annotation_text(spk, uid), uid)
        ivs = ann.canonical_intervals
        assert len(ivs) == len(u['labels']) == len(u['canonical_phones'])
        assert [iv.canonical_phone for iv in ivs] == u['canonical_phones']
        assert [int(iv.error_type != 'correct') for iv in ivs] == u['labels']
        positions = extract_word_positions(ivs, ann.words_intervals)
        annotations[(fold, spk, uid)] = ann
        for idx, iv in enumerate(ivs):
            center = (iv.xmin + iv.xmax) / 2
            mapped = any(w['xmin'] <= center <= w['xmax'] for w in ann.words_intervals)
            all_rows.append({'token_id': f'{fold}/{spk}/{uid}/{idx}', 'fold': fold, 'speaker': spk,
                             'utterance_id': uid, 'token_idx': idx, 'phone': iv.canonical_phone,
                             'error_type': iv.error_type, 'duration_sec': iv.xmax - iv.xmin,
                             'xmin': iv.xmin, 'xmax': iv.xmax,
                             'word_position': positions[idx] if mapped else 'unknown',
                             'archived_prob': u['matched_scores'][idx], 'mismatched_prob': u['mismatched_scores'][idx]})
    census = Counter(r['error_type'] for r in all_rows)
    assert census == {'correct': 35363, 'substitution': 3584, 'deletion': 629}
    reproduction = {}
    for et in ('all', 'substitution', 'deletion'):
        rs = all_rows if et == 'all' else [r for r in all_rows if r['error_type'] in ('correct', et)]
        y = [r['error_type'] != 'correct' for r in rs]
        reproduction[et] = {**metric(y, [r['archived_prob'] for r in rs]),
                            'acoustic_gap': ap(y, [r['archived_prob'] for r in rs]) - ap(y, [r['mismatched_prob'] for r in rs])}
    omission_rows = [r for r in all_rows if r['error_type'] == 'deletion']
    raw_short = [r for r in omission_rows if r['duration_sec'] < .04]
    rounded_short = [r for r in omission_rows if round(r['duration_sec'], 4) < .04]
    duration_disagreements = [r for r in raw_short if r not in rounded_short]
    save_csv(outdir / 'duration_boundary_disagreements.csv', duration_disagreements)
    duration_audit = {'raw_float_lt040': len(raw_short), 'rounded_4decimal_lt040': len(rounded_short),
                      'decimal_tick_lt040': sum(round(r['xmax'] * 10000) - round(r['xmin'] * 10000) < 400 for r in omission_rows),
                      'raw_float_lt080': sum(r['duration_sec'] < .08 for r in omission_rows),
                      'rounded_4decimal_lt080': sum(round(r['duration_sec'], 4) < .08 for r in omission_rows),
                      'interpretation': 'Raw subtraction at decimal boundaries is floating-point-sensitive; retain both conventions.'}
    # Exactly reproduce DIAG-02 selection: first ten omission-containing utterances per speaker, archive order.
    selected, counts = [], Counter()
    for u in pred['utterances']:
        ann = annotations[(u['fold'], u['speaker'], u['utterance_id'])]
        if any(iv.error_type == 'deletion' for iv in ann.canonical_intervals) and counts[u['speaker']] < args.samples_per_speaker:
            selected.append(u)
            counts[u['speaker']] += 1
    selected_keys = {(u['fold'], u['speaker'], u['utterance_id']) for u in selected}
    selected_rows = [r for r in all_rows if (r['fold'], r['speaker'], r['utterance_id']) in selected_keys]
    pairs, unmatched = matched_pairs(selected_rows)
    save_csv(outdir / 'matched_pairs.csv', pairs)
    # Waveform subset fixed from matched IDs BEFORE any inference, at most two pairs/speaker.
    wave_ids, wave_counts = set(), Counter()
    for p in pairs:
        if wave_counts[p['speaker']] < args.waveform_pairs_per_speaker:
            wave_ids.update([p['omission'], p['control']])
            wave_counts[p['speaker']] += 1
    token_records, head_records, interventions, legacy, parity, provenance = [], [], [], [], [], []
    by_utt = defaultdict(list)
    for r in selected_rows:
        by_utt[(r['fold'], r['speaker'], r['utterance_id'])].append(r)
    with torch.inference_mode():
        for fold in range(1, 5):
            ckpt_path = FOLD_CHECKPOINTS[fold]
            digest = sha256(ckpt_path)
            assert digest == pred['metadata']['checkpoint_sha256'][f'fold_{fold}']['fixed_budget']
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            cfg = ckpt['metadata']['config']
            assert cfg['fold'] == fold and cfg['pretrained_revision'] == '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'
            assert cfg['backbone_frozen'] and cfg['loss_weights'] == {'ctc': 1.0, 'binary': 1.0}
            assert ckpt['successful_updates'] == 3600
            model = LangfensWav2Vec2MDD(pretrained_model_name_or_path=cfg['pretrained_model'], revision=cfg['pretrained_revision'])
            model.load_state_dict(ckpt['model'], strict=True)
            model.eval().requires_grad_(False)
            provenance.append({'fold': fold, 'checkpoint': ckpt_path, 'sha256': digest,
                               'successful_updates': ckpt['successful_updates'], 'epoch': ckpt['epoch'],
                               'skipped_updates': ckpt.get('skipped_amp_attempts_total'),
                               'config': cfg, 'backbone_config': model.wav2vec2.config.to_dict(),
                               'strict_state_dict': True, 'cross_attention_heads': model.cross_attention_module.cross_attention.num_heads})
            del ckpt
            for u in (u for u in selected if u['fold'] == fold):
                spk, uid = u['speaker'], u['utterance_id']
                allowed_record(u, folds)
                wav, sr, _ = load_l2arctic_audio_from_zip(FOLD_ZIPS[fold], spk, uid)
                assert sr == 16000
                indices = torch.tensor([[PHONE_TO_CTC_INDEX[p] for p in u['canonical_phones']]])
                audio = wav.unsqueeze(0)
                kwargs = {'audio_lengths': torch.tensor([wav.numel()]), 'canonical_phone_indices': indices,
                          'canonical_phone_lengths': torch.tensor([indices.shape[1]])}
                output = model(waveforms=audio, **kwargs)
                repeated = model(waveforms=audio, **kwargs)
                assert torch.equal(output.binary_logits, repeated.binary_logits)
                hs = output.frame_features
                active = int(output.input_lengths[0])
                padding = torch.arange(hs.shape[1])[None] >= output.input_lengths[:, None]
                logits, heads = head_forward(model, hs, indices, padding)
                open_logits, _ = head_forward(model, hs, indices, padding, [(0, active)] * indices.shape[1])
                torch.testing.assert_close(logits, output.binary_logits, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(open_logits, output.binary_logits, atol=2e-5, rtol=2e-5)
                archive_diff = float(np.max(np.abs(output.binary_logits.sigmoid()[0].numpy() - u['matched_scores'])))
                archive_pass = archive_diff < 2e-4
                if not archive_pass:
                    save_json(outdir / 'ARCHIVE_DRIFT_NOTICE.json', {
                        'status': 'ARCHIVE_PARITY_FAILED', 'fold': fold, 'speaker': spk, 'utterance_id': uid,
                        'max_difference': archive_diff, 'unchanged_tolerance': 2e-4,
                        'continuation_explicit': args.continue_on_archive_drift})
                    if not args.continue_on_archive_drift:
                        raise AssertionError(f'Archived prediction parity failed: {archive_diff}')
                parity.append({'fold': fold, 'speaker': spk, 'utterance_id': uid, 'active_frames': active,
                               'active_tokens': indices.shape[1], 'head_vs_forward_logit_max': float((logits-output.binary_logits).abs().max()),
                               'open_mask_logit_max': float((open_logits-output.binary_logits).abs().max()),
                               'archive_probability_max': archive_diff, 'archive_parity_pass': archive_pass, 'repeat_exact': True})
                rows = by_utt[(fold, spk, uid)]
                windows = [frame_window(r['xmin'], r['xmax'], active) for r in rows]
                neighbors = [shifted_window(a, b, active) for a, b in windows]
                randoms = [random_window(a, b, active, rng) for a, b in windows]
                # Historical backbone shortcut replay, NOT used for valid interventions.
                old_hs = model.wav2vec2(audio).last_hidden_state
                old_pad = torch.zeros(1, old_hs.shape[1], dtype=torch.bool)
                old_logits, old_heads = head_forward(model, old_hs, indices, old_pad)
                old_avg = old_heads.mean(1)
                # Historical invalid path deliberately reproduced and labeled, never called valid MHA.
                raw_local = torch.stack([old_hs[0, a:b].mean(0) for a, b in windows])[None]
                ca = model.cross_attention_module
                old_forced = model.binary_head(torch.cat([ca.layer_norm(raw_local), ca.canonical_phone_embedding(indices)], -1)).squeeze(-1)
                def record(r, mode, val, a, b, unit):
                    idx = r['token_idx']
                    interventions.append({**r, 'mode': mode, 'window_start': a, 'window_end': b, 'window_unit': unit,
                                          'base_logit': float(logits[0, idx]), 'base_prob': float(logits[0, idx].sigmoid()),
                                          'logit': float(val), 'prob': float(val.sigmoid())})
                for mode, ws in [('attention_local', windows), ('attention_neighbor', neighbors), ('attention_random', randoms)]:
                    ls, _ = head_forward(model, hs, indices, padding, ws)
                    for r, (a, b) in zip(rows, ws):
                        record(r, mode, ls[0, r['token_idx']], a, b, 'frame')
                ctc = output.ctc_logits[0, :active].softmax(-1).numpy()
                head_np = heads[0, :, :, :active].numpy()
                ann = annotations[(fold, spk, uid)]
                for r, (a, b), (ra, rb) in zip(rows, windows, randoms):
                    idx = r['token_idx']
                    pc = ctc[a:b, PHONE_TO_CTC_INDEX[r['phone']]]
                    other = ctc[a:b].copy()
                    other[:, PHONE_TO_CTC_INDEX[r['phone']]] = 0
                    exp_a, exp_b, _ = find_expected_omission_span(ann.canonical_intervals, idx, ann.total_duration)
                    ea, eb = frame_window(exp_a, exp_b, active)
                    tr = {**r, 'base_logit': float(logits[0, idx]), 'base_prob': float(logits[0, idx].sigmoid()),
                          'active_frames': active, 'frame_start': a, 'frame_end': b,
                          'blank_mean': float(ctc[a:b, 0].mean()), 'canonical_mean': float(pc.mean()),
                          'negative_duration': -r['duration_sec'],
                          'canonical_peak': float(pc.max()), 'missing_canonical_mean': float(1-pc.mean()),
                          'missing_canonical_peak': float(1-pc.max()),
                          'blank_canonical_margin': float((ctc[a:b, 0]-pc).mean()),
                          'other_canonical_margin': float((other.max(-1)-pc).mean()),
                          'missing_emission_fraction': float(1-np.mean(ctc[a:b].argmax(-1) == PHONE_TO_CTC_INDEX[r['phone']])),
                          'canonical_emission_count': int(np.sum(ctc[a:b].argmax(-1) == PHONE_TO_CTC_INDEX[r['phone']])),
                          'blank_emission_fraction': float(np.mean(ctc[a:b].argmax(-1) == 0)),
                          'neighbor_canonical_peak': float(ctc[ea:eb, PHONE_TO_CTC_INDEX[r['phone']]].max())}
                    token_records.append(tr)
                    for h in range(head_np.shape[0]):
                        dist = head_np[h, idx]
                        k = b-a
                        starts = rng.integers(0, active-k+1, 100)  # inclusive last legal placement
                        rm = np.mean([dist[s:s+k].sum() for s in starts])
                        head_records.append({**r, 'head': h, 'mass': float(dist[a:b].sum()),
                                             'uniform_mass': k/active, 'random_mass': float(rm),
                                             'entropy': float(-np.sum(dist * np.log(np.maximum(dist, 1e-30))) / np.log(active)),
                                             'offset_sec': abs(int(dist.argmax())*.02-(r['xmin']+r['xmax'])/2),
                                             'peak_frame': int(dist.argmax()), 'max_weight': float(dist.max())})
                    # Per-token feature erasure changes K AND V but never re-encodes audio.
                    for mode, (ma, mb) in [('feature_local', (a, b)), ('feature_random', (ra, rb))]:
                        masked = hs.clone()
                        masked[:, ma:mb] = 0
                        ls, _ = head_forward(model, masked, indices, padding)
                        record(r, mode, ls[0, idx], ma, mb, 'frame')
                    old_masked = old_hs.clone()
                    old_masked[:, a:b] = 0
                    old_ls, _ = head_forward(model, old_masked, indices, old_pad)
                    legacy.append({**r, 'original_prob': float(old_logits[0, idx].sigmoid()),
                                   'mask_abs_delta': float((old_ls[0, idx].sigmoid()-old_logits[0, idx].sigmoid()).abs()),
                                   'invalid_forced_delta': float(old_forced[0, idx].sigmoid()-old_logits[0, idx].sigmoid()),
                                   'old_mass': float(old_avg[0, idx, a:b].sum()), 'uniform_mass': (b-a)/active,
                                   'old_backbone_vs_forward_max': float((old_hs-hs).abs().max())})
                    if r['token_id'] in wave_ids:
                        wa = min(wav.numel()-1, max(0, round(r['xmin']*sr)))
                        wb = min(wav.numel(), max(wa+1, round(r['xmax']*sr)))
                        wr_a, wr_b = random_window(wa, wb, wav.numel(), rng)
                        for mode, (ma, mb) in [('waveform_local', (wa, wb)), ('waveform_random', (wr_a, wr_b))]:
                            masked = audio.clone()
                            masked[:, ma:mb] = 0
                            val = model(waveforms=masked, **kwargs).binary_logits[0, idx]
                            record(r, mode, val, ma, mb, 'sample')
                print(f'fold {fold} {spk}/{uid}: parity max={archive_diff:.2g}, tokens={len(rows)}', flush=True)
            del model
    for name, rows in [('tokens', token_records), ('heads', head_records), ('interventions', interventions),
                       ('legacy_replay', legacy), ('parity', parity)]:
        save_csv(outdir / (f'{name}.csv' if name == 'parity' else f'{name}.csv.gz'), rows)
    intervention_report, speaker_rows = intervention_summary(interventions, args.bootstrap)
    save_csv(outdir / 'speaker_interventions.csv', speaker_rows)
    legacy_summary = {}
    for et in ('correct', 'substitution', 'deletion', 'all'):
        rs = legacy if et == 'all' else [r for r in legacy if r['error_type'] == et]
        legacy_summary[et] = {'n': len(rs), 'mask_abs_delta': np.mean([r['mask_abs_delta'] for r in rs]),
                              'invalid_forced_delta': np.mean([r['invalid_forced_delta'] for r in rs]),
                              'mean_token_uniform_enrichment': np.mean([r['old_mass']/r['uniform_mass'] for r in rs]),
                              'speaker_macro_mask': student_t_ci([np.mean([r['mask_abs_delta'] for r in rs if r['speaker'] == s])
                                                                 for s in sorted({r['speaker'] for r in rs})])}
    # Paired local-minus-random effects, never independent unpaired resamples.
    contrasts = {}
    lookup = {(r['mode'], r['token_id']): r for r in interventions}
    for family in ('attention', 'feature', 'waveform'):
        rs = []
        for r in interventions:
            if r['mode'] == family + '_local':
                control = lookup[(family + '_random', r['token_id'])]
                rs.append({'speaker': r['speaker'], 'error_type': r['error_type'],
                           'base_prob': r['base_prob'], 'local_prob': r['prob'], 'random_prob': control['prob'],
                           'delta': r['prob']-control['prob'],
                           'abs_sensitivity_difference': abs(r['prob']-r['base_prob'])-abs(control['prob']-control['base_prob'])})
        contrasts[family] = {'n': len(rs), 'mean_local_minus_random_prob': np.mean([r['delta'] for r in rs]),
                             'paired_cluster_ci': cluster_ci(rs, lambda ss: np.mean([r['delta'] for r in ss]), args.bootstrap),
                             'mean_abs_sensitivity_difference': np.mean([r['abs_sensitivity_difference'] for r in rs]),
                             'paired_abs_cluster_ci': cluster_ci(rs, lambda ss: np.mean([r['abs_sensitivity_difference'] for r in ss]), args.bootstrap),
                             'by_type': {et: {'n': sum(r['error_type'] == et for r in rs),
                                              'mean_local_minus_random': np.mean([r['delta'] for r in rs if r['error_type'] == et]),
                                              'paired_cluster_ci': cluster_ci([r for r in rs if r['error_type'] == et],
                                                  lambda ss: np.mean([r['delta'] for r in ss]), args.bootstrap)}
                                         for et in sorted({r['error_type'] for r in rs})}}
        def ap_contrast(items):
            yy = [r['error_type'] != 'correct' for r in items]
            return ap(yy, [r['local_prob'] for r in items]) - ap(yy, [r['random_prob'] for r in items])
        contrasts[family]['paired_local_minus_random_ap'] = ap_contrast(rs)
        contrasts[family]['paired_ap_cluster_ci'] = cluster_ci(rs, ap_contrast, args.bootstrap)
    archive_failures = [r for r in parity if not r['archive_parity_pass']]
    report = {'status': 'PARTIAL_ARCHIVE_PARITY_FAILED' if archive_failures else 'EXECUTED_BOUNDED_DIAGNOSTIC',
              'decision': 'NO_GO_ALIGNMENT_TRAINING_CAUSAL_GATE',
              'metadata': {'base': 'dd2dee9a2063284d2a0f9fbf1c18315f3a7cd774',
                           'runtime_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                           'script_sha256': sha256(__file__), 'args': vars(args), 'python': platform.python_version(),
                           'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
                           'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                           'cpu_threads': torch.get_num_threads(),
                           'mkl_cbwr': os.environ.get('MKL_CBWR'),
                           'authorized_package_sha256': {k: sha256(v) for k, v in FOLD_ZIPS.items()},
                           'seed': args.seed, 'elapsed_seconds': time.time()-started,
                           'prediction_sha256': sha256(PRED), 'split_sha256': sha256(SPLIT)},
              'scope': {'validation_utterances': len(pred['utterances']), 'census': census,
                        'sampled_utterances': len(selected), 'sampled_tokens': len(token_records),
                        'speaker_utterances': counts, 'waveform_tokens': len(wave_ids),
                        'ctc_matching': {'caliper_sec': .020, 'exact': ['speaker', 'phone', 'word_position'],
                                         'without_replacement': True, 'pairs': len(pairs), 'unmatched_omissions': unmatched}},
              'checkpoint_provenance': provenance, 'archive_metrics_reproduction': reproduction,
              'duration_reconciliation': duration_audit, 'legacy_replay_summary': legacy_summary,
              'parity': {'utterances': len(parity), 'repeat_exact': all(r['repeat_exact'] for r in parity),
                         'max_archive_probability_difference': max(r['archive_probability_max'] for r in parity),
                         'max_forward_logit_difference': max(r['head_vs_forward_logit_max'] for r in parity),
                         'max_open_mask_logit_difference': max(r['open_mask_logit_max'] for r in parity),
                         'archive_tolerance': 2e-4, 'archive_failures': archive_failures, 'logit_atol_rtol': 2e-5,
                         'sample_archive_metrics': metric([r['error_type'] != 'correct' for r in token_records],
                                                          [r['archived_prob'] for r in token_records]),
                         'sample_cpu_metrics': metric([r['error_type'] != 'correct' for r in token_records],
                                                      [r['base_prob'] for r in token_records]),
                         'full_1199_forward_replay': 'NOT_EXECUTED; bounded sample only'},
              'ctc_matched_discrimination': ctc_summary(token_records, pairs, args.bootstrap),
              'ctc_matching_10ms_sensitivity': {
                  'pairs': len(matched_pairs(token_records, .010)[0]),
                  'metrics': ctc_summary(token_records, matched_pairs(token_records, .010)[0], args.bootstrap)},
              'waveform_common_token_feature_metrics': intervention_summary(
                  [r for r in interventions if r['token_id'] in wave_ids and r['mode'].startswith('feature_')], args.bootstrap)[0],
              'attention_by_head_and_type': mass_summary(head_records),
              'intervention_metrics': intervention_report, 'paired_local_random': contrasts,
              'limitations': ['Only 8 validation speakers; percentile cluster bootstrap is exploratory and unstable at G=8.',
                             'Omission-enriched first-10 sample; not population prevalence or deployment AP.',
                             'Matched-case-control AP has 50% prevalence; not comparable to census AP.',
                             'Diagnostic FPR15 thresholds selected on same controls; not deployment calibration.',
                             'Waveform subset at most two matched pairs per speaker; low precision.',
                             'Gold omission windows are annotation conventions, not observable missing-phone acoustics.',
                             'Zeroing audio/features and attention restriction are artificial sensitivity probes, not natural omissions.',
                             'Contextual encoder features can retain local audio information outside the masked feature window.',
                             'No claim of natural-omission causal proof, unmeasured-population generalization, or TEST performance.']}
    save_json(outdir / 'REPORT.json', report)
    save_csv(outdir / 'aggregate_interventions.csv', [
        {'mode_subset': k, 'n': v['baseline']['n'], 'baseline_ap': v['baseline']['ap'],
         'intervened_ap': v['intervened']['ap'], 'paired_delta_ap': v['paired_delta_ap'],
         'mean_logit_delta': v['mean_logit_delta'], 'mean_prob_delta': v['mean_prob_delta'],
         'mean_abs_prob_delta': v['mean_abs_prob_delta'],
         'omission_recall_delta_at020': v['omission_recall_delta_at020']} for k, v in intervention_report.items()])
    save_csv(outdir / 'aggregate_ctc.csv', [
        {'score': k, 'n': v['n'], 'ap': v['ap'], 'roc_auc': v['roc_auc'],
         **v['diagnostic_fpr15']} for k, v in report['ctc_matched_discrimination'].items()
        if 'n' in v])
    save_json(outdir / 'ARTIFACT_SHA256.json', {p.name: sha256(p) for p in sorted(outdir.iterdir()) if p.name != 'RUN_STATE.json'})
    save_json(outdir / 'RUN_STATE.json', {'status': report['status'], 'optimizer_updates': 0,
                                         'backward_passes': 0, 'training': 'NOT_AUTHORIZED_NOT_EXECUTED',
                                         'elapsed_seconds': time.time()-started, 'last_valid_models': FOLD_CHECKPOINTS})
    print(f'Done: {outdir}/REPORT.json ({time.time()-started:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
