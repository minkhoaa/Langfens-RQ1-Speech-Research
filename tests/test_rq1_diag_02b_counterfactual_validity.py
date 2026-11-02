"""Executable inference regressions; no backward, optimizer or pretrained downloads."""
from types import SimpleNamespace

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from audit_scripts.diagnose_counterfactual_validity import (
    allowed_record, diagnostic_threshold, frame_window, head_forward,
    matched_pairs, metric, random_window, shifted_window,
)
from langfens_prep.model import CanonicalAcousticCrossAttention, LangfensWav2Vec2MDD


@pytest.fixture
def small():
    torch.manual_seed(1337)
    torch.set_num_threads(2)
    model = LangfensWav2Vec2MDD(hidden_dim=32, conv_dim=8, num_transformer_layers=1,
                               num_attention_heads=4, intermediate_dim=48,
                               phone_embed_dim=8, cross_attention_heads=4, dropout=0)
    return model.eval().requires_grad_(False)


def test_attention_baseline_and_open_mask_forward_parity(small):
    # Variable waveform AND token lengths exercise original padding-safe path.
    audio = torch.randn(2, 2000)
    lengths = torch.tensor([2000, 1400])
    tokens = torch.tensor([[3, 7, 12], [8, 2, 0]])
    with torch.inference_mode():
        out = small(waveforms=audio, audio_lengths=lengths,
                    canonical_phone_indices=tokens, canonical_phone_lengths=torch.tensor([3, 2]))
        pad = torch.arange(out.frame_features.shape[1])[None] >= out.input_lengths[:, None]
        logits, weights = head_forward(small, out.frame_features, tokens, pad)
        open_logits, _ = head_forward(small, out.frame_features, tokens, pad,
                                     [(0, out.frame_features.shape[1])] * 3)
        torch.testing.assert_close(logits, out.binary_logits)
        torch.testing.assert_close(open_logits, out.binary_logits)
        assert torch.count_nonzero(weights[1, :, :, out.input_lengths[1]:]) == 0
        repeated = small(waveforms=audio, audio_lengths=lengths, canonical_phone_indices=tokens)
        assert torch.equal(out.binary_logits, repeated.binary_logits)
        assert out.input_lengths.tolist() == [6, 4]
        # Padding token retained numerically exactly as forward, excluded by the active-token mask.
        assert tokens.ne(0).sum().item() == 5


def test_masked_attention_matches_independent_projected_individual_heads():
    torch.manual_seed(41)
    ca = CanonicalAcousticCrossAttention(16, phone_embed_dim=4, num_heads=4, dropout=0).eval()
    binary = torch.nn.Linear(20, 1)
    model = SimpleNamespace(training=False, cross_attention_module=ca, binary_head=binary)
    features = torch.randn(2, 7, 16)
    ids = torch.tensor([[1, 2, 3], [5, 6, 0]])
    pad = torch.tensor([[False]*7, [False]*5+[True]*2])
    windows = [(1, 4), (2, 5), (0, 3)]
    mha = ca.cross_attention
    with torch.inference_mode():
        logits, weights = head_forward(model, features, ids, pad, windows)
        embeds = ca.canonical_phone_embedding(ids)
        q = ca.phone_projection(embeds) + ca.canonical_pos_embedding(torch.arange(3))[None]
        wq, wk, wv = mha.in_proj_weight.chunk(3)
        bq, bk, bv = mha.in_proj_bias.chunk(3)
        project = lambda x, w, b: F.linear(x, w, b).reshape(2, -1, 4, 4).transpose(1, 2)
        qh, kh, vh = project(q, wq, bq), project(features, wk, bk), project(features, wv, bv)
        scores = qh @ kh.transpose(-1, -2) / 2
        mask = torch.ones(3, 7, dtype=torch.bool)
        for i, (a, b) in enumerate(windows):
            mask[i, a:b] = False
        scores.masked_fill_(mask[None, None] | pad[:, None, None], -torch.inf)
        expected_weights = scores.softmax(-1)
        joined = (expected_weights @ vh).transpose(1, 2).reshape(2, 3, 16)
        projected = F.linear(joined, mha.out_proj.weight, mha.out_proj.bias)
        expected = binary(torch.cat([ca.layer_norm(projected), embeds], -1)).squeeze(-1)
        torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-6)
        # Historical shortcut is not equivalent, even with genuine softmax weights.
        wrong = binary(torch.cat([ca.layer_norm(weights.mean(1) @ features), embeds], -1)).squeeze(-1)
        assert not torch.allclose(wrong, expected, atol=1e-3)


def test_restriction_rejects_all_padded_window(small):
    with torch.inference_mode(), pytest.raises(ValueError, match='all active'):
        head_forward(small, torch.randn(1, 4, 32), torch.tensor([[2]]),
                     torch.tensor([[False, False, True, True]]), [(2, 4)])


def test_windows_duration_and_determinism():
    a, b = frame_window(.039, .04, 100)
    assert b > a
    for a, b, length in [(0, 3, 20), (17, 20, 20), (4, 7, 20), (0, 20, 20)]:
        x, y = shifted_window(a, b, length)
        assert y-x == b-a and 0 <= x < y <= length
        r1 = random_window(a, b, length, np.random.default_rng(7))
        r2 = random_window(a, b, length, np.random.default_rng(7))
        assert r1 == r2 and r1[1]-r1[0] == b-a
        if b-a <= length/2:
            assert r1[1] <= a or r1[0] >= b
    # Last placement must be reachable (historical off-by-one regression).
    placements = {random_window(0, 1, 3, np.random.default_rng(i))[0] for i in range(50)}
    assert 2 in placements


def row(i, et='correct', speaker='S', phone='T', pos='word_final', duration=.04):
    return {'token_id': i, 'error_type': et, 'speaker': speaker, 'phone': phone,
            'word_position': pos, 'duration_sec': duration}


def test_matching_exact_factors_caliper_and_no_replacement():
    rows = [row('d1', 'deletion'), row('d2', 'deletion'), row('d3', 'deletion', phone='R'),
            row('c1', duration=.045), row('wrong_spk', speaker='X'), row('wrong_pos', pos='word_initial'),
            row('too_long', duration=.061)]
    pairs, missing = matched_pairs(rows)
    assert len(pairs) == 1 and pairs[0]['control'] == 'c1'
    assert missing == ['d2', 'd3']
    assert pairs[0]['duration_difference'] == pytest.approx(.005)


def test_aucs_ties_and_threshold_ties():
    result = metric([0, 1, 0, 1], [0, 1, 0, 1])
    assert result['roc_auc'] == result['ap'] == 1
    result = metric([0, 1], [.9, .9])
    assert result['roc_auc'] == result['ap'] == .5
    threshold = diagnostic_threshold([.2]*10)
    assert threshold > .2  # no false claimed FPR ceiling at a tied quantile


def test_partition_guard_rejects_before_data_access():
    folds = {1: {'val': {'utterances': ['S/u'], 'speakers': ['S']}, 'test': {'speakers': ['T']}}}
    allowed_record({'fold': 1, 'speaker': 'S', 'utterance_id': 'u'}, folds)
    with pytest.raises(ValueError, match='Non-validation'):
        allowed_record({'fold': 1, 'speaker': 'T', 'utterance_id': 'u'}, folds)
    folds[1]['test']['speakers'] = ['S']
    with pytest.raises(ValueError, match='TEST'):
        allowed_record({'fold': 1, 'speaker': 'S', 'utterance_id': 'u'}, folds)


RUN = Path('manifests/RQ1-DIAG-02B/evaluation-final')


def records(name):
    with gzip.open(RUN / f'{name}.csv.gz', 'rt') as f:
        return list(csv.DictReader(f))


def test_executed_artifact_digests_and_source_provenance():
    from audit_scripts.diagnose_counterfactual_validity import sha256
    ledger = json.loads((RUN / 'ARTIFACT_SHA256.json').read_text())
    for name, digest in ledger.items():
        assert sha256(RUN / name) == digest
    report = json.loads((RUN / 'REPORT.json').read_text())
    assert report['metadata']['script_sha256'] == sha256('audit_scripts/diagnose_counterfactual_validity.py')
    assert report['parity']['max_forward_logit_difference'] == 0
    assert report['parity']['max_open_mask_logit_difference'] == 0
    assert report['parity']['repeat_exact']
    # Archive parity must not be silently waived to manufacture acceptance.
    assert report['parity']['archive_failures']
    assert report['status'] == 'PARTIAL_ARCHIVE_PARITY_FAILED'
    assert report['parity']['max_archive_probability_difference'] > report['parity']['archive_tolerance']


def test_executed_matched_controls_and_ctc_metrics_recompute():
    by_id = {r['token_id']: r for r in records('tokens')}
    with (RUN / 'matched_pairs.csv').open() as f:
        pairs = list(csv.DictReader(f))
    assert len({p['control'] for p in pairs}) == len(pairs)
    ys, scores = [], []
    for pair in pairs:
        d, c = by_id[pair['omission']], by_id[pair['control']]
        assert d['error_type'] == 'deletion' and c['error_type'] == 'correct'
        for key in ('speaker', 'phone', 'word_position'):
            assert d[key] == c[key]
        assert abs(float(d['duration_sec'])-float(c['duration_sec'])) <= .020 + 1e-12
        ys.extend([1, 0])
        scores.extend([float(d['blank_mean']), float(c['blank_mean'])])
    report = json.loads((RUN / 'REPORT.json').read_text())
    measured = metric(ys, scores)
    expected = report['ctc_matched_discrimination']['blank_mean']
    assert measured['ap'] == expected['ap']
    assert measured['roc_auc'] == expected['roc_auc']
    assert expected['diagnostic_fpr15']['fpr'] <= .15


def test_executed_interventions_use_paired_duration_matched_controls():
    rs = records('interventions')
    lookup = {(r['token_id'], r['mode']): r for r in rs}
    for r in rs:
        if not r['mode'].endswith('_local'):
            continue
        rand = lookup[(r['token_id'], r['mode'].replace('_local', '_random'))]
        a, b = int(r['window_start']), int(r['window_end'])
        ra, rb = int(rand['window_start']), int(rand['window_end'])
        assert b-a == rb-ra > 0
        assert rb <= a or ra >= b
        assert r['base_logit'] == rand['base_logit']
        assert r['window_unit'] == rand['window_unit']
        if r['mode'].startswith('waveform'):
            assert r['window_unit'] == 'sample'
        else:
            assert r['window_unit'] == 'frame'
    wave = [r for r in rs if r['mode'] == 'waveform_local']
    assert len(wave) == 32 and len({r['speaker'] for r in wave}) == 8


def test_executed_intervention_ap_and_delta_recompute():
    report = json.loads((RUN / 'REPORT.json').read_text())
    rs = records('interventions')
    for mode in {r['mode'] for r in rs}:
        sub = [r for r in rs if r['mode'] == mode]
        y = [r['error_type'] != 'correct' for r in sub]
        baseline = metric(y, [float(r['base_prob']) for r in sub])
        changed = metric(y, [float(r['prob']) for r in sub])
        stored = report['intervention_metrics'][mode + '/all_errors']
        assert stored['paired_delta_ap'] == changed['ap'] - baseline['ap']
        assert stored['mean_abs_prob_delta'] == pytest.approx(np.mean([
            abs(float(r['prob'])-float(r['base_prob'])) for r in sub]), abs=1e-15)


def test_duration_reconciliation_and_historical_macro():
    report = json.loads((RUN / 'REPORT.json').read_text())
    recon = report['duration_reconciliation']
    assert recon['raw_float_lt040'] - recon['rounded_4decimal_lt040'] == 9
    assert recon['rounded_4decimal_lt040'] == recon['decimal_tick_lt040']
    with (RUN / 'duration_boundary_disagreements.csv').open() as f:
        disputed = list(csv.DictReader(f))
    assert len(disputed) == 9
    assert all(round(float(r['duration_sec']), 4) == .04 for r in disputed)
    with open('manifests/RQ1-DIAG-02/evaluation/speaker_paired_uncertainty_summary.csv') as f:
        spk = list(csv.DictReader(f))
    assert np.mean([float(r['deletion_local_mask_delta']) for r in spk]) == pytest.approx(.00172025)


def test_head_records_and_stable_random_baseline_aggregation():
    report = json.loads((RUN / 'REPORT.json').read_text())
    heads = records('heads')
    assert len(heads) == 4 * report['scope']['sampled_tokens']
    for group, stored in report['attention_by_head_and_type'].items():
        et, head = group.split('/head')
        sub = [r for r in heads if r['error_type'] == et and r['head'] == head]
        observed = sum(float(r['mass']) for r in sub)
        random = sum(float(r['random_mass']) for r in sub)
        assert stored['ratio_of_sums_random'] == pytest.approx(observed / random, abs=1e-12)
        assert 'mean_token_random_enrichment' not in stored
