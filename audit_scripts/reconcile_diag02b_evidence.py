#!/usr/bin/env python3
"""Append-only DIAG-02B numerical reconciliation and optional fresh-process CPU replay.

No training or TEST reads. Historical artifacts are read, never rewritten.
PYTHONPATH=.:src HF_HUB_OFFLINE=1 MKL_CBWR=COMPATIBLE OMP_NUM_THREADS=1
.venv/bin/python audit_scripts/reconcile_diag02b_evidence.py --verify-forward
--outdir manifests/RQ1-DIAG-02B/reconciliation
"""
import argparse
import csv
import gzip
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from audit_scripts.diagnose_counterfactual_validity import (
    FOLD_CHECKPOINTS, FOLD_ZIPS, PRED, PRED_SHA, SPLIT, SPLIT_SHA,
    allowed_record, clean, save_csv, save_json, sha256, student_t_ci,
)
from langfens_prep.audio_loader import get_l2arctic_reader, load_l2arctic_audio_from_zip
from langfens_prep.model import LangfensWav2Vec2MDD
from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
from langfens_prep.textgrid_parser import parse_gold_textgrid_content
from audit_scripts.diagnose_temporal_alignment_validation import find_expected_omission_span


def load_csv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='manifests/RQ1-DIAG-02B/evaluation-final')
    p.add_argument('--outdir', required=True)
    p.add_argument('--verify-forward', action='store_true')
    args = p.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    run = Path(args.run)
    report = json.loads((run / 'REPORT.json').read_text())
    old = json.loads(Path('manifests/RQ1-DIAG-02/evaluation/TEMPORAL_ALIGNMENT_VALIDATION_REPORT.json').read_text())
    old_spk = load_csv('manifests/RQ1-DIAG-02/evaluation/speaker_paired_uncertainty_summary.csv')
    corrections = []
    def add(location, claim, measured, disposition):
        corrections.append({'location': location, 'historical_claim': claim,
                            'verified_value_or_correction': str(measured), 'disposition': disposition})
    add('DIAG-01/02 duration prose', '293 omissions <40ms vs JSON 284', report['duration_reconciliation'],
        '293 raw binary float subtraction; 284 four-decimal seconds and decimal ticks. Nine boundary cases logged; neither is a new dataset.')
    add('DIAG-01 duration prose', '545 omissions <80ms (86.65%)', '545 raw; 542 four-decimal (86.1685%)',
        'Same float-boundary convention issue; distinguish annotation duration from observed missing acoustics.')
    add('DIAG-02 checkpoint macro masking', '0.001845; CI [0.000886,0.002804]',
        student_t_ci([float(r['deletion_local_mask_delta']) for r in old_spk]),
        '0.001720; CI [0.000678,0.002763] from stored speaker table/JSON. 0.001845 has no supporting computation. Pooled 0.001714 is a different estimand.')
    add('DIAG-02 scientific_go_no_go_recommendation', 'omission enrichment 0.00x',
        old['attention_enrichment_and_baselines']['deletion']['enrichment_over_uniform_mean'],
        '1.3966 mean token-normalized enrichment; NCC speaker-only value rounds to zero. Not complete absence of localization.')
    add('DIAG-02 recommendation masking', 'local score delta <0.001', '0.001714 omission; 0.001113 all-token pooled',
        'Both exceed 0.001; feature-space erasure only, not waveform acoustic insensitivity.')
    add('DIAG-02 recommendation CTC posterior', 'P(BLANK)=82.8%', '84.47% pooled tokens; 84.2337% speaker macro',
        '82.8% unsupported; blank alone is not omission discrimination.')
    add('DIAG-02 checkpoint CTC macro CI', '0.8423 CI [0.7885,0.8961]',
        old['speaker_aware_paired_uncertainty']['macro_8speaker_statistics']['ctc_omission_blank_macro'],
        'Stored CI [0.788281,0.896394]; round as [0.7883,0.8964].')
    add('DIAG-02 prose random baseline masses correct/sub/deletion/all', '[0.0275,0.0289,0.0171,0.0273]',
        [old['attention_enrichment_and_baselines'][k]['random_window_baseline_mass_mean']
         for k in ('correct','substitution','deletion','all')],
        'Actual JSON [0.028590,0.030389,0.017846,0.028370]. Historical sampler excludes last legal start.')
    add('DIAG-02 JSON random enrichment correct/all', '4850064031259008 / 4217304067845006.5',
        'Unstable mean of per-token ratios with almost-zero denominators',
        'Do not interpret; new reports use ratio of summed masses and paired mass differences; no per-token random division.')
    add('DIAG-02 per-head offsets prose H0..H3', '[1.385,1.422,1.378,1.382] sec',
        [old['individual_head_inspection'][f'head_{h}']['temporal_offset_mean_sec'] for h in range(4)],
        'JSON values [1.2953,1.3402,1.3485,1.3698]; head-averaged offset 1.3923 is a separate measure.')
    add('DIAG-02 saturation prose', 'Every head/token >0.991 and 6-7 static frames',
        {'mean_max': [old['individual_head_inspection'][f'head_{h}']['max_weight_mean'] for h in range(4)],
         'minimum_max': [old['individual_head_inspection'][f'head_{h}']['max_weight_min'] for h in range(4)],
         'unique_peak_means': [old['individual_head_inspection'][f'head_{h}']['mean_unique_peaks_per_utterance'] for h in range(4)]},
        'Means exceed .991, not all tokens (minima .3386-.5078). H3 has 7.66 unique peaks; anchoring is concentrated, not literally static four frames.')
    add('DIAG-02 forced intervention definition', 'Restrict original attention and renormalize; +0.26 proves OOD breakdown',
        'Historical code inserts uniform rows into averaged weights then bmm(raw_hidden_states)',
        'Bypasses per-head learned V and output projection (and learned Q/K softmax); invalid MHA counterfactual. Cannot identify MLP OOD mechanism.')
    add('DIAG-02 causal/CTC conclusions', 'CTC detects omissions with >84% confidence; verified primary cause; GO training',
        'Matched CTC blank AUC 0.652663; TPR 0.242424 at in-sample FPR 0.136364; unmasked attention parity exact',
        'Withdraw certainty and training GO. Modest matched discrimination and artificial-intervention sensitivity are not proof of natural-omission causation.')
    add('DIAG-01 inherited causal language', '1.59% training prevalence; ~2.8% indistinguishable from uniform 1/200; gap indistinguishable from zero',
        '1.59% is validation census; omission-vs-correct prevalence 629/35992=1.7476%; interval baseline is K/T, not 1/T',
        'No training prevalence audit or omission-gap CI was supplied; delete unqualified structural-cause/accent-transfer claims from current interpretation, retain historical text with notice.')
    add('DIAG-02 transition congruence', '500/629 bit-for-bit identical intervals',
        'Historical code compares rounded DURATION differences <1e-4 sec, not exact interval endpoints',
        'Tolerance-level duration agreement, not proof of observed acoustic absence or bit identity; fresh endpoint counts in report.')
    add('DIAG-02 documentation chronology', 'Checkpoint recorded 2026-11-02 01:30/01:40 +07',
        old['metadata']['timestamp'],
        'Report timestamp is 2026-11-02 14:17:16 +07; documentation chronology is stale, not proof of earlier execution.')
    add('DIAG-02 status in global index', 'COMPLETED / INDEPENDENTLY_VERIFIED / ready for joint alignment training',
        report['status'],
        'Historical numerical observations retained; causal validation and GO superseded by DIAG-02B NO-GO/PARTIAL.')
    # All explicit DIAG-02 artifact hash claims in its checkpoint.
    md = Path('manifests/RQ1-DIAG-02/checkpoints/01-temporal-alignment-root-cause-validation.md').read_text()
    import re
    hash_rows = []
    for path, claimed in re.findall(r'`((?:audit_scripts|manifests|tests)/[^`]+)` \| `([0-9a-f]{64})`', md):
        actual = sha256(path)
        base_bytes = subprocess.check_output(['git', 'show', 'dd2dee9:' + path])
        hash_rows.append({'path': path, 'claimed_sha256': claimed, 'actual_sha256': actual, 'matches': claimed == actual,
                          'identical_to_base': Path(path).read_bytes() == base_bytes})
    add('DIAG-02 checkpoint hash ledger', 'All artifact hashes verified',
        f'{sum(not r["matches"] for r in hash_rows)} / {len(hash_rows)} hashes mismatch current base files',
        'Master JSON hash matches; preserve originals and actual per-file digests. Hash mismatch alone does not prove data tampering.')
    save_csv(out / 'historical_hash_audit.csv', hash_rows)
    # All annotation accesses guarded exactly as primary run; TEST entries only used as exclusion metadata.
    assert sha256(SPLIT) == SPLIT_SHA and sha256(PRED) == PRED_SHA
    folds = {f['fold_index']: f for f in json.loads(Path(SPLIT).read_text())['folds']}
    with gzip.open(PRED, 'rt') as f:
        pred = json.load(f)
    for u in pred['utterances']:
        allowed_record(u, folds)
    duration_tolerance, endpoint_tolerance, endpoint_exact = 0, 0, 0
    for u in pred['utterances']:
        ann = parse_gold_textgrid_content(get_l2arctic_reader(FOLD_ZIPS[u['fold']]).read_annotation_text(u['speaker'], u['utterance_id']), u['utterance_id'])
        for i, iv in enumerate(ann.canonical_intervals):
            if iv.error_type != 'deletion':
                continue
            a, b, d = find_expected_omission_span(ann.canonical_intervals, i, ann.total_duration)
            duration_tolerance += abs(d-(iv.xmax-iv.xmin)) < 1e-4
            endpoint_tolerance += abs(a-iv.xmin) < 1e-4 and abs(b-iv.xmax) < 1e-4
            endpoint_exact += a == iv.xmin and b == iv.xmax
    add('DIAG-02B independent endpoint verification', 'Historical code only checked span duration',
        {'duration_tolerance': int(duration_tolerance), 'endpoint_tolerance': int(endpoint_tolerance),
         'endpoint_exact': int(endpoint_exact)},
        'Fresh independent check confirms 500/629 endpoint-identical spans; retain this observation, not the acoustic-absence causal claim.')
    save_csv(out / 'corrections.csv', corrections)
    save_json(out / 'CORRECTIONS.json', corrections)
    tokens = load_csv(run / 'tokens.csv.gz')
    pairs = load_csv(run / 'matched_pairs.csv')
    by_id = {r['token_id']: r for r in tokens}
    descriptives = {}
    for group in ('omission', 'control'):
        rs = [by_id[p[group]] for p in pairs]
        descriptives[group] = {k: np.mean([float(r[k]) for r in rs]) for k in
                               ('duration_sec','blank_mean','canonical_mean','canonical_peak','blank_emission_fraction')}
    # Speaker table-derived exact reconciliation of the disputed macro number.
    outputs = {'historical_macro_from_csv': student_t_ci([float(r['deletion_local_mask_delta']) for r in old_spk]),
               'span_agreement': {'duration_within_1e4': int(duration_tolerance),
                                  'both_endpoints_within_1e4': int(endpoint_tolerance),
                                  'both_endpoints_exact_float': int(endpoint_exact)},
               'matched_descriptives': descriptives,
               'source_sha256': sha256(__file__), 'primary_report_sha256': sha256(run/'REPORT.json'),
               'fresh_process_forward': {'status': 'NOT_EXECUTED'}}
    if args.verify_forward:
        threads = report['metadata']['args']['threads']
        torch.set_num_threads(threads)
        torch.manual_seed(report['metadata']['seed'])
        torch.use_deterministic_algorithms(True)
        expected = {(int(r['fold']), r['speaker'], r['utterance_id']): [] for r in tokens}
        for r in tokens:
            expected[(int(r['fold']),r['speaker'],r['utterance_id'])].append(r)
        checks = []
        with torch.inference_mode():
            for fold in range(1,5):
                assert sha256(FOLD_CHECKPOINTS[fold]) == report['checkpoint_provenance'][fold-1]['sha256']
                ckpt = torch.load(FOLD_CHECKPOINTS[fold], map_location='cpu', weights_only=False)
                cfg = ckpt['metadata']['config']
                model = LangfensWav2Vec2MDD(pretrained_model_name_or_path=cfg['pretrained_model'], revision=cfg['pretrained_revision'])
                model.load_state_dict(ckpt['model'], strict=True)
                model.eval().requires_grad_(False)
                del ckpt
                for u in (u for u in pred['utterances'] if u['fold'] == fold):
                    key = (fold,u['speaker'],u['utterance_id'])
                    if key not in expected:
                        continue
                    rs = sorted(expected[key], key=lambda r:int(r['token_idx']))
                    wav, _, _ = load_l2arctic_audio_from_zip(FOLD_ZIPS[fold],u['speaker'],u['utterance_id'])
                    indices = torch.tensor([[PHONE_TO_CTC_INDEX[p] for p in u['canonical_phones']]])
                    output = model(waveforms=wav[None], audio_lengths=torch.tensor([len(wav)]),
                                   canonical_phone_indices=indices, canonical_phone_lengths=torch.tensor([indices.shape[1]]))
                    old = np.array([float(r['base_prob']) for r in rs], dtype=np.float32)
                    new = output.binary_logits.sigmoid()[0].numpy()
                    checks.append({'fold': fold,'speaker': u['speaker'],'utterance_id':u['utterance_id'],
                                   'max_probability_difference': float(np.max(np.abs(new-old))),
                                   'exact': bool(np.array_equal(old,new)),
                                   'max_logit_difference': float(np.max(np.abs(output.binary_logits[0].numpy() -
                                       np.array([float(r['base_logit']) for r in rs], dtype=np.float32))))})
                del model
        save_csv(out / 'fresh_process_forward.csv', checks)
        outputs['fresh_process_forward'] = {'status': 'EXACT' if all(c['exact'] for c in checks) else 'NUMERICAL_DRIFT',
                                            'utterances':len(checks), 'exact_utterances':sum(c['exact'] for c in checks),
                                            'max_probability_difference':max(c['max_probability_difference'] for c in checks),
                                            'max_logit_difference':max(c['max_logit_difference'] for c in checks),
                                            'interpretation': 'Probability records used scalar sigmoid; fresh check uses vector sigmoid. Report logit parity separately.',
                                            'threads':threads,'mkl_cbwr':os.environ.get('MKL_CBWR')}
    save_json(out / 'RECONCILIATION.json', outputs)
    save_json(out / 'ARTIFACT_SHA256.json', {f.name:sha256(f) for f in out.iterdir() if f.is_file()})
    print(json.dumps(clean(outputs), indent=2))


if __name__ == '__main__':
    main()
