"""Independent standard-library audit of frozen TRN-04R outputs and B metrics.

No model, audio, TextGrid parser, or production metric implementation is imported.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ap(records, score, positive, ops):
    bins = defaultdict(lambda: [0, 0])
    for r in records:
        if r['gold_canonical_index'] is None or r['operation_target'] not in ops or r.get(score) is None:
            continue
        y = int(r['operation_target'] != 0 if positive is None else r['operation_target'] == positive)
        bins[r[score]][y] += 1
    total_positive = sum(b[1] for b in bins.values())
    if not total_positive:
        return None
    cumulative = true = 0
    contributions = []
    for value in sorted(bins, reverse=True):
        negatives, positives = bins[value]
        cumulative += negatives + positives
        true += positives
        contributions.append(positives / total_positive * true / cumulative)
    return math.fsum(contributions)


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def equal(a, b, label):
    if a is None or b is None:
        assert a is b, (label, a, b)
    else:
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14), (label, a, b)


FAMILIES = {'binary': ('s_error', None, {0, 1, 2}),
            'omission_legacy': ('s_error', 2, {0, 2}),
            'omission_dedicated': ('s_omit', 2, {0, 2}),
            'substitution_legacy': ('s_error', 1, {0, 1}),
            'substitution_dedicated': ('s_sub', 1, {0, 1})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--endpoint', choices=['fixed_budget_3600', 'best'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists(), 'Do not overwrite an audit'
    report = json.loads(args.metrics.read_text())
    checks = []
    for arm in ('C0', 'T1', 'T2', 'T3'):
        candidates = [args.root / 'arms' / arm / f'{arm}_{args.endpoint}',
                      args.root / f'{arm}_{args.endpoint}']
        existing = [p for p in candidates if p.is_dir()]
        assert len(existing) == 1, ('ambiguous/missing arm output', arm)
        directory = existing[0]
        manifest_root = directory.parent
        freeze = json.loads((manifest_root / 'Z_RUNTIME_FREEZE.json').read_text())
        assert freeze['gold_reads_before_freeze'] == 0
        for relative, sha in freeze['runtime_hashes'].items():
            assert digest(manifest_root / relative) == sha
        primary = rows(directory / 'tokens_Z.jsonl')
        mismatch = rows(directory / 'mismatched_tokens_Z.jsonl')
        cal = rows(directory / 'cal_tokens_Z.jsonl')
        recognition = rows(directory / 'recognition_Z.jsonl')
        assert {r['speaker_id'] for r in primary} == {'NJS', 'RRBI'}
        assert {r['speaker_id'] for r in cal} == {'NCC', 'THV'}
        gold = [r for r in primary if r['gold_canonical_index'] is not None]
        assert Counter(r['operation_target'] for r in gold) == {0:8799, 1:1012, 2:145}
        assert len(gold) == 9956
        answered = [r for r in gold if r['prompt_index'] is not None]
        assert Counter(r['operation_target'] for r in answered) == {0:8774, 1:1008, 2:137}
        assert len(primary)-len(gold) == 37
        assert len(gold)-len(answered) == 37
        for name in ('runtime_Z.jsonl', 'cal_runtime_Z.jsonl', 'mismatched_runtime_Z.jsonl'):
            for r in rows(directory/name):
                assert not any(any(s in k.lower() for s in ('gold', 'target', 'textgrid', 'realiz', 'timestamp')) for k in r)
                assert all(math.isfinite(v) for v in r['scores'].values())
                if name.startswith('mismatched'):
                    assert r['audio_source_id'] != r['speaker_id']+'/'+r['utterance_id']
                    assert r['audio_source_id'].split('/')[0] == r['speaker_id']
        result = report['cell_results'][arm]
        for records, expected in ((primary, result), (mismatch, result['mismatched_acoustic_control'])):
            strata = [(records, expected['metrics'], 'all')]
            for speaker, metrics in expected['by_speaker'].items():
                strata.append(([r for r in records if r['speaker_id']==speaker], metrics, speaker))
            for phone, metrics in expected['by_phone'].items():
                strata.append(([r for r in records if r['canonical_phone']==phone], metrics, phone))
            for subset, metrics, label in strata:
                for family, (score, positive, ops) in FAMILIES.items():
                    equal(ap(subset, score, positive, ops), metrics[family]['ap'], (arm, label, family))
            for family, (score, positive, ops) in FAMILIES.items():
                negatives = sorted([r[score] for r in cal if r['gold_canonical_index'] is not None and r['operation_target']==0 and r[score] is not None])
                for ceiling in (.05, .1, .15):
                    k = len(negatives)-math.floor(ceiling*len(negatives))-1
                    threshold = math.nextafter(negatives[k], math.inf)
                    opoint = expected['metrics'][family]['cal_operating_points'][str(ceiling)]
                    equal(threshold, opoint['threshold'], (arm, family, ceiling, 'threshold'))
                    subset = [r for r in records if r['gold_canonical_index'] is not None]
                    alarms = [r for r in subset if r.get(score) is not None and r[score]>=threshold]
                    tp = sum(r['operation_target']!=0 if positive is None else r['operation_target']==positive for r in alarms)
                    fp = len(alarms)-tp
                    counts = opoint['counts']
                    assert (tp, fp) == (counts['TP'], counts['FP']), (arm, family, ceiling)
                    if positive==2:
                        assert counts['FN']==145-tp and tp<=137
        edits = sum(distance(r['ref_phones'],r['hyp_phones']) for r in recognition)
        references = sum(len(r['ref_phones']) for r in recognition)
        assert references==9881
        assert edits==result['recognition']['edits']
        equal(edits/references,result['recognition']['PER'],(arm,'PER'))
        for speaker in ('NJS','RRBI'):
            group=[r for r in recognition if r['speaker_id']==speaker]
            equal(sum(distance(r['ref_phones'],r['hyp_phones']) for r in group)/sum(len(r['ref_phones']) for r in group),result['recognition']['by_speaker'][speaker]['PER'],(arm,speaker,'PER'))
        checks.append(dict(arm=arm,gold=9956,aligned=9919,unmapped=37,extras=37,ctc_edits=edits,ctc_references=references,status='VERIFIED'))
    output=dict(status='INDEPENDENT_RECOMPUTATION_AGREES',model_executed=False,endpoint=args.endpoint,
                source_sha256=digest(__file__),metrics_sha256=digest(args.metrics),checks=checks,
                scope='global/speaker/phone AP; CAL thresholds; deployed TP/FP; omission FN; global/speaker greedy PER; runtime freeze and gold-field audit')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(output))


if __name__=='__main__':
    main()
