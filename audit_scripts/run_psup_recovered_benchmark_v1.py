"""CPU-FP32 inference and saved-record recomputation for recovered P-SUP endpoints.

No training API is called. Original frozen backbone/head implementations are used
unchanged. Batch-one backbone features are reused across heads and protocols.
Z runtime files are frozen before the first TextGrid read. O is explicitly oracle.
Raw outputs must be outside Git and subsequently persisted to private storage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from audit_scripts.verify_psup_archival_v1 import average_precision, operating_points

SEALED = {'ABA', 'ASI', 'EBVS', 'HJK', 'LXC', 'PNV'}
REVISION = '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'
BACKBONE = 'facebook/wav2vec2-base-960h'
ZIP_SHA = 'b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9'
SPLIT_SHA = '7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b'
CONTRACT_SHA = '4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2'
CONFIGS = {
    't_op_fixed_budget_3600': ('t_op', 'fixed_budget_3600.pt', 'operation_ce', 3600),
    'b_control_fixed_budget_3600': ('b_control', 'fixed_budget_3600.pt', 'binary_collapsed', 3600),
    't_op_best': ('t_op', 'best.pt', 'operation_ce', 2700),
    'b_control_best': ('b_control', 'best.pt', 'binary_collapsed', 2250),
}
FAMILIES = (
    ('binary', 's_error', None, (0, 1, 2)),
    ('omission_legacy', 's_error', 2, (0, 2)),
    ('omission_dedicated', 's_omit', 2, (0, 2)),
    ('substitution_legacy', 's_error', 1, (0, 1)),
    ('substitution_dedicated', 's_sub', 1, (0, 1)),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_ids(ids):
    return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def create_output(path):
    if Path(path).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Raw benchmark output must be outside the Git repository')
    Path(path).mkdir(parents=True, exist_ok=False)


def cal_threshold(correct_scores, ceiling):
    """Correct-only CAL order statistic, score>=tau; never splits tied negatives."""
    if not correct_scores or not 0 <= ceiling < 1 or not all(math.isfinite(x) for x in correct_scores):
        raise ValueError('Finite nonempty CAL correct scores and ceiling in [0,1) required')
    descending = sorted(correct_scores, reverse=True)
    allowed = math.floor(ceiling*len(descending))
    return math.nextafter(descending[allowed], math.inf)


def cyclic_map(ids):
    groups = defaultdict(list)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate utterance IDs')
    for uid in ids:
        speaker = uid.split('/')[0]
        if speaker in SEALED:
            raise ValueError('Sealed speaker forbidden')
        groups[speaker].append(uid)
    result = {}
    for group in groups.values():
        if len(group) < 2:
            raise ValueError('Cyclic control requires at least two utterances per speaker')
        result.update(zip(group, group[1:]+group[:1]))
    return result


def confusion(rows, score, positive, threshold):
    """Operational no-alarm counts include abstentions; FPR bounds disclose them."""
    population = [r for r in rows if r.get('gold_canonical_index', 0) is not None
                  and (positive is None or r['operation_target'] in (0, positive))]
    is_positive = lambda r: bool(r['binary_target']) if positive is None else r['operation_target'] == positive
    alarm = lambda r: r.get(score) is not None and r[score] >= threshold
    tp = sum(is_positive(r) and alarm(r) for r in population)
    fp = sum(not is_positive(r) and alarm(r) for r in population)
    positives = sum(is_positive(r) for r in population)
    negatives = len(population)-positives
    fn, tn = positives-tp, negatives-fp
    unanswered_correct = sum(not is_positive(r) and r.get(score) is None for r in population)
    unanswered_error = sum(is_positive(r) and r.get(score) is None for r in population)
    deployed = [r for r in rows if r.get('gold_canonical_index', 0) is not None and alarm(r)]
    deployed_tp = sum(is_positive(r) for r in deployed)
    precision = tp/(tp+fp) if tp+fp else None
    return dict(threshold=threshold, TP=tp, FP=fp, TN=tn, FN=fn, n=len(population),
                positives=positives, negatives=negatives,
                recall=tp/positives if positives else None, precision=precision,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                achieved_correct_fpr=fp/negatives if negatives else None,
                correct_fpr_lower=fp/negatives if negatives else None,
                correct_fpr_upper=(fp+unanswered_correct)/negatives if negatives else None,
                unanswered_correct=unanswered_correct, unanswered_true_errors=unanswered_error,
                deployed_predicted_count=len(deployed), deployed_event_TP=deployed_tp,
                deployed_event_precision=deployed_tp/len(deployed) if deployed else None,
                TN_answered=tn-unanswered_correct,
                abstentions_counted_no_alarm_in_primary_confusion=True)


def edit_counts(ref, hyp):
    n, m = len(ref), len(hyp)
    table = [[(0, 0, 0, 0) for _ in range(m+1)] for _ in range(n+1)]
    for i in range(1, n+1):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, m+1):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = int(ref[i-1] != hyp[j-1])
            a, s, d, ins = table[i-1][j-1]
            diagonal = (a+cost, s+cost, d, ins)
            a, s, d, ins = table[i-1][j]
            deletion = (a+1, s, d+1, ins)
            a, s, d, ins = table[i][j-1]
            insertion = (a+1, s, d, ins+1)
            table[i][j] = min((diagonal, deletion, insertion), key=lambda value: value[0])
    return table[n][m]


def join_positions(gold, prompt_phones, runtime, oracle=False):
    from langfens_prep.alignment import align_canonical_sequences
    if oracle:
        if [r['canonical_phone'] for r in gold] != prompt_phones:
            raise ValueError('Oracle canonical sequence mismatch')
        pairs, missing, extra = [(i, i, 0) for i in range(len(gold))], [], []
    else:
        alignment = align_canonical_sequences([r['canonical_phone'] for r in gold], prompt_phones, tie_break='sub_first')
        pairs, missing, extra = alignment['aligned_pairs'], alignment['unaligned_gold'], alignment['unaligned_g2p']
    mapping = {g: p for g, p, _ in pairs}
    identity = {k: runtime[0][k] for k in ('speaker_id', 'utterance_id')}
    result = []
    for g, original in enumerate(gold):
        p = mapping.get(g)
        scores = runtime[p]['scores'] if p is not None else dict(s_error=None, s_omit=None, s_sub=None)
        result.append(dict(identity, **original, gold_canonical_index=g, prompt_index=p,
                           mapping_status='aligned' if p is not None else 'unmapped_gold_abstention', **scores))
    for p, _ in extra:
        result.append(dict(identity, gold_canonical_index=None, prompt_index=p,
                           operation_target=None, binary_target=None, canonical_phone=None,
                           word_final_consonant=False, mapping_status='prompt_extra', **runtime[p]['scores']))
    return result


def score_metrics(rows, calibration, treatment):
    gold = [r for r in rows if r.get('gold_canonical_index') is not None]
    answered = [r for r in gold if r.get('s_error') is not None]
    results = {}
    for name, score, positive, operations in FAMILIES:
        if not treatment and score != 's_error':
            results[name] = dict(evidence_status='unavailable', ap=None, reason='binary_collapsed subtype logits unidentified')
            continue
        eligible = [r for r in answered if r['operation_target'] in operations]
        y = [r['binary_target'] if positive is None else int(r['operation_target'] == positive) for r in eligible]
        scores = [r[score] for r in eligible]
        correct_cal = [r[score] for r in calibration if r.get('operation_target') == 0 and r.get(score) is not None]
        calibrated = {}
        for ceiling in (.05, .10, .15):
            threshold = cal_threshold(correct_cal, ceiling)
            point = confusion(gold, score, positive, threshold)
            point.update(nominal_CAL_correct_fpr_ceiling=ceiling, CAL_correct_count=len(correct_cal),
                         CAL_achieved_correct_fpr=sum(s >= threshold for s in correct_cal)/len(correct_cal),
                         threshold_provenance='CAL NCC/THV correct scores only; not VAL-tuned')
            calibrated[str(ceiling)] = point
        diagnostics = operating_points(y, scores) if y and sum(y) and len(y)-sum(y) else None
        if diagnostics:
            for point in diagnostics.values():
                point['all_gold_operating_point'] = confusion(gold, score, positive, point['threshold'])
        results[name] = dict(evidence_status='reproduced', score=score, n=len(y), positives=sum(y),
                             negatives=len(y)-sum(y), ap=average_precision(y, scores),
                             calibrated_operating_points=calibrated, retrospective_VAL_fpr_points=diagnostics)
    keys = sorted((r['speaker_id'], r['utterance_id'], r['gold_canonical_index']) for r in answered)
    return dict(evidence_status='reproduced', ap_population='Finite aligned/answered scores only; no full-gold AP imputation',
                gold_positions=len(gold), answered_positions=len(answered),
                label_counts=dict(Counter(str(r['operation_target']) for r in gold)),
                answered_label_counts=dict(Counter(str(r['operation_target']) for r in answered)),
                extra_prompt_positions=len(rows)-len(gold),
                eligible_gold_key_sha256=hashlib.sha256(('\n'.join('/'.join(map(str, k)) for k in keys)+'\n').encode()).hexdigest(),
                metrics=results)


def recognition_metrics(rows):
    totals = [0, 0, 0, 0]
    reference_count = 0
    speakers = defaultdict(lambda: dict(edits=0, references=0))
    for row in rows:
        counts = edit_counts(row['ref_phones'], row['hyp_phones'])
        assert list(counts) == row['edit_counts'], 'Saved decoding edit-count mismatch'
        totals = [a+b for a, b in zip(totals, counts)]
        reference_count += len(row['ref_phones'])
        speakers[row['speaker_id']]['edits'] += counts[0]
        speakers[row['speaker_id']]['references'] += len(row['ref_phones'])
    return dict(evidence_status='reproduced', edits=totals[0], substitutions=totals[1],
                deletions=totals[2], insertions=totals[3], reference_phones=reference_count,
                per=totals[0]/reference_count if reference_count else None,
                by_speaker={s: dict(v, per=v['edits']/v['references']) for s, v in speakers.items()},
                semantics='CTC recognition edits against realized phones; NOT human omission/substitution event counts')


def recompute(input_root, output_root):
    """Reads saved rows/decodings and CAL records, with no model or audio loading."""
    create_output(output_root)
    source_manifest = json.loads((input_root/'RUN_MANIFEST.json').read_text())
    for relative, digest in source_manifest['output_hashes'].items():
        if sha(input_root/relative) != digest:
            raise ValueError(f'Raw artifact hash mismatch: {relative}')
    summary = {}
    for config, (_, _, objective, updates) in CONFIGS.items():
        summary[config] = dict(successful_updates=updates, protocols={})
        for protocol in ('protocol_o', 'protocol_z'):
            directory = input_root/config/protocol
            cal = read_rows(directory/'calibration/offline_token_evaluation.jsonl')
            matched = read_rows(directory/'offline_token_evaluation.jsonl')
            mismatch = read_rows(directory/'mismatch/offline_token_evaluation.jsonl')
            report = score_metrics(matched, cal, objective == 'operation_ce')
            report['protocol'] = 'O-ARCHIVE' if protocol == 'protocol_o' else 'Z-PROMPT-V1'
            report['ctc'] = recognition_metrics(read_rows(directory/'utterance_predictions.jsonl'))
            assert report['ctc']['reference_phones'] == 9881
            mm = score_metrics(mismatch, cal, objective == 'operation_ce')
            report['mismatch_control'] = dict(evidence_status='reproduced',
                matched_binary_ap=report['metrics']['binary']['ap'], mismatched_binary_ap=mm['metrics']['binary']['ap'],
                binary_ap_gap=report['metrics']['binary']['ap']-mm['metrics']['binary']['ap'],
                policy='same batch1 frozen features, same-speaker cyclic shift1 audio, same prompt/gold cohort')
            report['by_speaker'] = {s: score_metrics([r for r in matched if r['speaker_id'] == s], cal, objective == 'operation_ce') for s in ('NJS', 'RRBI')}
            report['word_final_consonants'] = score_metrics([r for r in matched if r.get('word_final_consonant')], cal, objective == 'operation_ce')
            report['canonical_only_prior_control'] = dict(evidence_status='unavailable',
                reason='Frozen FIT1799 canonical-phone/position prior lookup not recovered; a new prior fit is not authorized')
            write_json(output_root/config/protocol/'metrics.json', report)
            write_json(output_root/config/protocol/'mismatch/metrics.json', mm)
            summary[config]['protocols'][protocol] = report
    write_json(output_root/'SUMMARY.json', summary)
    write_json(output_root/'RECOMPUTATION_MANIFEST.json', dict(source_manifest_sha256=sha(input_root/'RUN_MANIFEST.json'),
        recomputation_source_sha256=sha(__file__), input_root=str(input_root), model_executed=False))
    return summary


def make_frontend():
    from langfens_prep.g2p import CMUDictG2P
    return CMUDictG2P(dict_path=ROOT/'resources/CMUDict/cmudict.dict', verify_checksum=True, oov_policy='strict')


def infer(output_root, archive, threads):
    import numpy as np
    import torch
    from transformers import Wav2Vec2Model
    from transformers.utils.hub import cached_file
    from langfens_prep.dataset import ZeroOracleL2ArcticDataset
    from langfens_prep.g2p import CMUDictG2P
    from langfens_prep.p_sup.model import CompactOperationHead, HeadConfig
    from langfens_prep.p_sup.trainer import FrozenWav2Vec2AudioEncoder, greedy_ctc_decode
    from langfens_prep.p_sup.runtime import operation_runtime_rows, validate_complete_run
    from langfens_prep.phoneme_normalizer import CTC_INDEX_TO_PHONE, STANDARD_39_ARPABET

    started = time.perf_counter()
    source_sha = sha(__file__)
    split_path = ROOT/'manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json'
    contract_path = ROOT/'manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md'
    prompts_path = ROOT/'resources/L2-ARCTIC/prompts.json'
    dict_path = ROOT/'resources/CMUDict/cmudict.dict'
    for path, expected in ((archive, ZIP_SHA), (split_path, SPLIT_SHA), (contract_path, CONTRACT_SHA),
        (prompts_path, '87843d363bc7e2bbe849842efe16974cb6634dfe0539efb42d666aed7e17f11e'),
        (dict_path, '81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22')):
        if sha(path) != expected:
            raise ValueError(f'Input identity mismatch: {path}')
    weight = Path(cached_file(BACKBONE, 'model.safetensors', revision=REVISION, local_files_only=True))
    if sha(weight) != '8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a':
        raise ValueError('Pinned backbone cache hash mismatch')
    fold = next(f for f in json.loads(split_path.read_text())['folds'] if f['fold_index'] == 1)
    ids = dict(val=fold['val']['utterances'], cal=[u for u in fold['train']['utterances'] if u.split('/')[0] in {'NCC', 'THV'}])
    assert len(ids['val']) == len(ids['cal']) == 300
    assert digest_ids(ids['val']) == '6d34642b15bbf80cbd5bd102b5574ab36f5adac74be6d6d0ae7218b002eff065'
    assert not any(u.split('/')[0] in SEALED for values in ids.values() for u in values)
    permutation = cyclic_map(ids['val'])
    verification = json.loads((ROOT/'manifests/RQ1-P-SUP-BENCHMARK-V1/CHECKPOINT_VERIFICATION.json').read_text())
    recovery = json.loads((ROOT/'manifests/RQ1-P-SUP-BENCHMARK-V1/RECOVERY_MANIFEST.json').read_text())
    torch.set_num_threads(threads)
    torch.manual_seed(1337)
    np.random.seed(1337)
    safe = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
    models, checkpoint_metadata = {}, {}
    for config, (arm, filename, objective, updates) in CONFIGS.items():
        record = recovery['sources'][arm]['files'][filename]
        path = Path(record['local_path'])
        if sha(path) != record['sha256']:
            raise ValueError(f'Checkpoint mismatch: {config}')
        with torch.serialization.safe_globals(safe):
            payload = torch.load(path, map_location='cpu', weights_only=True)
        assert payload['successful_updates'] == updates and payload['skipped_amp_attempts_total'] == 0
        assert payload['metadata']['objective'] == objective
        assert verification['checkpoints'][f'{arm}/{filename}']['sha256'] == record['sha256']
        head = CompactOperationHead(HeadConfig()).float().eval().requires_grad_(False)
        head.load_state_dict(payload['head'], strict=True)
        models[config] = head
        checkpoint_metadata[config] = dict(path=str(path), sha256=record['sha256'], successful_updates=updates,
                                           skipped_updates=0, metadata=payload['metadata'])
    encoder = FrozenWav2Vec2AudioEncoder(Wav2Vec2Model.from_pretrained(BACKBONE, revision=REVISION, local_files_only=True).float()).eval().requires_grad_(False)
    def fingerprint(module):
        digest = hashlib.sha256()
        for key, value in sorted(module.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    before = dict(backbone=fingerprint(encoder), **{c: fingerprint(m) for c, m in models.items()})
    create_output(output_root)
    write_json(output_root/'AUDIO_MISMATCH_MAPPING.json', dict(mapping=permutation, ordered_VAL_ids=ids['val'], mapping_sha256=digest_ids([u+'->'+permutation[u] for u in ids['val']])))
    cache, features_receipt = {}, {}
    g2p = make_frontend()
    with torch.no_grad():
        for partition in ('val', 'cal'):
            dataset = ZeroOracleL2ArcticDataset(str(split_path), 1, 'val' if partition == 'val' else 'train',
                                               prompts_source=str(prompts_path), g2p_frontend=g2p, zip_path=str(archive))
            allowed = set(ids[partition])
            dataset.utterance_entries = [entry for entry in dataset.utterance_entries if '/'.join(entry) in allowed]
            assert ['/'.join(entry) for entry in dataset.utterance_entries] == ids[partition]
            for i in range(len(dataset)):
                item = dataset[i]
                uid = item['speaker_id']+'/'+item['utterance_id']
                audio = item['audio_waveform_16k'].unsqueeze(0)
                features, lengths = encoder(audio, torch.tensor([audio.shape[1]], dtype=torch.long))
                assert torch.isfinite(features).all()
                cache[uid] = dict(features=features, lengths=lengths, phones=item['canonical_phones'], prompt_ids=item['canonical_phone_indices'])
                features_receipt[uid] = dict(shape=list(features.shape), frames=int(lengths[0]),
                    fp32_feature_sha256=hashlib.sha256(features.numpy().tobytes()).hexdigest(),
                    fp32_waveform_sha256=hashlib.sha256(audio.numpy().tobytes()).hexdigest())
                if (i+1) % 50 == 0:
                    print(f'features {partition}: {i+1}/300', flush=True)
    write_json(output_root/'FROZEN_FEATURE_IDENTITIES.json', features_receipt)
    print('Frozen feature extraction finished; no TextGrid read yet.', flush=True)
    z_groups, hypotheses = {}, {}
    def forward(config, uid, audio_uid, phone_ids, protocol, partition, condition):
        features = cache[audio_uid]
        with torch.no_grad():
            output = models[config](features=features['features'], frame_lengths=features['lengths'],
                prompt_phone_ids=phone_ids.unsqueeze(0), prompt_lengths=torch.tensor([len(phone_ids)], dtype=torch.long))
        speaker, utterance = uid.split('/')
        rows = operation_runtime_rows(logits=output['operation_logits'][0, :len(phone_ids)],
            prompt_phone_ids=phone_ids.tolist(), run_id=f'{config}:{protocol}:{partition}:{condition}',
            condition=condition, speaker_id=speaker, utterance_id=utterance, objective=CONFIGS[config][2])
        if protocol == 'O-ARCHIVE':
            for row in rows:
                row['protocol'] = protocol
        logits = output['ctc_logits'][0, :int(features['lengths'][0])]
        frames = logits.argmax(-1).tolist()
        hyp = [CTC_INDEX_TO_PHONE[x] for x in greedy_ctc_decode(frames, blank_id=0)]
        return rows, dict(speaker_id=speaker, utterance_id=utterance, audio_source_id=audio_uid,
                          hyp_phones=hyp, frames=len(frames), blank_frames=frames.count(0))
    def group_path(config, protocol, partition, condition):
        base = output_root/config/protocol
        return base/('calibration' if partition == 'cal' else 'mismatch') if partition == 'cal' or condition == 'within_speaker_cyclic' else base
    for config in CONFIGS:
        for partition, condition in (('val', 'matched'), ('cal', 'matched'), ('val', 'within_speaker_cyclic')):
            runtime, utterances, expected = [], [], {}
            for uid in ids[partition]:
                audio_uid = permutation[uid] if condition != 'matched' else uid
                phone_ids = cache[uid]['prompt_ids']
                rows, utterance = forward(config, uid, audio_uid, phone_ids, 'Z-PROMPT-V1', partition, condition)
                runtime.extend(rows)
                utterances.append(utterance)
                expected[tuple(uid.split('/'))] = phone_ids.tolist()
                if condition == 'matched':
                    hypotheses[config, uid] = utterance['hyp_phones']
            validate_complete_run(runtime, run_id=f'{config}:Z-PROMPT-V1:{partition}:{condition}', condition=condition, expected_prompts=expected)
            directory = group_path(config, 'protocol_z', partition, condition)
            write_rows(directory/'runtime_predictions.jsonl', runtime)
            write_rows(directory/'utterance_runtime_predictions.jsonl', utterances)
            z_groups[config, partition, condition] = (runtime, utterances)
        print(f'Z frozen predictions: {config}', flush=True)
    z_freeze = {str(p.relative_to(output_root)): sha(p) for p in sorted(output_root.glob('*/protocol_z/**/*.jsonl'))}
    write_json(output_root/'Z_RUNTIME_FREEZE.json', dict(output_hashes=z_freeze, gold_reads_before_this_freeze=0))

    # First annotation access is below this boundary. It cannot affect Z predictions.
    from langfens_prep.audio_loader import get_l2arctic_reader
    from langfens_prep.textgrid_parser import parse_gold_textgrid_content
    from langfens_prep.p_sup.labels import build_operation_labels
    reader = get_l2arctic_reader(str(archive))
    vowels = {'AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY','OW','OY','UH','UW'}
    consonants = set(STANDARD_39_ARPABET)-vowels
    gold = {}
    for partition in ('val', 'cal'):
        for uid in ids[partition]:
            speaker, utterance = uid.split('/')
            annotation = parse_gold_textgrid_content(reader.read_annotation_text(speaker, utterance), utterance+'.TextGrid')
            labels = build_operation_labels(annotation)
            positions = []
            for i, interval in enumerate(annotation.canonical_intervals):
                midpoint = (interval.xmin+interval.xmax)/2
                word = next((w for w in annotation.words_intervals if w['xmin'] <= midpoint <= w['xmax']), None)
                indices = [j for j, iv in enumerate(annotation.canonical_intervals) if word and word['xmin'] <= (iv.xmin+iv.xmax)/2 <= word['xmax']]
                wfc = len(indices) > 1 and i == indices[-1] and interval.canonical_phone in consonants
                op = int(labels.operation[i])
                positions.append(dict(canonical_phone=interval.canonical_phone, operation_target=op, binary_target=int(op != 0), word_final_consonant=wfc))
            gold[uid] = dict(positions=positions, canonical_ids=labels.canonical_ids,
                             references=[CTC_INDEX_TO_PHONE[int(x)] for x in labels.ctc])
    def offline(config, protocol, partition, condition, runtime, utterances):
        grouped = defaultdict(list)
        for row in runtime:
            grouped[row['speaker_id']+'/'+row['utterance_id']].append(row)
        joined, recognition = [], []
        directory = group_path(config, protocol, partition, condition)
        runtime_hash = sha(directory/'runtime_predictions.jsonl')
        for uid in ids[partition]:
            phones = cache[uid]['phones'] if protocol == 'protocol_z' else [r['canonical_phone'] for r in gold[uid]['positions']]
            rows = join_positions(gold[uid]['positions'], phones, grouped[uid], oracle=protocol == 'protocol_o')
            for row in rows:
                row.update(runtime_sha256=runtime_hash, condition=condition)
            joined.extend(rows)
        for utterance in utterances:
            reference = gold[utterance['audio_source_id']]['references']
            recognition.append(dict(utterance, ref_phones=reference, edit_counts=list(edit_counts(reference, utterance['hyp_phones']))))
        write_rows(directory/'offline_token_evaluation.jsonl', joined)
        write_rows(directory/'utterance_predictions.jsonl', recognition)
        if partition == 'val':
            assert len([r for r in joined if r['gold_canonical_index'] is not None]) == 9956
            if protocol == 'protocol_z':
                assert sum(r['mapping_status'] == 'unmapped_gold_abstention' for r in joined) == 37
                assert sum(r['mapping_status'] == 'prompt_extra' for r in joined) == 37
    for (config, partition, condition), (runtime, utterances) in z_groups.items():
        offline(config, 'protocol_z', partition, condition, runtime, utterances)
    for config in CONFIGS:
        for partition, condition in (('val', 'matched'), ('cal', 'matched'), ('val', 'within_speaker_cyclic')):
            runtime, utterances = [], []
            for uid in ids[partition]:
                audio_uid = permutation[uid] if condition != 'matched' else uid
                rows, utterance = forward(config, uid, audio_uid, gold[uid]['canonical_ids'], 'O-ARCHIVE', partition, condition)
                if condition == 'matched':
                    assert utterance['hyp_phones'] == hypotheses[config, uid], 'CTC must be query independent'
                runtime.extend(rows)
                utterances.append(utterance)
            directory = group_path(config, 'protocol_o', partition, condition)
            write_rows(directory/'runtime_predictions.jsonl', runtime)
            write_rows(directory/'utterance_runtime_predictions.jsonl', utterances)
            offline(config, 'protocol_o', partition, condition, runtime, utterances)
        print(f'O predictions and joins: {config}', flush=True)
    after = dict(backbone=fingerprint(encoder), **{c: fingerprint(m) for c, m in models.items()})
    assert before == after, 'Frozen parameters changed'
    assert source_sha == sha(__file__), 'Inference source changed during execution'
    for relative, digest in z_freeze.items():
        assert sha(output_root/relative) == digest, 'Z runtime changed after gold access'
    sources = ['audit_scripts/run_psup_recovered_benchmark_v1.py','audit_scripts/verify_psup_archival_v1.py',
        'src/langfens_prep/p_sup/model.py','src/langfens_prep/p_sup/trainer.py','src/langfens_prep/p_sup/runtime.py',
        'src/langfens_prep/dataset.py','src/langfens_prep/p_sup/labels.py','src/langfens_prep/g2p.py',
        'src/langfens_prep/alignment.py','src/langfens_prep/phoneme_normalizer.py','src/langfens_prep/textgrid_parser.py',
        'src/langfens_prep/audio_loader.py']
    manifest = dict(schema_version='psup-recovered-benchmark-v1', status='INFERENCE_RECORDS_FROZEN',
        inference_source_sha256=source_sha, source_hashes={p: sha(ROOT/p) for p in sources},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
        command=sys.argv, interpreter=sys.executable, python=platform.python_version(), hardware=platform.platform(),
        precision='CPU FP32; batch1 for all endpoints, protocols and conditions', threads=torch.get_num_threads(),
        cache_policy='Original frozen encoder forward once per utterance, unchanged head consumes exact cached tensor',
        dependencies={p: importlib.metadata.version(p) for p in ['torch','transformers','numpy','soundfile','scipy','huggingface-hub']},
        backbone=dict(repository=BACKBONE, revision=REVISION, weight_sha256=sha(weight)),
        dataset=dict(archive=str(archive),sha256=ZIP_SHA,split_sha256=SPLIT_SHA,contract_sha256=CONTRACT_SHA),
        prompts_sha256=sha(prompts_path),dictionary_sha256=sha(dict_path),
        allowed_ids=ids, allowed_id_hashes={p:digest_ids(u) for p,u in ids.items()},
        configurations=checkpoint_metadata, parameter_fingerprints_before=before, parameter_fingerprints_after=after,
        elapsed_inference_and_gold_join_seconds=time.perf_counter()-started,
        optimizer_updates_this_process=0, backward_passes_this_process=0, gold_free_Z_runtime_freeze_sha256=sha(output_root/'Z_RUNTIME_FREEZE.json'))
    write_json(output_root/'inference_source.json', dict(filename=__file__,source_text=Path(__file__).read_text()))
    manifest['output_hashes'] = {str(p.relative_to(output_root)): sha(p) for p in sorted(output_root.rglob('*')) if p.is_file()}
    write_json(output_root/'RUN_MANIFEST.json', manifest)
    summary = recompute(output_root, output_root/'recomputed')
    for config, result in summary.items():
        print(config, {p: values['metrics']['binary']['ap'] for p, values in result['protocols'].items()}, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['infer', 'recompute'])
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--zip-path', type=Path, default=ROOT/'datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == 'infer':
        infer(args.output_root, args.zip_path, args.threads)
    else:
        if args.input_root is None:
            parser.error('--input-root is required for recompute')
        recompute(args.input_root, args.output_root)


if __name__ == '__main__':
    main()
