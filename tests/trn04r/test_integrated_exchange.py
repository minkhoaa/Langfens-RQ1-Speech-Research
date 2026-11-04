"""Coordinator's production runtime/independent-metrics exchange integration."""
import torch

from audit_scripts.recompute_trn04r_metrics import validate_recognition_pair
from langfens_prep.alignment import align_canonical_sequences
from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
from langfens_prep.trn04.model import CompactFactorialOperationHead, FactorialHeadConfig
from langfens_prep.trn04.runtime import runtime_rows, offline_recognition_row, join_gold_rows
from langfens_prep.trn04r.metrics import recognition_summary


def fixture():
    with torch.random.fork_rng():
        torch.manual_seed(1337)
        head = CompactFactorialOperationHead(FactorialHeadConfig(feature_dim=4, width=8, heads=2, dropout=0)).eval().requires_grad_(False)
        phones = ['B', 'AE', 'T']
        ids = [PHONE_TO_CTC_INDEX[p] for p in phones]
        with torch.inference_mode():
            output = head(features=torch.randn(1, 4, 4), acoustic_features=torch.zeros(1, 4, 4),
                frame_lengths=torch.tensor([4]), prompt_phone_ids=torch.tensor([ids]), prompt_lengths=torch.tensor([3]))
        runtime, recognition = runtime_rows(logits=output['operation_logits'][0], ctc_logits=output['ctc_logits'][0],
            prompt_phone_ids=ids, canonical_phones=phones, speaker_id='NJS', utterance_id='synthetic',
            arm='T1', endpoint='fixed_budget_3600', condition='matched', audio_source_id='NJS/synthetic')
    return runtime, recognition, phones, ids


def test_focal_runtime_and_offline_recognition_exchange_matches_independent_consumer():
    runtime, recognition, phones, ids = fixture()
    offline = offline_recognition_row(recognition, ids)
    validate_recognition_pair([recognition], [offline])
    summary = recognition_summary([offline])
    assert summary['edits'] == offline['edit_counts'][0]
    assert summary['by_speaker']['NJS']['edits'] == summary['edits']
    assert summary['PER'] == summary['by_speaker']['NJS']['PER']
    assert all(r['scores']['s_omit'] is not None for r in runtime)
    assert all(not any(x in key for x in ('gold', 'target', 'textgrid')) for r in runtime for key in r)


def test_frozen_prediction_unchanged_across_nonidentical_offline_gold_joins():
    import copy
    runtime, _, phones, _ = fixture()
    before = copy.deepcopy(runtime)
    gold_a = [dict(canonical_phone=p, operation_target=0, binary_target=0) for p in phones]
    gold_b = [dict(canonical_phone=p, operation_target=2, binary_target=1) for p in ['HH', 'B', 'AE', 'T']]
    first = join_gold_rows(gold_positions=gold_a, prompt_phones=phones, runtime=runtime, align=align_canonical_sequences)
    second = join_gold_rows(gold_positions=gold_b, prompt_phones=phones, runtime=runtime, align=align_canonical_sequences)
    assert len(first) == 3 and len(second) == 4
    assert second[0]['mapping_status'] == 'unmapped_gold_abstention'
    assert runtime == before
    assert [r['prompt_phone'] for r in runtime] == phones
