"""Synthetic in-memory labels/features only. No files, model loading or training."""
from dataclasses import replace

import pytest
import torch

from langfens_prep.p_sup.collate import SupervisedFeatures, collate_supervised_features
from langfens_prep.p_sup.inputs import PromptFeatures, collate_prompt_features
from langfens_prep.p_sup.labels import LabelContractError, build_operation_labels
from langfens_prep.textgrid_parser import parse_gold_textgrid_content


def synthetic_annotation(labels):
    intervals = '\n'.join(f'intervals [{i + 1}]:\nxmin = 0\nxmax = 0\ntext = "{label}"' for i, label in enumerate(labels))
    return parse_gold_textgrid_content('item [1]:\nclass = "IntervalTier"\nname = "phones"\nxmin = 0\nxmax = 0\n' + intervals, 'synthetic')


def test_strict_same_base_unknown_deletion_addition_chronology():
    y = build_operation_labels(synthetic_annotation(['AA1', 'AA,AA*,s', 'T,sil,d', 'N,err,s', 'sil,R,a', 'sil', 'AA']))
    assert y.canonical_ids.tolist() == [1, 1, 31, 23, 1]
    assert y.operation.tolist() == [0, 1, 2, 1, 0]
    assert y.binary.tolist() == [0, 1, 1, 1, 0]
    assert y.realization.tolist() == [0, 0, -100, -100, 0]
    assert y.ctc.tolist() == [1, 1, 28, 1]
    assert y.addition_gaps == (4,)
    assert y.unknown_realized_intervals == (3,)


@pytest.mark.parametrize('label', ['NOTAPHONE', 'err', 'NOTAPHONE,AA,s', 'NOTAPHONE,sil,d'])
def test_unknown_canonical_fails_not_shortens(label):
    with pytest.raises(LabelContractError, match='unknown canonical'):
        build_operation_labels(synthetic_annotation(['AA', label, 'AA']))


@pytest.mark.parametrize('labels', [[], ['sil'], ['sil,AA,a']])
def test_empty_canonical_supervision_rejected(labels):
    with pytest.raises(LabelContractError, match='no canonical'):
        build_operation_labels(synthetic_annotation(labels))


def test_index_and_binary_corruption_fail():
    a = synthetic_annotation(['AA'])
    a.intervals[0].token_index = 1
    with pytest.raises(LabelContractError, match='indices'):
        build_operation_labels(a)
    a.intervals[0].token_index = 0
    a.intervals[0].binary_label_strict = 1
    with pytest.raises(LabelContractError, match='binary'):
        build_operation_labels(a)


def example(labels, frames):
    y = build_operation_labels(synthetic_annotation(labels))
    return SupervisedFeatures(PromptFeatures(y.canonical_ids, torch.ones(frames, 8)), y)


def test_unequal_lengths_and_no_supervision_in_net_input():
    a, b = example(['AA', 'AA,AA*,s', 'T,sil,d'], 5), example(['N'], 2)
    batch = collate_supervised_features([a, b])
    assert set(batch['net_input']) == {'features', 'frame_lengths', 'prompt_phone_ids', 'prompt_lengths'}
    assert batch['targets']['operation'].tolist() == [[0, 1, 2], [0, -100, -100]]
    assert batch['targets']['operation_mask'].tolist() == [[True, True, True], [True, False, False]]
    assert batch['targets']['realization_mask'].tolist() == [[True, True, False], [True, False, False]]
    assert batch['targets']['ctc_lengths'].tolist() == [2, 1]
    assert batch['targets']['ctc'].tolist() == [[1, 1], [23, 0]]


def test_deletion_only_has_empty_realized_ctc_not_blank_target():
    batch = collate_supervised_features([example(['T,sil,d'], 2)])
    assert batch['targets']['ctc'].shape == (1, 0)
    assert batch['targets']['ctc_lengths'].tolist() == [0]
    assert batch['targets']['operation_mask'].all()


@pytest.mark.parametrize('kind', ['operation', 'binary', 'realization', 'ctc', 'prompt'])
def test_corrupt_supervision_rejected(kind):
    x = example(['AA'], 2)
    if kind == 'prompt':
        x = replace(x, inputs=PromptFeatures(torch.tensor([2]), x.inputs.features))
    else:
        x = replace(x, labels=replace(x.labels, **{kind: torch.tensor([99])}))
    with pytest.raises(ValueError):
        collate_supervised_features([x])


def test_blinded_runtime_and_mixed_batches():
    x = example(['AA'], 2)
    assert collate_prompt_features([x.inputs])['prompt_lengths'].tolist() == [1]
    with pytest.raises(ValueError, match='mixed'):
        collate_supervised_features([x, x.inputs])
    with pytest.raises(ValueError):
        collate_prompt_features([x])


@pytest.mark.parametrize('key', ['gold', 'operation', 'perceived_sequence', 'TextGrid', 'xmin', 'word_times'])
def test_runtime_input_whitelist_rejects_gold(key):
    with pytest.raises(ValueError, match='only'):
        PromptFeatures.from_mapping({'prompt_phone_ids': torch.tensor([1]), 'features': torch.ones(2, 8), key: []})


@pytest.mark.parametrize('phones', [torch.tensor([]), torch.tensor([0]), torch.tensor([40]), torch.tensor([1.0])])
def test_invalid_prompt_phones(phones):
    with pytest.raises(ValueError):
        PromptFeatures(phones, torch.ones(2, 8)).validate()
