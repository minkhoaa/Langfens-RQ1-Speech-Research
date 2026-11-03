"""Synthetic CPU no-grad forward/loss algebra only; no backward or optimizer calls."""
import pytest
import torch
from torch import nn

from langfens_prep.p_sup.inputs import PromptFeatures, collate_prompt_features
from langfens_prep.p_sup.model import FrozenAudioAdapter, HeadConfig, matched_initial_pair, parameter_inventory
from langfens_prep.p_sup.objectives import collapsed_binary_logit, realized_ctc_loss, task_loss
from langfens_prep.p_sup.planning import UpdateLedger, accumulation_groups, independent_rng_pair, learning_rate_multiplier


@pytest.fixture(autouse=True)
def no_grad():
    with torch.no_grad():
        yield


def models():
    return matched_initial_pair(HeadConfig(feature_dim=8, width=8, heads=2, max_prompt_length=5, dropout=0.1))


def inputs():
    return collate_prompt_features([PromptFeatures(torch.tensor([1, 1, 31]), torch.ones(5, 8)),
                                   PromptFeatures(torch.tensor([23]), torch.full((2, 8), 2.0))])


def test_identical_initialization_capacity_storage_and_rng_isolation():
    before = torch.random.get_rng_state().clone()
    a, b = models()
    assert torch.equal(before, torch.random.get_rng_state())
    assert parameter_inventory(a) == parameter_inventory(b)
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()
    r1, r2 = independent_rng_pair()
    assert r1 is not r2 and torch.equal(r1.get_state(), r2.get_state())
    torch.rand(2, generator=r1)
    assert not torch.equal(r1.get_state(), r2.get_state())


def test_forward_shapes_mask_and_matching():
    a, b = models()
    a.eval(); b.eval()
    x = inputs()
    out = a(**x)
    assert out['operation_logits'].shape == (2, 3, 3)
    assert out['ctc_logits'].shape == (2, 5, 40)
    assert out['prompt_mask'].tolist() == [[True, True, True], [True, False, False]]
    assert torch.equal(out['operation_logits'], b(**x)['operation_logits'])
    assert torch.equal(out['attention'][1, :, 2:], torch.zeros(3, 3))
    assert torch.equal(out['operation_logits'][1, 1:], torch.zeros(2, 3))
    assert not out['operation_logits'].requires_grad
    assert all(p.grad is None for p in a.parameters())


def test_acoustic_access_and_padding_invariance_not_scientific_grounding():
    a, _ = models(); a.eval()
    x = inputs(); reference = a(**x)['operation_logits']
    changed = dict(x, features=x['features'].clone())
    changed['features'][0] *= -3
    assert not torch.allclose(reference[0], a(**changed)['operation_logits'][0])
    changed = dict(x, features=x['features'].clone())
    changed['features'][1, 2:] = 999
    assert torch.equal(reference, a(**changed)['operation_logits'])


def test_dummy_audio_encoder_frozen_eval_and_gold_rejected():
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__(); self.scale = nn.Parameter(torch.ones(8))
        def forward(self, audio, lengths):
            return audio[..., None] * self.scale, lengths
    head, _ = models()
    adapter = FrozenAudioAdapter(DummyEncoder(), head).train()
    assert not adapter.encoder.training and not adapter.encoder.scale.requires_grad
    result = adapter(audio=torch.ones(1, 4), audio_lengths=torch.tensor([4]),
                     prompt_phone_ids=torch.tensor([[1, 1]]), prompt_lengths=torch.tensor([2]))
    assert result['operation_logits'].shape == (1, 2, 3)
    with pytest.raises(TypeError):
        adapter(audio=torch.ones(1, 4), audio_lengths=torch.tensor([4]), prompt_phone_ids=torch.tensor([[1]]),
                prompt_lengths=torch.tensor([1]), gold_intervals=[])


@pytest.mark.parametrize('field,value', [('frame_lengths', torch.tensor([0, 2])), ('prompt_lengths', torch.tensor([4, 1])),
                                        ('prompt_phone_ids', torch.tensor([[0, 1, 31], [23, 0, 0]]))])
def test_invalid_model_input_fails(field, value):
    a, _ = models(); x = inputs(); x[field] = value
    with pytest.raises(ValueError):
        a(**x)


def test_objective_algebra_and_all_three_logits_active_in_binary_control():
    z = torch.tensor([[[1., 2., 3.], [2., 1., 0.], [0., 0., 0.]]])
    y = torch.tensor([[0, 1, 2]]); mask = torch.ones(1, 3, dtype=torch.bool)
    p = z.softmax(-1)
    assert torch.allclose(collapsed_binary_logit(z).sigmoid(), p[..., 1:].sum(-1))
    expected = -torch.stack([p[0, 0, 0].log(), p[0, 1, 1:].sum().log(), p[0, 2, 1:].sum().log()]).mean()
    assert torch.allclose(task_loss(z, y, mask, 'binary_collapsed'), expected)
    assert torch.isfinite(task_loss(z, y, mask, 'operation_ce'))
    for component in range(3):
        altered = z.clone(); altered[0, 0, component] += 0.5
        assert not torch.equal(task_loss(z, y, mask, 'binary_collapsed'), task_loss(altered, y, mask, 'binary_collapsed'))
    mask[0, 2] = False; z[0, 2] = float('nan'); y[0, 2] = -100
    assert torch.isfinite(task_loss(z, y, mask, 'operation_ce'))
    with pytest.raises(ValueError, match='no operation'):
        task_loss(z, y, torch.zeros_like(mask), 'operation_ce')


def test_realized_ctc_repeats_empty_and_infeasible():
    z = torch.zeros(1, 3, 40)
    loss = realized_ctc_loss(z, torch.tensor([3]), torch.tensor([[1, 1]]), torch.tensor([2]))
    assert torch.isfinite(loss)
    empty = realized_ctc_loss(z, torch.tensor([3]), torch.empty(1, 0, dtype=torch.long), torch.tensor([0]))
    assert torch.allclose(empty, torch.tensor(3.) * torch.tensor(40.).log())
    with pytest.raises(ValueError, match='infeasible'):
        realized_ctc_loss(z, torch.tensor([2]), torch.tensor([[1, 1]]), torch.tensor([2]))
    with pytest.raises(ValueError, match='blank'):
        realized_ctc_loss(z, torch.tensor([3]), torch.tensor([[0]]), torch.tensor([1]))


def test_metadata_update_schedule_tail_and_budget_only():
    a, b = UpdateLedger(), UpdateLedger()
    assert a is not b
    assert not a.record_attempt(overflow=True)
    assert a.record_attempt(overflow=False)
    assert (a.attempts, a.successful, a.skipped) == (2, 1, 1)
    assert (b.attempts, b.successful, b.skipped) == (0, 0, 0)
    assert learning_rate_multiplier(0) == 0 and learning_rate_multiplier(100) == 1
    assert learning_rate_multiplier(3600) > 0
    assert accumulation_groups(3) == (2, 1)
    assert len(accumulation_groups(900)) == 450  # FIT 1799, batch 2
    a.successful = 3600
    assert a.fixed_budget_reached
    with pytest.raises(ValueError, match='exhausted'):
        a.record_attempt(overflow=False)


def test_synthetic_label_to_head_loss_runtime_integration():
    from langfens_prep.p_sup.collate import SupervisedFeatures, collate_supervised_features
    from langfens_prep.p_sup.labels import build_operation_labels
    from langfens_prep.p_sup.runtime import operation_runtime_rows, validate_complete_run
    from langfens_prep.textgrid_parser import parse_gold_textgrid_content
    annotation = parse_gold_textgrid_content('''item [1]:
class = "IntervalTier"
name = "phones"
xmin = 0
xmax = 0
intervals [1]:
xmin = 0
xmax = 0
text = "AA"
intervals [2]:
xmin = 0
xmax = 0
text = "T,sil,d"
''', 'synthetic')
    y = build_operation_labels(annotation)
    batch = collate_supervised_features([SupervisedFeatures(PromptFeatures(y.canonical_ids, torch.ones(4, 8)), y)])
    treatment, control = models(); treatment.eval(); control.eval()
    out = treatment(**batch['net_input']); targets = batch['targets']
    for objective in ('operation_ce', 'binary_collapsed'):
        loss = task_loss(out['operation_logits'], targets['operation'], targets['operation_mask'], objective)
        loss += realized_ctc_loss(out['ctc_logits'], out['frame_lengths'], targets['ctc'], targets['ctc_lengths'])
        assert torch.isfinite(loss) and not loss.requires_grad
        predictions = operation_runtime_rows(logits=out['operation_logits'][0], prompt_phone_ids=[1, 31],
            run_id='synthetic', condition='matched', speaker_id='SYNTHETIC', utterance_id='fixture', objective=objective)
        result = validate_complete_run(predictions, run_id='synthetic', condition='matched',
                                      expected_prompts={('SYNTHETIC', 'fixture'): [1, 31]})
        assert result['ok'] == 2
