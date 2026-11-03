"""Synthetic CPU forward-only arrays. No model, weights, corpus, or autograd."""

import itertools
import math

import pytest

from langfens_prep.p_score.ctc import (
    alternatives, ctc_forward, score_prompt, validate_emissions,
)
from langfens_prep.p_score.controls import greedy_decode, levenshtein, self_aligned_gop


def probabilities_to_logs(rows):
    return [[math.log(p) if p else -math.inf for p in row] for row in rows]


def brute_paths(rows):
    """Independent acoustic-path enumeration, not traversal of scorer states."""
    classes = len(rows[0]) if rows else 3
    masses, rewards = {}, {}
    for path in itertools.product(range(classes), repeat=len(rows)):
        sequence, counts, previous = [], [], None
        for symbol in path:
            if symbol != previous and symbol:
                sequence.append(symbol)
                counts.append(0)
            if symbol:
                counts[-1] += 1
            previous = symbol
        sequence = tuple(sequence)
        weight = math.prod(rows[t][symbol] for t, symbol in enumerate(path))
        masses[sequence] = masses.get(sequence, 0.0) + weight
        rewards.setdefault(sequence, [0.0] * len(sequence))
        for i, count in enumerate(counts):
            rewards[sequence][i] += weight * count
    return masses, rewards


PROMPTS = [p for n in range(4) for p in itertools.product((1, 2), repeat=n)]


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("frames", range(5))
@pytest.mark.parametrize("distribution", range(3))
def test_p_score_ctc_exhaustive_tiny_lattice_oracle(prompt, frames, distribution):
    patterns = ((0.2, 0.3, 0.5), (1 / 3, 1 / 3, 1 / 3), (0.0, 0.25, 0.75))
    rows = [patterns[(t + distribution) % 3] for t in range(frames)]
    log_rows = probabilities_to_logs(rows)
    masses, rewards = brute_paths(rows)
    canonical = ctc_forward(log_rows, prompt).log_mass
    assert math.exp(canonical) == pytest.approx(masses.get(prompt, 0), abs=1e-12)
    for i in range(len(prompt)):
        forward = ctc_forward(log_rows, prompt, target_index=i)
        if forward.occupancy is not None:
            assert forward.occupancy == pytest.approx(rewards[prompt][i] / masses[prompt], abs=1e-10)
        for variant in ("S", "SD"):
            sequences = alternatives(prompt, i, variant, 3)
            # Construct expected language independently (including canonical q).
            expected = {prompt[:i] + (q,) + prompt[i + 1:] for q in (1, 2)}
            if variant == "SD":
                expected.add(prompt[:i] + prompt[i + 1:])
            assert set(sequences) == expected
            expected_mass = sum(masses.get(seq, 0.0) for seq in expected)
            occupancy_mass = sum(rewards.get(seq, [0] * len(seq))[i]
                                 for seq in expected if len(seq) == len(prompt))
            score = score_prompt(log_rows, prompt, variant, num_classes=3)[i]
            assert math.exp(score.relaxed_log_mass) == pytest.approx(expected_mass, abs=1e-12)
            assert canonical <= score.relaxed_log_mass + 1e-12
            if expected_mass:
                assert score.posterior_target_occupancy == pytest.approx(occupancy_mass / expected_mass, abs=1e-10)
            if frames and masses.get(prompt, 0) > 0:
                expected_error = math.log(expected_mass / masses[prompt])
                assert score.s_error == pytest.approx(expected_error, abs=1e-10)
                assert score.raw_gop == pytest.approx(-expected_error, abs=1e-10)
                assert score.occupancy_normalized_error == pytest.approx(
                    expected_error / max(1.0, occupancy_mass / expected_mass), abs=1e-10)
            else:
                assert score.status == "abstain"
                assert score.s_error is None


def test_p_score_ctc_deletion_exposes_repeat_boundary_requires_blank():
    rows = probabilities_to_logs([(0.1, 0.8, 0.1)] * 2)
    # Deleting middle 2 from 1,2,1 gives 1,1, NOT a collapsed single 1.
    deleted = (1, 1)
    assert ctc_forward(rows, deleted).log_mass == -math.inf
    three = rows + probabilities_to_logs([(0.8, 0.1, 0.1)])
    assert math.isfinite(ctc_forward(three, deleted).log_mass)
    assert alternatives((1, 1), 0, "SD", 3).count((1,)) == 1
    assert alternatives((1, 1), 1, "SD", 3).count((1,)) == 1


def test_p_score_ctc_union_sum_not_mean_max_or_duplicate():
    scores = score_prompt(probabilities_to_logs([(0.2, 0.3, 0.5)]), [1], "SD", num_classes=3)
    assert scores[0].relaxed_log_mass == pytest.approx(0)
    assert scores[0].s_error == pytest.approx(-math.log(0.3))
    assert scores[0].posterior_target_occupancy == pytest.approx(0.8)
    # All phones plus empty accept every one-frame path once, not twice.


@pytest.mark.parametrize("variant", ("S", "SD"))
def test_p_score_ctc_padding_empty_and_no_frame_accounting(variant):
    rows = probabilities_to_logs([(0.2, 0.3, 0.5)] * 4)
    original = score_prompt(rows, [1, 2], variant, num_classes=3)
    assert score_prompt(rows + [[-math.inf] * 3], [1, 2], variant, valid_frames=4, num_classes=3) == original
    assert score_prompt([], [], variant, num_classes=3) == []
    score = score_prompt([], [1], variant, num_classes=3)[0]
    assert score.reason == "no_valid_frames" and score.s_error is None
    assert ctc_forward([], []).log_mass == 0
    assert ctc_forward(rows, []).log_mass == pytest.approx(4 * math.log(0.2))


@pytest.mark.parametrize("bad", (math.nan, math.inf, 0.1, True, "0.1"))
def test_p_score_ctc_reject_invalid_emissions(bad):
    with pytest.raises(ValueError):
        validate_emissions([[bad, -math.inf, -math.inf]], num_classes=3)


def test_p_score_ctc_reject_nan_in_padding_and_all_zero_valid_row():
    rows = probabilities_to_logs([(0.2, 0.3, 0.5)])
    with pytest.raises(ValueError):
        validate_emissions(rows + [[math.nan] * 3], 1, num_classes=3)
    with pytest.raises(ValueError):
        validate_emissions([[-math.inf] * 3], num_classes=3)
    with pytest.raises(ValueError):
        validate_emissions(rows, True, num_classes=3)
    with pytest.raises(ValueError):
        validate_emissions(rows, 2, num_classes=3)
    with pytest.raises(ValueError):
        validate_emissions([[0, 0]], num_classes=3)


@pytest.mark.parametrize("phones", ([0], [40], [True], [-1], [1.0]))
def test_p_score_ctc_reject_phone_map_misuse(phones):
    with pytest.raises(ValueError):
        score_prompt([], phones, "S")


def test_p_score_ctc_sdi_explicitly_unsupported():
    with pytest.raises(ValueError, match="supported"):
        score_prompt([], [1], "SDI")


def test_p_score_ctc_zero_mass_context_and_canonical_abstain():
    rows = probabilities_to_logs([(1.0, 0, 0)] * 3)
    score = score_prompt(rows, [1], "SD", num_classes=3)[0]
    assert score.relaxed_log_mass == 0 and score.posterior_target_occupancy == 0
    assert score.reason == "canonical_infeasible_or_zero_mass"
    score = score_prompt(rows, [1, 2], "SD", num_classes=3)[0]
    assert score.reason == "relaxed_context_infeasible_or_zero_mass"


def test_p_score_ctc_log_space_long_sequence_avoids_probability_underflow():
    rows = probabilities_to_logs([(0.2, 0.4, 0.4)] * 1200)
    scores = score_prompt(rows, [1, 2], "SD", num_classes=3)
    assert all(math.isfinite(s.s_error) for s in scores)
    assert all(s.canonical_log_mass < -745 for s in scores)
    assert math.exp(scores[0].canonical_log_mass) == 0.0


def test_p_score_controls_collapse_blank_and_deterministic_ties():
    rows = probabilities_to_logs([(0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    assert greedy_decode(rows) == (1, 1, 2)
    assert greedy_decode(probabilities_to_logs([(0.5, 0.5, 0)])) == ()
    aligned = levenshtein((1, 1), (1,))
    assert aligned.ambiguous and aligned.distance == 1
    assert [(s.prompt_index, s.decoded_index, s.recognition_edit) for s in aligned.steps] == [
        (0, None, "deletion"), (1, 0, "match")]
    assert levenshtein((), (1,)).steps[0].prompt_index is None
    assert not levenshtein((1, 2), (1, 2)).ambiguous


def test_p_score_controls_self_alignment_viterbi_brute_force():
    rows = [(0.2, 0.3, 0.5), (0.5, 0.3, 0.2), (0.3, 0.4, 0.3), (0.4, 0.3, 0.3)]
    logs = probabilities_to_logs(rows)
    for prompt in PROMPTS:
        alignment = self_aligned_gop(logs, prompt)
        candidates = []
        for path in itertools.product(range(3), repeat=len(rows)):
            if greedy_decode([[0 if j == p else -math.inf for j in range(3)] for p in path]) == prompt:
                candidates.append(math.prod(rows[t][p] for t, p in enumerate(path)))
        if not candidates:
            assert alignment is None
        else:
            assert math.exp(alignment.log_path_mass) == pytest.approx(max(candidates), abs=1e-12)
            assert len(alignment.phone_frames) == len(prompt)
            assert all(len(fs) >= 1 for fs in alignment.phone_frames)
            for i, frames in enumerate(alignment.phone_frames):
                assert alignment.error_scores[i] == pytest.approx(-sum(logs[t][prompt[i]] for t in frames) / len(frames))
    assert self_aligned_gop([], [1]) is None
    assert self_aligned_gop([], []).states == ()
