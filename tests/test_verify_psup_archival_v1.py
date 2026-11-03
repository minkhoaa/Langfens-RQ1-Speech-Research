"""Small independent tests for the archival verification audit (no models)."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "archival_v1", Path(__file__).parents[1] / "audit_scripts/verify_psup_archival_v1.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_perfect_ranking():
    assert audit.average_precision([0, 1, 1], [.1, .8, .9]) == 1.0


def test_tied_scores_are_grouped_not_row_ordered():
    assert audit.average_precision([1, 0], [.3, .3]) == .5
    assert audit.average_precision([0, 1], [.3, .3]) == .5


def test_undefined_ap_is_null():
    assert audit.average_precision([0, 0], [.1, .2]) is None
    assert audit.average_precision([], []) is None


def test_fpr_ties_cannot_be_split():
    points = audit.operating_points([1, 0, 0], [.9, .9, .1])
    for result in points.values():
        assert result['TP'] == result['FP'] == 0
        assert result['FN'] == 1
        assert result['TN'] == 2
        assert result['precision'] is None
        assert result['recall'] == 0
        assert result['threshold'] > .9


def test_fpr_ceiling_uses_correct_denominator():
    labels, scores = [1] + [0]*20, [.95] + [i/100 for i in range(20)]
    point = audit.operating_points(labels, scores)['0.1']
    assert (point['TP'], point['FP'], point['TN'], point['FN']) == (1, 2, 18, 0)
    assert point['threshold'] == .18
    assert point['achieved_fpr'] == .1
    assert point['precision'] == pytest.approx(1/3)


def test_noninterpolated_ap():
    assert audit.average_precision([0, 1, 0, 1], [.9, .8, .7, .6]) == .5
