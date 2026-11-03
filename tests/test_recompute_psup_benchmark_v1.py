"""Independent metric recomputation tests (no models, gold data, or network)."""
import importlib.util
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('recompute',Path(__file__).parents[1]/'audit_scripts/recompute_psup_benchmark_v1.py')
recompute=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recompute)


def test_fpr_threshold_includes_positive_score_between_negatives():
    # Two allowed correct-token false positives; the positive is between the
    # third- and second-highest negative scores. Negative-only cutoffs miss it.
    labels=[0]*20+[1]
    scores=[i/100 for i in range(20)]+[.175]
    point=recompute.metrics.operating_points(labels,scores)['0.1']
    assert point['threshold']==.175
    assert (point['TP'],point['FP'],point['TN'],point['FN'])==(1,2,18,0)


def test_omission_deployed_precision_counts_substitutions_as_wrong_event():
    rows=[]
    for op,err,omit,sub in [(0,.1,.05,.05),(0,.2,.1,.1),(1,.99,.95,.04),(2,.95,.9,.05)]:
        rows.append(dict(operation_target=op,binary_target=int(op!=0),s_error=err,s_omit=omit,s_sub=sub))
    result=recompute.summarize(rows,True)
    point=result['omission_dedicated']['retrospective_val_fpr_points']['0.15']
    assert point['precision']==1.0  # correct-vs-deletion subset
    assert point['deployed_event_precision']==.5  # substitution is not omission
    assert point['deployed_wrong_event_count']==1
    assert result['omission_legacy']['n']==3
    assert result['substitution_legacy']['n']==3


def test_binary_control_has_no_identified_subtypes():
    rows=[dict(operation_target=op,binary_target=int(op!=0),s_error=score,s_omit=None,s_sub=None)
          for op,score in [(0,.1),(0,.2),(1,.8),(2,.9)]]
    result=recompute.summarize(rows,False)
    assert result['omission_dedicated']['evidence_status']=='unavailable'
    assert result['omission_dedicated']['ap'] is None
    assert result['substitution_dedicated']['ap'] is None
    assert result['binary']['ap']==1.0
