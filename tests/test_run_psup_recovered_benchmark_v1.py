"""No model/backward execution: tests the final runner's accounting boundaries."""
import importlib.util
from pathlib import Path

import pytest


def load():
    path=Path(__file__).parents[1]/'audit_scripts/run_psup_recovered_benchmark_v1.py'
    assert path.is_file(), 'Final executable benchmark runner has not been implemented'
    spec=importlib.util.spec_from_file_location('final_runner',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_cutoff_uses_correct_scores_and_retains_between_score_positive():
    m=load()
    threshold=m.cal_threshold([i/100 for i in range(20)],.1)
    assert .17 < threshold < .175
    assert sum(x>=threshold for x in [i/100 for i in range(20)])==2
    assert .175>=threshold


def test_calibration_never_splits_a_tied_correct_group():
    m=load()
    threshold=m.cal_threshold([.1]*10+[.9]*10,.05)
    assert threshold>.9
    assert .95>=threshold


def test_cyclic_control_is_manifest_ordered_same_speaker_without_fixed_points():
    m=load()
    assert m.cyclic_map(['NJS/a','NJS/b','RRBI/c','RRBI/d'])=={'NJS/a':'NJS/b','NJS/b':'NJS/a','RRBI/c':'RRBI/d','RRBI/d':'RRBI/c'}
    with pytest.raises(ValueError):
        m.cyclic_map(['ABA/a','ABA/b'])
    with pytest.raises(ValueError):
        m.cyclic_map(['NJS/a'])


def test_omission_confusion_retains_abstentions_and_deployed_wrong_subtype():
    m=load()
    rows=[dict(operation_target=op,binary_target=int(op!=0),s_error=score)
          for op,score in [(0,.1),(1,.95),(2,.9),(2,None),(0,None)]]
    result=m.confusion(rows,'s_error',2,.8)
    assert (result['TP'],result['FP'],result['TN'],result['FN'])==(1,0,2,1)
    assert result['recall']==.5
    assert result['precision']==1
    assert result['deployed_event_precision']==.5
    assert result['unanswered_correct']==1
    assert result['correct_fpr_lower']==0
    assert result['correct_fpr_upper']==.5


def test_saved_decoding_edit_accounting_substitution_deletion_not_human_events():
    m=load()
    assert m.edit_counts(['A','B','C'],['A','D'])==(2,1,1,0)
    assert m.edit_counts([],['A'])==(1,0,0,1)


def test_zero_oracle_join_retains_prompt_extras_without_gold_imputation():
    m=load()
    gold=[{'canonical_phone':'AA','operation_target':0},{'canonical_phone':'B','operation_target':2}]
    rows=[{'speaker_id':'NJS','utterance_id':'u','prompt_index':i,'scores':{'s_error':.1+i*.1,'s_omit':None,'s_sub':None}}
          for i in range(3)]
    joined=m.join_positions(gold,['AA','D','B'],rows)
    assert [(r['gold_canonical_index'],r['prompt_index']) for r in joined]==[(0,0),(1,2),(None,1)]
    assert joined[-1]['mapping_status']=='prompt_extra'
    joined2=m.join_positions(gold,['AA'],rows[:1])
    assert joined2[1]['gold_canonical_index']==1
    assert joined2[1]['prompt_index'] is None
    assert joined2[1]['s_error'] is None


def test_pinned_prompt_frontend_uses_real_constructor_and_first_pronunciation():
    m=load()
    frontend=m.make_frontend()
    assert frontend.phonemize('cat')==['K','AE','T']
    assert frontend.oov_policy=='strict'


def test_raw_output_inside_git_repository_is_refused():
    m=load()
    path=m.ROOT/'.pytest_cache'/'psup-forbidden-output-test'
    assert not path.exists()
    try:
        with pytest.raises(ValueError):
            m.create_output(path)
    finally:
        if path.exists():
            path.rmdir()


def test_final_output_creation_refuses_any_existing_directory(tmp_path):
    m=load()
    path=tmp_path/'run'
    m.create_output(path)
    (path/'original.txt').write_text('retain')
    with pytest.raises(FileExistsError):
        m.create_output(path)
    assert (path/'original.txt').read_text()=='retain'
