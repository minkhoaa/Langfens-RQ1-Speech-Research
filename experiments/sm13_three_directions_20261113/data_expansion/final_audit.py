from __future__ import annotations
import csv,json
from pathlib import Path
HERE=Path('/home/khoa/Projects/New-Method-Research/experiments/sm13_three_directions_20261113/data_expansion')
folds=list(csv.DictReader((HERE/'FOLD_SEED_RESULTS.csv').open()))
pooled=json.loads((HERE/'POOLED_OOF_RESULTS.json').read_text())
assert len(folds)==8
assert len(pooled['results'])==4
rows=[]
for r in folds:
    assert int(r['optimizer_updates']) in (696,708)
    assert float(r['candidate_ap_insertion_exact_gap'])==float(r['control_ap_insertion_exact_gap'])
    assert int(r['pseudo_pred_correct'] if 'pseudo_pred_correct' in r else r['teacher_pseudo_class_correct']) + int(r['teacher_pseudo_class_substitution']) + int(r['teacher_pseudo_class_omission']) == int(r['pseudo_target_rows'])
    receipt=json.loads((HERE/'results'/f"sm13_scalar_mix_seed{r['seed']}_fold{r['fold']}_training_receipt.json").read_text())
    assert receipt['pseudo_data']['teacher_confidence_min']>=0.90
    assert receipt['target_train']['optimizer_updates']==int(r['optimizer_updates'])
    rows.append({'fold':int(r['fold']),'seed':int(r['seed']),'updates':int(r['optimizer_updates']),'pseudo_target_rows':int(r['pseudo_target_rows']),'insertion_unchanged':True,'candidate_fwr':float(r['candidate_joint_fwr'])})
for r in pooled['results']:
    m=r['metrics']
    assert m['total_utterances']==1800 and m['total_targets']==59562 and m['total_gaps']==61362
    assert r['classes']['INSERTION']['ap']==m['ap_insertion_exact_gap']
cache_audit=json.loads((HERE/'CACHE_LEAKAGE_AUDIT.json').read_text())
assert cache_audit['unique_extra_utterances_total']==600
candidate_fwr={r['seed']:r['metrics']['joint_fwr'] for r in pooled['results'] if r['system']=='candidate'}
assert all(candidate_fwr[s] > 0.20 for s in (101,202))
summary={'status':'PASS','paired_fold_seeds':len(folds),'pooled_seed_systems':len(pooled['results']),'pooled_topology':{'utterances':1800,'target_events_N':59562,'insertion_events_N_plus_1':61362},'all_insertions_unchanged':True,'unique_extra_utterances_across_12_speakers':600,'candidate_pooled_realized_fwr':candidate_fwr,'candidate_exceeds_B02_both_seeds':True,'folds':rows}
(HERE/'FINAL_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'paired_fold_seeds':len(folds),'pooled_systems':len(pooled['results']),'topology':summary['pooled_topology']},indent=2))
