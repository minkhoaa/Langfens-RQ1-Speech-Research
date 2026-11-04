"""Generate lightweight TRN-04R scientific reports from executed saved metrics.

No inference, fitting, gold modification or threshold selection occurs here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ARMS=('C0','T1','T2','T3')

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadrows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def fmt(v):return 'NA' if v is None else (str(v) if isinstance(v,(str,int)) else f'{v:.9f}')
def table(headers,records):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in records])+'\n'
def write(p,s):
 for before,after in {'CPUFP32 batch1':'CPU FP32, batch size 1','in155.':'in 155.','seconds, all8endpoints':' seconds, all 8 endpoints','Earlier4parallel':'Earlier 4 parallel','immutable600utterancecache':'immutable 600-utterance cache','across allheads':'across all heads','all12explicit':'all 12 explicit','exactly3,600':'exactly 3,600','All9,956gold':'All 9,956 gold','has9,919aligned+37gold abstentions+37extra':'has 9,919 aligned + 37 gold abstentions + 37 extra','makes621gold-linked':'makes 621 gold-linked','(17TP+604FP)':'(17 TP + 604 FP)','C0\'s1138':'C0\'s 1,138','precision2.74%':'precision 2.74%','recall11.72%':'recall 11.72%','None reaches25%':'None reaches 25%','reaches25%/30%':'reaches 25%/30%','all8corrected':'all 8 corrected','Same9919aligned':'Same 9,919 aligned','all8endpoints':'all 8 endpoints','all12explicit':'all 12 explicit'}.items():
  s=s.replace(before,after)
 p.write_text(s.rstrip()+'\n')
def jsonwrite(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def scalar(r):
 m=r['metrics'];o=m['omission_dedicated']['cal_operating_points']['0.15']['counts']
 return dict(dedicated_omission_ap=m['omission_dedicated']['ap'],legacy_omission_ap=m['omission_legacy']['ap'],
 strict_deployed_omission_precision=o['precision'],all_gold_omission_recall=o['recall_all_gold'],aligned_omission_recall=o['recall_aligned'],
 false_alarms_per_1000_canonical=o['false_alarms_per_1000_full_gold'],binary_ap=m['binary']['ap'],
 legacy_substitution_ap=m['substitution_legacy']['ap'],dedicated_substitution_ap=m['substitution_dedicated']['ap'],
 greedy_per=r['recognition']['PER'],mismatched_audio_binary_ap=r['mismatched_acoustic_control']['metrics']['binary']['ap'],
 acoustic_reliance_gap=m['binary']['ap']-r['mismatched_acoustic_control']['metrics']['binary']['ap'],**{k:o[k] for k in ('TP','FP_correct','FP_sub','FN')})


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--parallel-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 repo=Path(__file__).resolve().parents[1];out=args.output;out.mkdir(parents=True,exist_ok=True)
 reports={e:read(args.raw_root/'metrics'/f'{e}.json') for e in ('fixed_budget_3600','best')}
 cells={e:r['cell_results'] for e,r in reports.items()};s={e:{a:scalar(cells[e][a]) for a in ARMS} for e in cells}
 fixed=s['fixed_budget_3600'];old=read(repo/'manifests/RQ1-TRN-04/ALL_EVALUATION_METRICS.json')
 manifest=read(args.raw_root/'predictions/RUN_MANIFEST.json')
 assert manifest['parameter_fingerprints_before']==manifest['parameter_fingerprints_after']
 assert manifest['git_dirty'] is False
 note='Primary operating point: dedicated omission score, CAL-correct 15% FPR ceiling on NCC/THV; VAL NJS/RRBI never tunes thresholds. AP is finite aligned-only, not full-gold AP. Rates are fractions unless marked %. FA/1,000 uses 9,956 full gold positions. Extra-prompt alarms are separate and unlabelled.\n'
 limits='One seed and two repeatedly used VAL speakers support exploratory descriptive contrasts only. No population-superiority, causal acoustic-localization, deployment, universal focal-loss/acoustic-feature invalidation, or new-training claim follows. Historical training-source commit, empirical FIT acoustic-normalization receipt, and frozen FIT-only prior control remain unverified/unrecovered.\n'
 headers=['Arm','Dedicated omit AP','Legacy omit AP','Precision','Recall all','Recall aligned','FA/1,000','Binary AP','Legacy sub AP','Dedicated sub AP','PER']
 keys=['dedicated_omission_ap','legacy_omission_ap','strict_deployed_omission_precision','all_gold_omission_recall','aligned_omission_recall','false_alarms_per_1000_canonical','binary_ap','legacy_substitution_ap','dedicated_substitution_ap','greedy_per']
 comparison=table(headers,[[a]+[fixed[a][k] for k in keys] for a in ARMS])
 confusion=table(['Arm','TP','FP_correct','FP_substitution','FN all','Matched binary AP','Mismatched binary AP','Gap'],[[a]+[fixed[a][k] for k in ('TP','FP_correct','FP_sub','FN','binary_ap','mismatched_audio_binary_ap','acoustic_reliance_gap')] for a in ARMS])
 write(out/'FOUR_ARM_FIXED_BUDGET_COMPARISON.md','# Four-arm fixed-budget genuine-Z comparison\n\n'+note+'\nAll endpoints are exactly3,600 successful updates; original bytes unchanged.\n\n'+comparison+'\n'+confusion+'\n'+limits)
 best=table(headers,[[a]+[s['best'][a][k] for k in keys] for a in ARMS])
 write(out/'SELECT_BEST_COMPARISON.md','# Existing SELECT-best checkpoints — secondary only\n\n'+note+'\nSuccessful updates: C0=2247,T1=897,T2=3600,T3=2247. These unequal-exposure snapshots are not a fixed-budget factorial causal contrast. T2 best and fixed have identical model-state fingerprints despite differing serialized training-state bytes. Registered primary gates target fixed-budget, not a replacement checkpoint selected from this table.\n\n'+best+'\n'+limits)
 pr=[]
 for a in ARMS:
  for c,v in cells['fixed_budget_3600'][a]['metrics']['omission_dedicated']['cal_operating_points'].items():
   q=v['counts'];pr.append([a,c,v['threshold'],v['cal_achieved_fpr'],q['TP'],q['FP_correct'],q['FP_sub'],q['FN'],q['precision'],q['recall_all_gold'],q['recall_aligned'],q['false_alarms_per_1000_full_gold'],cells['fixed_budget_3600'][a]['extra_prompt_alarms']['s_omit']['alarms_at_thresholds'][c]])
 write(out/'PRECISION_RECALL_AND_FALSE_ALARM_REPORT.md','# CAL-selected precision, recall and false alarms\n\n'+note+'\nCAL5/CAL10 are secondary predeclared operating points, not VAL-selected substitutes for CAL15. None reaches25% precision at these three points. Abstentions are no-alarm; the eight unmapped omissions remain false negatives. No claim is made about every conceivable threshold.\n\n'+table(['Arm','CAL ceiling','Threshold','CAL achieved FPR','TP','FP_correct','FP_sub','FN all','Precision','Recall all','Recall aligned','FA/1,000','Extra-prompt alarms'],pr))
 effects=reports['fixed_budget_3600']['factorial_contrasts']
 rows=[[k,v['direction'],v['matched_contrasts']['T1_minus_C0'],v['matched_contrasts']['T2_minus_C0'],v['matched_contrasts']['T3_minus_C0'],v['interaction_T3_minus_T2_minus_T1_plus_C0']] for k,v in effects.items()]
 write(out/'FACTORIAL_EFFECT_RECOMPUTATION.md','# Matched factorial effects — fixed budget\n\n'+note+'\n'+table(['Endpoint','Favorable direction','T1-C0','T2-C0','T3-C0','T3-T2-T1+C0'],rows)+'\nT3 has favorable descriptive omission AP/precision/recall interaction terms and improves these endpoints over T1, but remains far below C0. Interaction is scale-dependent and does not prove beneficial practical synergy. Substitution-AP interactions are unfavorable. T2 improves binary/substitution AP and lowers false alarms, but worsens omission AP/precision/recall; strict Pareto domination by C0 is unsupported.\n\n'+limits)
 differences=[]
 for e in cells:
  for a in ARMS:
   previous=old[a.lower()][e]['metrics']
   for k,v in s[e][a].items():
    if k in previous and isinstance(previous[k],(int,float)):
     differences.append([e,a,k,previous[k],v,v-previous[k]])
 write(out/'ORIGINAL_VS_CORRECTED_METRICS.md','# Original versus corrected values\n\nOriginal TRN-04 numbers are preserved claims, not valid genuine-Z measurements. Changes combine protocol/input/population and score-family corrections, not new training effects. Focal dedicated channels were formerly missing and legacy AP was relabelled dedicated; old deployed focal s_error was compared with a deletion-score CAL threshold. Such differences are not same-estimand performance gains. WFC old heuristic and adjudicated estimates have no valid direct delta and are not reconstructed.\n\n'+table(['Endpoint','Arm','Metric','Original claimed','Corrected','Delta corrected-original'],differences))
 count_changes=[]
 for e in cells:
  for a in ARMS:
   previous=old[a.lower()][e]['cal_operating_points']['CAL_15%']
   current=cells[e][a]['metrics']['omission_dedicated']['cal_operating_points']['0.15']['counts']
   for oldkey,newkey in [('tp','TP'),('fp_cor','FP_correct'),('fp_sub','FP_sub'),('fn','FN')]:
    count_changes.append([e,a,newkey,previous[oldkey],current[newkey],current[newkey]-previous[oldkey]])
 changes_path=out/'ORIGINAL_VS_CORRECTED_METRICS.md'
 write(changes_path,changes_path.read_text()+'\n## Deployed count changes (CAL15)\n\nHistorical FN includes the invalid extra-eight correction; retained as a conflicting original claim, not validated accounting. Aligned-only recall had no valid original counterpart and is separately reported.\n\n'+table(['Endpoint','Arm','Count','Original claimed','Corrected','Delta'],count_changes))
 ledger={}
 for e in cells:
  ledger[e]={}
  for a in ARMS:
   directory=args.raw_root/'predictions'/f'{a}_{e}';tokens=loadrows(directory/'tokens_Z.jsonl');gold=[r for r in tokens if r['gold_canonical_index'] is not None];aligned=[r for r in gold if r['prompt_index'] is not None];missing=[r for r in gold if r['prompt_index'] is None]
   keys_canonical=[(r['speaker_id'],r['utterance_id'],r['gold_canonical_index']) for r in aligned]
   cohort=hashlib.sha256(('\n'.join('/'.join(map(str,k)) for k in sorted(keys_canonical))+'\n').encode()).hexdigest()
   historical=read(repo/'manifests/RQ1-P-SUP-BENCHMARK-V1/final-v2/t_op_fixed_budget_3600/protocol_z/metrics.json')['eligible_gold_key_sha256']
   assert cohort==historical
   ledger[e][a]=dict(gold=len(gold),aligned=len(aligned),unmapped=len(missing),extras=len(tokens)-len(gold),gold_labels=dict(Counter(str(r['operation_target']) for r in gold)),aligned_labels=dict(Counter(str(r['operation_target']) for r in aligned)),unmapped_labels=dict(Counter(str(r['operation_target']) for r in missing)),eligible_gold_key_sha256=cohort,historical_V2_cohort_equal=True,coverage=cells[e][a]['coverage'],by_speaker={sp:dict(gold=sum(r['speaker_id']==sp for r in gold),unmapped=sum(r['speaker_id']==sp for r in missing)) for sp in ('NJS','RRBI')})
 jsonwrite(out/'POPULATION_AND_ABSTENTION_LEDGER.json',dict(status='COUNTED_FROM_ACTUAL_SAVED_ROWS',cells=ledger,all_gold_FN_rule='145-TP_aligned; TP_aligned<=137',AP_policy='aligned finite scores only; no fabricated full-population AP'))
 gates=reports['fixed_budget_3600']['registered_absolute_gates'];gaterows=[]
 for a in ARMS:
  for level,v in gates[a].items():
   for metric,c in v['checks'].items():gaterows.append([a,level,metric,c['value'],c['operator'],c['threshold'],c['status']])
 ni=[]
 for a in ARMS[1:]:
  for k,margin,direction in [('binary_ap',-.005,'>='),('legacy_substitution_ap',-.005,'>='),('greedy_per',.005,'<=')]:
   delta=fixed[a][k]-fixed['C0'][k];passed=delta>=margin if direction=='>=' else delta<=margin;ni.append([a,k,delta,direction,margin,'descriptive margin met' if passed else 'descriptive margin failed'])
 write(out/'SCIENTIFIC_ACCEPTANCE_REPORT.md','# Registered scientific acceptance recovery\n\nAll four fixed-budget arms fail both Level A and Level B. No arm reaches25%/30% strict omission precision at the primary operating point. C0/T2 meet recall gates; T1/T3 do not. All fail primary false-alarm limits. T1 also fails the external binary-AP floor. Fixed-budget PER and legacy substitution floors pass. Every acoustic mismatch gap exceeds.010, a sensitivity control, not causal localization.\n\n'+table(['Arm','Level','Metric','Value','Rule','Threshold','Result'],gaterows)+'\n## Internal C0 descriptive non-inferiority margins\n\nThese point comparisons use registered .005 margins; they are not inferential non-inferiority tests. External absolute gates above remain distinct.\n\n'+table(['Arm','Metric','Delta vs C0','Rule','Margin','Point disposition'],ni)+'\n'+limits)
 speaker=[];phone=[]
 for a in ARMS:
  for sp,m in cells['fixed_budget_3600'][a]['by_speaker'].items():
   op=m['omission_dedicated']['cal_operating_points']['0.15']['counts'];speaker.append([a,sp,m['omission_dedicated']['ap'],m['binary']['ap'],op['precision'],op['recall_all_gold'],op['TP'],op['FP_correct'],op['FP_sub'],cells['fixed_budget_3600'][a]['recognition']['by_speaker'][sp]['PER']])
  for ph,m in cells['fixed_budget_3600'][a]['by_phone'].items():
   z=m['omission_dedicated'];op=z['cal_operating_points']['0.15']['counts'];phone.append([a,ph,z['n_population'],z['positives_scored'],z['ap'],op['TP'],op['FP_correct'],op['FP_sub'],'sparse' if z['positives_scored']<10 else 'descriptive'])
 write(out/'SPEAKER_AND_PHONEME_ANALYSIS.md','# Speaker and per-phone analysis\n\nNJS is Spanish L1/female; RRBI Hindi L1/male. Two speakers are not a Vietnamese-generalization cohort. CAL thresholds are global and unchanged for these strata. Speaker differences are a descriptive heterogeneity/uncertainty summary, not a confidence interval or independent-seed replication.\n\n'+table(['Arm','Speaker','Omit AP','Binary AP','Precision','Recall all','TP','FP_correct','FP_sub','PER'],speaker)+'\n## Per-phone strict omission error analysis\n\nAP population is correct+omission only; deployed FP_sub separately retained. Positive counts are scored omissions. Small support is flagged; no phone-level significance. Full per-phone legacy/dedicated/binary metrics are in private metrics JSON.\n\n'+table(['Arm','Gold phone','Full omission cohort','Scored omissions','Omit AP','TP','FP_correct','FP_sub','Support'],phone))
 history=[]
 for a in ('t_op_fixed_budget_3600','b_control_fixed_budget_3600'):
  m=read(repo/f'manifests/RQ1-P-SUP-BENCHMARK-V1/final-v2/{a}/protocol_z/metrics.json');history.append([a]+[m['metrics'][k].get('ap') for k in ('binary','omission_legacy','omission_dedicated','substitution_legacy','substitution_dedicated')])
 write(out/'HISTORICAL_V2_COMPARISON.md','# Matching historical V2 genuine-Z reference\n\nSame9919aligned gold keys verified by hash for all8corrected cells, same frozen prompt/CMUdict/sub_first policy, same score-family/AP populations and CPUFP32 batch1. T-OP is an external architecture/training-run reference, NOT the matched factorial control C0. B-Control subtype AP remains unsupported/null. Separate CPU executions can produce small floating-point differences; no bitwise equality to old V2 encoder features is claimed.\n\n'+table(['Historical fixed3600','Binary AP','Legacy omit AP','Dedicated omit AP','Legacy sub AP','Dedicated sub AP'],history)+'\n'+comparison+'\nFrozen FIT-only canonical prior remains unavailable; no replacement was fitted. This gap prevents a blanket full-control-completion claim.\n')
 decision='''# Revised research decision

**Disposition: limited negative result for these treatments; original scientific conclusions only partially supported.** Complete the reproducibility/provenance record and continue bounded research planning; do not deploy, unseal TEST, or automatically authorize additional training.

1. All four fixed-budget checkpoints and all four existing SELECT-best snapshots were actually evaluated under genuine written-prompt Protocol Z.
2. All9,956gold positions are accounted for in every cell.
3. Each has9,919aligned+37gold abstentions+37extra prompt positions, exactly the frozen V2 policy.
4. Every comparable historical scalar delta is in ORIGINAL_VS_CORRECTED_METRICS.md; original mislabelled focal score channels and unsupported adjudication are not valid same-estimand benchmarks.
5. T1 does not reproduce the historical near-universal omission-alarm explosion. At primary CAL15 it makes621gold-linked alarms (17TP+604FP), fewer than C0's1138, but precision2.74% and recall11.72% remain poor. The extreme old alarm claim is overturned, not repaired by a new model.
6. T2 underperforms C0 on omission AP/precision/recall, but improves binary/substitution AP and reduces false alarms. Global underperformance or strict Pareto domination is unsupported.
7. T3 has favorable descriptive omission interaction, but remains substantially worse than C0 and fails practical gates. This is not a confirmed generalizable beneficial interaction or architecture winner.
8. C0 is the strongest fixed-budget internal omission control on primary omission AP/precision/recall, not the strongest model on every endpoint and not deployment-ready.
9. No candidate improves practical omission precision over matched C0 while preserving acceptable recall and all other requirements at the registered primary operating point.
10. All-arm practical gate failure and the tested focal omission deficit are supported; near-universal focal alarms, zero beneficial interaction, global C0 superiority and mandatory adjudicated performance claims are overturned/qualified. Do not retain 'all hypotheses conclusively rejected' wording.
11. These measurements do not by themselves justify additional training. No training is authorized by this campaign.
12. Document this bounded negative result and corrected evaluator; resolve provenance/prior gaps and prospectively define a genuinely independent speaker-disjoint confirmation only if an operationally viable candidate emerges. No present arm warrants promotion to independent confirmation as a practical winner. RQ1 is not universally closed.

One seed/two repeatedly used VAL speakers cannot establish general population superiority. The tested objective/feature choices do not invalidate every focal-loss or acoustic-feature approach. Broad-event adjudicated performance is disabled absent appropriate case-level records/inclusion probabilities.
'''
 write(out/'NEXT_RESEARCH_DECISION.md',decision)
 write(out/'EXECUTIVE_REBENCHMARK_REPORT.md','# LANGFENS RQ1 — TRN-04R executive rebenchmark\n\n**Executed corrected benchmark; practical acceptance failed; scientific conclusions partially supported, not blanket PASS.**\n\nFrozen inference revision `'+manifest['git_tip']+'`; final shared-cache CPUFP32 batch1 execution completed in'+fmt(manifest['elapsed_seconds'])+'seconds, all8endpoints. Earlier4parallel runs are preserved but not final factorial evidence because one process had nonidentical encoder feature hashes. Final version shares one immutable600utterancecache across allheads, with unchanged model/source fingerprints. Independent production and standard-library saved-row recomputation agree, including global/speaker/phone AP, CAL thresholds, confusion counts and greedy PER.\n\n'+note+'\n'+comparison+'\nNo arm passes Level A/B. Historical focal alarm explosion is not reproduced; score-family fallback was invalid. T2 is not globally Pareto-dominated; T3 has favorable descriptive omission interaction without practical benefit over C0. C0 remains an omission reference, not a universal architecture champion. Unsupported broad-event HT claims are withdrawn from decision evidence.\n\n'+limits+'\nSee NEXT_RESEARCH_DECISION.md for all12explicit decisions; original-vs-corrected table, fixed/best, gates, speaker/phone analyses, provenance ledger, independent audit and private persistence receipts provide detailed evidence. Final process/durability state is in PROCESS_CHECKPOINT.md.\n')
 jsonwrite(out/'CORRECTED_METRIC_SUMMARY.json',dict(status='EXECUTED_SHARED_CACHE_GENUINE_Z',inference_source_commit=manifest['git_tip'],metrics={e:{'sha256':sha(args.raw_root/'metrics'/f'{e}.json'),'cells':s[e]} for e in cells},factorial_effects=effects,gates=gates,limits=limits))
 print('Generated scientific reports from',args.raw_root)

if __name__=='__main__':main()
