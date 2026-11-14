"""Aggregate disjoint held-speaker folds into per-seed pooled OOF EVAL-V3.1."""
from __future__ import annotations

import csv
import json
import sys
import statistics
from pathlib import Path

import numpy as np

ROOT=Path("/home/khoa/Projects/New-Method-Research")
BASE=ROOT/"experiments/sm13_three_directions_20261113"
sys.path.insert(0,str(ROOT))
from research.evaluation_v3.evaluator import UnifiedEvaluatorV3  # noqa: E402


def rows_for(model_dir: Path, tag: str, seed: int):
    target=[]; gap=[]; folds=[]
    for fold in range(4):
        stem=f"{tag}_seed{seed}_fold{fold}"
        p=model_dir/f"{stem}.npz"
        z=np.load(p,allow_pickle=False)
        ti=z["target_event_id"].astype(str); gi=z["gap_event_id"].astype(str)
        if len(set(ti))!=len(ti) or len(set(gi))!=len(gi): raise ValueError(f"duplicate IDs in {p}")
        for eid,gold,pred,prob in zip(ti,z["target_gold"].astype(str),z["target_pred"].astype(str),z["target_probabilities"]):
            ds,sp,uid,_,pos=eid.split("|")
            target.append({"dataset_id":ds,"speaker_id":sp,"utterance_id":uid,"position":int(pos),
                "gold_label":gold,"pred_label":pred,"prob_correct":float(prob[0]),
                "prob_substitution":float(prob[1]),"prob_omission":float(prob[2])})
        for eid,gold,pred,prob in zip(gi,z["gap_gold"],z["gap_pred"],z["gap_probability"]):
            ds,sp,uid,_,pos=eid.split("|")
            gap.append({"dataset_id":ds,"speaker_id":sp,"utterance_id":uid,"gap_position":int(pos),
                "gold_has_insertion":bool(gold),"pred_has_insertion":bool(pred),"prob_insertion":float(prob)})
        meta=json.loads((model_dir/f"{stem}.json").read_text())
        folds.append(meta)
    assert len({r["speaker_id"] for r in target})==12
    assert len(target)==59562 and len(gap)==61362
    tgold=[{k:r[k] for k in ("dataset_id","speaker_id","utterance_id","position","gold_label")} for r in target]
    tpred=[{k:v for k,v in r.items() if k!="gold_label"} for r in target]
    ggold=[{k:r[k] for k in ("dataset_id","speaker_id","utterance_id","gap_position","gold_has_insertion")} for r in gap]
    gpred=[{k:v for k,v in r.items() if k!="gold_has_insertion"} for r in gap]
    result=UnifiedEvaluatorV3().evaluate(tpred,gpred,tgold,ggold)
    return result,folds


def flat(seed, tag, result, folds):
    s=result["summary"]; c=result["classes"]
    row={"seed":seed,"model":tag,"targets":s["total_targets"],"gaps":s["total_gaps"],
         "utterances":s["total_utterances"],"macro3_error_ap":s["macro3_error_ap"],
         "macro3_error_f1":s["macro3_error_f1"],"joint_fwr":s["joint_fwr"],
         "false_warning_count":s["false_warning_count"],"fwr_reference_0_20_exceeded":s["fwr_reference_0_20_exceeded"]}
    for cls in ("SUBSTITUTION","OMISSION","INSERTION"):
        key=cls.lower()
        for metric in ("precision","recall","f1","ap"):
            row[f"{key}_{metric}"]=c[cls].get(metric)
    row["fold_macro3_ap"]=[x["metrics"]["macro3_error_ap"] for x in folds]
    row["fold_macro3_f1"]=[x["metrics"]["macro3_error_f1"] for x in folds]
    row["fold_joint_fwr"]=[x["metrics"]["joint_fwr"] for x in folds]
    return row


def pooled_oracle_r_at_precision(model_dir: Path, tag: str, seed: int):
    """Pooled held labels only; include whole tied score groups, never split ties."""
    target_scores=[]; target_gold=[]; gap_scores=[]; gap_gold=[]
    for fold in range(4):
        z=np.load(model_dir/f"{tag}_seed{seed}_fold{fold}.npz",allow_pickle=False)
        labels=z["target_gold"].astype(str)
        probs=np.asarray(z["target_probabilities"],dtype=float)
        target_scores.append(probs);target_gold.append(labels)
        gap_scores.append(np.asarray(z["gap_probability"],dtype=float))
        gap_gold.append(np.asarray(z["gap_gold"],dtype=bool))
    scores=np.concatenate(target_scores,axis=0);gold=np.concatenate(target_gold)
    gscore=np.concatenate(gap_scores);ggold=np.concatenate(gap_gold)
    rows=[]
    for cls,name,sc,gy in (
        ("SUBSTITUTION","S",scores[:,1],gold=="SUBSTITUTION"),
        ("OMISSION","O",scores[:,2],gold=="OMISSION"),
        ("INSERTION","I",gscore,ggold),
    ):
        order=np.argsort(-sc,kind="mergesort");total=int(gy.sum())
        tp=0;n=0;i=0;eligible=[]
        while i<len(order):
            j=i+1
            while j<len(order) and sc[order[j]]==sc[order[i]]:j+=1
            block=order[i:j];tp+=int(gy[block].sum());n+=len(block)
            prec=tp/n;rec=tp/max(1,total)
            for wanted in (.75,.80):
                if prec>=wanted:
                    eligible.append((wanted,rec,prec,n,float(sc[order[i]]),total))
            i=j
        for wanted in (.75,.80):
            choices=[r for r in eligible if r[0]==wanted]
            if choices:
                # Max recall, then highest precision, then fewer whole events.
                best=max(choices,key=lambda r:(r[1],r[2],-r[3]))
                _,rec,prec,nsel,threshold,pos=best
            else: rec=0.0;prec=None;nsel=0;threshold=None;pos=total
            rows.append({"model":tag,"seed":seed,"error_class":name,"class_name":cls,
                "requested_precision_minimum":wanted,"achieved_precision":prec,"recall":rec,
                "selected_events":nsel,"positive_support":pos,"score_threshold":threshold,
                "ties_split":False,"population":"four held folds pooled per seed",
                "status":"HELD_ORACLE_DIAGNOSTIC_NOT_DEPLOYABLE"})
    return rows


def main():
    all_dirs={
        "sm13_all13":BASE/"temporal/baseline_predictions",
        "l6_pooled_target_shared_sm13_gap":BASE/"temporal/l6_pooled_predictions",
        "temporal_target_shared_sm13_gap":BASE/"temporal/target_only_predictions",
        "full_temporal_target_gap":BASE/"temporal/full_temporal_predictions",
        "data_expansion_labelled_control":BASE/"data_expansion/control_results",
        "data_expansion_target":BASE/"data_expansion/results",
        "adapter_frozen_control":BASE/"encoder_adapter/folds",
        "adapter_lora":BASE/"encoder_adapter/folds",
    }
    tags={"sm13_all13":"sm13_scalar_mix","l6_pooled_target_shared_sm13_gap":"l6_pooled_target_shared_sm13_gap",
          "temporal_target_shared_sm13_gap":"temporal_cnn_target_shared_sm13_gap",
          "full_temporal_target_gap":"temporal_cnn_target_gap_l6",
          "data_expansion_labelled_control":"data_expansion_labelled_control",
          "data_expansion_target":"data_expansion_target",
          "adapter_frozen_control":"encoder_adapter_control_prefix",
          "adapter_lora":"encoder_adapter_lora"}
    dirs={}
    matched_adapter_pairs=all((BASE/"encoder_adapter/folds"/f"matched_pair_seed{s}_fold{f}.json").exists()
                              for s in (101,202) for f in range(4))
    for method,folder in all_dirs.items():
        if method.startswith("adapter_") and not matched_adapter_pairs:
            print(f"skip adapter branch until all matched-pair receipts exist: {method}")
            continue
        if all((folder/f"{tags[method]}_seed{s}_fold{f}.npz").exists() for s in (101,202) for f in range(4)):
            dirs[method]=folder
        elif method not in ("sm13_all13","l6_pooled_target_shared_sm13_gap"):
            print(f"skip incomplete OOF aggregate: {method}")
    rows=[]; detailed={}; oracle_rows=[]
    for method,folder in dirs.items():
        detailed[method]={}
        for seed in (101,202):
            result,folds=rows_for(folder,tags[method],seed)
            row=flat(seed,method,result,folds); rows.append(row)
            detailed[method][str(seed)]={"summary":result["summary"],"classes":result["classes"],"folds":folds}
            oracle_rows.extend(pooled_oracle_r_at_precision(folder,tags[method],seed))
    out=BASE/"shared"; (out/"baseline_pooled_oof.json").write_text(json.dumps(detailed,indent=2,allow_nan=False)+"\n")
    numeric=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k!="seed"]
    with (out/"baseline_pooled_oof.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["seed","model",*numeric]); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in w.fieldnames})
    deltas=[]
    by={(r["model"],r["seed"]):r for r in rows}
    for seed in (101,202):
        base=by[("sm13_all13",seed)]
        for method in (m for m in dirs if m!="sm13_all13"):
            candidate=by[(method,seed)]
            for metric in ("macro3_error_ap","macro3_error_f1","joint_fwr"):
                deltas.append({"seed":seed,"fold":"pooled_oof","candidate":method,"metric":metric,
                    "baseline":base[metric],"candidate_value":candidate[metric],"delta":candidate[metric]-base[metric]})
            for fold in range(4):
                bfold=next(x for x in detailed["sm13_all13"][str(seed)]["folds"] if x["fold"]==fold)
                cfold=next(x for x in detailed[method][str(seed)]["folds"] if x["fold"]==fold)
                for metric in ("macro3_error_ap","macro3_error_f1","joint_fwr"):
                    b=bfold["metrics"][metric];v=cfold["metrics"][metric]
                    deltas.append({"seed":seed,"fold":fold,"candidate":method,"metric":metric,
                        "baseline":b,"candidate_value":v,"delta":v-b})
    with (out/"baseline_deltas.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["seed","fold","candidate","metric","baseline","candidate_value","delta"])
        w.writeheader();w.writerows(deltas)
    seed_summary={}
    for method in dirs:
        selected=[r for r in rows if r["model"]==method]
        seed_summary[method]={}
        for metric in ("macro3_error_ap","macro3_error_f1","joint_fwr"):
            vals=[float(r[metric]) for r in selected]
            seed_summary[method][metric]={"mean":statistics.mean(vals),"min":min(vals),"max":max(vals),"values_by_seed":{str(r["seed"]):r[metric] for r in selected}}
    (out/"baseline_seed_summary.json").write_text(json.dumps(seed_summary,indent=2,allow_nan=False)+"\n")
    oracle_path=BASE/"temporal/pooled_held_oracle_precision_recall.csv"
    with oracle_path.open("w",newline="") as f:
        fields=["model","seed","error_class","class_name","requested_precision_minimum","achieved_precision","recall","selected_events","positive_support","score_threshold","ties_split","population","status"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(oracle_rows)
    print(f"wrote {out/'baseline_pooled_oof.json'} and CSVs")


if __name__=="__main__": main()

