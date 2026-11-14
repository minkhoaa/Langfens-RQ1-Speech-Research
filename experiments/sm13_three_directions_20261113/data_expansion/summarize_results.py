"""Pair the data-expansion candidate and labelled-only fold receipts."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/khoa/Projects/New-Method-Research")
sys.path.insert(0, str(ROOT))
from research.evaluation_v3.evaluator import UnifiedEvaluatorV3  # noqa: E402

HERE = ROOT / "experiments/sm13_three_directions_20261113/data_expansion"
RESULTS = HERE / "results"
CONTROL = HERE / "control_results"
CLASSES = ("SUBSTITUTION", "OMISSION", "INSERTION")
METRICS = (
    "macro3_error_ap", "macro3_error_f1", "joint_fwr", "false_warning_count",
    "ap_substitution", "ap_omission", "ap_insertion_exact_gap",
    "total_targets", "total_gaps", "total_utterances",
)


def get_pair(seed: int, fold: int):
    base_path = CONTROL / f"data_expansion_labelled_control_seed{seed}_fold{fold}.json"
    candidate_path = RESULTS / f"data_expansion_target_seed{seed}_fold{fold}.json"
    receipt_path = RESULTS / f"sm13_scalar_mix_seed{seed}_fold{fold}_training_receipt.json"
    if not (base_path.exists() and candidate_path.exists() and receipt_path.exists()):
        return None
    base = json.loads(base_path.read_text())
    cand = json.loads(candidate_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    row = {"seed": seed, "fold": fold,
           "fit_speakers": ";".join(cand["fit_speakers"]),
           "calibration_speakers": ";".join(cand["calibration_speakers"]),
           "held_speakers": ";".join(cand["held_speakers"]),
           "pseudo_target_rows": receipt["pseudo_data"]["pseudo_target_rows_retained"],
           "teacher_pseudo_class_correct": receipt["pseudo_data"]["teacher_predicted_class_counts"]["CORRECT"],
           "teacher_pseudo_class_substitution": receipt["pseudo_data"]["teacher_predicted_class_counts"]["SUBSTITUTION"],
           "teacher_pseudo_class_omission": receipt["pseudo_data"]["teacher_predicted_class_counts"]["OMISSION"],
           "pseudo_confidence_min": receipt["pseudo_data"]["teacher_confidence_quantiles"][0],
           "pseudo_confidence_q25": receipt["pseudo_data"]["teacher_confidence_quantiles"][1],
           "pseudo_confidence_median": receipt["pseudo_data"]["teacher_confidence_quantiles"][2],
           "pseudo_confidence_q75": receipt["pseudo_data"]["teacher_confidence_quantiles"][3],
           "pseudo_confidence_max": receipt["pseudo_data"]["teacher_confidence_quantiles"][4],
           "optimizer_updates": receipt["target_train"]["optimizer_updates"],
           "candidate_elapsed_seconds": receipt.get("elapsed_seconds"),
           "labeled_fit_exposures": receipt["target_train"]["labeled_fit_exposures"],
           "pseudo_exposures": receipt["target_train"]["pseudo_exposures"]}
    for tag, obj in (("control", base), ("candidate", cand)):
        metrics = obj["metrics"]
        for name in METRICS:
            row[f"{tag}_{name}"] = metrics.get(name)
        for cls in ("CORRECT", *CLASSES):
            cm = obj["classes"][cls]
            for name in ("precision", "recall", "f1", "ap"):
                row[f"{tag}_{cls.lower()}_{name}"] = cm.get(name)
    for name in METRICS:
        a, b = row.get(f"candidate_{name}"), row.get(f"control_{name}")
        row[f"delta_{name}"] = a - b if a is not None and b is not None else None
    for cls in ("CORRECT", *CLASSES):
        for name in ("precision", "recall", "f1", "ap"):
            a, b = row.get(f"candidate_{cls.lower()}_{name}"), row.get(f"control_{cls.lower()}_{name}")
            row[f"delta_{cls.lower()}_{name}"] = a - b if a is not None and b is not None else None
    row["candidate_oracle_diagnostics_only"] = json.dumps(cand.get("oracle_fixed_precision_diagnostics_only", {}), sort_keys=True)
    return row


def mean_sd(values):
    xs = [float(x) for x in values if x is not None]
    if not xs:
        return "NA"
    return f"{statistics.mean(xs):.4f} ± {statistics.stdev(xs) if len(xs)>1 else 0.0:.4f}"


def load_fold_predictions(path: Path):
    with np.load(path) as z:
        ti, tg, tp, td = z["target_event_id"], z["target_gold"], z["target_probabilities"], z["target_pred"]
        gi, gg, gp, gd = z["gap_event_id"], z["gap_gold"], z["gap_probability"], z["gap_pred"]
        pt=[]; gt=[]; pg=[]; ggold=[]
        for key,gold,pred,prob in zip(ti,tg,td,tp):
            ds,sp,uid,_,pos=str(key).split("|")
            pt.append({"dataset_id":ds,"speaker_id":sp,"utterance_id":uid,"position":int(pos),
                       "pred_label":str(pred),"prob_correct":float(prob[0]),
                       "prob_substitution":float(prob[1]),"prob_omission":float(prob[2])})
            gt.append({"dataset_id":ds,"speaker_id":sp,"utterance_id":uid,"position":int(pos),"gold_label":str(gold)})
        for key,gold,pred,prob in zip(gi,gg,gd,gp):
            ds,sp,uid,_,pos=str(key).split("|")
            pg.append({"dataset_id":ds,"speaker_id":sp,"utterance_id":uid,"gap_position":int(pos),
                       "pred_has_insertion":bool(pred),"prob_insertion":float(prob)})
            ggold.append({"dataset_id":ds,"speaker_id":sp,"utterance_id":uid,"gap_position":int(pos),
                          "gold_has_insertion":bool(gold)})
    return pt,pg,gt,ggold


def pooled_result(seed: int, system: str):
    paths=[]
    for fold in range(4):
        if system == "control":
            paths.append(CONTROL/f"data_expansion_labelled_control_seed{seed}_fold{fold}.npz")
        else:
            paths.append(RESULTS/f"data_expansion_target_seed{seed}_fold{fold}.npz")
    if not all(p.exists() for p in paths):
        return None
    pt=[];pg=[];gt=[];gg=[]
    for path in paths:
        a,b,c,d=load_fold_predictions(path);pt.extend(a);pg.extend(b);gt.extend(c);gg.extend(d)
    ev=UnifiedEvaluatorV3().evaluate(pt,pg,gt,gg)
    return {"seed":seed,"system":system,"metrics":ev["summary"],"classes":ev["classes"]}


def main():
    pairs = [r for fold in range(4) for seed in (101, 202)
             if (r := get_pair(seed, fold)) is not None]
    if not pairs:
        raise SystemExit("No complete candidate/control pairs found")
    csv_path = HERE / "FOLD_SEED_RESULTS.csv"
    fields = list(pairs[0])
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(pairs)

    pooled = [r for seed in (101, 202) for system in ("control", "candidate")
              if (r := pooled_result(seed, system)) is not None]
    (HERE / "POOLED_OOF_RESULTS.json").write_text(json.dumps({
        "standard": "EVAL-V3.1",
        "pooling": "Within each seed and system, concatenate held predictions from all four outer folds after fold-specific thresholds are frozen, then evaluate once.",
        "results": pooled,
    }, indent=2, allow_nan=False) + "\n")

    n_expected = 8
    lines = [
        "# Fit-speaker pseudo-label data expansion",
        "",
        "**Question.** Does adding a fixed 50 previously unannotated prompted utterances per fit speaker with high-confidence, fold-fit-teacher soft pseudo-targets improve exact target-event prediction at a matched optimizer-update budget?",
        "",
        f"**Paired runs available:** {len(pairs)}/{n_expected} fold-seed pairs. Primary results pool held predictions across all four folds separately for each seed and system, then run EVAL-V3.1 once. Fold-level results are supplementary; seed means are descriptive, not confidence intervals.",
        "",
        "## Pooled out-of-fold results",
        "",
        "Each seed row pools all 1,800 distinct held-out utterances, exact N target events, and exact N+1 gaps. Per-fold calibration-frozen decisions are preserved; continuous scores are pooled only after held predictions are fixed. Candidate insertion probabilities are unchanged shared baseline scores.",
        "",
        "| Seed | System | Utterances | Targets | Gaps | Macro-3 AP | Macro-3 F1 | FWR | Sub AP | Omi AP | Ins AP |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in pooled:
        m=item["metrics"]
        lines.append(f"| {item['seed']} | {item['system']} | {m['total_utterances']} | {m['total_targets']} | {m['total_gaps']} | {m['macro3_error_ap']:.4f} | {m['macro3_error_f1']:.4f} | {m['joint_fwr']:.4f} | {m['ap_substitution']:.4f} | {m['ap_omission']:.4f} | {m['ap_insertion_exact_gap']:.4f} |")
    pooled_index={(item['seed'],item['system']):item['metrics'] for item in pooled}
    mean_values={system:{metric:statistics.mean(pooled_index[(seed,system)][metric] for seed in (101,202))
                         for metric in ('macro3_error_ap','macro3_error_f1','joint_fwr')}
                 for system in ('control','candidate')}
    candidate_fwr=[pooled_index[(seed,'candidate')]['joint_fwr'] for seed in (101,202)]
    lines.extend([
        "",
        f"Across the two seeds, averaging the per-seed pooled values gives control→candidate Macro-3 AP {mean_values['control']['macro3_error_ap']:.4f}→{mean_values['candidate']['macro3_error_ap']:.4f}, Macro-3 F1 {mean_values['control']['macro3_error_f1']:.4f}→{mean_values['candidate']['macro3_error_f1']:.4f}, and realized FWR {mean_values['control']['joint_fwr']:.4f}→{mean_values['candidate']['joint_fwr']:.4f}.",
        f"Candidate Macro-3 AP is lower for both seeds. Although thresholds were frozen using calibration-only B=0.2, candidate realized FWR exceeds 0.20 for both seeds ({candidate_fwr[0]:.4f} and {candidate_fwr[1]:.4f}); this experiment does not pass that operating target. The metric changes are mixed and do not establish a consistent gain from this self-training expansion.",
    ])
    lines.extend(["", "### Pooled class P/R/F1/AP", "",
                  "| Seed | System | Class | Precision | Recall | F1 | AP |", "|---:|---|---|---:|---:|---:|---:|"])
    for item in pooled:
        for cls in ("CORRECT", *CLASSES):
            c=item["classes"][cls]
            lines.append(f"| {item['seed']} | {item['system']} | {cls.title()} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['ap']:.4f} |")
    lines.extend(["", "### Fold-seed variation", "",
                  "These per-held-group rows are supplementary and show variation between folds.", "",
                  "| Fold | Seed | Held speakers | Control AP | Candidate AP | Δ AP | Control FWR | Candidate FWR | Δ FWR | Pseudo rows | Teacher-predicted pseudo C/S/O |",
                  "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for r in pairs:
        lines.append(f"| {r['fold']} | {r['seed']} | {r['held_speakers']} | {r['control_macro3_error_ap']:.4f} | {r['candidate_macro3_error_ap']:.4f} | {r['delta_macro3_error_ap']:+.4f} | {r['control_joint_fwr']:.4f} | {r['candidate_joint_fwr']:.4f} | {r['delta_joint_fwr']:+.4f} | {r['pseudo_target_rows']} | {r['teacher_pseudo_class_correct']}/{r['teacher_pseudo_class_substitution']}/{r['teacher_pseudo_class_omission']} |")
    fit_counts = sorted({r["optimizer_updates"] for r in pairs})
    lines.extend([
        "",
        "## Protocol and provenance",
        "",
        "- Outer folds use `GROUPS[f]` as held, `GROUPS[(f+1)%4]` as calibration, and the remaining six TRAIN speakers as fit. No held/calibration speaker audio enters pseudo-labeling or fit adaptation.",
        "- Exactly 50 unique extra utterances per fit speaker were selected before teacher prediction by SHA-256 ordering of nonmanifest utterance IDs. Extra acoustic features are cached once per speaker and reused only on folds where that speaker is in the fit set.",
        "- A fold-fit SM13 scalar-mixing target teacher generates three-class probabilities. Only phone rows with maximum teacher probability ≥0.90 are used, as soft pseudo-targets. The prompt supplies intended phones for alignment/reference vectors and is not an actual-pronunciation correctness label.",
        "- Candidate training uses the same target head, class weights, 12 epochs, batch size 512, AdamW learning rate 0.001, weight decay 0.01, and optimizer-update count as the labelled-only target baseline. Each batch has 75% labelled fit rows and 25% pseudo rows, so pseudo examples replace labelled exposures; this is reported in `FOLD_SEED_RESULTS.csv`.",
        "- The insertion head and its continuous probabilities are unchanged from the shared fold baseline. EVAL-V3.1 scores exact N target events and N+1 gaps, with thresholds frozen on calibration speakers under B=0.2. Held labels only enter EVAL-V3.1. Oracle fixed-precision metrics are diagnostics only.",
        "- Source: L2-ARCTIC v5.0 TRAIN speaker archives from the pinned public mirror revision in `SOURCE_MANIFEST.json`, CC BY-NC 4.0; source licence, README, checksums, per-utterance hashes, extraction receipts, pseudo-target files, and keyed held predictions are retained in this directory.",
        "- Per-run pseudo confidence min/Q25/median/Q75/max and teacher-predicted C/S/O counts are in `FOLD_SEED_RESULTS.csv`; full row-level teacher probability vectors and confidence values are in each pseudo manifest and retained-pseudo NPZ.",
        f"- Update counts observed: {', '.join(map(str,fit_counts)) or 'none'}.",
        "",
        "## Limits",
        "",
        "This tests only the within-TRAIN-speaker pseudo-label self-training subcase of data expansion: it adds 600 extra clips across the same 12 speakers, not new speaker diversity or an external corpus, and no human error annotation. The 0.90 confidence filter and pseudo classes do not establish pseudo-label correctness because the extra utterances have no adjudicated realized-phone labels. Any benefit or harm is limited to the selected 50 utterances per fit speaker, this frozen teacher, and this training schedule. The eight fold-seed runs overlap in speakers and are not independent samples.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "for fold in 0 1 2 3; do",
        "  flock -n experiments/sm13_three_directions_20261113/gpu.lock env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/sm13_three_directions_20261113/data_expansion/extract_pseudo_features.py --fold \"$fold\"",
        "done",
        "for fold in 0 1 2 3; do",
        "  for seed in 101 202; do",
        "    flock -n experiments/sm13_three_directions_20261113/gpu.lock env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/sm13_three_directions_20261113/data_expansion/train_eval.py --fold \"$fold\" --seed \"$seed\"",
        "  done",
        "done",
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/sm13_three_directions_20261113/data_expansion/summarize_results.py",
        "```",
        "",
        "The actual run log is stored in `RUN_LOG.md`; per-run commands and hardware are recorded there.",
    ])
    (HERE / "BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path} and BENCHMARK_REPORT.md from {len(pairs)} paired runs")


if __name__ == "__main__":
    main()
