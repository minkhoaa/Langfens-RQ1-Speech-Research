# Unified student event evaluator

Run from repository root (absolute interpreter required):

```bash
/home/khoa/Projects/New-Method-Research/.venv/bin/python experiments/student_evaluation/evaluate_student_predictions.py \
  --dev-cal experiments/student_training/ARM_seed101_dev_cal.npz \
  --dev-search experiments/student_training/ARM_seed101_dev_search.npz \
  --dev-cal-manifest data/derived/l2arctic/manifests/dev_cal.jsonl \
  --dev-search-manifest data/derived/l2arctic/manifests/dev_search.jsonl \
  --output experiments/student_evaluation/individual/ARM_seed101.json
```

Each input NPZ must contain exactly usable arrays `p_target` (N,3; CORRECT/SUBSTITUTION/OMISSION probabilities), `p_gap` (G,2; NO_INSERTION/INSERTION), `target_meta`, `gap_meta`, `target_Y` (N,), and `gap_Y` (G,). Metadata may be dict records including `speaker_id`, `utterance_id`, and event index (optional `phone_family` enables family stratification), or canonical `[utterance_id, index]` pairs; the latter infer speaker from the utterance-ID prefix. Rows preserve canonical NPZ order. IDs in DEV_CAL and DEV_SEARCH must be disjoint.

Threshold policies are selected only on DEV_CAL using 21 coarse quantiles plus fixed upper-tail quantiles (0.97, 0.98, 0.99, 0.995, 0.9975, 0.999, 0.9995, 0.9999) and a no-warning nextafter(max) threshold. The selection objective is macro-3 error F1 = mean(SUB F1, OMISSION F1, exact-gap INSERTION F1), with macro-4 only a tie-breaker, subject to each empirical joint-FWR budget. Protocol is `student-eval-v2-upper-tail`; candidate counts are recorded. Frozen DEV_SEARCH reports the actual count and explicitly flags realized budget breaches; it is never retuned. The output includes one always-CORRECT/no-warning control, per-class counts/PRF1/AUPRC/ECE/Brier, correct/error confusion, speaker summaries, phone-family breakdowns joined by canonical target event ID from the DEV_SEARCH manifest, and prediction warning ledger plus event-ID hashes. Gold phones are used only for evaluation stratification, never prediction. Joint false warnings follow UnifiedEvaluator semantics: each target event predicted as an incorrect error type contributes one class FP, and each gap false alarm contributes once. Onset thresholds and floating-score quantile-grid resolution are empirical, not certified optimal.

Both DEV_CAL and DEV_SEARCH arrays are checked for complete canonical event order and exact labels against their authorized split manifests. Output event keys are `l2arctic:speaker:utterance:target|gap:index`. The module evaluates prediction probability files; it does not create student predictions or claim student training. For the corrected all-arm run, first run `run_corrected_v2_evaluation.py`; it refuses to evaluate unless receipt/checkpoint/trainer/prediction hashes and checkpoint seed metadata pass preflight. It then selects policies on DEV_CAL and writes v2 JSONs in `individual/`. The all-arm runner also performs canonical-manifest checks. Use `summarize_student_experiments.py` to write metrics, Pareto, paired-seed/speaker tables, and detailed results; `evaluate_historical_controls.py` plus `summarize_historical_controls.py` re-score complete frozen control ledgers. Frozen historical decisions are distinct from DEV_CAL-recalibrated student policies. DEV_SEARCH is development-exposed and exploratory.
