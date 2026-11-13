## Review

- Correct: The KD repair uses each event class’s marginal from the full student softmax, applies Bernoulli KL only to valid slots, and averages dual insertion-teacher losses (`research/student/train_kd_student.py`). Tests cover analytic KL, padding/gradient invariance, dual-teacher averaging, and empty-reference inference (`research/student/test_student.py`).
- Correct: OOF teacher scores use speaker-grouped TRAIN folds; calibration is fitted on DEV_CAL, and DEV_SEARCH is only transformed/scored (`research/student/kd/run_teacher_oof.py`; `experiments/student_training/kd/provenance_v2_correct_i_sources.json`).
- Correct: The evaluator checks event order and labels against canonical DEV_CAL/DEV_SEARCH manifests, selects thresholds on DEV_CAL using macro-3 error F1, counts wrong-type target warnings and gap false alarms, and reports DEV_SEARCH budget breaches (`experiments/student_evaluation/evaluate_student_predictions.py`). The recorded corrected run covers 14 arm/seed pairs and reports no breaches (`experiments/student_evaluation/PHASE3_EVALUATOR_PROVENANCE.json`).
- Correct: The audio adapter accepts audio and intended phones, produces indexed target and N+1 gap probabilities, and handles empty references. Its receipt records a local CUDA sample and identifies that orthographic text-to-phone conversion is outside the adapter (`research/student/infer_student.py`; `experiments/student_training/student_audio_inference_receipt.json`).
- Limitations: DEV_SEARCH is exploratory and limited to four speakers; OOF teacher heads are economical replicas, not equivalent reproductions of the original training budgets. Results appropriately do not establish KD superiority.

No issues found.

- Merge verdict: OK with notes.