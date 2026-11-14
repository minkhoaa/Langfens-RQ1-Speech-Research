# Superseded preliminary KD runs

The preliminary Phase 3 KD student runs used teacher score caches whose I-F1/I-Precision source mapping was incorrect. Do not use their checkpoints, predictions, receipts, or evaluator outputs for matched scientific claims. Corrected teacher cache version 2 and provenance are at `kd/teacher_scores_v2_{train,dev_cal,dev_search}.npz` and `kd/provenance_v2_correct_i_sources.json`.

The incorrect score NPZs and fold checkpoints were preserved verbatim in `kd/superseded_wrong_i_teacher_sources/`; the prior OOF provenance is marked superseded at `research/student/kd/PHASE3_TEACHER_OOF_PROVENANCE.md`. Corrected matched arms overwrote their earlier same-name student checkpoint/prediction paths. The final per-arm receipts describe only the corrected-source, TRAIN-derived-weight, packed-length-masked runs. Historical preliminary student checkpoint files were not separately retained after those paths were deliberately replaced.
