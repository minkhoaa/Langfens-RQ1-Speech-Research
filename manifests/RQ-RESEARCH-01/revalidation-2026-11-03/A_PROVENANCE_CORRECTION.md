## Lane A report — provenance correction addendum

**Correction to the preceding report:** E1 was **trained**, not an untrained model. I inspected the additional read-only artifacts supplied by the parent:

- `/tmp/e1_train_output/langfens-rq-model-01-e1-fold1-train.log`: the epoch-6 record reports `successful_updates_total: 3600` and `skipped_amp_attempts_total: 0`; the final endpoint reports `COMPLETED_3600_UPDATES`.
- `/tmp/e1_train_output/checkpoints/fold1/e1_fold1_alignment_aware/training_endpoint_report.json`: confirms Fold 1, 3,600 successful updates, epoch index 5, and fixed-budget SHA-256 `3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24`.
- That SHA matches the E1 checkpoint recorded in the archived evaluation report. The benchmark’s **inference-only** description applies to the evaluation job, not E1’s training history.

Thus the archived E0/E1 comparison is a trained E1 fixed-budget checkpoint evaluated on Fold-1 validation—not an untrained E1 comparison. The benchmark’s recorded E1 deletion AP, recall-at-FPR, binary AP, CTC PER, and zero deletion skip mass remain archive evidence; I did not inspect weights or rerun evaluation. The archived training history also records a best validation binary AP of 0.3156, while the later benchmark evaluation records 0.309293; this is a checkpoint/metric-provenance detail that should be reconciled before interpreting the difference, not treated as proof of a new effect.

I also inspected `/tmp/langfens-thesis-normalized-20261103.txt`, which corroborates the thesis scope excerpts previously cited: phoneme-level acoustic assessment on referenced, labeled speech; substitutions, final-consonant deletion, and confusable pairs; confidence-aware corrective feedback; and RQ2’s separate Multi-Agent exercise-generation scope. The separate RQ1/omission proposal remains **not located**.

**Disposition unchanged:** research/pilot consideration only, not execution authorization. The E1 training provenance is now verified from the specified logs and endpoint JSON; E1’s effectiveness remains limited to the reported validation evidence and does not establish that CTC blank or a decoder deletion equals gold omission.