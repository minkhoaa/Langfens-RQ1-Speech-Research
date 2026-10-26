# Blockers and next actions

## Completed

- Configured Drive remote verified and all five original production artifacts recovered read-only.
- Sizes and local SHA-256 recorded; all downloaded JSON files matched tracked Git copies byte-for-byte.
- Checkpoint dictionary metadata inspected on CPU. Execution SHA, split checksum, zero-based epoch indices, update counts, and best-ever AP are recorded.
- No official test audio/labels were accessed. No training, optimizer update, checkpoint rewrite, or remote write was performed.

## Blocking remaining acceptance

No fresh validation predictions or independent Fold 1 replay were generated in this recovery. Consequently no reproduced AP/prevalence, threshold confusion matrices, per-speaker scores, CTC edit counts, decoded-length/blank statistics, canonical-only baseline fit, or matched-versus-mismatched audio control can be certified. The saved JSON reports are historical evidence only, not substitutes for replay.

The checked-in training driver currently exits before honoring `--eval-only` when CUDA is unavailable (`audit_scripts/train_fold1_ctc_binary.py`, device guard near line 280). Do not work around this by invoking training on CPU. Implement a separate inference-only CPU entrypoint or narrow evaluation-only device allowance, enforce `model.eval()` plus `torch.inference_mode()`, strictly load the recovered best checkpoint, and audit that no parameter changes occur. Add regression tests proving inference-only behavior and absence of optimizer/backward calls. Measure runtime/RAM with bounded batches before full replay.

Although a local authorized train/validation-only archive and a cached Hugging Face model are present, the extraction/indexing/provenance and model compatibility must be checked before use. Validate that only NJS/RRBI validation records are evaluated, that canonical inputs exclude gold timing alignment, and that training-only canonical baseline fitting uses only Fold 1 TRAIN labels. Then compare fresh outputs with the saved evaluation and history artifacts and run paired controls/uncertainty analysis.

## Gate

`RQ1-TRN-02A-EVAL-R1: PARTIAL` — original checkpoints recovered and integrity metadata verified, but acceptance requires genuine independent replay and remaining controls. Do not promote to PASS until those steps are evidenced. No official test leakage occurred.
