# Independent Fold 1 validation replay

Status: **NOT RUN — BLOCKED**. This is not a replay report and contains no fresh predictions.

The evaluation driver currently rejects CPU even with `--eval-only`. A training/evaluation entrypoint also creates an optimizer and includes training code, so merely bypassing its CUDA guard is not sufficient evidence of an isolated inference-only protocol. This recovery deliberately did not invoke the driver, perform a backward pass, execute an optimizer step, alter checkpoints, or use official test data.

No evaluation-only script or tests are claimed as complete here. Before replay, implement and test an independent entrypoint that loads `best.pt` strictly, constructs the registered model using the pinned pretrained revision, calls `eval()`, runs exclusively under `torch.inference_mode()`, and asserts model parameters remain unchanged. It must enumerate exactly Fold 1 validation (300 utterances, NJS/RRBI), use bounded batches, and write prediction-level provenance sufficient to independently recompute Binary AP/prevalence, threshold sweep and confusion matrices, per-speaker metrics, greedy CTC PER/S/D/I, decoded length ratio, and blank fraction.

Required subsequent outputs: fresh validation predictions or a suitably auditable prediction ledger; a train-only canonical-phone empirical prior; and matched-versus-mismatched waveform control predictions. Compare these to the original report/history without substituting saved metrics. Record peak RAM and wall time. None of these results is available from this checkpoint recovery.
