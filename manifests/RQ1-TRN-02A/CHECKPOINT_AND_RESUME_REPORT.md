# Checkpoint and resume report

- **Bounded T4 feasibility checkpoint:** saved with model, optimizer, and GradScaler states; model reloaded and CTC-head tensor equality verified. Downloaded to `checkpoints/RQ1-TRN-02A/feasibility/rq1_fold1_colab_t4_micro.pt` (local SHA256 `fb3e8608dcb1bd4023460109de4c43e2fdaa4f923c9210e3edc59dced9ad39f9`). Remote Drive presence was verified at `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-02A/feasibility/rq1_fold1_colab_t4_micro.pt`.
- This is an isolated feasibility artifact, not a training initialization; it must not be used to start the full run. Full training must initialize task heads freshly from the pinned pretrained backbone.
- The production driver writes model, optimizer, scheduler, scaler, RNG, update/micro-step counters, Git SHA, manifest checksum, configuration, and history to `last.pt` and `best.pt` after each completed epoch and supports resume.
- **Full Fold 1 training checkpoint/history: NOT GENERATED.** No full training checkpoint was stored or claimed.
