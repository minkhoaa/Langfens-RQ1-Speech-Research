# GPU feasibility — Colab T4

**Result: bounded micro-batch feasibility PASS; full-epoch/full-run feasibility NOT ESTABLISHED.**

Runtime: Google Colab CLI, Tesla T4 (14,912 MiB VRAM), torch `2.11.0+cu130`, torchaudio `2.11.0+cu130`, transformers `5.20.2`. The configured pinned model revision was loaded; sample records below are verified members of Fold 1 TRAIN only. A generated source subset archive contained only these train speakers/utterances, not validation or test.

| Real TRAIN audio pair | Durations (s) | CTC/Binary loss | Peak allocated/reserved MiB | CTC frame lengths |
|---|---:|---:|---:|---:|
| ERMS/arctic_a0098 + MBMPS/arctic_a0298 | 0.999 + 9.347 (longest Fold 1 TRAIN) | 19.9974 / 0.7999 | 525.6 / 634 | 49, 467 |
| SVBI/arctic_a0003 + MBMPS/arctic_a0298 | 3.637 + 9.347 | 18.8259 / 0.3781 | 569.5 / 638 | 181, 467 |

Both used batch size 2, one optimizer step per probe, fp16 autocast and GradScaler initial scale 4096. Losses and gradients were finite; no AMP step was skipped; short-vs-long padding yielded expected per-item frame lengths. CTC and Binary heads updated. Frozen backbone and diagnosis-head weights remained unchanged; diagnosis gradients were absent. Model+optimizer+scaler checkpoint reloaded successfully (412,355,238 bytes).

This is not proof of sustained full-dataset training safety, wall time, full validation stability, or quota availability. The initially default GradScaler scale 65,536 produced transient inf gradients on local probing; a controlled scale sweep and the Colab T4 real-audio probes showed finite gradients at 4,096. The driver therefore uses 4,096 and counts later scaler-skipped steps, aborting after 8 consecutive skipped attempts.

The T4 session disappeared after the bounded probe; its feasibility checkpoint was downloaded and verified on Drive. The exact 2,699-utterance TRAIN+validation audio/TextGrid whitelist is now staged on Drive (762 MB; all other utterance members and six test speakers excluded), ready for an authorized runtime after interactive Drive mount. No full training run occurred. Colab environment differed from `requirements-lock.txt` (torch 2.11.0 vs lock 2.14.1); compatibility for this bounded probe passed, but exact lockfile environment parity is unresolved.
