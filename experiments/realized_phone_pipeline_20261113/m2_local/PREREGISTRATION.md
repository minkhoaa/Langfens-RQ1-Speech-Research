# M2 local realized-phone recognizer — 2026-11-13

Registered before M2 training. User explicitly replaced Colab/Kaggle with local
training. GTX 1650 4 GiB, 31 GiB host RAM; one exclusive shared GPU lease.
New compute wall budget: 3,600 seconds (smoke + training + exports); no sweep.
Development TRAIN only. Official test/dev_replication remain sealed.

Question: does additional training of the frozen H6 / IF-style MLP recognizer
retain useful realized error phones? This is a schedule experiment, not a new
architecture or an end-to-end paper replication. Keep the audited pretrained
W2V2-base-960h revision, inventory, strict supervision and speaker folds from M1.
Only the 1,243 complete realized sequences are eligible for CTC; missing or
deviation-marked phones never become fabricated gold. Evaluate MDD on all 1,800
held utterances per seed. No gold boundaries in model inputs.

Eight runs: seeds 101/202 × four 6-fit/3-cal/3-held folds. Fresh random head,
frozen encoder H6 cached fp16 -> fp32, IFStyleHead(768,384,384,41), AdamW
lr=.001, weight decay=.0001, batch 16, clip norm 5, float32 CTC. No augmentation.
32 epochs, cosine LR to .0001. Calibration PER at epochs 4,8,...,32; choose
lowest calibration PER, earliest tie. No held data in checkpoint selection.
Save epoch-8 checkpoint as a within-schedule control. It is not bitwise matched
to M1 CPU constant-LR training. Record all learning curves and optimizer updates.
Atomic last/best snapshots each epoch include model, optimizer, scheduler,
disabled scaler state, Python/NumPy/CPU/CUDA RNG, fit IDs, metadata and source
hashes; support restart of interrupted runs. No full encoder weights duplicated.

Smoke: first 12 strict fit, six cal and six held utterances of fold 0, 20 updates,
full train/predict/calibrate/evaluate/checkpoint reload. Then eight full runs.
Export greedy acoustic-only phones, hard edit N targets/N+1 gaps and cal-only
policies at FWR .1/.2/.5 for selected and epoch-8 checkpoints. Greedy edit scores
are uncalibrated; no full event posterior claim. Store original fp32-decoded
phones separately from compressed posterior snapshots.

Report pooled/per-speaker PER S/D/I/support, per-phone confusion/deletion and
insertion, strict realized substitution retention (with counts), C/S/O/I metrics,
FWR and exact-gap diagnostics. Compare with M1 frozen MLP, M1 end-to-end and
SM13, with compute/selection differences disclosed. Research gate follows M0 v2
(both-seed Macro3 AP gain >= .02 and no class AP loss > .01, or the registered
precision/recall branch). PER alone cannot promote; speaker/seed repetition is
not independent evidence. Three seeds plus fresh speakers are still required
for confirmation. If no useful error evidence, report M2 quality gate unfulfilled
and return to data/recognizer work; do not mark M3 or the goal complete.

Preserve M1's sealed GOAL hash using GOAL_before_M2.md when verifying historical
artifacts after updating the live GOAL. Do not rewrite any historical manifest.
