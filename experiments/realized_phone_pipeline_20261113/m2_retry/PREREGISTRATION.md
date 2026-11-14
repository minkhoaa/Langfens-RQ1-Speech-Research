# M2 retry: acoustic evidence, temporal context and regularization

2026-11-13. User explicitly authorized trying training again, locally.
Registered before any retry training. GTX1650 4GiB, one exclusive GPU lease;
new wall budget 3600 seconds for smoke/train/exports, no further sweep.

Correction: prior m2_report.py hardcodes M2_quality_gate_complete=False and
uses the MDD AP advancement branch to discuss M2. This cannot itself test M2's
GOAL §8 criterion: useful acoustic evidence of real error phones. Keep that
historical receipt immutable; assess M2 and MDD advancement separately here.
There is no agreed numerical product acceptance or absolute M2 PER threshold.
Do not retroactively pretend this new assessment was registered for old runs.

Prior evidence: near-zero fit CTC while calibration PER plateaus at 26–27%;
strict realized S retention 47–48%, vowels 33–36%, inserted phones 40–41%.
These support an overfitting hypothesis, not proof that H6 or all fine-tuning
cannot work. Retry changes local temporal context AND uses regularization.

Fresh matched arms, 4 folds × seeds101/202 each, interleaved per fold/seed:
- pointwise: shared Linear768->384/LeakyReLU/.2 dropout; depthwise kernel1;
  Linear384->384/LeakyReLU/dropout residual; LayerNorm; Linear384->41.
- temporal: identical except depthwise kernel5, same-centered padding.
  Actual context is ±2 encoder frames (20ms/frame); frozen H6 itself contextual.
  Both depthwise layers initialize as identity; all shared initial weights and
  initial dropout RNG state match. Parameter difference is 1536, disclose it.
  Mask zero-padded frames before convolution. Verify padding-invariant logits
  and finite CTC backward; no TextGrid boundaries are inputs.

Both use audited continuous H6 fp16->fp32 features from the frozen pinned
encoder, exact same 1243 complete-sequence supervision population, 6/3/3 TRAIN
speaker folds, AdamW lr .001, weight_decay .001, cosine to .0001, clip5,
batch16, fp32 CTC, 32 epochs. Cal PER at4/8/.../32, earliest minimum; full1800
held MDD population/seed. Same train order/update count, no augmentation,
no held tuning, no missing/deviation-phone fabrication. Test/dev_rep sealed.
These differ from old M2 in residual architecture/dropout/weight decay; only
the new matched pair supports temporal-context attribution. Old temporal CNN
used SM13 target/gap supervision; this retry uses continuous realized-sequence
CTC, not a renamed repeat of that exact experiment.

Smoke each arm:12 fit/6cal/6held,20updates, full train/predict/calibrate/
evaluate/checkpoint reload. Reuse isolated, immutable M2 local harness, with
honest architecture/source metadata; each epoch atomic resumable last/best.
Retain epoch8 controls and all failed attempts; don't overwrite old results.

Acoustic evaluation (M2): greedy phones, independent PER S/D/I, per-speaker/
phone confusion and retention. Report realized-gold alignment token retention
AND direct source-slot substitution retention / exact-gap insertion phone
recovery. Hard alignment to supplied reference is only a diagnostic adapter.
Compare with reference-only output and a fixed same-prompt wrong-speaker
audio control: group by prompt suffix AND identical supplied reference phones,
sort held IDs and rotate predictions by one within each group of >=2. A held
speaker's supervision is never training input. For fairness, compare natural
and swapped hypotheses on exactly the same eligible error events; disclose
coverage and zero/unsupported groups. The swapped control holds prompt and
lexical identity constant; its error realization can differ between speakers.

Evidence interpretation: positive audio-specific recovery of source error
phones over prompt-only and same-prompt swapped control, consistent across
both seeds and reported by speaker, supports M2's useful-evidence criterion
for studying M3. Report speaker-level descriptive bootstrap intervals; do not
treat overlapping fold/seed runs as independent or claim significance.
If the paired control is inconclusive, say so; no hardcoded failure/success.
This criterion does not establish product quality or complete M3–M6.

MDD evaluation separately: original frozen policies at B=.1/.2/.5; full C/S/O/I
precision/recall/F1/AP/support/FWR and hard exact-gap diagnostics. Apply old
AP advancement branch vs old M2 but never substitute it for the M2 acoustic
criterion. Binary hard-edit scores + hash tie breaks are not calibrated event
posteriors, so weak operating recall can reflect adapter/policy limitations
as well as recognizer errors. Do not claim a proven causal explanation.

Preserve old M2 seal with GOAL_before_retry.md when live GOAL changes. Retain
all old code/manifests/checkpoints. Report environment, parameter/update and
wall/VRAM costs, seed/speaker distribution; no runtime-speed superiority claim.
