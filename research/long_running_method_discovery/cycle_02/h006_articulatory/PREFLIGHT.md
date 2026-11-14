# H-006 articulatory auxiliary learning — preflight

Status: **PROPOSED / NOT EXECUTED**. This is a CPU-side source and protocol review. It does not establish that auxiliary learning helps LANGFENS.

## Existing evidence and interface

- `tools/phonological_features.py` contains a hand-authored 16-value ARPAbet table. It encodes broad binary properties such as voiced, nasal, continuant, strident, labial, coronal, dorsal, vowel height/backness/rounding, rhotic and glottal. It also implements a weighted Hamming distance; it is not itself an auxiliary-learning experiment.
- The feature distinctions are deliberately coarse. Labial groups bilabial/labiodental together; coronal groups several tongue places together; the current vector does not distinguish all manners as categorical labels. Some English vowel contrasts share exactly the same vector (for example IY/IH, EY/EH, UH/UW); the comments mentioning lax/tense are not represented in those rows. A new auxiliary label mapping needs a versioned, reviewed contract and explicit `UNKNOWN`/masked handling rather than silently treating absent features as a phonetic negative.
- Phase 3 student input is 51D per target: 39D intended-phone one-hot plus 12 acoustic/CTC summaries; gap input is 6D. The Gold-Only GRU has 78,661 parameters and separate 3-way target / binary gap heads. It is reference-conditioned and has no frame-level contextual hidden representation beyond sequence context over cached summaries.
- Baseline receipt `experiments/student_training/STU-01-GOLD-ONLY_seed101_receipt.json`: 12 epochs, 684 updates, CUDA GTX 1650, checkpoint SHA-256 `8ce4ebe5f5e4ebb6fcbd02a9599328059e51d5ecc3bd9f52bdb66c822f994090`. Canonical EVAL-V3 release records DEV_SEARCH seed-101 AP S/O/I `0.3646021 / 0.1594330 / 0.1867152`, Macro-3 Error AP `0.2369168`; these are exposed development scores, not independent validation. Three Gold-Only seeds exist, but this Cycle 2 report makes no seed aggregation claim.
- Existing `research/teachers/substitution/train_teacher_s.py` and `research/main/exp_a103_articulatory_feature_net.py` already inject deterministic mismatch/prototype features into specialist/older models. Those are not this hypothesis: H-006 requires auxiliary-task supervision in a matched Gold-Only student and must avoid reusing gold realized-phone/error data as inference features.

## Minimal matched experiment

Freeze Gold-Only seeds 101/202/303, data and split membership, optimizer, update count (684), initialization scheme, class weights, batch order policy, checkpoint criterion (unweighted Gold DEV_CAL target+gap loss), and evaluator. Keep the main target/gap heads and their loss exactly the same. Add identically sized auxiliary heads to every arm so parameter count is matched; in the control, auxiliary losses have coefficient zero. Do not choose weights or checkpoint epochs using DEV_SEARCH.

Derive auxiliary labels from intended `target_phone` only, using a frozen ARPAbet table. `place` needs a clear vocabulary and a masked task for phones whose place is undefined; `manner` uses a documented categorical mapping; `voicing` uses a documented binary label, with silence and any ambiguous symbols explicitly masked. Fit class weights from TRAIN only. Run separately: place-only, manner-only, voicing-only, then combined only if single-task results justify it. Fix one auxiliary weight before training; an optional class-balanced/target-focused weight is a separately registered ablation.

To test audio dependence instead of copying the reference one-hot:

1. **Reference-only control:** train/evaluate the same target head with the 39D intended-phone identity and no acoustic summaries, retaining the same full event population.
2. **Audio-only auxiliary control:** auxiliary heads receive acoustic/context representation with the intended-phone one-hot blocked at the auxiliary branch; the primary task head retains its baseline reference+audio input.
3. **Audio-plus-reference arm:** expose both to the auxiliary branch. Compare with the blocked-reference auxiliary branch; a gain only in this arm may be a phone-identity shortcut.
4. Report acoustic ablation sensitivity of the main score, but do not treat a shuffled audio/reference pair as a valid label-scored prediction.

Preserve exactly N target events and N+1 gaps for every utterance. Use only TRAIN speakers for training and grouped validation; DEV_CAL can support frozen checkpoint/calibration decisions per the common protocol. DEV_SEARCH stays a development-exposed report set and cannot choose the architecture/auxiliary objective.

## Measurements and error analysis

Use EVAL-V3 for raw AP S/O/exact-gap I, Macro-3 Error AP, per-class PR/F1 and Macro-4 F1. Report paired per-seed deltas, speaker summaries, class supports, runtime/VRAM, parameter count, and all seeds. Preserve baseline AP regardless of FWR; show operating behavior at multiple policies, with no `FWR <= 0.20` success gate.

For phonetic strata, use gold realized phones only after inference as diagnostic metadata if the canonical annotation supports them; do not feed them to the model. Report substitution contrast distance and place/manner/voicing strata separately from end-to-end AP. Measure task interference by per-task gradient cosine on shared encoder parameters at fixed, predeclared batches/epochs, and report auxiliary train/validation loss. High auxiliary accuracy alone is not evidence of improved error discrimination.

## Falsification

The hypothesis is unsupported if auxiliaries fail to improve paired raw error AP / Macro-3 Error AP over matched Gold-Only, or if any gain disappears when the auxiliary branch cannot read the reference-phone one-hot. Report harmful S/O/I trade-offs and speaker inconsistency. No hypothesis result is VERIFIED without reproducible CUDA receipts, prediction/checkpoint hashes, full event-key conservation and independent evaluator/source checks.

## Execution blocker

At preflight, `nvidia-smi` could not communicate with the driver, so no CUDA job, checkpoint, prediction or metric was produced. The existing Gold-Only receipt is historical baseline evidence only. Neural CPU fallback is prohibited. Resume after checking the live driver and shared CUDA lease, then rerun the frozen Gold-Only control and single-task arms under the same three seeds.
