# Independent listening protocol for the SM13 TRAIN packet

The [playback page](AUDIO_REVIEW.html) contains 136 randomized clips. The [blind sheet](AUDIO_REVIEW_BLIND.csv) shows only the intended phone, prompt, and clip path. The keyed [evidence sheet](AUDIO_REVIEW_WITH_CLIPS.csv) contains the corpus label, raw TextGrid notation, and model score for later adjudication.

1. Two listeners independently hear every clip with the intended phone and prompt visible. They do not inspect the corpus label, raw notation, model score, or each other's answers before submitting the first pass.
2. Each listener records one code in their own column: `0` = acceptable realization, no corrective feedback; `1` = clear, actionable deviation from the intended phone; `2` = uncertain or audio insufficient. The decision is about feedback for a learner, not merely whether an accent is audible.
3. After both first passes are frozen, reveal the raw annotation and compare disagreements. Record `0`, `1`, or `2` in `adjudication`. For `2`, preserve the uncertainty; do not force a binary label.
4. Summarize agreement and outcomes separately for same-normalized-phone `*` events, clear different-phone substitutions, and high-scoring clean phones. Do not change TRAIN labels or model checkpoints directly from this convenience sample. Any systematic relabeling policy needs a separately specified corpus-wide protocol.

The source [L2-ARCTIC documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/) defines `*` as a perceived-phone deviation from standard American English. The 836 starred TRAIN labels are therefore meaningful annotation claims, though their pedagogical actionability has not been independently established by this packet. No AI-generated judgment has been entered as a reviewer vote.
