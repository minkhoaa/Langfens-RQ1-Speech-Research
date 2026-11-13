# RQ1-METHOD-BENCH-02 — Next research

## Decision

Do not integrate the new window-level methods into U2 yet. The current H04 acoustic scores and frame-decoded exact-gap insertion method are negative. Continue with a genuine phoneme-CTC model only after obtaining checkpoint weights with verified training split, blank/token contract, and phone inventory mapping. Keep H04 closed unless a materially different acoustic alternative mechanism can beat unrestricted alternatives and the TRAIN-only prior.

## Evidence behind the decision

1. The optional-phone Viterbi score increased O AP from 0.03677 to 0.08793 versus its matched forced-path score on DEV_SEARCH, but remains below the Gold-Only O AP 0.15943. It preserves insertion scores from the old gap head, so it did not test whether edit paths improve exact-gap insertion.
2. Frame-argmax decoded extra-phone counts mapped to exact gaps only when all minimum-edit paths agreed. It scored I AP 0.02825 versus 0.18672 for the inherited Gold-Only gap head; 1,015 gaps were ambiguous and abstained.
3. The unit bigram adaptation reduced binary target-error AP versus the matched unigram (0.11736 vs 0.14318); stop that transition variant.
4. Ordered-state OT adaptation delivered S AP 0.15842 and O AP 0.01058; the missing gap head prevents Macro-3 AP. Its sequence-level advantage remains untested because the local cache contains isolated target windows.
5. A genuine phoneme-CTC model's config/tokenizer exists in the local HF cache, but its weight files do not. Public CROTTC-IF source and corpus-specific checkpoint bundles exist; no auditable TRAIN-only bundle was used.
6. Current DEV_SEARCH contains four speakers and is repeatedly exposed; no reported result here confirms generalization.

## Ranked follow-up experiments

| Priority | Experiment | Required evidence and controls | Stop condition |
|---:|---|---|---|
| 1 | Genuine phoneme-CTC recognition + edit lattice | Checkpoint provenance, blank/token contract, eSpeak/IPA-to-LANGFENS mapping, free-running phone sequence and timestamps; report PER and S/O; explicit ambiguity and repeated-phone mapping; N+1 insertion scores including both edge gaps | Do not proceed with no verified model weights, token map, or audio-derived spans. |
| 2 | Revised H04 acoustic alternative mechanism (optional) | Must beat unrestricted acoustic alternatives and TRAIN prior on frozen full target IDs; report speaker strata and retain prior-only control | Current frame-margin configuration is closed after the 0.19487 vs 0.29923 AP-S result. |
| 3 | Train-only CROTTC/IF-MDD reimplementation from public method | Cache/weights with split-safe training, full utterance sequences, CPU/GPU budget, no corpus-specific checkpoint leakage; compare matched Gold-Only and unconditioned acoustic controls | Stop if split provenance or full sequence inputs cannot be established. |
| 4 | Text-guided acoustic-unit DTW | Audio-unit and text-to-unit model or a faithful smaller trained pair; controls: unigram surprisal, transcript-only prior, shuffled text, timing-only; evaluate target S/O and gap I separately | Stop if candidate gains are explained by phone/reference priors rather than acoustic matching. |
| 5 | Speaker-grouped repeat of a supported method | Only after a meaningful DEV_CAL/SEARCH signal: matched seeds, TRAIN speaker-grouped validation; freeze model and CAL policy before SEARCH | Require consistency across speaker groups, not a single pooled AP increase. |

## Exact-gap sequence mapping contract

For every utterance with `N` intended phones, preserve exactly `N` target IDs and `N+1` gap IDs. Map each acoustically decoded extra-phone span to the gap bounded by its adjacent reference alignments; leading and trailing unmatched phones map to gap 0 and gap N. If repeated adjacent phones or multiple insertions yield tied edit paths, retain an ambiguity flag and either emit a predeclared marginal score over tied paths or abstain. Never use gold phone labels/times to choose among inference paths. Report event-level gap coverage and exact-gap AP only when every gap has a prediction or explicit abstention.

## Broader discovery to retain

The 2026 prompt-free CROTTC/indirect-fusion framework and OTTC topology are distinct from target-centered U2. Their papers and public code justify a future sequence-level reproduction; current local results do not validate their published performance on LANGFENS. A fresh split-safe training run or artifact-backed test would change this recommendation.
