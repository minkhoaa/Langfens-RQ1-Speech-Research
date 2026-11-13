# METHOD-BENCH-01 Novel hypotheses (literature-led)

These are proposals, not measured results. Novelty is relative to the inspected LANGFENS research registry and current repository snapshot.

## MB01-H01 — Realized-phone sequence plus soft similarity supervision

**Source:** Zhou et al., Interspeech 2025, DOI 10.21437/Interspeech.2025-2417. **Mechanism:** train an acoustic phone recognizer to represent the phone actually produced; soften targets according to phonetic similarity; diagnose by alignment with the supplied canonical sequence. **Novelty:** existing systems score expected phones / mismatch heads; repository search did not find this verbatim-recognition formulation. **Why it could help:** acoustic recognition may distinguish substitutions that a target-conditioned score ranks using priors. Alignment can naturally represent deletions and inserted realized phones. **Small experiment:** use compact/frozen acoustic frame embeddings and TRAIN labels; train hard-label vs phoneme-similarity-softened phone heads, then sequence edit map to each reference event. Ensure extra phones map to exact gap boundaries. **Falsify:** no paired gain in AP S/O/I or substitutions improve only through phoneme prevalence; insertion mapping is unstable under alternative valid alignments. **Resource:** compact head low; full SSL fine-tune moderate/high (likely batch/sequence constrained on 4GB).

## MB01-H02 — Correct-speech prototype / open-set anomaly scoring

**Source:** SylPh, Jin et al., Interspeech 2025, DOI 10.21437/Interspeech.2025-1285. **Mechanism:** positive bags represent normal in-distribution syllable acoustics; pseudo-negative bags and adaptive text embeddings train contrastive separation; localize abnormality despite weak syllable labels. **Novelty:** anomaly detection from target-conditioned normal prototype departs from gold error classification and specialist fusion. **Why it could help:** each phoneme's normal acoustics can be modeled despite sparse error types; unseen error realizations may still deviate. **Experiment:** build target-conditioned normal prototypes from TRAIN correct phones only; score acoustic token residuals at phone windows and exact gaps. **Falsify:** distance captures speaker/channel or phone identity more than correctness, or fails to improve held-out-speaker ranking. **Resource:** frozen embeddings + prototype statistics cheap; contrastive model modest.

## MB01-H03 — Realized-extra-phone likelihood for exact-gap insertion

**Source:** inspired by verbatim phoneme recognition (EXT01), discrete token surprisal (EXT11), and alignment-free recognition literature. **Mechanism:** instead of classifying an insertion at an empty location, decode acoustic phone mass between adjacent canonical phone emissions; use extra-phone posterior/surprisal and duration as gap evidence. **Novelty:** explicit acoustic “extra realized segment” score, independent of gap binary classifier. **Experiment:** cached CTC/frame logits, reference phone sequence and TRAIN-derived blank/gap mapping; compare to current exact-gap score. **Falsify:** score dominated by silence/boundary duration and cannot rank annotated insertions versus clean gaps. **Resource:** inference only if logits available; no new GPU training initially.

## MB01-H04 — Confusion-graph restricted alternatives

**Source:** Parikh et al., Interspeech 2025, DOI 10.21437/Interspeech.2025-829 and 2026 ISCA listing “Domain-Aware ... Statistical Graphs”. **Mechanism:** compare the target posterior only against linguistically plausible/articulatorily related alternatives derived from TRAIN confusions/phonological graph. **Novelty:** alternative-set scoring is distinct from a flat all-phone maximum. **Experiment:** TRAIN-only confusion graph; score margin between target and plausible alternatives using frozen acoustic logits. **Falsify:** gains disappear with speaker-held-out graph or score does not improve S AP beyond priors. **Resource:** low.

## MB01-H05 — Alignment-free text-to-acoustic cross-attention as a localization proposal

**Source:** OpenPA, Chao et al. arXiv:2610.13826. **Mechanism:** use pretrained encoder-decoder cross-attention to associate expected text/phone tokens with waveform regions, avoiding forced alignment. **Novelty:** token-conditioned attention maps instead of CTC/forced boundaries. **Experiment:** only if a compact checkpoint runs locally or CPU extraction is feasible; evaluate attention overlap with event windows/gaps and add fixed attention summaries to a train-only classifier. **Falsify:** attention localization is weak, unstable across decoding contexts, or merely reproduces transcript position. **Resource:** likely high unless frozen small Whisper model and CPU used.

## Existing line, not novel: phonological attributes

Phonological-feature MDD is already represented by H-006 and prior LANGFENS literature. New papers justify refreshing evidence but do not constitute a new method family for this project. Re-test only if a concrete architecture or feature decomposition is missing.
