# Data expansion: fit-speaker pseudo-labeling

## Question

Does adding a fixed 50 prompt-matched, previously unannotated utterances per *fit* speaker, labeled with high-confidence soft pseudo-targets from a fold-specific SM13 teacher, improve held-speaker exact target-event prediction at the same optimizer-update budget?

## Scope and provenance

- The core labelled corpus is `data/derived/l2arctic/manifests/train.jsonl` (1,800 utterances; 150 per each of 12 TRAIN speakers).
- The normalized local audio contains only the 3,599 utterances in the existing split manifests. The experiment needs additional TRAIN-speaker utterances.
- The public mirror is `chikingsley/l2-arctic-release-v5.0`, pinned to repository revision `cc02fd37197966e5de11ad737f7affd577c61992`. The mirror reports `cc-by-nc-4.0`; official corpus information and use limits are in the L2-ARCTIC PSI Lab site. Source archive SHA-256 values and per-utterance SHA-256 values will be recorded after download.
- For every speaker, select exactly 50 unique WAVs absent from `train.jsonl`, sorted by SHA-256 of the canonical utterance ID (then ID as a tie-break). Selection is fixed before model predictions are generated. No non-TRAIN speaker archive is acquired or read for this experiment.

## Fold isolation

For outer fold `f`, the fit set is the six speakers outside `GROUPS[f]` and `GROUPS[(f+1)%4]`. Both calibration speakers and held speakers, including their extra unannotated recordings, are excluded from feature extraction, pseudo-labeling, teacher fit, and candidate fit. A speaker's WAVs are opened only for folds where that speaker is in `fit_speakers`.

## Method

1. Fit the shared 13-layer scalar-mixing SM13 target head on labelled fit-speaker target rows only. Use the existing 12 epoch AdamW schedule, class weights, batch size, seed, and architecture. This fit-only model is the pseudo-label teacher.
2. For the predetermined 50 extra WAVs for each fit speaker, derive intended ARPAbet phones from the corpus prompt text using the checked-in CMUdict. Align prompt phones with frozen CharSiu CTC output and extract the same 13-layer Wav2Vec2-base-960h pooled window features and one-hot intended-phone references as the shared target head.
3. For each pseudo target row, retain the teacher's full three-class probability vector only if its maximum probability is at least 0.90. These are model predictions, not human correctness labels. A canonical prompt phone denotes intended text only; it is never treated as proof that the phone was actually pronounced.
4. Retrain a fresh target head with the baseline architecture, loss, optimizer, epochs, batch size, and **the same optimizer-update count**. Each batch contains 75% labelled fit rows and 25% retained pseudo rows; this replaces 25% of labelled exposures at each update. Soft pseudo targets use the teacher probability vector. Report effective labelled and pseudo exposures and pseudo-label class/confidence counts.
5. Reuse the shared baseline insertion probabilities unchanged. This is a target-only experiment; insertion results are carried through EVAL-V3.1 as an unchanged reference component.
6. Use the shared calibration-only B=0.2 thresholds and EVAL-V3.1 exact N/N+1 evaluation. Held labels are used only by the evaluator. Oracle fixed-precision results remain labelled diagnostics.

## Controls and reporting

The acoustic path uses the repository's torchaudio 44.1→16 kHz resampling and PCM16 WAV save/decode, passes that waveform unnormalized to frozen CharSiU, and applies the checkpoint-configured `AutoFeatureExtractor` normalization to Wav2Vec2. A phone-to-column assertion checks the 39 supported phone references against the full TRAIN reference matrix. The annotated probe parity results are in `INPUT_PARITY_AUDIT.json`.

- Compare candidate target probabilities against the shared fit-only labelled baseline, with the same folds and seeds 101/202.
- Report aggregate target class precision/recall/F1/AP, macro-3 error AP/F1, unchanged insertion AP/F1, and realized FWR. Preserve keyed held predictions and checkpoints.
- Report pseudo-label coverage, class counts, confidence quantiles, source utterance IDs and hashes, model revision, commands, environment, and fold exclusion sets.
- No class weighting or resampling change is the experimental treatment. No human annotation is inferred; pseudo-label error rates cannot be estimated from held labels.

## Stopping / interpretation

If no valid extra audio, canonical prompt phones, or high-confidence pseudo rows can be obtained under the fixed rules, record the failed feasibility/run and do not relabel or rebalance existing examples to claim added data. The result applies only to the selected 50 extra utterances per fit speaker and does not add speakers or human error annotations.
