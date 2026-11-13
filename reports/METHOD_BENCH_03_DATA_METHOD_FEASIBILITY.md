# METHOD-BENCH-03: feasibility note

**Checked:** 2026-11-08. This is the initial implementation and data review; the experiment tracks continue.

## Recovered baseline

METHOD-BENCH-02 and UNIFIED-02 artifacts are present under `reports/` and `experiments/`. Their SEARCH values are treated as exposed development results, not test performance. Key METHOD-BENCH-02 findings are: frame-classifier GOP is not CTC; optional-phone omission scoring reached AP-O 0.08793; frame-argmax edit mapping reached exact-gap AP-I 0.02825; and no evaluated new method reached the existing neural/specialist frontier. UNIFIED-02 records U2 insertion overfit and no repeatable benefit from the tested gap interventions. Existing workspace changes are extensive and were left untouched.

## TRAIN supervision

The authoritative file is `data/derived/l2arctic/manifests/train.jsonl`. Its target records contain `target_phone`, `label`, and `realized_phone`; gap records contain exact `gap_idx`, `has_insertion`, and ordered `insertions`. Thus the TRAIN annotations include realized-phone supervision rather than only canonical phones. Agent A's count audit found 50,356 correct targets, 7,381 substitutions, 1,825 omissions, and 560 positive gaps (507 one-phone, 46 two-phone, 7 three-phone insertions). A realized utterance sequence can be constructed by interleaving non-omitted realized targets with annotated gap insertions. These gold annotations are TRAIN targets and evaluation labels only, never inference inputs.

## Checkpoint and compute

The Hugging Face weights were recovered from `facebook/wav2vec2-lv-60-espeak-cv-ft`, revision `fdf22f0210322b799533bc499dd3e489a9452bdc`. The model page lists Apache-2.0 licensing and 1.26 GB weights; the downloaded file is 1,263,535,127 bytes with SHA-256 `3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e`. Its eSpeak/IPA vocabulary was not validated against the LANGFENS phone inventory, so it is recorded as a recovered and probed checkpoint without a LANGFENS PER/MDD score. Cached H-002 target contexts are available (`train_frame_context.npy`: 59,562 × 17 × 768, float16), but are canonical-center crops rather than continuous utterance sequences; concatenation would duplicate overlapping context and omit free-running gap speech. Two utterance-level CTC arms were trained: continuous waveform-derived log-mel plus BiGRU, and a linear CTC head over continuous frozen `facebook/wav2vec2-base-960h` hidden states[6]. The latter improved realized-phone PER and event AP over log-mel on held-out development; the three-seed replication is complete. W2V2 beam-10 realized PER averaged .3303 on TRAIN speaker holdout, .3458 DEV_CAL and .3216 DEV_SEARCH. At campaign execution, `nvidia-smi` could not communicate with the NVIDIA driver and zero-byte historical lease files did not indicate an active workload; all training, extraction and evaluation ran on CPU.

## Active work

The feasibility and experiment phases are complete. Agents A/B/C delivered recognition, probabilistic edit-path scoring, speaker/generalization and external-method findings. The three W2V2 seeds were run with fixed settings; DEV_SEARCH did not drive checkpoint or coefficient selection. No protected holdout was accessed.
