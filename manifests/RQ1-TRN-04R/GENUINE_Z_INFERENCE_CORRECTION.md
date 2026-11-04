# TRN-04R Genuine Z Inference Correction

Status: **IMPLEMENTATION_READY_FOR_REVIEW; INFERENCE_NOT_EXECUTED**. This document corrects protocol/output semantics only; it reports no newly generated scores or benchmark results.

## Correction

TRN-04 checkpoint evaluation must be called **Z-PROMPT-V1** only when all inference inputs are the 16 kHz waveform and the written prompt passed through the frozen CMUDict G2P path. Gold canonical identities, TextGrid boundaries, word alignment, operation labels, and realization labels are not runtime inputs. A deterministic `sub_first` canonical-gold/prompt DP join is performed offline after freezing and hashing every CAL and VAL raw runtime row. Prompt mismatch/unmapped rows are mapping abstentions, not acoustic deletion predictions.

Each TRN-04 factorial arm (C0/T1/T2/T3) and checkpoint endpoint (`fixed_budget_3600`/`best`) is a distinct inference cell. Scores are the full-precision three-way softmax-derived `s_error`, `s_omit`, and `s_sub`; raw three-logit vectors remain available in private rows. CTC output is greedy recognition only and its edit counts do not represent human omissions. Matched and same-speaker cyclic-mismatch audio preserve the prompt; mismatch inference recomputes both Wav2Vec2 frames and the 4D waveform-derived features from the shifted audio.

No historic O, historical Z, P-SUP, E0, or prior TRN-04 metric is imported or relabeled as a result of this runner. In particular, old checkpoint evaluation code can use gold-derived canonical queries and is not evidence for genuine Z prompt-only inference. Any historical figures remain historical and are outside the TRN-04R result namespace.

## Evidence boundary

The implementation may be checked using gold-free synthetic model inputs only. Real CAL/VAL execution is reserved for the coordinator after the integrated revision is frozen. No training, gradient/backward pass, optimizer update, fitting, threshold selection/tuning, TEST payload access, or publication/privacy change is authorized by this implementation. Runtime output path must be a new private path outside Git. Reports must distinguish implemented/pending from executed evidence.
