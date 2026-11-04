# TRN-04R Genuine Z Inference Correction

Current coordinator status: **EXECUTED_ALL_EIGHT_ENDPOINTS_SHARED_CACHE; PRIVATE_READBACK_VERIFIED**. Frozen inference source `c60a84fb044f049dc30f61046f819ff0ac36dc72`; exact runtime/raw hashes are indexed in `PROTOCOL_Z_RUNTIME_MANIFEST.json` and `TRN04R_RAW_PREDICTION_HASHES.json`. Initial parallel results are preserved separately; final V2 uses a shared immutable feature cache. See `PROCESS_CHECKPOINT.md` and `FINAL_PRIVATE_STORAGE_RECEIPT.json`.

Historical lane-delivery status: **IMPLEMENTATION_READY_FOR_REVIEW; INFERENCE_NOT_EXECUTED**. The implementation-stage account below did not assert scores or benchmark results; coordinator execution subsequently occurred.

## Correction

TRN-04 checkpoint evaluation must be called **Z-PROMPT-V1** only when all inference inputs are the 16 kHz waveform and the written prompt passed through the frozen CMUDict G2P path. Gold canonical identities, TextGrid boundaries, word alignment, operation labels, and realization labels are not runtime inputs. A deterministic `sub_first` canonical-gold/prompt DP join is performed offline after freezing and hashing every CAL and VAL raw runtime row. Prompt mismatch/unmapped rows are mapping abstentions, not acoustic deletion predictions.

Each TRN-04 factorial arm (C0/T1/T2/T3) and checkpoint endpoint (`fixed_budget_3600`/`best`) is a distinct inference cell. Scores are the full-precision three-way softmax-derived `s_error`, `s_omit`, and `s_sub`; raw three-logit vectors remain available in private rows. CTC output is greedy recognition only and its edit counts do not represent human omissions. Matched and same-speaker cyclic-mismatch audio preserve the prompt; mismatch inference reuses batch-one cached Wav2Vec2 frames and 4D waveform-derived features keyed to the shifted audio source. Raw `recognition_runtime_Z.jsonl` remains gold-free; offline `recognition_Z.jsonl` joins the realized CTC reference, hypothesis and `(total, substitution, deletion, insertion)` edit counts after the runtime freeze. All four heads serialize their three-class outputs, including class-focal T1/T3, with non-null dedicated subtype scores and explicit uncalibrated score semantics.

No historic O, historical Z, P-SUP, E0, or prior TRN-04 metric is imported or relabeled as a result of this runner. In particular, old checkpoint evaluation code can use gold-derived canonical queries and is not evidence for genuine Z prompt-only inference. Any historical figures remain historical and are outside the TRN-04R result namespace.

## Evidence boundary

The implementation may be checked using gold-free synthetic model inputs only. Real CAL/VAL execution is reserved for the coordinator after the integrated revision is frozen. No training, gradient/backward pass, optimizer update, fitting, threshold selection/tuning, TEST payload access, or publication/privacy change is authorized by this implementation. Runtime output path must be a new private path outside Git. Reports must distinguish implemented/pending from executed evidence.
