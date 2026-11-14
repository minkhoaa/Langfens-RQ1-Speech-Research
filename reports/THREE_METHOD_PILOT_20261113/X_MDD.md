# X-MDD: implementation and fidelity record

**Paper:** X-MDD: LLM-Based Mispronunciation Detection and Diagnosis for Unseen Languages with Canonical References, arXiv:2610.23079v1 (October 2026). [Primary paper](https://arxiv.org/html/2610.23079v1)

**Status:** Bounded real Qwen SFT ran; no valid phone predictions or GRPO learning update resulted. The pinned Qwen3-ASR-1.7B checkpoint and official audio tower/projector/decoder were loaded. The run used 85.6 seconds; final RSS was 12.0 GB under the 600-second, two-thread, 16 GiB cap. It used two fit utterances for two LoRA SFT updates and one group of two sampled responses. This is a smoke-scale adaptation, not a reproduction or efficacy result.

## Method in the paper

X-MDD initializes its audio transformer, audio projector, and language decoder from Qwen3-ASR-1.7B. Each query supplies canonical synthesized speech, the canonical phone string, and learner speech; the model autoregressively emits the learner's actual phone sequence. In-context training demonstrations contain the same three inputs followed by annotated actual phones. Paper SFT updates all model parameters other than the three initial audio Conv2D layers. GRPO samples multiple sequences from a frozen old-policy snapshot, scores each with a mixture of detection F1 and phone error rate (PER), normalizes rewards within the query's group, and optimizes token log-probability ratios with a KL penalty against frozen SFT weights. Evaluation uses greedy actual-phone generation. These points are described in §§3.1–3.3 and §4.2 of the paper.

The paper reports English detection F1 of 58.78 and Arabic Quranic F1 of 38.07, compared with 44.14 for its Arabic supervised challenge baseline. These results are single-paper evidence, not evidence about this local implementation. The preprint's reported setup uses two NVIDIA L40S GPUs, a far larger budget than the local GTX 1650 and the registered CPU cap.

## Local implementation

- `research/three_method_pilot/x_mdd/model_adapter.py` uses the official Qwen3-ASR thinker API: `thinker.get_audio_features(...)` for both audio streams, the pretrained decoder embedding table for canonical phone BPE, `thinker.model(inputs_embeds=...)` for causal decoding, and `thinker.lm_head` for token scores. It assembles inputs in canonical audio → canonical phone sequence → learner audio order with trainable segment boundary vectors. It supports teacher-forced response logits, per-token log probabilities, and greedy or sampled autoregressive generation.
- `research/three_method_pilot/x_mdd/x_mdd.py` implements the combined reward, detached group-normalized advantages, response-only token masking, and length-normalized GRPO surrogate with current/old/reference policy roles kept explicit. PER is computed against gold actual phones. Detection F1 compares canonical-relative edit-event locations in the gold and generated phone sequences, so a model is rewarded for detecting an error even when its phone diagnosis is still wrong. This event alignment is a local operationalization of the paper's detection reward and should be checked against the authors' exact scorer if released.
- The decoder LoRA helper freezes pretrained weights and attaches LoRA to decoder projections; the audio tower remains frozen. This is a resource-driven deviation from paper SFT, which updates almost all weights. The CPU pilot uses a frozen SFT LoRA snapshot as its GRPO reference and does not clone a second full model.
- `run_x_mdd.py --preflight` writes a JSON receipt and fails closed if checkpoint files, official `qwen_asr`, pilot requests, or an auditable phone-conditioned TTS receipt are missing. It never downloads a model or begins training. A valid eSpeak receipt must explicitly state phoneme input and `deviation_from_paper: true`; ordinary text-to-speech output or copied learner audio is rejected.

The input adapter relies on the official QwenLM/Qwen3-ASR implementation and its declared `transformers==4.57.6` dependency. The wrapper and model expose the official thinker, audio-feature encoder, decoder, and output head; absent interfaces raise errors rather than falling back to a text-only language model.

## Actual bounded run

The SFT stage completed both updates with finite losses and nonzero gradients:

| Fit update | UID | SFT loss | Gradient norm |
|---|---|---:|---:|
| 1 | HQTV_arctic_a0434 | 3.2765 | 22.9226 |
| 2 | HQTV_arctic_a0113 | 1.5777 | 10.6678 |

The GRPO group generated ordinary English transcript text for both samples, so each received reward 0 and failed CMU39 phone-output parsing. The group had zero reward variance; its normalized advantages, loss gradient, and optimizer update were all zero. The run records this as `DEGENERATE_ZERO_ADVANTAGE_NO_UPDATE`: one group was attempted, zero optimizer updates were effective. The GRPO checkpoint is retained as a provenance artifact, not as evidence that GRPO improved the model.

Greedy generation over the two fit requests, one calibration request, and one held request produced ordinary English text for all four. `recognized_phones` is `null` with `OUT_OF_INVENTORY` status; the raw strings remain available. No PER or detection score is reported for these outputs. Held data contributed learner audio only to generation, not to fitting or reward calculation.

Run evidence: `experiments/three_method_pilot_20261113/x_mdd_run/run_receipt.json`, `recognized_phones.jsonl`, and the SFT/GRPO checkpoint pairs. The canonical-audio receipt is `experiments/three_method_pilot_20261113/x_mdd_tts_receipt.json` and states the eSpeak phone-input deviation explicitly.

## Fidelity and evidence limits

| Component | Paper | Registered local pilot | Consequence |
|---|---|---|---|
| Backbone | Qwen3-ASR-1.7B | Same checkpoint and official multimodal components | Backbone identity is preserved if the pinned checkpoint loads successfully |
| Inputs | Canonical speech voice-cloned from learner audio, canonical phones, learner speech | Phone-conditioned eSpeak synthesis is allowed only with a receipt; no learner voice cloning | Canonical audio acoustics and voice matching differ materially |
| SFT | Broad model update except first three audio Conv2D layers | Decoder LoRA plus trainable boundary vectors; audio tower frozen | This is parameter-efficient adaptation, not the published optimization recipe |
| GRPO | Multiple generations, group-normalized detection-F1/PER reward, frozen SFT KL reference | At most one group-size-two update, LoRA snapshot reference | Objective structure is retained; sampling and update counts are only a smoke-scale check |
| Compute | Two L40S GPUs | Up to 600 CPU seconds, two threads | Throughput and learning signal are not comparable |
| Evaluation | Larger paper datasets and reported metrics | Tiny train-only pilot; held requests remain for root orchestration | No efficacy, generalization, or diagnostic-readiness claim is supported |

## Verification

The isolated synthetic objective tests pass 5/5 and cover the reward trade-off, canonical-copy penalty, detection-versus-recognition separation, detached advantages, zero-variance groups, ratio/KL gradients, response-token masks, and unlabelled query validation. An additional tiny official-Qwen class check verified adapter construction, LoRA wrapping, and finite decoder gradients. Neither check is efficacy evidence.

**Trust for method idea:** Promising. The paper aligns actual-phone generation with both detection and recognition and uses acoustic and phone references, but this is a recent preprint with limited compute replication evidence and a large hardware mismatch for this pilot. **Trust in local efficacy:** Not established. The SFT loss fell on two fit examples, but the model continued to generate orthographic transcripts instead of phone sequences and therefore produced no scorable MDD output.
