# X-MDD bounded local pilot

This implementation uses the real `Qwen/Qwen3-ASR-1.7B` multimodal thinker. It assembles the paper's input order as canonical speech embeddings, canonical-phone BPE embeddings, and learner speech embeddings, then scores or generates actual-phone tokens with the Qwen decoder. It does not use a text-only model or a toy replacement.

The paper's full update and TTS recipe cannot be reproduced inside the registered 600-second, two-thread CPU budget. The bounded adaptation is decoder LoRA with the audio tower frozen, plus phone-conditioned eSpeak output if the registered TTS receipt documents actual phone input. This differs from paper SFT (all parameters except initial Conv2D layers updated) and from paper CosyVoice3 voice cloning. Report all outcomes as a constrained pilot, not a reproduction.

## Runtime

Use the pre-existing isolated runtime only:

```sh
PYTHONPATH=/tmp/langfens_qwen_runtime:/tmp/qwen3-asr-src \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m research.three_method_pilot.x_mdd.run_x_mdd \
  --config research/three_method_pilot/x_mdd/config.json --preflight \
  --receipt experiments/three_method_pilot_20261113/x_mdd_preflight.json
```

The command checks local weights, official package importability, resources, the shared pilot manifest, and canonical TTS provenance. It does not download weights or begin training. The actual Qwen model entry points are `Qwen3ASRForConditionalGeneration.thinker.get_audio_features(...)`, `thinker.model(inputs_embeds=...)`, and `thinker.lm_head`; package pin is `transformers==4.57.6` from QwenLM/Qwen3-ASR's official `pyproject.toml`.

The registered CPU pilot command is:

```sh
PYTHONPATH=/tmp/langfens_qwen_runtime:/tmp/qwen3-asr-src \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -u -m research.three_method_pilot.x_mdd.real_pilot
```

It is capped at 600 seconds and 16 GiB RSS, uses only two fit labels for SFT and one fit prompt for the sampled group, and writes atomic verified LoRA checkpoints plus a receipt. If generated strings are not valid CMU39 phone sequences they are retained verbatim with an explicit grammar failure; no invalid token is deleted or converted into a phone metric.

The policy functions in `x_mdd.py` compute canonical-relative edit-event detection F1 separately from PER against gold actual phones. `grpo_loss` uses sampled old-policy log probabilities, differentiable current-policy log probabilities, frozen-reference KL, response-only token masks, and detached per-query group advantages. The model adapter exposes teacher-forced token log probabilities so GRPO can keep those three policy roles separate.

## Checks

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  research/three_method_pilot/x_mdd/test_x_mdd.py
```

These tests validate objective math only; they are not model-efficacy evidence.
