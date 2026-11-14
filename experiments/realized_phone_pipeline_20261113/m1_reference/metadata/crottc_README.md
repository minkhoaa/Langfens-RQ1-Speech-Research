---
tags:
- speech
- audio
- automatic-speech-recognition
- pronunciation-assessment
- speechbrain
datasets:
- l2_arctic
---

# CROTTC-IF for L2-ARCTIC

This repository is a standalone IF-MDD inference bundle. It includes the
complete `model.ckpt` and `perceived_ssl.ckpt` inference state, the verified 44-label encoder,
bundle-local WavLM configuration, and all runtime modules needed by the wrapper, so loading
does not contact Hugging Face.

## Download

```bash
python -m pip install -U huggingface_hub
hf download Haopeng/CROTTC-IF-l2-arctic --local-dir CROTTC-IF-l2-arctic
cd CROTTC-IF-l2-arctic
```

## Single WAV

```bash
pip install -r requirements.txt
python custom_interface.py \
  --hparams-file hyperparams.yaml \
  --audio /path/to/audio.wav \
  --override ctc_decode_weight=0.99
```

## Repository Eval

With the [`main`](https://github.com/Secondtonumb/IF-MDD/tree/main) branch:

```bash
python train.py hparams/CROTTC_IF.yaml \
  --mode eval \
  --inference_ckpt /path/to/CROTTC-IF-l2-arctic
```

Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to enforce offline loading.
`manifest.json` records the required recoverables, architecture signature,
defaults, and label-encoder SHA256.
