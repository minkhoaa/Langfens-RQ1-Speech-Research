# X-MDD full-data scaling estimate and go/no-go

## Findings

The frozen M0 fold-0 split contains 900 FIT, 450 CAL, and 450 HELD utterances. Joining the frozen FIT IDs to M0 supervision yields 628 exact-sequence-eligible FIT examples; 272 are ineligible for X-MDD sequence SFT under the current strict policy. The 628 eligible utterances contain about 2,193.5 seconds of learner audio (mean 3.49 s, maximum 7.95 s). HELD labels are not used for selection or training.

The Qwen pilot used the pinned Qwen3-ASR-1.7B checkpoint. Its two SFT completion timestamps were 21.3927 s and 28.2448 s from process start. Their difference gives one measured steady update interval of 6.8521 s, including the pilot's two atomic checkpoint writes per update. Multiplying that single interval by 628 gives a **linear timing estimate** of 4,303 s, or about **71.7 minutes for one SFT pass**. This is not a mathematical lower bound, benchmark, or confidence interval. It is based on one steady interval from two examples, so audio length, batching, caching, and storage can move the result in either direction.

The run's 57.4 s after its second SFT update combines two sampled generations, scoring and checkpoint activity, and four greedy generations. It does not isolate per-token generation cost. Those outputs were all ordinary English text instead of CMU39 phones. Both GRPO rewards were zero, producing zero group variance and **zero effective GRPO optimizer updates**. No X-MDD full-data curve or MDD score exists, and the 71.7-minute estimate excludes full-pool TTS, feature encoding, CAL/HELD inference (900 requests), GRPO, checkpoint verification, and recovery time.

## Resource and method constraints

The real pilot loaded the 1.7B checkpoint in CPU fp32 at 9.41 GB RSS, with final receipt RSS near 12.0 GB. Initial system memory available was about 21.8 GB. The 4 GiB GTX 1650 had about 3.7 GiB free at preflight; fp16 model weights alone are about 3.4 GB before gradients, optimizer states, and activations. The paper's full-parameter SFT recipe and two-L40S setup are therefore not a fit for this laptop GPU. CPU LoRA is the only measured adaptation recipe here and differs from the paper's full update except for three frozen Conv2D layers.

The canonical speech was synthesized from phones with eSpeak-NG, without learner voice cloning. This is a declared deviation from the paper's CosyVoice3 recipe. The larger obstacle is the output format: the pilot model continued generating orthographic text after two SFT updates. A full-scale GRPO run must wait until generated phone output and the joint detection-F1/PER reward work on calibration data with nonzero group variance.

## Scaling decision

Do not start a 628-example X-MDD run from the current checkpoint. First establish a phone-output codec and validate the canonical-relative detection scorer and PER on CAL only, then verify nondegenerate sampled-group rewards. Register the number of updates, CPU wall-time and memory caps, TTS cost, checkpoint cadence, and stop conditions before the next optimizer step. Do not use HELD for grammar, decoding, or reward tuning.

The baseline and SM13 control curves have since been measured on the registered 628/450/450 protocol for seeds 101 and 202. Their per-epoch predictions are under `experiments/three_method_fulldata_20261113/controls/retry_01/`; the run receipt records complete FIT-only training, feature hash checks, and no HELD selection. The SM13 outputs come from an instrumented matched variant, not a bitwise historical replay: this runner clips gradients at 5, interleaves target/gap epochs (freezing gap after epoch 5), and uses a joint RNG schedule. Its receipt carries these deviations. X-MDD's two-update loss trace remains non-comparable and **unmeasured** as a full-data curve. The frozen X budget authorizes zero new optimizer updates, so do not draw or impute an X curve.

## Reproduce the read-only scaling calculation

```sh
python research/three_method_scaling/x_scaling_profile.py
```

The command prints the scaling plan as JSON and verifies the M0 split and supervision hashes and the expected 628 FIT count. It performs no model inference or training. The saved model integrity receipt is `experiments/three_method_pilot_20261113/x_model_integrity.json`; the Qwen SFT/GRPO timing evidence is `experiments/three_method_pilot_20261113/x_mdd_run/run_receipt.json`.
