# MB08 Agent A — reference-conditioned optional-gap insertion alignment

**Status: MEASURED, completed seed 101 and seed 202 replication.** Both seeds ran all four frozen speaker-held-out TRAIN folds on CPU. The learned alignment is a negative result relative to the MB03 baseline; MB03 remains the preferred insertion producer. No claim is made that all alignment approaches fail.

## Method and leakage controls

A fold-specific, single-layer 768D linear softmax acoustic emission model was trained for 8 fixed epochs on labeled TRAIN frames from the **nine** non-held speakers (1,350 utterances). The fixed 39-phone ARPAbet inventory is declared in `align.py`; it does not depend on held-speaker labels or insertion-phone vocabulary. W2V2 H6 features are continuous 20-ms frame sequences. Fold feature mean/std and emission weights were fit only on the nine training speakers. Frame labels use training-only gold phone intervals: realized phone when in the fixed inventory (otherwise intended phone), and a separate insertion class on gold insertion intervals; OMISSION intervals are ignored. Gold timing/realized-phone/error fields are not consumed by held-out inference. The reference phone sequence is the intended canonical sequence.

For N reference phones, inference uses states `G0,P0,G1,...,P(N-1),GN`. Each state can self-loop; ordinary transitions advance one state; a phone-to-phone transition may skip its intervening optional gap. Initial state may be G0 or P0, and accepted terminal states are P(N-1) or GN, so edge gaps are optional too. Forward/backward normalizes only over accepted terminal states. Gap score is posterior path probability of visiting the corresponding gap state, computed from posterior entry-transition marginals (not summed duration, which can exceed one). No rank warp or fixed rank window is used.

The no-gap-state ablation keeps the same trained acoustic emission head and instead scores insertion evidence at phone-only chain boundary posteriors: first/last phone posterior at edge gaps, adjacent-phone posterior product internally, normalized over frames. It therefore differs mathematically from explicit insertion-state visit probability. The acoustic null permutes held-out utterance acoustic frames, retains intended reference phones, and reruns the optional-gap model. Tiny-state exhaustive path enumeration checks forward/backward partition, edge optionality, posterior occupancy normalization, and gap visitation probabilities before training.

## Results

Frozen MB03 historical direct neural baseline: AP-I `.162125`, P@560 `.24286` (reused, not rerun). MB07 rank-window CNN baseline: AP-I `.012828` (context only). New seed 101 optional-gap model: pooled AP-I `.094538`, P@560 `.189286`. Seed 202: `.101253`, `.191071`. Neither meets the frozen adoption rule of beating MB03 on both metrics. Retain MB03; do not generalize this failure to all alignment methods.

### Seed 101 fold results (all TRAIN held-out gaps)

| Fold | Speakers | N / I | Optional-gap AP-I | P@560 | No-gap AP-I / P@560 | Shuffle AP-I / P@560 |
|---|---|---:|---:|---:|---:|---:|
| F0 | ABA, ASI, BWC | 15,266 / 134 | .089117 | .087500 | .040015 / .060714 | .008140 / .003571 |
| F1 | EBVS, HJK, HKK | 15,298 / 127 | .088784 | .101786 | .039427 / .042857 | .009768 / .007143 |
| F2 | HQTV, LXC, NJS | 15,439 / 150 | .126868 | .105357 | .035399 / .044643 | .010872 / .010714 |
| F3 | PNV, SKA, SVBI | 15,359 / 149 | .119994 | .108929 | .061599 / .064286 | .009877 / .007143 |

Seed-101 pooled metrics: explicit gap AP-I `.094538`, P@560 `.189286`; no-gap AP-I `.040030`, P@560 `.091071`; acoustic shuffle AP-I `.009147`, P@560 `.014286`. Framewise mean state entropies F0–F3: `.21881`, `.21458`, `.21185`, `.23084`. Gap-visit score mean by label: insertion `.56575`, non-insertion `.11795`; this is a posterior model score, not a calibrated deployment probability.

Seed-202 per-fold optional-gap AP-I / P@560: F0 `.083992 / .083929`, F1 `.100447 / .101786`, F2 `.140680 / .110714`, F3 `.127713 / .114286`. Pooled metrics: explicit gap AP-I `.101253`, P@560 `.191071`; no-gap AP-I `.042691`, P@560 `.087500`; acoustic shuffle AP-I `.009089`, P@560 `.012500`. Per-fold no-gap AP-I: `.033822`, `.059847`, `.033712`, `.058055`; shuffle AP-I: `.008870`, `.009267`, `.010100`, `.009571`. Framewise mean state entropies F0–F3: `.2173`, `.2196`, `.2116`, `.2303`. Gap-visit score mean by label: insertion `.57333`, non-insertion `.11776`.

### Diagnostic operating points (ranking diagnostics only, not deployment policies)

| Seed | Model | P@100 / R@100 | P@500 / R@500 |
|---|---|---:|---:|
| 101 | Explicit optional gap | .160 / .02857 | .198 / .17679 |
| 101 | No-gap ablation | .170 / .03036 | .096 / .08571 |
| 101 | Acoustic shuffle | .010 / .00179 | .014 / .01250 |
| 202 | Explicit optional gap | .260 / .04643 | .200 / .17857 |
| 202 | No-gap ablation | .170 / .03036 | .090 / .08036 |
| 202 | Acoustic shuffle | .010 / .00179 | .014 / .01250 |

Per-speaker top-560 ranking precision (each speaker contributes only its own top 560; diagnostic, not same support as pooled P@560), seed 101: ABA `.09821`, ASI `.01786`, BWC `.02321`, EBVS `.08393`, HJK `.00536`, HKK `.03929`, HQTV `.06786`, LXC `.04643`, NJS `.04464`, PNV `.01607`, SKA `.11429`, SVBI `.02143`. Seed-101 speaker AP-I: ABA `.14974`, ASI `.04137`, BWC `.05715`, EBVS `.10492`, HJK `.03684`, HKK `.11918`, HQTV `.17943`, LXC `.08058`, NJS `.16013`, PNV `.06313`, SKA `.21585`, SVBI `.07504`. Lowest precision/recall is concentrated in HJK/ASI/BWC and F0/F1, but no fold reaches the MB03 fold baseline uniformly. Seed-202 speaker AP and all other detailed metrics are in the results JSON. Scores and diagnostic K values are not a calibrated policy.

## Reproducibility and artifacts

Source revision: `aa4e6ea2147180a4a820d77a71911f333fcb6721`. Frozen config at `experiment_config.json` was corrected and frozen at 2026-11-08T11:18:27Z, after the interrupted invalid attempt but before valid model training/outcomes. Initial incomplete attempt config is preserved as `experiment_config_initial_attempt.json`; the interrupted attempt produced no valid outcomes/checkpoints and was not used. The exact runnable recipe, fold list, seed, loss, transition graph, controls, hashes, and output details are recorded in config/results.

Exact execution commands (CPU, two threads):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python /home/khoa/Projects/New-Method-Research/experiments/rq1_method_bench_08/agent_a/align.py 101
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python /home/khoa/Projects/New-Method-Research/experiments/rq1_method_bench_08/agent_a/align.py 202
```

Runtime: 47.46s seed 101 and 47.96s seed 202 as measured inside each run; total <2 minutes. Each seed writes keyed raw predictions (61,362 rows), 4 fold checkpoints, results JSON and run log. Main artifacts: `align.py`, `experiment_config.json`, `predictions_seed101.jsonl`, `predictions_seed202.jsonl`, `results_seed101.json`, `results_seed202.json`, `checkpoint_seed{101,202}_fold{0..3}.pt`.

Input SHA-256: TRAIN manifest `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff`; MB03 W2V2 H6 extraction receipt `4f7f438a57284248f06c9dc99f6108de5692a5ca13131cf68a65e6786bff1eba`. Script SHA-256: `e9d2b02a92aa0d5992b0d39b32ef54a8be3709666b1353cfb5baa80ea02878bd`. Config SHA-256: `7442c502e8f6d414a78906c67c32f69190e28f6795e8f1d145cb968846084939`. Results JSON SHA-256 seed101 `669b18fbcd0329882090c0bde3b237128e8c06480f3495457acd7cca22843435`, seed202 `558d48249ae7b5966cad63d185ba8a962358fd570528cdf60750165740420b0a`. Prediction SHA-256 seed101 `ad9ab6d3b2c92deeb36f66305cb0f674bbdf402df6f357d4c1ac9fdf38c8009c`; seed202 `ef37d8ecd1ea1197164c5bdaa01215e48a11f1dce8831ba74fb0ada868a59d56`. Checkpoint hashes are recorded in each seed results JSON. No TEST_SEALED or DEV_SEARCH data were read. No repository paths outside Agent A cwd were written.

**Supervisor verification errata:** The preserved frozen config and one source comment say “eight” training speakers; the executable selects all 12 speakers except the three held speakers, hence actually **nine**. This is a documentation typo, not a changed experiment. The fixed 39-phone inventory maps four unsupported intended tokens to its first phone (AA); the four IDs/tokens are listed in `supervisor/unknown_reference_audit.json`. This fallback is a limitation and is not adopted for formal training. Runtime above excludes input/supervision preparation. Independent supervisor replay reproduced all metrics and ran 13 tiny DP enumeration fixtures. Structural no-gap ablation reuses the same trained emissions; it is not a separately trained alignment loss. See `supervisor/new_insertion_audit.json`.

**Caveats:** Linear frame classifier and frame-level interval supervision are intentionally small/weak; forward-backward state transition weights are fixed uniform rather than learned. All results use the same speakers/folds, so seed replication is not an independent cohort. Acoustic shuffle confirms dependence on acoustic sequence, but performance remains below historical MB03. Preserve MB03 as baseline and stop without broader tuning.
