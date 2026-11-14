# Exact-gap insertion positive-weight sweep (seed 101)

## Protocol and selection
The predeclared selection rule is **maximum DEV_CAL exact-gap insertion AP** over gap positions; ties go to the smaller positive weight. Each candidate checkpoint uses the same existing minimum unweighted target-plus-gap DEV_CAL CE selection rule. Four fits ran sequentially on the GTX 1650; TRAIN alone supplied model fitting, normalization, target weights, and positive/negative gap counts. Architecture (gold-only bidirectional GRU-64), seed 101, 12 epochs, AdamW (0.001, 1e-4), and batch 32 were fixed; target loss stayed at baseline sqrt-inverse TRAIN class weights. Initialization, shuffle RNG, and TRAIN row order were reset identically per arm. The TRAIN gap ratio was 60,802 / 560 = 108.575, so capped ratio candidate = 20.

| Gap positive weight | DEV_CAL insertion AP | Best CE epoch | Checkpoint bytes |
|---:|---:|---:|---:|
| 1 | 0.15214 | 9 | 320,342 |
| 3 | **0.17944** | 11 | 320,342 |
| 10 | 0.17704 | 11 | 320,357 |
| 20 (ratio capped at 20) | 0.17319 | 11 | 320,357 |

Thus **weight 3** was selected using the declared DEV_CAL criterion. Its checkpoint SHA-256 is `6ea9644ecf23cb360ba02cd8b661a736f97b0e0d60b39e0d5ad8280af44acf48`. Candidate checkpoints and DEV_CAL insertion score arrays are retained in this namespace; `results.json` includes hashes, scores, checkpoint epochs, and epoch histories. No DEV_SEARCH result was viewed for non-selected weights.

## Selected model: DEV_SEARCH EVAL-V3
One EVAL-V3 evaluation was run on DEV_SEARCH after selection. Per-class AP / F1:

- Correct: **0.97603 / 0.89810**
- Substitution: **0.36475 / 0.40651**
- Omission: **0.15747 / 0.26190**
- Exact-gap insertion: **0.20186 / 0.24920**
- Macro3 error AP / F1: **0.24136 / 0.30587**
- Macro4 mixed-domain F1 (includes CORRECT; not a four-way softmax): **0.45393**
- Joint FWR: **4.90985** (2,941 false warnings); shown without filtering. This exceeds 0.20.

The selected DEV_CAL EVAL-V3 policy was `CAL_MAX_ERROR_MACRO_F1` (S=.37414, O=.60498, I=.24791); it selected macro3 error F1 .45497 with no FWR budget. This policy selection is distinct from candidate selection by insertion AP.

## Runtime and command
Aggregate runtime was **203.43 s**, including candidate training, candidate DEV_CAL scoring, selected-model inference, and evaluation; 684 updates per candidate. Peak allocated VRAM was approximately 44.7 MiB. Selected checkpoint: `gap_positive_3_seed101.pt`, 320,342 bytes. Command: `source .venv/bin/activate && python experiments/exact_gap_insertion_weight_sweep/run.py` (stdout captured in `run.log`).

## Next step
Weight 3 leads this one-seed calibration-selected sweep, but DEV_SEARCH insertion AP (0.20186) is only a modest result and FWR remains high. Do not treat it as robust evidence; the next focused step is a preregistered independent-seed replication of weight 3 against a matched baseline, with candidate choice separated from the new held-out split. No broader audit was performed.
