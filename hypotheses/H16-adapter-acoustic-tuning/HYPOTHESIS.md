# Hypothesis H16: Non-Native Acoustic Adapter Tuning (NNA-Adapter)

## 1. Proposition
All preceding models (EXP-04 to EXP-18) operated on frozen acoustic features extracted from a Wav2Vec2 backbone trained predominantly on native English speech. Native backbones exhibit an inherent perceptual bias, over-penalizing non-native allophonic formant variations and compressing weak non-native closure intervals.

By introducing a parameter-efficient Non-Native Acoustic Adapter (NNA-Adapter) that projects frame-level SSL representations through an articulatory bottleneck with residual identity skips:
$$\mathbf{e}'_t = \mathbf{e}_t + \mathbf{W}_{\text{up}} \cdot \text{GELU}(\mathbf{W}_{\text{down}} \cdot \mathbf{e}_t)$$
and optimizing it jointly with the Hierarchical Articulatory multi-task objective on `TRAIN` (1,800 utterances across 6 L1s):
1. The adapted representation space will warp non-native accent clusters toward their target phonological prototypes, eliminating false substitution alarms caused by L1 transfer.
2. Low-energy closure and burst intervals will be enhanced, clarifying true omissions from weak reductions.
3. On `DEV_SEARCH` and `DEV_REPLICATION`, this will push:
   - `SUBSTITUTION` F1 beyond **0.40**,
   - `OMISSION` F1 beyond **0.40**,
   - Macro F1 closer to the **0.60 milestone**.

## 2. Experimental Plan
- Method ID: `M21-adapter-fine-tuning`
- Run ID: `EXP-21-ENDLESS-EXPLORATION-ACOUSTIC-REFIT`
- Train NNA-Adapter + HM-ADN on `TRAIN`.
- Pareto calibration on `DEV_CAL`.
- Primary locked evaluation on `DEV_SEARCH` and `DEV_REPLICATION`.
