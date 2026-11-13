# RQ1-METHOD-BENCH-04 B — Speaker-generalizable event learning

## Scope and existing evidence

This track examined whether the observed TRAIN random-utterance to speaker-grouped gap reflects a representation problem that can be addressed with a small direct event model. Existing METHOD-BENCH-03 grouped OOF results were reused as historical reference, not rerun: 4 folds hold out three speakers at a time, train on nine speakers, and score every TRAIN event once. The fixed folds are ABA/ASI/BWC, EBVS/HJK/HKK, HQTV/LXC/NJS, and PNV/SKA/SVBI.

The original plain-head grouped OOF AP was S .423209, O .276706, exact-gap I .162125, Macro-3 .287347 (supports 7,381 / 1,825 / 560). Matched utterance-random OOF was .478970 / .359747 / .220765, Macro-3 .353161. This split gap is descriptive, not a causal estimate of speaker identity because label composition and utterances also vary. Per-speaker omission prevalence on TRAIN ranged .0008–.0947 and insertion .0014–.0206; per-speaker AP varied sharply, including omission AP .0025–.4507 and insertion AP .0034–.3424. Small supports in several speakers make individual AP unstable.

The previous GRL attempt is a negative control, not repeated here: grouped Macro-3 .288631 vs plain .287347, with S .423371, O .276634, I .165887; three of four fold insertion APs declined and the binary MDD AP also declined slightly. It does not establish that all speaker-invariant learning fails.

## Executed experiments

All new runs use frozen H-002 layer-6 acoustic event crops plus intended-phone reference features; gold phones, error labels, or gold timings are not model inputs. TRAIN labels supply supervision. No DEV data was used for training or selection. Model development uses the same four speaker groups, a lightweight 128/64 GELU head, AdamW at 8e-4, batch 1,024, square-root class weights, five epochs and one seed. Direct event classifiers do not emit realized-phone sequences, so PER is not applicable; recognition quality was not claimed.

| Method | S AP | O AP | Exact-gap I AP | Macro-3 AP | Matched comparison |
|---|---:|---:|---:|---:|---|
| MB03 plain grouped OOF | .423209 | .276706 | .162125 | .287347 | Historical baseline |
| Gaussian feature jitter, AUG S101 | .424092 | .277921 | .166973 | .289662 | Versus paired PLAIN from same run: Δ −.000665 / +.006509 / +.000170; Macro-3 Δ +.002005 |
| Utterance-centered acoustic residuals S101 | .406526 | .257928 | .163893 | .276116 | Versus MB03 plain: Δ −.016683 / −.018778 / +.001768; Macro-3 Δ −.011231 |

The jitter result and its same-run PLAIN arm are in `experiments/rq1_method_bench_04/agent_b/speaker_grouped_oof_aug_receipt.json`. It adds Gaussian noise with standard deviation .15 to fold-z-scored training minibatches only. The paired omission AP gain is +.006509, while substitution decreases slightly and insertion is nearly unchanged. Omission fold AP gains are mixed (two of four improve); insertion fold AP changes are also mixed. This is a promising but weak one-seed screen, not stable evidence.

The new residualization screen subtracts each utterance's mean from its 768-dimensional acoustic event vector; for insertion, left and right acoustic vectors are centered separately. Reference features remain unchanged. This targets utterance-common channel/speaker offset without using speaker ID at inference. It reduced S and O AP enough to lower Macro-3 by .011231; I gained only .001768. Fold APs for residualized O were .2534/.1809/.3454/.3084 and I .1947/.1111/.1427/.2154. The intervention is rejected under this feature construction; no seed repeats are justified.

### Speaker and class variability

Fold supports are fixed by the event population: S 1,859/1,524/2,079/1,919; O 357/301/773/394; I 134/127/150/149. The largest O fold has more than twice the support of the smallest. Plain historical fold AP spans .3610–.4595 (S), .1903–.3555 (O), and .1027–.2515 (I). With residualization, fold AP spans .3409–.4598 (S), .1809–.3454 (O), and .1111–.2154 (I). Thus centering does not consistently improve a held-out speaker group. The jitter source receipt reports its own paired fold metrics; the small pooled O gain does not replicate across all groups.

## Decision

1. **Speaker-generalizable training gain:** no reproducible gain established. The feature-jitter omission gain is +.0065 pooled AP in one seed but heterogeneous by fold; the distinct residualization mechanism harms S/O and overall Macro-3.
2. **Class behavior:** insertion is largely unchanged by jitter and slightly higher under centering, but remains weak. Omission is the only jitter class with a material pooled improvement; this is exploratory because one seed is insufficient. Substitution remains the strongest class and declines marginally under jitter.
3. **PER vs MDD AP:** these event-oriented interventions do not train a phone recognizer, so no PER/AP relationship can be measured from them. Existing report evidence should remain separate from this direct-event experiment.
4. **Repeat decision:** no three-seed repeat is warranted from the present evidence. The intervention signal is small relative to speaker-group variation, and the distinct normalization experiment is clearly negative. Preserve jitter as an unconfirmed candidate if later work has capacity, not as an established component.
5. **Limits:** this is one small frozen-feature detector on four TRAIN speaker folds, with rare O/I classes and no threshold selected. AP is reported; F1 and calibration are unsupported. The grouped OOF population is not interchangeable with DEV_CAL or exploratory DEV_SEARCH.

## Reproduction

New residualization run:

```bash
.venv/bin/python experiments/rq1_method_bench_04/agent_b/train_utterance_centered_oof.py
```

Receipt and predictions: `experiments/rq1_method_bench_04/agent_b/utterance_centered_s101_receipt.json` and `utterance_centered_s101_predictions.npz`. Runtime was 20.61 s CPU, PyTorch 2.6.0+cu124, NumPy 2.4.6; peak memory was not instrumented. Inputs are hashed in the receipt. An initial invocation with `/usr/bin/python` stopped before loading data because sklearn was unavailable there; it produced no prediction artifact. The successful rerun used the repository `.venv`.

Historical grouped baseline, speaker supports, and analysis: `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_receipt.json`, `speaker_support_and_ap.csv`, and `reports/METHOD_BENCH_03_SPEAKER_GENERALIZATION.md`. Feature-jitter source and results: `experiments/rq1_method_bench_04/agent_b/train_aug.py`, `speaker_grouped_oof_aug_receipt.json`, and `speaker_grouped_oof_aug_metrics.json`.
