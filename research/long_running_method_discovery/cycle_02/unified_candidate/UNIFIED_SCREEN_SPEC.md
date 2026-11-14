# RC2 unified model: screened design and decision gates

Status: `SUPPORTED_EXPLORATORY` — screened score bundle assembled and replayed; no joint training run or merged checkpoint.

## Purpose

Prepare a small unified candidate that can consume acoustic evidence for target phones and exact insertion gaps. It is not a claim that EXP24, the optional alignment score, or any new method is superior. The initial design preceded RC2 neural screens; their completed results and the score assembly are appended below. The current H-2 target plus Gold-Only gap artifact is an analysis bundle, not one executable neural model.

## Evidence inspected

- Frozen source revision at inspection: `aa4e6ea2147180a4a820d77a71911f333fcb6721`.
- Phase 3 model: [`research/student/train_light_student.py`](../../../student/train_light_student.py), SHA-256 `80c506b8118d4e210ba75e66674d31fd041481c99de775482f479104983ce11b`.
- Existing 51D target / 6D gap summaries are derived from frame logits plus acoustic summaries. `Student.forward` checks CUDA-resident tensors and the `N+1` gap contract.
- Official closure rows in `experiments/evaluation_v3/closure/release_final/RESULTS_PRIMARY_COMPARABLE.csv` use all 19,755 target events and 20,354 exact gaps on DEV_SEARCH, but score bundles differ by source and policy rows duplicate. These are development-exposed ranking references, not matched controlled comparisons.
- Selected source-bundle AP references (S/O/I): EXP24 native neural direct heads `0.35774 / 0.22509 / 0.24763`; EXP29 composition `0.37806 / 0.19654 / 0.27207`; EXP36 composition `0.38709 / 0.20469 / 0.27592`; EXP43 composition `0.36345 / 0.20577 / 0.28506`. All four carry `ANNOTATION_DERIVED_REFERENCE_INDEPENDENCE_UNVERIFIED`; EXP29/36/43 are compositions, not standalone learned models. Their EVAL-V3 CSV rows do not provide directly comparable Macro3 AP/F1 for those source bundles.
- Historical STU-01 Gold-Only seed 101 has the same population and target/gap denominators: AP S/O/I `0.36460 / 0.15943 / 0.18672`, Macro3 AP `0.23692`. This is the relevant neural-control ranking profile; search was exposed during prior work.
- Fresh CUDA 13.0 Gold-Only control: job `RC2-GOLD-ONLY-CUDA130-S101`, `.venv-cuda` Torch 2.14.1+cu130, GTX 1650, wrapper lease exit 0, 12 epochs/684 updates, 78,661 parameters, peak allocated VRAM 44.09 MiB, runtime 44.39 s. Checkpoint SHA-256 `52c6c26bb52c119915942da8a3aaf9fdbc109911b77a55df8cf813ca46f7dc4f`; DEV_CAL prediction SHA `5a79c6181ea69aa580f53ab0eaf436392b3c6ea0a37896d19802eb52fd39bd87`; DEV_SEARCH prediction SHA `9b6e36adf250a46cf47b44f3ee5df9de48648ae2e7dbb48e58047b14ed586d5c`. EVAL-V3 run: [`RC2-GOLD-ONLY-CUDA130-S101-eval.json`](../baseline_control/RC2-GOLD-ONLY-CUDA130-S101-eval.json).
- At the evaluator's 0.05 DEV_CAL-selected budget: AP S/O/I `0.36460 / 0.15943 / 0.18672`, Macro3 AP `0.23692`; class F1 S/O/I `0 / 0 / 0.07519`; Macro3 Error F1 `0.02506`; Macro4 F1 `0.25561`. The zero S/O F1 is an operating-point result, while AP measures the raw ranking; do not conflate the two.
- Existing optional-path H-001 evidence is exploratory and shows an omission ranking change on frozen emissions. It does not establish an acoustic neural improvement. Do not promote it into the unified model unless the H-001 owner supplies valid provenance and H-002/H-001 integration passes a matched control.

## Candidate architecture after component gates

Keep the Phase 3 lightweight shared bidirectional GRU and both output domains. Add no component until its separate control/intervention screen meets the gate below.

1. **Target stream**: the existing 51D per-phone summaries plus only an H-002 frame representation or H-001 deletion-evidence scalar if that component has positive TRAIN speaker-grouped validation evidence. Project to GRU-64 and use the existing mutually exclusive CORRECT/SUBSTITUTION/OMISSION head.
2. **Gap stream**: retain exactly `N+1` leading/internal/trailing gaps. Add local acoustic pooling or transition evidence only if H-003/H-002 establishes positive grouped TRAIN validation evidence. Keep a separate binary gap head; never fold gaps into target softmax.
3. **Auxiliary objective**: add one articulatory task only if H-006 demonstrates audio-dependent gain on grouped TRAIN validation. Reference-derived labels can supervise training; reference identities may be input, but no realized gold-phone or error labels may be used at inference.
4. **Distillation**: add teacher losses only if H-004 passes the OOF provenance audit and improves held-out TRAIN speaker groups against Gold-Only at matched seed/budget. Keep S/O/C target simplex and independent gap semantics. No EXP36 pseudo-teacher target.

This is a gated family, not a proposal to combine every method. At most one newly supported representation/evidence change enters the first unified ablation. Further components require new matched arms.

## Falsifiable execution plan

### Stage 0 — baseline lock

Re-run neither the control nor prior scores. Verify the listed checkpoint/prediction hashes, EVAL-V3 evaluator hash and complete event coverage. Use the fresh CUDA130 Gold-Only seed 101 control as the initial paired reference. Keep its files immutable.

### Stage 1 — component gate

For each component independently, compare its registered single-intervention arm with the same Gold-Only model, training utterances, seed 101, 12-epoch budget, event definitions and EVAL-V3 evaluator. Any architecture/capacity difference must be reported and matched with a parameter-count control where practical. Use TRAIN-only speaker-grouped validation for selection. DEV_CAL is reserved for policy/calibration. DEV_SEARCH is used only after choices are frozen and must be labeled exploratory.

Proceed only when the intervention improves its predeclared target metric on grouped TRAIN validation without changing event population or leaking annotations. The one-seed run is a screen only. A surviving component needs three matched seeds and speaker-level reporting before a repeatability claim or integration recommendation.

### Stage 2 — unified interaction test

After individual gates, compare:

| Arm | Intervention |
|---|---|
| U0 | Gold-Only 51D/6D control |
| U1 | Gold-Only plus the single strongest eligible new acoustic representation/evidence component |
| U2 | U1 plus one additional independently supported component, only if its primary component gate passed |

Keep seed, optimizer, epochs, train population and model-selection process matched. Do not use SEARCH to choose between U1/U2. If H-002 frame features are not available from a lawful local checkpoint, U1 uses the best independently supported non-SSL feature or remains unimplemented; no remote checkpoint download.

### EVAL-V3 outcomes

Report AP S/O/I and Macro3 AP first; then class precision/recall/F1, Macro3 Error F1, Macro4 F1, binary MD AP/F1, calibration, and operating curves/FWR as supplementary. Report all 19,755 SEARCH target events and 20,354 N+1 gap events, speaker breakdown, score distributions, baseline deltas, exact prediction/checkpoint SHA-256, and CUDA wrapper ledger entry. Do not make FWR <= 0.20 an acceptance gate.

## Stop conditions and integration decision

- Reject a component if grouped TRAIN validation fails its declared target metric or if the gain is reference-only, leakage-driven, population-dependent, or explained by an unmatched training budget.
- Mark it inconclusive if the TRAIN groups disagree materially or the candidate is selected only by SEARCH.
- Do not edit or replace Phase 4. A future Phase 4 recommendation requires a three-seed matched unified result, provenance audit, reliable audio-dependence control, and an operator review of the separate RC2 artifacts.

## Reproduction entry points

Baseline reproduction command already executed:

```bash
.venv/bin/python coordination/phase3_gpu_job.py \
  --job RC2-GOLD-ONLY-CUDA130-S101 \
  -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u \
  research/student/train_light_student.py \
  --arm RC2-GOLD-ONLY-CUDA130-CONTROL --epochs 12 --seed 101
```

EVAL-V3 command already executed:

```bash
.venv/bin/python experiments/student_evaluation/evaluate_student_predictions.py \
  --dev-cal experiments/student_training/RC2-GOLD-ONLY-CUDA130-CONTROL_seed101_dev_cal.npz \
  --dev-search experiments/student_training/RC2-GOLD-ONLY-CUDA130-CONTROL_seed101_dev_search.npz \
  --dev-cal-manifest data/derived/l2arctic/manifests/dev_cal.jsonl \
  --dev-search-manifest data/derived/l2arctic/manifests/dev_search.jsonl \
  --output research/long_running_method_discovery/cycle_02/baseline_control/RC2-GOLD-ONLY-CUDA130-S101-eval.json
```

These commands reproduce only the current control. Candidate command and hashes must be added only after the parent supervisor approves a GPU slot and the component owners publish their registered results.

## Screened-result update and current decision

The H-002 owner trained a frozen, ASR-finetuned Wav2Vec2 layer-6 reference-conditioned attention target head with seed 101. It is the only screen in this group with positive AP changes on both target errors while holding the Gold-Only exact-gap scores fixed: AP S/O/I `0.377810 / 0.167294 / 0.186715`, Macro3 AP `0.243940` versus Gold-Only `0.364602 / 0.159433 / 0.186715`, Macro3 AP `0.236917`. At the DEV_CAL-selected 0.10 warning budget, Macro3 F1 was `0.054638` vs `0.034250`. This remains one-seed exploratory evidence on development-exposed SEARCH with four speakers, not independent validation.

Acoustic-dependence checks show sensitivity to the acoustic and reference inputs: full H-002 Macro3 AP `0.243940`; reference-only perturbation `0.168350`; slot-shuffled acoustics `0.149948`; zero-reference perturbation `0.098116`. These are inference-time perturbations of one trained model, not matched retrained controls, and can be out of distribution. They support only the limited statement that this predictor's scores use both inputs under these perturbations.

The cross-run candidate bundle uses H-002 target scores and the independently trained Gold-Only gap scores. The complete, unique event-key join and EVAL-V3.1 replay are recorded in [`RC2_SCREENED_UNIFIED_BUNDLE.json`](RC2_SCREENED_UNIFIED_BUNDLE.json), [`assembly receipt`](outputs/RC2-UNIFIED-SCREENED-H2-TARGET-GOLDONLY-GAP-S101-ASSEMBLY_RECEIPT.json), and [`EVAL-V3.1 receipt`](outputs/RC2-UNIFIED-SCREENED-H2-TARGET-GOLDONLY-GAP-S101-EVAL-V3.1.json). CAL has 10,043 target / 10,343 gap events; SEARCH has 19,755 / 20,354. Replayed AP S/O/I, Macro3 AP/F1, Macro4 F1 and joint FWR exactly match H-002 at CAL policy budgets 0.10 and 0.20. There are no TRAIN component predictions for this bundle; none were synthesized.

### Tradeoffs against the official closure rows

The comparison below subtracts each official source-bundle AP from the H-002 target + Gold-Only gap bundle, in S/O/I order. The official rows come from distinct bundles; EXP29/36/43 are compositions. The closure sources flag intended-reference independence as unverified, and the rows do not provide comparable Macro3 AP/F1. Treat these as exposed development context, not a matched winner comparison.

| Official reference | Official AP S/O/I | Screened bundle minus reference AP S/O/I |
|---|---|---|
| EXP24 | 0.357737 / 0.225092 / 0.247628 | +0.020074 / -0.057798 / -0.060913 |
| EXP29 | 0.378062 / 0.196539 / 0.272068 | -0.000252 / -0.029245 / -0.085353 |
| EXP36 | 0.387088 / 0.204687 / 0.275924 | -0.009278 / -0.037393 / -0.089209 |
| EXP43 | 0.363453 / 0.205771 / 0.285056 | +0.014358 / -0.038477 / -0.098341 |

The H-001 optional-phone scalar + acoustics arm was not supported for discrimination: AP S/O/I `0.356376 / 0.161252 / 0.169413`, Macro3 AP `0.229014`. H-004 raw OOF scalar KD had Macro3 AP `0.239828`, but F1 at CAL budget 0.20 was `0.104513` vs `0.104897`; its deltas were small and mixed. H-004 feature KD Macro3 AP was `0.231038`, with all three class APs down. H-006 combined articulatory auxiliary had mixed small deltas, Macro3 AP `0.239060`; its 0.20-budget Macro3 F1 rose to `0.113712` from `0.104897`, but it is one seed and not a robust component result. No H-001/H-004/H-006 component enters the screened bundle.

### Decision and limits

Retain the H-002 + Gold-Only gap pair as a **screened candidate score bundle**, not a winner. Do not integrate it into Phase 4. Run TRAIN speaker-grouped selection and matched multi-seed repeats before any unified neural training. The ASR-finetuned encoder is not pretraining-only SSL; no inference-time gold boundaries, realized phones, or error labels were used by the H-002 feature pipeline. Intended-reference independence remains unresolved. TEST_SEALED and protected replication data were not accessed, so there is no sealed-test generalization claim.
