# Full-FIT curves — protocol registered before outcome inspection

Question: with the entire eligible FIT population in frozen M0 fold 0, does the local compact CROTTC adaptation or paper-encoder PER retrieval improve strict S/O/I detection over matched CTC and SM13 controls?

Frozen configuration: `experiments/three_method_fulldata_20261113/scaling.json`. FIT is 628 exact-sequence-eligible utterances out of 900 candidates, CAL 450, HELD 450; speakers are disjoint. All selected audio/H6 hashes and 26 SM13 feature hashes were compared with earlier receipts before execution. No official test access. Reference vectors without an earlier recorded hash are frozen now, not retrospectively authenticated.

- CROTTC compact: 16 epochs, seeds 101/202, two CPU threads, 1,800 seconds total cap.
- CTC baseline: eight epochs; SM13: 12 target epochs, five gap epochs; seeds 101/202, two CPU threads, 1,800 seconds total control cap.
- PER: fixed k=10, threshold=0.7, paper HuBERT-large encoder, nested FIT pools 24/96/256/628; no optimizer updates; shared GPU lock, longest-waveform memory profile, 1,800-second cap.
- X-MDD: no new optimizer updates in this phase. Report measured pilot throughput and explicitly conditional cost extrapolation; no synthetic efficacy curve.

Primary diagnostics: strict exact-position S/O/I macro F1 and false warnings per utterance, evaluated on all 450 HELD utterances including those without exact realized-phone sequences. Secondary realized PER uses only explicitly eligible sequences (386 CAL, 229 HELD). Zero missing/duplicate event predictions required.

HELD curves are registered diagnostics, not a blind confirmatory test. Never choose an epoch or threshold by maximizing HELD. Report fixed final epochs, and optionally CAL-only checkpoint selection with its rule stated. Show seed mean and full two-seed range, not a confidence interval. Do not compare the numerical values of different training losses as one objective.

Architectures, feature interfaces, supervision objectives, epoch counts and optimization budgets differ. This is a workload-matched practical comparison, not a compute-matched causal ranking or reproduction of the three papers. SM13 features inherit annotation-derived alignment/reference limitations. The compact CROTTC implementation retains the pilot's documented deviations from the paper. Excluding 272 FIT utterances remains an unresolved supervision limitation.
