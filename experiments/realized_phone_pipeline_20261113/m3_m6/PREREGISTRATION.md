# Remaining milestones: bounded local implementation and evidence

2026-11-13, user explicitly authorized all remaining milestones and local
training. Existing feature branch retained; no publication/merge authorization.
Budgets: GPU wall cap3600s, CPU analysis cap3600s; all failed attempts retained.
No sweep. TRAIN grouped CV only, no official test/dev_replication access.
All outcomes reported, negative results do not become milestone success.

M3: same normalized fp16-exported M2 temporal posteriors for all decoder arms,
4fold ×seeds101/202. Compare greedy deterministic edits, all minimum-cost edit
alignment marginals for the greedy acoustic sequence, and single-edit CTC
likelihood ratios. Unlike old alphabetical E4 alternatives, substitution top4
and insertion top8 come from verified strict FIT realized error phones, with
FIT phone frequency fallback. No gold timestamps in inference. Exact CTC
sequence likelihood computed in fp32 batches on local GPU, parity checked with
independent float64 C++/PyTorch; differences measured, no fp32-vs64 gain claim.
Candidate mass is sum of DISTINCT evaluated sequence probabilities; do not call
finite LR a full event posterior. Beam1/4/8 convergence audit on first CAL
utterance of each fold, seed101, fixed before scoring. Measure all S/O/I,
exact-gap, edges/multiple insertion/S-vs-O+I; keep CharSiU/SM13 historical
baseline reference, no claim new alignment superior to CharSiU without matched
direct evidence. Frozen policies B=.1/.2/.5 fit CAL only.

M4: new categorical ARPAbet attribute mapping version1, sources CMUdict phone
classes and IPA chart; additional dimensions are explicit English engineering
conventions, not corpus human gold. AX=schwa; diphthongs have own categories,
do not invent single monophthong position. Actual realized gold only. Train
phone-only / attribute-only / joint; same acoustic temporal residual head,
features, splits, seeds101/202, optimizer and16epoch budgets. Batch16, AdamW
lr.001 wd.001, cosine to.0001, clip5, fp32. Joint phoneCTC +.2*mean categorical
attribute CTC. Attribute-only inference uses factorized attribute log scores
over registered phones and mean blank score; disclose ambiguity/collisions.
No reference input to recognizer. CAL PER every4epochs, earliest minimum.
Attribute CTC feasibility audited before updates (adjacent repeats preserved).
All arms include same attribute heads (inactive gradients in phone-only noted).
Same hard/soft decoder for full comparisons; finite-LR evaluation of all M4
arms only if within budget, otherwise clearly scoped, no asymmetric claim.

M5: physical audit existing corpora; no new gold labels or speaker acquisition
claimed. Use preselected600 unlabeled utterances from the12 TRAIN speakers in
existing local archives, verify source SHA. Frozen continuous H6 extraction
uses repository resampling + PCM16 + extractor normalization, two TRAIN waveform
parity probes. Never use old CharSiU target-window caches as continuous frames.
Per fold, only FIT speakers' extras enter adaptation. Teacher is corresponding
M2 temporal FIT-only checkpoint, acoustic greedy labels, emission confidence
>=.6, nonempty and CTC-feasible. Record all accepted/rejected/source hashes;
no gold quality claim for pseudo labels. Compare joint model real+pseudo vs
real+repeated-real at same16epochs/update budget, additional training rows count
matched; same fixed real base, all exposure counts recorded. Seeds101/202.
Measure CAL-teacher PER on true strict labels as transfer diagnostic, not
unbiased pseudo-gold validation. Existing M0 inventory for SpeechOcean says no
complete realized/exact-gap gold: do not relabel its score as S/O. M5 new-speaker
true-gold expansion remains unfulfilled unless data actually exists/authorized.

M6: freeze primary candidate BEFORE seed303: M4 joint,16epochs, same split,
FIT-derived finite-LR decoder, B.20 CAL policy. Train four seed303 folds; all
three seeds reported, not best-seed selection. Seed303 is seed confirmation on
development speakers, NOT fresh-speaker confirmation. Fixed fold0/seed303 used
for prototype handoff, no held-based selection. Package pinned backbone/head,
attribute map, decoder/threshold config, checksum manifest, inference CLI and
environment. Verify waveform -> N target/N+1gap, saved/load, reference perturbation
acoustic invariance, independent CTC checks, and protected-field rejection.
Fresh human-annotated speaker-disjoint confirmation and product acceptance
are external requirements; report pending, never fabricate or open sealed TEST.

History: old GOAL hashes resolve through snapshots; preserve all earlier
artifacts/source/checkpoints. Implementation, measured research benefit and
product acceptance are separate statuses, with explicit blockers.
