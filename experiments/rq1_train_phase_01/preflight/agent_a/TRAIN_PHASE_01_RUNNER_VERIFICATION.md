# TRAIN-PHASE-01 runner bounded verification — corrective continuation

## Result

**MEASURED:** one bounded eligible TRAIN smoke completed on CPU with the independent target, CTC/E4 and gap branches. **VERIFIED:** synthetic edge/topology, E4 producer usage, update and resume contracts. This is infrastructure evidence only, not formal training, model evaluation, or readiness approval. No formal folds/seeds, full epoch budgets, outer predictions, DEV_SEARCH or protected data were used.

Corrective work retained the earlier implementation, tests, freeze and failed receipts under `prior_version/`; failures also remain in `tests_bcf2.log` and `tests_bcf2_r1.log`. Final implementation/config were hash-frozen in `SOURCE_FREEZE_BCF2.json` before final tests and smoke.

## Implemented and corrected

- `runner.py` keeps three independent modules: target H-002 (2343→256→128→3), CTC linear 768→43 (blank 42), and MB03 exact-gap head (1614→128→64→2); no shared trainable trunk.
- `gap_input` now emits **N+1** gap examples for N target crops. At gap j, the left context/reference comes from target j−1 and right context/reference from target j; missing edge neighbors are zero. N=0 yields one all-zero gap. Exact valued tests cover N=0,1,3.
- Grouped split plan uses local inner indices 0/1/2 for seed scheduling; each fit uses six speakers excluding outer fold 0 and its inner-held group.
- Bounded TRAIN adapter reads only cached frozen H-002 `train_frame_context.npy`, `train_reference.npy`, `train_labels.npy`, keyed event IDs, TRAIN manifest/gap labels, and audited MB03 continuous H6 feature files. No encoder extraction or legacy main was run.
- Smoke CLI rejects >1 epoch, >4 utterances/speaker, and formal requests. Label-free target/gap inference accepts cached contexts, reference39 and IDs; output includes exactly each target and each per-utterance N+1 gap ID. Raw keyed JSONL and NPZ are retained.
- CTC/E4 path imports audited MB03 phone normalization and MB07 `make_alternatives`/`event_scores` backed by MB05 `score_sequences`; no legacy main or kernel compilation was invoked. Four documented aliases normalize via MB03 `PHONE_NORMALIZATION`; unknown/unregistered tokens fail. SPN remains PHONE_CLASSES index 32. Unknown target-reference vectors remain 39D zero vectors and are not mapped to phone 0/AA.
- Each checkpoint preserves model, optimizer, Python/NumPy/Torch RNG, sampler RNG/order/cursor, epoch, fit IDs, scaler values, seed and frozen config hash. Full training defaults remain spec values; the smoke launcher passes exactly one epoch and cannot launch formal folds.

## Validation evidence

Final command, CPU-only/two threads/no bytecode:

`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python - <<'PY' ... pytest.main(['-q','test_runner.py']) ... PY`

**7 passed** (`tests_bcf2_acceptance.log`; 3.282s test process, 799424 KiB max RSS). Synthetic coverage includes all branches' finite forward/backward and parameter update; independent parameters; target/gap scaler and class-weight behavior; N/N+1 ID checks; exact N=0/1/3 feature values and edges; local grouped indices; unknown and alias/SPN handling; actual MB07 E4 event-scoring API fixture; one-epoch target/gap/CTC checkpoint creation; and exact resume next-update equivalence across target/gap multi-batch+epoch continuation and CTC multi-batch plus next-epoch continuation, including optimizer/RNG/sampler state.

Bounded run command:

`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/khoa/Projects/New-Method-Research/.venv/bin/python smoke.py --outer-fold 0 --inner-local 0 --utterances-per-speaker 4 --epochs 1 --out smoke_artifacts_final`

**MEASURED:** 24 utterances (four each from six fit speakers `HQTV,LXC,NJS,PNV,SKA,SVBI`), disjoint from outer held `ABA,ASI,BWC` and inner-held `EBVS,HJK,HKK`; 738 target events, 762 gap events, 1,500 unique keyed rows. TRAIN fit support: target C/S/O `[635,89,14]`; gap NO_INSERTION/INSERTION `[754,8]`. Wall 61.955s; process maximum RSS 865252 KiB, CPU and two Torch threads. This observed time/RSS is for this 24-utterance, one-epoch smoke only, not full-budget feasibility.

`smoke_artifacts_final/receipt.json` includes frozen source/config, manifest/feature and **per-utterance CTC feature hashes**, fit IDs/speakers, split, class counts, checkpoint hashes, keyed output hashes, runtime/RSS, E4 competitor lists and phone-token provenance. Checkpoints: `target.pt` SHA256 `784f3f5236de08ca6de4fff7eec46323ab309a324c87c5b2c8e0f547a04e2523`; `gap.pt` `c64f51874723169cf7c1a14fa353663ffa07ea543a14de6b48d91974b18a394c`; `ctc.pt` `8869ac5ee8b935ec511f7e92d713b8823eb29721623e7ba0b876ff66a40b0350`. Keyed JSONL SHA256 `d13dddc61f03b348d0e41fcb56dd92cb1893b92d7949652f12609d6b3a4c9ef9`; NPZ `ec73fc32226cd384d2d4deebf6c05ed2ebc2b024737a19564c2f9dc5d75fa85d`. Independent post-run identity/count/hash checks passed in `smoke_artifacts_final/validation.log`.

## Qualifications and remaining blockers

- The smoke deliberately uses the same 24 utterances to fit and exercise CTC/E4 scoring; it is plumbing only and yields no valid generalization or metric claim.
- One epoch does not verify fixed terminal target12/gap5/CTC8 selection or complete formal runner execution. Formal controlled training remains NOT_RUN.
- The bounded cached H6 adapter is lineage-audited feature scope, not a new full waveform feature-extraction run. No prompt-text-only or deployment claim is made.
- Small smoke class support and compute must not be extrapolated as full fold/seed feasibility.
- The documented alias/unknown flow is tested synthetically; the selected TRAIN smoke contains one explicit zero-reference unknown (`SVBI_arctic_a0006::7`) and records SPN class 32. No fallback to AA/phone0 occurs.

## Status labels / authority

- **MEASURED:** bounded 24-utterance training and keyed inference plumbing; time/RSS and hashes in receipt.
- **VERIFIED:** tests, N/N+1 identity/edge rules, fit/held membership, independent branches, producer E4 pathway and checkpoint state round-trip.
- **NOT_TESTED:** formal folds/seeds/epochs, full-system policy, full-budget resource estimate and external generalization.
- **INCONCLUSIVE:** readiness for formal launch is supervisor-owned; this lane makes no GO/NO-GO verdict.

All new writable artifacts, including preserved initial version, failures, tests, receipts, checkpoints and predictions, are within this lane. No staged files.