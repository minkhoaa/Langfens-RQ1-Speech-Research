# MB08 readiness gate

## Verdict: CONDITIONAL_GO

**The methodology is ready for bounded implementation and preflight, not an unconditional training launch.** After the conditions below pass, begin the frozen modular `RQ1-TRAIN-PHASE-01` protocol. MB08 did **not** launch formal training.

### What we will train

- **S/O primary:** independent H-002 C/S/O neural head and CTC/E4 branch; fixed 50/50 score fusion. **Fallback:** direct H-002.
- **Insertion:** MB03 PLAIN exact-gap adjacent-edge neural head. Retain it; the new learned-emission monotonic aligner lost on both primary metrics in both seeds.
- **Architecture:** independently trained modules over a shared **frozen** acoustic extractor. A shared trainable multi-head model and staged target-to-gap model remain untested architectural alternatives, not the primary design.

Exact inputs, architecture, losses, seeds, folds, checkpoints, policies, budgets, ablations and reproduction commands: [TRAIN-PHASE-01 specification](METHOD_BENCH_08_TRAIN_PHASE_01_SPEC.md).

## Evidence supporting the decision

| Gate | Status | Evidence |
|---|---|---|
| Authoritative data and event mapping | VERIFIED | TRAIN: 1,800 utterances, 12 speakers, 59,562 targets (S 7,381; O 1,825), 61,362 exact gaps (I 560); unique keyed joins and official four speaker folds |
| Primary held-speaker fitting | VERIFIED | Outer speakers excluded from fits/transforms/competitors; 24 reused inner CTC checkpoint hashes checked; selected fixed scorer avoids invalid meta-calibration dependence |
| Reproducible central scores | VERIFIED, qualified | MB03 AP-I .162124897/P@560 .242857 recovered; source/config/input/predictions available; historical direct MLP checkpoints were not saved |
| Useful S/O evidence | MEASURED | Fixed neural+E4 AP-S/O .402742/.330783 (101), .402074/.333998 (202); favorable exploratory matched-achieved-FWR omission behavior |
| Insertion baseline retained | VERIFIED | New alignment AP-I .094538/.101253 and P@560 .189286/.191071 lose to MB03; acoustic and no-gap controls executed |
| Complete S/O/I score assembly | VERIFIED | Selected hybrid Macro-3 Error AP .298550/.299399; insertion seed stays 101 in both assemblies |
| Complete-system operating policy | INCOMPLETE | No eligible archived inner insertion threshold provenance; full-system P/R/F1, Macro-3 F1 and joint FWR remain null |
| Feasible training design | SPECIFIED, not fully profiled | Frozen features and compact independent heads; CPU fallback; full-run wall/RSS pilot and new runner still required |
| Protection and compute governance | VERIFIED for MB08 | CPU-only execution; no protected data access, DEV_SEARCH tuning, GPU allocation or formal training |

## Conditions before formal training

1. **Implement and test the fresh phase01 runner.** Do not overwrite legacy producers. Freeze source/config/split/seed hashes; pass event/topology, label-free inference and speaker-exclusion tests. Fixed terminal epochs and identical seed/fold budgets must match the specification.
2. **Finish bounded joint-policy integration.** Generate eligible inner OOF target **and gap** predictions from fits excluding their speakers. Freeze wrong-type-inclusive target/gap false-warning thresholds before outer outcomes. Never fit on archived outer labels or substitute top-K diagnostics for policies.
3. **Certify the intended-phone input contract and explicit unknown handling.** Supplied intended phonemes are permitted; no prohibited inference dataflow was found. Keep four unsupported tokens explicit, retain event coverage, and avoid silent phone-0/AA fallbacks. A stronger audio+prompt-text-only claim additionally needs upstream text-to-phone lineage certification.
4. **Time and memory preflight.** CPU default, two threads; register any uniform budget amendment before new outer outcomes. If CUDA is later used, require a valid cooperative lease with start/end/release receipts. Historical MB07's missing lease is not precedent.

These are bounded engineering/calibration conditions, not another method search. Formal controlled training can begin **after** this preflight passes; current reports do not assert it has passed or authorize immediate execution.

## Why not GO or NO_GO?

**Not GO:** complete-system decision provenance and an executed new-runner preflight are missing. Pooled S/O FWR also hides speaker risk: at nominal inner budget .2, selected per-speaker achieved FWR reaches .287/.360 in seeds 101/202. This is not deployment or safety readiness.

**Not NO_GO:** primary event mapping, held-speaker score evidence and a valid insertion baseline survive independent audit. The negative insertion result alone is insufficient for NO_GO. Historical warning-rate mislabeling, prior indexing and meta-calibration limitations are exposed and excluded from primary-policy claims, not concealed.

**Escalate to NO_GO** if later audit finds prohibited inference features, outer-speaker fitting, broken event identity or irreproducible primary evidence. **Upgrade to GO** after the bounded runner/data/policy/resource conditions pass with receipts—not after an arbitrary metric improvement.

## Limitations and authority

Evidence is repeatedly explored TRAIN speaker-grouped 4-fold OOF, not independent external confirmation. Learned meta AP is usable as outer-held ranking evidence, but its strict inner calibration is not established. Intended-phone conditioning is distinct from an independently verified text-only frontend. No complete-system F1/FWR was manufactured.

Revision: `aa4e6ea2147180a4a820d77a71911f333fcb6721`, dirty user work preserved. Three native `openai-codex/gpt-6-luna` lanes plus supervisor; retained-child recovery only for unfinished/audit-correction work. Evidence, raw predictions, eight new alignment checkpoints and receipts: `experiments/rq1_method_bench_08/`. Full findings: [MB08 final report](METHOD_BENCH_08_FINAL.md). Final artifact hashes: `experiments/rq1_method_bench_08/EVIDENCE_MANIFEST_SHA256.json`.

**Final action:** stop MB08 here. No formal unified training, protected holdout access, remote push or destructive repository operation.
