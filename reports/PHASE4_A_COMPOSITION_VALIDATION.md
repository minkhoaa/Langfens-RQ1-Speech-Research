# LANGFENS Phase 4 Agent A — Composition validation

**Scope.** Read-only audit of the EVAL-V3 closure and the specified TMCI reports; new work is limited to this report and `experiments/phase4/composition/`. No existing artifact, checkpoint, report, or other namespace was changed. No neural inference or training was run. The three existing `coordination/.phase3_gpu_slot*.lock` files were inspected; because there was no GPU work required, no lease was acquired and no slot was assumed available.

## Executive result

The reproducible, published composition result is the **new canonical CAL-selected score mixture**, not a historical EXP-36 replay: target and gap coefficients were chosen on DEV_CAL from a finite source grid using post-arbitration Macro-3 F1, then frozen and applied to DEV_SEARCH. Independent CPU-only recomposition from the frozen component outputs reproduced the published score arrays for EXP-29, EXP-36, and EXP-43 exactly (`max abs delta = 0` for targets and gaps). This verifies arithmetic/provenance linkage, not model training, historical parity, or unbiased superiority on an unexposed test set.

EVAL-V3 publication reports EXP-36 highest Macro-3 error AP among the four requested systems (0.289233), while EXP-24 remains the standalone model and has the highest omission AP (0.225092). EXP-43 has the highest insertion AP (0.285056). DEV_SEARCH is development-exposed; its scores support descriptive comparison, not blind generalization claims.

## Provenance and reconstruction

### EXP-24 and EXP-34

- **EXP-24-FINAL-ROADMAP-EXPANSION (M24):** standalone `JointSoftAttentionCTCNetwork` (BiLSTM, monotonic local attention, articulatory multitask); 533,272 parameters; target/gap dimensions 55/6. Frozen checkpoint SHA-256: `78e906d0f5369c23196dab3b8a7816b1f061c380dbfb7ddadd31149e88321b59`. SEARCH score file SHA-256: `bf157001d92e225c24c9da3e329dc70648d8bf6b944961333444ec946299a53a`. Its checkpoint replay receipt labels this new frozen inference, not historical parity, and uses non-oracle native head averaging.
- **EXP-34-EXTENDED-PHONOTACTIC-PROBING (M34):** standalone M34 phonotactic n-gram prober; 535,640 parameters per checkpoint receipt. Receipt records TRAIN-only normalizers and `labels_loaded_or_used: false`; no hierarchical routing. Checkpoint SHA-256: `1c6b167c87ced83a6873d79f96205d119b4bba4ff0dcbf9e597d7e53826a8ac6`. SEARCH score file SHA-256: `76cfad9475547b289a4d9ca3ff39dab3ecbc985eb90b179db891a467fbca20c9`.
- EXP-24 and EXP-34 are standalone component score sources, not mixture outputs. Their source arrays have 19,755 target rows and 20,354 gap rows, each with aligned metadata. EVAL-V3 manifest/report defines DEV_SEARCH as 599 utterances / four speakers and explicitly marks it development-exposed. Intended reference remains annotation-derived and its independence is unverified.

### Frozen score composition recipe

EXP-29 includes six sources in coefficient order: 22, 17, 21, 10, 24, 28. EXP-36 adds 34 as the seventh. EXP-43 adds 38 and 42 to those seven. The frozen coefficients are distinct between target and insertion-gap scores:

| Mixture | Target weights (source order above) | Gap weights |
|---|---|---|
| EXP-29 | .10, .25, .15, .10, .25, .15 | .10, .15, .15, .25, .15, .20 |
| EXP-36 | .10, .15, .10, .10, .20, .15, .20 | .08, .12, .12, .18, .12, .18, .20 |
| EXP-43 | .08, .12, .08, .08, .12, .14, .14, .16, .08 | .06, .08, .08, .12, .10, .14, .16, .14, .12 |

Each coefficient vector sums to 1. EVAL-V3 receipts state source coefficients are normalized, weights are selected on DEV_CAL using the actual post-arbitration V3 Macro-3 F1 grid objective, and no FWR constraint is applied. EXP-36 frozen recipe SHA-256: `1b1a5a644e51e399def0e41b7d87e885fa5420ee9e9da434bf3c38524a282f01`; recipe cites DEV_CAL manifest hash `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6`, speakers ERMS/THV, component receipt/score/checkpoint hashes. Its stored calibration objective is Macro-3 F1 0.4870945304. This is a newer, explicitly documented canonical mixture, not evidence that historical mixtures used the same selection protocol.

### Leakage and selection controls

- The closure report identifies historical gold-conditioned `target_m` routing in some methods and says current replay uses `target_m=None` / `NONORACLE_NATIVE_HEAD_AVERAGE`. EXP-34 receipt explicitly states labels not loaded and routing null. No evidence examined indicates EXP-24 or EXP-34 inference used realized-phone target labels.
- The mixture recipe is selected on DEV_CAL only; no SEARCH label or SEARCH score is part of the published weight-selection rule. Nonetheless, DEV_SEARCH has been repeatedly examined in the wider experiment campaign and is not blind. Any new SEARCH-guided ablation, ordering choice, or model selection remains exploratory and must not be described as an independent confirmation.
- Training-data speaker disjointness does not establish intended-reference independence: closure audits explicitly leave annotation-derived reference lineage unverified. No assertion of zero leakage or sealed-test validity is made.

## Official recomputation and comparison

The added verifier `experiments/phase4/composition/recompute_official_compositions.py` reopens the saved standalone score arrays, reads frozen coefficient files, recomputes weighted `p_target` / `p_gap` vectors in CPU float64 arithmetic, checks target/gap metadata identity, and compares against stored official score files. It performs no fitting, label access, neural forward, or writes to source artifacts. Machine-readable evidence is `experiments/phase4/composition/official_recomposition_audit.json`.

| Published canonical variant | AP-S | AP-O | AP-I (exact gap) | Macro-3 error AP |
|---|---:|---:|---:|---:|
| EXP-36 | 0.387088 | 0.204687 | 0.275924 | **0.289233** |
| EXP-43 | 0.363453 | 0.205771 | **0.285056** | 0.284760 |
| EXP-29 | 0.378062 | 0.196539 | 0.272068 | 0.282223 |
| EXP-24 | 0.357737 | **0.225092** | 0.247628 | 0.276819 |

These figures are from `reports/EVAL_V3_CLOSURE_REPORT.md` and TMCI capability report. The composition score-file SHA-256 values checked/recorded by the verifier are EXP-29 `601d877deeb9dca72d489263970954a83a2d0b4efad5b72ebccf9abf979014ab`, EXP-36 `ad06fde5813406089fccb3a7ff92aa07c95a8e7bd07371940888c54204b4b1ae`, EXP-43 `0d9651b9a7c5288997a727039e041e7465de572420f2c015824423fa8f620eff`.

## Ablations, correlations, and interpretation

TMCI Agent A provides a cumulative-build table and leave-one-out results, but its original script and the full underlying variant arrays/weights are not present in the scoped closure evidence. Treat those ablations as **historical exploratory evidence**, not as independently recomputed canonical results. In its reported fixed-renormalized EXP-36 ablations, removal of EXP-34 lowers Macro-3 AP by 0.00682, EXP-28 by 0.00385, and EXP-24 by 0.00325; removing EXP-10 increases it by 0.00222. Under CAL refitting, removal of EXP-34/28/24 reduces reported Macro-3 AP by 0.01779/0.01766/0.01662, while removal of EXP-10 changes it by only -0.00026. These are recipe-dependent associations, not isolated causal effects; each refit increases selection degrees of freedom and requires its own untouched evaluation set for confirmatory inference.

The legacy cumulative path reports EXP-24+EXP-34 Macro-3 AP 0.292012, greater than the full EXP-36 reported 0.289233, with AP-O 0.242612 versus 0.204687. This is explicitly **exploratory**: the artifact does not provide enough information here to audit whether its weights/step ordering were chosen without SEARCH-guided iteration. Do not call it a validated improvement or replace the frozen official recipe.

The independent recomposition audit calculates component-wise Pearson correlations for raw substitution target scores and gap scores (full matrices in JSON). In EXP-36, target substitution score pairwise correlations are high, from about 0.723 to 0.914; the added EXP-34 correlates 0.861 with EXP-22 and 0.906 with EXP-28. This indicates substantial shared ranking signal as well as potential complementarity; correlation alone neither identifies independent information nor proves causal contribution. The legacy `candidate_agreement_matrix.csv` is additional descriptive evidence, not a treatment-effect analysis.

### Compute and operational cost

EVAL-V3 closure gives parameter counts: EXP-24 533,272; EXP-34 535,640; EXP-29 constituents total **2,999,732** (six sources); EXP-36 total **3,535,372**; EXP-43 total **4,626,882** across nine neural constituents. Summed counts describe deployed constituent capacity, not measured latency or memory. Mixtures require multiple constituent forwards (6/7/9) versus one for EXP-24 plus score composition. The official closure report cautions that cached-head timings and CPU score-composition timings are not comparable end-to-end benchmarks and no standardized cold latency result establishes a speedup. Costs should be measured under a future explicit device lease and a matched end-to-end workload; none was run here.

## Limitations and commands

- No GPU lease/inference was requested or run; existing slot locks were not modified.
- No labels were used in recomposition. AP values are quoted from the published evaluator reports; verifier checks score composition, not a second independent AP implementation.
- Historical TMCI ablation/cumulative claims are retained with provenance limits; the TMCI reproduction manual describes scripts whose source is not available in that legacy investigation tree here.
- DEV_SEARCH exposure and annotation-derived reference limitations remain material. No sealed/protected data was accessed.

Commands run:

```bash
PYTHONPATH=. python experiments/phase4/composition/recompute_official_compositions.py
sha256sum experiments/evaluation_v3/closure/checkpoint_replays/EXP-24-FINAL-ROADMAP-EXPANSION/CHECKPOINT_INFERENCE_RECEIPT.json experiments/evaluation_v3/closure/checkpoint_replays/EXP-24-FINAL-ROADMAP-EXPANSION/DEV_SEARCH.npz experiments/evaluation_v3/closure/checkpoint_replays/EXP-34-EXTENDED-PHONOTACTIC-PROBING/CHECKPOINT_INFERENCE_RECEIPT.json experiments/evaluation_v3/closure/checkpoint_replays/EXP-34-EXTENDED-PHONOTACTIC-PROBING/dev_search_scores.npz
```

The recomposition command passed for all three variants; target and gap maximum absolute differences were exactly zero. The hash command emitted the hashes listed above.
