# Phân tích nguyên nhân — RQ1-UNIFIED-02

## Kết luận điều hành

**Có đủ bằng chứng attribution cho biết nhánh gap score của U2 seed101 làm giảm Macro-3 AP; chưa đủ bằng chứng để nói nguyên nhân cơ học là dimensionality, localization hay imbalance.** Đây là kết luận từ phép thay nhánh score trên đúng population, không phải chứng minh mô hình neural gap cụ thể gây root cause production.

## Attribution branch score (frozen substitution)

EVAL-V3.1 dùng 19,755 target events, 20,354 exact gaps, cùng gold/IDs; chỉ thay prediction branch. DEV_CAL threshold được chọn riêng cho từng composition; AP không phụ thuộc threshold.

| Configuration | AP-S | AP-O | AP-I exact-gap | Macro3 AP | Δ Macro3 so với U2/U2 |
|---|---:|---:|---:|---:|---:|
| U2 target + U2 gap | .371360 | .185318 | .117992 | .224890 | — |
| U2 target + GoldOnly gap | .371360 | .185318 | .186715 | .247798 | +.022908 |
| H002 target + U2 gap | .377810 | .167294 | .117992 | .221032 | −.003858 |
| H002 target + GoldOnly gap | .377810 | .167294 | .186715 | .243940 | +.019050 |
| U2 target + EXP43 gap (diagnostic composition) | .371360 | .185318 | .285056 | .280578 | +.055688 |

GoldOnly/H002 baseline Macro3 AP=.236917/.243940. U2 gap substitution recovers .022908 points relative to U2 and brings U2/GoldOnly composition to .247798, which is .010881 above GoldOnly because U2 improves target AP, especially omission. The insertion AP shortfall is exactly .068723 relative to the GoldOnly gap score. H002 target substitution has the same gap-branch impact (+.022908) on Macro3 AP. EXP43 row is a **diagnostic score composition**, not a trained unified model.

## What the learned gap-head diagnostics show

TRAIN has 61,362 gaps, 560 positive (0.9126%); CAL 102/10,343; SEARCH 125/20,354. U2's actual 1,614D vector is `[768D left pooled frames, 768D right pooled frames, 39D previous intended phone, 39D next intended phone]`. Internal edges pool left crop offsets `[11:16]` and right `[1:6]` from 17-frame crop around predicted phone centers; missing utterance-boundary neighbor is a zero sentinel. No gold timing, labels or realized phone features feed this vector.

| Arm | Input / capacity | Train AP-I (in-sample) | CAL AP-I | SEARCH AP-I | SEARCH Macro3 AP | CAL20 Macro3 F1 |
|---|---|---:|---:|---:|---:|---:|
| G0 U2 original | 1614→256→64→2, 430,018 gap params | .976979 | .138111 | .117992 | .224890 | .125421 |
| G1 bottleneck128 | same 1614D input, 1614→128→64→2, 215,106 gap params | .978514 | .136409 | .125929 | .227536 | .140180 |
| Balanced minibatches | same original architecture, 64 positive + 64 negative sampled per batch | .992556 | .094039 | .125692 | .227457 | .149231 |

G1 halves parameter count and gains +.007937 AP-I in exposed SEARCH but loses .001702 AP-I on CAL. Balanced sampling similarly has a small SEARCH gain (+.007699), but CAL insertion AP declines .044072. Both are single seed. G1 train AP remains virtually perfect; these results do not show that reducing dimension solved overfit. Balanced CAL-vs-SEARCH reversal warns against selecting the sampler from SEARCH. No intervention result is repeated across new matched seeds.

The train/evaluation mismatch is stronger than the summary table alone: U2 train AP-I=.97698/AUC=.99986; CAL AP=.13811/AUC=.81530; SEARCH AP=.11799/AUC=.78343. Median positive score is .99764 TRAIN but .000527 CAL and .000833 SEARCH. This supports severe overfit/poor transfer of this head. It still does not identify dimensionality as cause. A 10k TRAIN sample gives effective covariance rank ≈236/768 for each acoustic block (top eigenvalue ≈3.9%, top 10 ≈25.6% variance): the features are redundant but have nontrivial variance. Severe imbalance is real, but one balanced screen does not make it the dominant bottleneck. G6 tested paired crop-relative context: 18×768 frames (left offsets 8:17 + right 0:9), Conv1d head, 8 epochs, same frozen W2V layer-6 cache, seed101. AP-I CAL=.06280/SEARCH=.10900; Macro3 SEARCH=.22189, below U2 .22489. G6 omitted phone vectors; G6-REF restored the two 39D phone vectors with the same sequence/CNN: CAL AP-I rises to .10078, but SEARCH AP-I is .09967, Macro3 .21878. Thus removal of reference explains part of G6's CAL weakness, but paired crop context remains below U2 on SEARCH; no repeat. Neither sequence is a continuous waveform interval. G2 with only the existing 6D `gap_X` summaries had AP-I .03634 CAL/.02594 SEARCH, much lower than U2; the low-dimensional summary alone is weak. G3 concatenating the same 6D summaries with U2's 1,614D features yielded CAL AP-I deltas +.03616/+.00945/+.01770 on seeds 101/202/303, but SEARCH deltas +.01617/−.00183/−.03332. Mean SEARCH AP-I .13267 vs matched U2 .13900; mean Macro3 AP .22579 vs .22789. This is CAL separation without SEARCH generalization; do not assemble an integrated candidate from it. G8 all-17-frame crop context also failed to improve: AP-I .07922 CAL/.11651 SEARCH, Macro3 .22439 vs U2 .22489. Historical HNM GRU64 TRAIN hard-negative mining recovered AP-I .18507 and Macro3 .23766, only +.00075 M3 AP over its GoldOnly reference and with lower M3 F1; it suggests hard-negative objective can help one architecture but not a robust gap fix. G9 auxiliary CharSiU CTC posterior/uncertainty features were appended to W2V+gap_X on seeds101/202: AP-I CAL/SEARCH .15978/.15596 and .13422/.13226. Seed101 exceeds matched U2 SEARCH, but seed202 SEARCH is below U2 and G3; CAL remains below G3 in both. This is not a repeatable improvement and mixes encoders. Direct G6-PREDICTED-BOUNDARY then extracted full-wave normalized W2V hidden_states[6], mapped each predicted CTC gap center onto the verified W2V frame clock, and trained on a contiguous 17-frame sequence plus intended phones. No gold timing was used; exact event IDs and target-cache parity (<2e-3) passed. Seed101 AP-I TRAIN/CAL/SEARCH=.83238/.10229/.08015; Macro3 SEARCH AP=.21228 vs U2 .22489, CAL20 Macro3 F1=.13033 vs .12542. Thus this specific continuous predicted-center sequence test is negative on both raw AP splits, not evidence that all localization or sequence hypotheses are false.

## Other causal explanations

- **Localization/geometry:** code inspection shows pooled predicted-center crop edges, not a continuous predicted interval. On SEARCH 124/125 positives are internal; leading has 0/599 positives and trailing 1/599 (AP .3333 is based on one event). Boundary-specific ranking cannot be estimated. The same-encoder continuous predicted-center 17-frame G6 intervention was negative (CAL/SEARCH AP-I .10229/.08015), but it did not vary center/width/region systematically. Wrong localization remains **unresolved**, not demonstrated.
- **Joint training:** target and gap modules have separate trainable weights and separate inputs; shared trainable parameter count is zero. Hence there is no classic shared-parameter gradient conflict. Independent-vs-joint scheduling has not been causally tested.
- **Rare positives:** evidenced prevalence is low; balanced-minibatch screen has conflicting CAL/SEARCH effects and no seed replication. Not dominant-established.
- **Target/gap trade-off:** U2 target scores improve O vs H002 (+.018024 AP) and are slightly lower on S (−.006451). The two-headed checkpoint has not demonstrated universal target superiority.
- **Operating policy:** AP ranking and threshold decisions differ. U2 CAL20 S recall is .0124 despite AP-S .3714; unrestricted F1 increases at FWR 4.654. Do not mistake operating-point failure for absence of ranking signal.

## Root-cause verdict and action

Supported: weak insertion score branch explains U2's seed101 Macro3 AP regression in frozen attribution. Not established: why that branch is weak. Falsified: a classic shared-trainable-representation gradient-conflict explanation. A CPU synthetic-update check with the actual U2 module, same per-head losses/batches/AdamW states and dropout disabled produced max absolute parameter delta 0 between joint and separate updates for all four steps; it verifies the algebra for matching schedules, not different real-data schedules. Inconclusive: 1,614D overfit as a causal mechanism, incorrect gap location, imbalance dominance, and broad multi-task schedule interference.

The direct predicted-boundary sequence screen is complete and did not improve CAL or SEARCH ranking. Do not repeat this identical configuration absent a new CAL-driven hypothesis. Remaining mechanistic questions are the exact best gap region/window width and other inference-valid representations; these are exploratory and not required to judge the tested U2 failure. Preserve the boundary test receipt and frame-mapping validation. Do not expand architectural search based on exposed SEARCH uplift alone.
