# LANGFENS RQ1-METHOD-BENCH-07 — Báo cáo khoa học cuối

**Ngày kiểm kê:** 08-11-2026 · **EVAL contract:** V3.1 · **Quần thể mới:** TRAIN, 1,800 utterances, 12 speakers, 59,562 target events, 61,362 exact gaps. Không dùng DEV_SEARCH để fit/chọn và không truy cập protected/sealed data.

## Tóm tắt quyết định

- **RQ-A (E4 + neural):** hoàn tất nested 4-outer × 3-inner cross-fit cho seed101 và seed202. Learned neural+phone-prior+E4 đạt pooled AP-S/O `.42328/.32509` và `.42300/.32862`; cải thiện so với nested direct H-002 neural trong cả hai seeds và **4/4 outer folds**. So với fixed neural+prior, gain cũng lặp lại. E4-only yếu hơn neural; substitution so với historical MB03 PLAIN gần như ngang, còn omission tăng mạnh hơn.
- Target-only warning-budget policy cho thấy ranking AP không bảo đảm F1: fixed neural+E4 0.5 blend đạt Macro-2 F1 cao nhất ở mỗi budget/seed; learned fusion không thắng nhất quán theo policy. Không có insertion warning/Macro-3.
- E4-only yếu hơn neural-only (`.27222/.22901` vs `.38408/.26565`). Target-only warning-budget evaluation shows the ranking gain does not transfer uniformly to F1.
- **RQ-B:** mô hình Conv1d trên chuỗi frame Wav2Vec2 layer 6, dùng rank-warped temporal window để chấm N+1 gaps, đã train/eval 4-fold seed101. AP-I pooled `.012828`, chỉ nhỉnh hơn static-local `.011473` khoảng `.00136`, nhưng P@560 `.01964` thấp hơn static `.02143`; cả hai rất gần prevalence `.009126` và xa MB03 neural `.162125`. Không có bằng chứng localization có ích hoặc cạnh tranh; đây là negative result cho rank-warp + local CNN, không bác bỏ mọi alignment method.
- **RQ-C:** adaptation phonological-attribute của Shahin & Ahmed (Interspeech 2024) đạt AP-S/O `.23462/.12643`; dưới historical neural PLAIN `.42321/.27671` và E4 `.31972/.25936` trên grouped TRAIN references. Đây là adaptation, không phải reproduction trung thực; không có insertion output.
- Không có full S/O/I operating result hoặc Macro-3 F1; không có cơ sở train unified model.

## VERIFIED — triển khai, dữ liệu và provenance

### Population và folds

Bốn outer folds giống official TRAIN grouping: F0 ABA/ASI/BWC; F1 EBVS/HJK/HKK; F2 HQTV/LXC/NJS; F3 PNV/SKA/SVBI. Supports target theo fold: S/O = 1,859/357; 1,524/301; 2,079/773; 1,919/394. Exact-gap insertion supports = 134/127/150/149 (560 tổng). H-002 extraction receipt xác nhận Wav2Vec2 Base 960h ASR-finetuned, hidden state 6, predicted CharSiU monotonic alignment + intended phone, không dùng gold timing, realized phones hay labels để tạo features. MB03 frozen Wav2Vec2 hidden state 6 receipt xác nhận tương tự không dùng gold timing/realized phones/error labels.

### Nested direct neural + phone prior (Agent A)

`RQ1-MB07-A-NESTED-DIRECT-S101` chạy CPU trong 138.06 s. Với mỗi outer fold: giữ kín 3 speaker; ba inner folds giữ 3 speaker từng lượt; base direct C/S/O MLP fit trên 6 speaker và chấm inner-held speakers; meta-model train bằng inner OOF predictions + fold-specific phone priors. Inner prior fit riêng trên inner training speakers; outer prior và outer base fit trên 9 outer-training speakers. Standardization và regularized multinomial logistic (`C=.1`) fit inner-only; fixed control là 0.5 average giữa neural/prior simplex. Outer probability đầu ra giữ C/S/O simplex.

Independent file audit: 59,562 unique `(utterance_id,slot_idx)` IDs khớp authoritative TRAIN manifest, speaker/label counts chính xác; mỗi inner OOF index set đầy đủ outer-train và loại outer-test. Outer prediction SHA-256 `f6b6279ca8569052f49b8a8566b92d99383d19b1cb2a8e05e21cf10a2282b823`; inner OOF SHA-256 `6cfbd3379e2afdf574d7edbbeab844da842b2178898c4a316784b530c58663cd`.

Pooled AP-S/O: direct `.38408/.26565`; prior `.30696/.08833`; fixed `.38251/.24001`; learned neural+prior without E4 `.36748/.21949`. Arithmetic mean fold AP-S/O: direct `.38635/.27055`; fixed `.39299/.26052`; learned-no-E4 `.38378/.23804`. Learned neural+prior alone giảm O ở 4/4 folds; S chỉ tăng ở F2. Fixed 0.5 mean tăng S ở 3/4 folds nhưng giảm S pooled nhẹ, và giảm O ở cả 4 folds. H-002 MLP không cùng kiến trúc MB03 PLAIN.

### Nested CTC/E4 + neural integration (Agent A, seed101)

12 genuine CTC linear heads được train trên sáu TRAIN speakers mỗi inner fold (8 epochs, cached continuous Wav2Vec2-H6 frames). CTC/E4 competitor statistics fit trên chính sáu speakers đó; held inner three speakers được chấm. Outer E4 scores được recover từ MB05 seed101 với chín speaker fit khớp chính xác từng outer fold. Neural inner OOF, phone prior inner-fit và E4 scores join cùng target events. Scaler và multinomial logistic (`C=.1`) fit hoàn toàn trên inner OOF; outer labels chỉ dùng evaluation. Auxiliary E4 scores được biểu diễn bằng `softmax([0,SLLR,OLLR])` cho model fusion; đây không phải tuyên bố E4 đã calibrated.

| Outer fold | Held speakers | Neural AP-S/O | E4-only AP-S/O | Fixed neural+prior AP-S/O | Fixed neural+E4 AP-S/O | Learned neural+prior+E4 AP-S/O |
|---|---|---:|---:|---:|---:|---:|
| F0 | ABA/ASI/BWC | .40295/.23219 | .25082/.20845 | .41317/.21203 | .40196/.31024 | **.44105/.32227** |
| F1 | EBVS/HJK/HKK | .33507/.18259 | .25663/.15172 | .33825/.17650 | .36565/.19655 | **.38479/.21394** |
| F2 | HQTV/LXC/NJS | .41098/.37090 | .30908/.32852 | .44732/.36571 | .43698/.46124 | **.50390/.46618** |
| F3 | PNV/SKA/SVBI | .39640/.29652 | .27367/.19158 | .37321/.28783 | .41013/.35186 | **.41156/.36790** |
| **Pooled OOF** | all 12 TRAIN speakers | .38408/.26565 | .27222/.22901 | .38251/.24001 | .40274/.33078 | **.42328/.32509** |

Fixed neural+E4 is the predeclared 0.5 arithmetic mean of two valid simplexes (not a trained model, no outer-population normalization); pooled gain vs neural is AP-S `+.01866`, AP-O `+.06513`. It wins S in 3/4 and O in 4/4 outer-fold AP. Learned E4 fusion vs neural-only gains in every outer fold: AP-S `+.03810,+.04971,+.09292,+.01515`; AP-O `+.09008,+.03135,+.09528,+.07137`. It also beats fixed neural+prior in both classes in all folds. Compared with fixed neural+E4, learned AP is better in every fold; pooled AP-S is higher, but pooled AP-O `.32509` is slightly below `.33078` for the fixed blend, while arithmetic mean fold AP favors learned (`.34257` vs `.32997` O). This pooled/mean-fold reversal reflects cross-fold score distribution and must be reported, not hidden. Against historical MB03 neural PLAIN `.423209/.276706`, pooled seed101 S is essentially tied (`+.000073`) while O is higher by `.04838`; S does not win in every fold. The seed202 replication also ties pooled S and improves O; both seeds show the learned model above its newly fitted direct H-002 base in every fold. Against the stronger historical MB03 PLAIN reference, O AP improves in 4/4 folds for both seeds, while S improves in only 2/4 folds and pooled S is effectively tied. This is not a blind external cohort.

Receipts: `agent_a/ctc_nested_e4_seed101_receipt.json`, `agent_a/nested_e4_fusion_seed101_receipt.json`; 12 inner checkpoints `agent_a/ctc_outer*_inner*_seed101.pt`. The fusion receipt verifies 59,562 unique outer IDs, 178,686 inner outer-fold event scores and no outer-test labels in fitting.

### Seed202 repetition

Same folds, training budgets, CTC/E4 construction and inner-only meta procedure; a second 12 inner CTC heads trained on six-speaker sets. Direct nested refit runtime 145.45 s CPU; CTC/E4 inner runtime 613.05 s CPU. Learned fusion again improves over its direct neural base in S/O in all folds:

| Fold | Neural AP-S/O | Learned neural+prior+E4 AP-S/O |
|---|---:|---:|
| F0 | .40655/.27215 | .43423/.34264 |
| F1 | .34530/.14365 | .39302/.21931 |
| F2 | .40522/.34507 | .50129/.46263 |
| F3 | .38008/.27040 | .40632/.35310 |
| **Pooled** | .38017/.25608 | **.42300/.32862** |

Pooled gains are `+.04283` AP-S and `+.07254` AP-O (seed101: `+.03920/+ .05943`). Fixed 0.5 neural+E4 composition gives `.40207/.33400`; its pooled O AP slightly exceeds learned meta, another reason not to claim learned fusion is uniformly best. Both seeds show a replicated ranking gain over neural and neural+prior controls, with omission gains also exceeding historical MB03 PLAIN O AP. This replication shares the same 12 speakers/folds and is not an independent cohort.

### Acoustic insertion (Agent B)

`RQ1-MB07-B-SEQACOUSTIC-S101`, seed101, CUDA GTX 1650, 10 epochs × 150 steps/fold, batch 256, 314,017 parameters, runtime 80.86 s. Input is frozen `facebook/wav2vec2-base-960h` snapshot `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, layer 6 continuous hidden frames (768D, 20 ms). For gap j, center is rank-warp estimate `round((j+0.5)T/(N+1)-0.5)`, then 9-frame window; temporal CNN two Conv1d layers + MLP. Matched static control uses local mean/std/endpoints. No held-out gold timing/realized-phone inference; no DEV use.

| Fold | MB03 neural AP-I | MB07 temporal AP-I | Static-local AP-I | Gap positives |
|---|---:|---:|---:|---:|
| F0 | .188792 | .011852 | .017313 | 134 |
| F1 | .102708 | .010431 | .009209 | 127 |
| F2 | .109449 | .015074 | .011194 | 150 |
| F3 | .251541 | .015844 | .012924 | 149 |
| **Pooled** | **.162125** | **.012828** | **.011473** | **560** |

Temporal P@560 `.01964` (11 positives); static `.02143` (12); neural `.24286` (136). MB03 NPZ lacks row event IDs; source-level manifest order/producer function and fold counts were audited. Separately, new MB07 gap scores have 61,362 unique explicit IDs (`utterance_id::gap_idx`) and exact manifest labels. No second seed.

**GPU lease deviation:** MB07-B receipt says CUDA but no MB07 entry exists in shared GPU lease ledger. GPU was idle when initially checked, but the required cooperative lease was not receipted. Result retained; no redundant rerun.

### External method (Agent C)

Shahin & Ahmed, “Phonological-Level Mispronunciation Detection and Diagnosis,” Interspeech 2024, DOI `10.21437/Interspeech.2024-2217` (paper’s stated approach: SSL acoustic features predict phonological attributes; compare canonical and recognized attributes). LANGFENS adaptation predicts 16 local articulatory attributes from H-002 11-frame crop means using only correct TRAIN targets, then computes mean absolute mismatch against intended-phone attribute vector. Four grouped folds, 10 epochs, CPU, 26.03 s. Fidelity is partial/adaptation: not the 35-attribute sequence-alignment method, original data recipe or auxiliary corpus. Public code/weights availability was not documented and is unverified.

| Fold | n | supports S/O | AP-S | AP-O |
|---|---:|---:|---:|---:|
| F0 | 14,816 | 1,859/357 | .24979 | .09215 |
| F1 | 14,848 | 1,524/301 | .19646 | .08440 |
| F2 | 14,989 | 2,079/773 | .24148 | .22210 |
| F3 | 14,909 | 1,919/394 | .25833 | .10289 |
| **Pooled** | **59,562** | **7,381/1,825** | **.23462** | **.12643** |

Supervisor joined all 59,562 rows by `(utterance_id,slot_idx)` to TRAIN manifest; labels, reference slots, speaker IDs and supports verified. The method emits target-only scores; Macro-3 and insertion are unsupported/not reported.

## Target-only warning-policy evaluation

For each outer fold and system, a max(P(S),P(O)) threshold was selected from inner OOF to target budgets 0.2/0.5/1/2 warnings per utterance, then frozen on the untouched outer fold. Learned meta thresholds used a second inner-group cross-fit of the meta learner. A fixed SHA-256 event-ID tie-break handles equal scores; no outer-population ranking or outer labels set thresholds. This is target-only and is not probability reliability calibration.

| Seed | Budget | Neural target Macro-2 F1 (realized FWR) | Fixed neural+E4 Macro-2 F1 (FWR) | Learned neural+prior+E4 Macro-2 F1 (FWR) |
|---:|---:|---:|---:|---:|
| 101 | 0.2 | .030 (.105) | **.093 (.223)** | .027 (.099) |
| 101 | 0.5 | .076 (.294) | **.158 (.495)** | .078 (.344) |
| 101 | 1.0 | .142 (.686) | **.226 (.985)** | .171 (.902) |
| 101 | 2.0 | .248 (1.602) | **.319 (1.966)** | .290 (1.841) |
| 202 | 0.2 | .044 (.160) | **.092 (.231)** | .031 (.112) |
| 202 | 0.5 | .089 (.398) | **.150 (.515)** | .084 (.377) |
| 202 | 1.0 | .154 (.802) | **.227 (1.009)** | .175 (.942) |
| 202 | 2.0 | .252 (1.792) | **.325 (2.021)** | .282 (1.845) |

Fixed neural+E4 has the highest target Macro-2 F1 at every budget in both seeds; learned fusion beats neural-only at budgets 1/2 but not consistently at 0.2/0.5. Thus AP improvement does not imply a superior warning policy. Full class P/R/F1, false warnings, thresholds and realized target FWR are in `METHOD_BENCH_07_TARGET_WARNING_BUDGETS.csv` and `_SUMMARY.csv`. No insertion warning was invented, so these are not complete-system FWR or Macro-3 results.

## MEASUREMENTS versus exploration versus published claims

**Executed measurements:** nested neural/E4/prior/fixed/meta AP and target-only warning-budget P/R/F1 above; insertion grouped AP/P@K; phonological-feature adaptation grouped AP. These are TRAIN cross-speaker results on the 12 known speakers, not independent external confirmation. AP is ranking-only, not calibrated posterior.

**Historical controls:** MB03 PLAIN grouped OOF AP-S/O/I `.423209/.276706/.162125`; MB05 E4 S101 mean `.31972/.25936/.03159`; MB06 pooled rank-mean `.44547/.35708` remains exploratory due pooled score normalization. Historical outer OOF scores are controls, not inner training features. MB06 neural+E4 rank blend is not an established improvement.

**Published claim:** original Shahin & Ahmed paper describes phonological-attribute scoring and reports its own dataset results. Those published results do not establish performance on LANGFENS. No second external method was implemented in this campaign.

**Exploratory/inconclusive:** nested E4 AP direction replicated at seed202, but only two seeds/four known speaker groups were evaluated. Normalization-choice sensitivity, probability calibration and external confirmation remain open. Temporal rank warping lacks predicted alignment uncertainty. DEV_CAL policy selection, DEV_SEARCH tuning and protected test were not used.

## Answers to the final scientific questions

1. **Full nested E4+neural integration completed?** Yes for seeds101 and202: each has four outer × three inner grouped folds, 12 new six-speaker CTC models, inner-fold E4 statistics and exact-fold outer E4 scores.
2. **Inner predictions leakage-safe?** Yes for neural, phone prior and E4. Inner base models/statistics use only six inner-training speakers; 59,562 outer event IDs join uniquely; outer-test labels are evaluation-only.
3. **Learned E4+neural beats neural-only?** Yes in both seeds: seed101 pooled `.42328/.32509` vs `.38408/.26565`; seed202 `.42300/.32862` vs `.38017/.25608`. Both S and O improve in all four outer folds at each seed.
4. **Beats neural+phone-prior?** Yes: learned E4 fusion beats the fixed 0.5 neural+prior control pooled and in every fold. E4-only remains weaker than neural-only.
5. **Gain concentrated S or O?** Both improve; pooled Δ AP-S `+.03920`, AP-O `+.05943`. Against historical MB03 PLAIN frontier, S is essentially tied while O improves `.04838`.
6. **Fold/seed stability?** Both seeds improve AP-S/O versus their direct base in 4/4 folds; pooled gains are similar across seeds. This supports repeatability on these 12 speakers, not external generalization.
7. **Normalization robustness?** Only one inner-only mean/std feature transform tested. No sensitivity comparison; no outer-test percentile normalization.
8. **F1/warning trade-off improved?** Target-only S/O budgets 0.2/0.5/1/2 were evaluated with inner OOF threshold selection and untouched outer evaluation. Fixed neural+E4 had highest target Macro-2 F1 at every budget/seed; learned fusion was not uniformly best. No insertion warnings or complete S/O/I Macro-3/FWR.
9. **What insertion model trained?** Frozen W2V2 layer-6 continuous frame-window rank-warp CNN, exact N+1 output.
10. **New acoustic localization information?** Not demonstrated. Weak AP near prevalence and only +.00136 over static AP; P@560 below static. Model did not estimate true phone boundaries.
11. **Beat MB03 AP-I .162125?** No; `.012828` pooled versus `.162125`.
12. **Stable across speakers?** No insertion gain: AP-I folds `.01185/.01043/.01507/.01584`, all far below neural. Insertion seed repetition was not run.
13. **External methods investigated?** Shahin & Ahmed Interspeech 2024 phonological-attribute method was source-checked and adapted. CTC monotonic alignment literature (Graves et al. 2006) informed the insertion rationale; no claim of reproducing that paper.
14. **New method and benchmark result?** 16-attribute correct-only predictor; pooled `.23462/.12643`, below neural/E4 references. It is an adaptation, not faithful reproduction.
15. **Beat matched control?** It beats class prevalence but no relevant strong model; comparison to neural/E4 is same grouped population but not one-factor-matched architecture/supervision. No insertion output.
16. **Negative hypotheses established?** Rank-warped temporal CNN does not improve on neural and is near static/chance; current phonological-attribute adaptation does not compete with direct neural/E4. These do not refute all temporal alignment or phonological methods.
17. **Evidence to train a unified model?** No. E4 ranking complementarity replicated, but target warning policy does not favor the learned fusion, there is no full insertion-inclusive policy gain, and insertion model is negative.
18. **Next justified experiment?** Test a reference-conditioned monotonic alignment with uncertainty and an explicit optional insertion state for exact gaps, on the same speaker folds. Then repeat target warning policies with a valid fixed insertion scorer to produce a complete S/O/I system; do not use held-out gold timing or tune on DEV_SEARCH.

## Negative, unresolved and limitations

- Seed101/202 AP direction is replicated; normalization sensitivity, probability calibration, full S/O/I operating policy, external speaker test and unified architecture are unresolved/not established.
- The rank-warp model does not test learned alignment; insertion negative scope is narrow.
- The CTC prediction artifact MB03 has no embedded gap IDs; provenance relies on verified producer ordering. New MB07 insertion artifact and direct nested target artifact persist explicit event IDs.
- TRAIN folds have been explored in prior campaigns; nested design prevents stack leakage but is not a blind external validation.
- Target-only S/O warning budgets were evaluated, but no insertion warning was added; no full-system FWR or Macro-3 F1 claim is made.

## Artifact index

- Agent A: `experiments/rq1_method_bench_07/agent_a/`; `reports/METHOD_BENCH_07_A_NESTED_COMPLEMENTARITY.md`.
- Agent B: `experiments/rq1_method_bench_07/agent_b/`; `reports/METHOD_BENCH_07_B_ACOUSTIC_INSERTION.md`.
- Agent C: `experiments/rq1_method_bench_07/agent_c/`; `reports/METHOD_BENCH_07_C_EXTERNAL_METHODS.md`.
- Nested folds/P@K: `reports/METHOD_BENCH_07_NESTED_FOLD_RESULTS.csv`; target PR curves `reports/METHOD_BENCH_07_NESTED_PR_CURVES.csv.gz`.
- Insertion ablations/PR: `reports/METHOD_BENCH_07_INSERTION_ABLATIONS.csv`; `reports/METHOD_BENCH_07_INSERTION_PR_CURVES.csv.gz`.
- Target-only warning policy: `reports/METHOD_BENCH_07_TARGET_WARNING_BUDGETS.csv` and `_SUMMARY.csv`; generator `experiments/rq1_method_bench_07/supervisor/evaluate_nested_target_warning_budgets.py`.
- Leaderboard, verdicts, reproduction and handoff: `reports/METHOD_BENCH_07_EXPERIMENT_LEADERBOARD.csv`, `reports/METHOD_BENCH_07_HYPOTHESIS_VERDICTS.md`, `reports/METHOD_BENCH_07_REPRODUCTION.md`, `reports/METHOD_BENCH_07_NEXT_RESEARCH.md`.
