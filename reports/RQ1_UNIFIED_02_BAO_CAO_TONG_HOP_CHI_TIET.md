# LANGFENS RQ1-UNIFIED-02 — Báo cáo tổng hợp chi tiết

**Ngày chốt:** 2026-11-08  
**Phạm vi:** campaign RQ1-UNIFIED-02; các score, mô hình, ablation và kiểm tra đã lưu trong repository.  
**Kết luận ngắn:** attribution ở cấp score xác định nhánh insertion/gap của U2-S101 là nguồn gần trực tiếp làm giảm Macro-3 AP. Tuy nhiên các thí nghiệm chưa xác định được nguyên nhân cơ học. Thử nghiệm liên tục quanh predicted CTC gap boundary đã chạy và cho kết quả âm tính trên raw AP. Không có unified neural candidate mới nào cho cải thiện ổn định vượt Gold-Only/H-002.

## 1. Kết luận điều hành

1. Trên cùng population và exact N+1 gap identities, thay riêng U2 gap scores bằng Gold-Only gap scores làm AP-I tăng **.117992→.186715** và Macro-3 AP **.224890→.247798**. Vì vậy gap-score branch gây regression ở phép attribution đóng băng này; đây chưa phải bằng chứng xác định vì sao neural gap head yếu.
2. U2 gap head overfit TRAIN mạnh: AP-I/AUC **.976979/.99986**, so CAL **.138111/.81530** và SEARCH **.117992/.78343**. Bottleneck, balanced sampling, 6D controls, concatenation, crop-sequence, broad context và auxiliary CTC không tạo cải thiện chuyển miền lặp lại.
3. G3 acoustic+6D tăng CAL AP-I trên ba seed, nhưng paired SEARCH deltas là **+.01617/−.00183/−.03332**; mean AP-I **.132672 so với U2 .138999**, mean Macro-3 AP **.225786 so với .227894**. Không chọn G3 làm candidate.
4. G9 thêm sáu thống kê CTC CharSiU vào G3: SEARCH AP-I **.155957** ở seed101 nhưng **.132255** ở seed202 (matched U2 .137553); không lặp lại được uplift, và đây là auxiliary second-encoder arm.
5. Thử nghiệm G6-PREDICTED-BOUNDARY dùng contiguous 17-frame full-wave W2V sequence quanh predicted CTC centers, không gold timing. AP-I TRAIN/CAL/SEARCH **.832383/.102291/.080151**; SEARCH Macro-3 AP **.212276**, thấp hơn U2 **.224890**. Cấu hình cụ thể này không cải thiện transfer.
6. Raw AP và CAL20 threshold-policy F1/confusions được tách riêng. FWR được báo cáo mô tả, không dùng **≤.20** làm success gate. DEV_SEARCH đã exposed, do đó tất cả SEARCH so sánh chỉ mang tính exploratory.
7. Không truy cập protected holdout; không dùng gold timing, realized phones hay error/insertion labels làm inference features. Score compositions EXP29/36/43 và Phase-4 fusion không được gọi là learned unified checkpoints.

## 2. Phạm vi, population và giao thức đánh giá

### 2.1 Roles và sự kiện

Giữ nguyên roles **TRAIN / DEV_CAL / DEV_SEARCH**. DEV_SEARCH được dùng để mô tả kết quả sau các lần chạy; vì vậy không dùng như dữ liệu huấn luyện/model-selection và các SEARCH finding không được diễn giải confirmatory.

| Split | Target events | Exact gaps (N+1) | Insertion positives | Tỷ lệ insertion |
|---|---:|---:|---:|---:|
| TRAIN | 59,562 | 61,362 | 560 | 0.9126% |
| DEV_CAL | 10,043 | 10,343 | 102 | 0.9862% |
| DEV_SEARCH | 19,755 | 20,354 | 125 | 0.6141% |

DEV_SEARCH gồm 599 utterances. Trong 125 positive insertion gaps, **124 internal**, leading **0/599**, trailing **1/599**. Vì vậy AP/rates ở leading/trailing riêng không ổn định.

### 2.2 EVAL-V3.1 và metric semantics

- Target classes: substitution (S), omission (O), correct; insertion (I) được xếp hạng trên từng exact gap của N+1 gaps.
- Macro-3 AP là trung bình AP-S/AP-O/AP-I; đây là ranking metric, không phụ thuộc threshold.
- CAL20 là threshold policy được chọn trên DEV_CAL theo budget cảnh báo 0.20 warnings/utterance. P/R/F1, confusion và FWR phụ thuộc threshold. Báo raw AP riêng với policy output.
- EVAL-V3.1 yêu cầu đủ target/gap coverage; event IDs và thứ tự exact gaps phải khớp.
- FWR được báo cáo như trade-off; **không** là ngưỡng thành công.

## 3. Baselines và branch attribution (Q1)

EVAL-V3.1 trên cùng 19,755 target events/20,354 exact gaps; chỉ thay score branch trong các composition được chỉ rõ.

| Target scores | Gap scores | AP-S | AP-O | AP-I exact-gap | Macro-3 AP | Chú thích |
|---|---|---:|---:|---:|---:|---|
| U2-S101 | U2-S101 | .371360 | .185318 | .117992 | .224890 | U2 baseline |
| U2-S101 | Gold-Only | .371360 | .185318 | .186715 | .247798 | Frozen score substitution |
| H-002 | U2-S101 | .377810 | .167294 | .117992 | .221032 | Frozen substitution |
| H-002 | Gold-Only | .377810 | .167294 | .186715 | .243940 | H-002 baseline |
| U2-S101 | EXP43 gap | .371360 | .185318 | .285056 | .280578 | Diagnostic composition, **không phải trained unified model** |

- U2→Gold-Only gap: AP-I **+.068723**, Macro-3 AP **+.022908**.
- Gold-Only→U2 net Macro-3 AP delta là **−.012027**; gap replacement phục hồi **+.022908**. U2 target branch giúp đặc biệt ở omission, bù một phần insertion loss.
- So Gold-Only: U2 target AP-S tăng **+.006758**, AP-O tăng **+.025885**, trong khi AP-I của gap branch thấp hơn **−.068723**.
- Kết luận giới hạn: attribution chứng minh U2 gap score gây giảm ranking tại phép thay score seed101; không chứng minh nguyên nhân training/architecture của gap head.

## 4. Feature geometry và overfit gap head

### 4.1 U2 input thực tế

Trong `experiments/rq1_unified_01/agent_a/train_twohead.py`, mỗi internal gap ghép:

- 768D mean 5 frame cuối crop trái, offsets `[11:16]`;
- 768D mean 5 frame đầu crop phải, offsets `[1:6]`;
- hai intended-phone one-hot 39D (trái/phải).

Tổng **1,614D**; crops 17 frame lấy quanh predicted phone centers trên frozen normalized Wav2Vec2 `hidden_states[6]`. Đây là crop-relative pooled features gần các rìa predicted-center crops, không phải contiguous sequence của gap interval. Utterance-edge neighbor dùng zero sentinel. Không dùng gold timing, realized phone hoặc error labels làm feature.

### 4.2 U2 overfit

| Split | Gaps | Positives | AP-I | ROC-AUC | Positive-score median |
|---|---:|---:|---:|---:|---:|
| TRAIN | 61,362 | 560 | .976979 | .99986 | .99764 |
| DEV_CAL | 10,343 | 102 | .138111 | .81530 | .000527 |
| DEV_SEARCH | 20,354 | 125 | .117992 | .78343 | .000833 |

Đây là evidence mạnh của train/evaluation mismatch và poor transfer; không quy được nguyên nhân duy nhất cho 1,614D. Covariance sample TRAIN cho mỗi acoustic block: effective rank khoảng **235.6/768**, top eigenvalue khoảng **3.9%**, top 10 khoảng **25.6% variance**; có redundancy nhưng không suy biến hoàn toàn.

### 4.3 Gap-head screens

Raw AP-I, Macro-3 AP trên SEARCH; policy F1 là CAL20. SEARCH exposed, không phải model selection.

| Arm / seed | Params | TRAIN AP-I | CAL AP-I | SEARCH AP-I | SEARCH Macro-3 AP | CAL20 Macro-3 F1 | GPU info |
|---|---:|---:|---:|---:|---:|---:|---|
| G0 U2 1614→256→64 / 101 | 430,018 | .976979 | .138111 | .117992 | .224890 | .125421 | 24.881 s; 41.327 MiB |
| G1 bottleneck 1614→128→64 / 101 | 215,106 | .978514 | .136409 | .125929 | .227536 | .140180 | 7.860 s; 27.899 MiB |
| Balanced 64+/64− / 101 | 430,018 | .992556 | .094039 | .125692 | .227457 | .149231 | 6.841 s; 36.811 MiB |
| G2 6D gap_X only / 101 | 9,282 | — | .036339 | .025938 | .194205 | .087606 | 7.204 s; 18.169 MiB |
| G3 1614D+6D / 101 | 431,554 | .976238 | .174269 | .134162 | .230280 | .124811 | 10.341 s; 30.063 MiB |
| G3 1614D+6D / 202 | 431,554 | .989427 | .138015 | .135722 | .222813 | .113683 | 10.455 s; 30.063 MiB |
| G3 1614D+6D / 303 | 431,554 | .986811 | .134810 | .128132 | .224263 | .102671 | 10.134 s; 30.063 MiB |
| G6 paired crop-edge sequence, no refs / 101 | 168,258 | — | .062795 | .108997 | .221892 | .110666 | 32.109 s; 83.962 MiB |
| G6-REF paired crop-edge + phone refs / 101 | 173,250 | — | .100776 | .099673 | .218784 | .119996 | 35.797 s; 84.130 MiB |
| G6 continuous predicted-boundary W2V17 + refs / 101 | 173,250 | .832383 | .102291 | .080151 | .212276 | .130327 | 193.920 s GPU lease*; 1,136 MiB monitored |
| G8 all 17 crop frames per neighbor + refs / 101 | 173,250 | — | .079218 | .116505 | .224394 | .146075 | 54.808 s; 140.161 MiB |
| G9 G3+CTC6 / 101 | 433,090 | .996735 | .159784 | .155957 | .237545 | .118343 | 10.272 s; 30.099 MiB |
| G9 G3+CTC6 / 202 | 433,090 | .991220 | .134220 | .132255 | .221657 | .113672 | 9.930 s; 30.099 MiB |

`*` G6-PREDICTED lease time includes full-wave extraction, training/inference and process overhead, not pure optimizer time. `nvidia-smi` monitored up to 1,136 MiB; PyTorch allocated peak was not captured. Original process returned nonzero only at JSON serialization after training/predictions; receipt was reconstructed CPU-only without model forward or CUDA rerun.

#### G3 paired seed results

| Seed | U2 CAL AP-I | G3 CAL AP-I | Δ CAL | U2 SEARCH AP-I | G3 SEARCH AP-I | Δ SEARCH | U2 SEARCH M3 AP | G3 SEARCH M3 AP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | .138111 | .174269 | +.036158 | .117992 | .134162 | +.016170 | .224890 | .230280 |
| 202 | .128568 | .138015 | +.009447 | .137553 | .135722 | −.001831 | .223423 | .222813 |
| 303 | .117111 | .134810 | +.017699 | .161451 | .128132 | −.033319 | .235369 | .224263 |
| Mean | .127930 | .149031 | +.021101 | .138999 | .132672 | −.006327 | .227894 | .225786 |

CAL AP-I tăng ở cả ba seed, nhưng exposed SEARCH trung bình giảm và S303 đảo mạnh. Đây không phải repeatable generalization gain; không lắp integrated candidate từ kết quả này.

#### G9 paired results

| Seed | AP-I CAL | Matched U2 CAL | Matched G3 CAL | AP-I SEARCH | Matched U2 SEARCH | Matched G3 SEARCH | SEARCH Macro-3 AP | CAL20 Macro-3 F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | .159784 | .138111 | .174269 | .155957 | .117992 | .134162 | .237545 | .118343 |
| 202 | .134220 | .128568 | .138015 | .132255 | .137553 | .135722 | .221657 | .113672 |

G9-S101 cho SEARCH gain, G9-S202 thì thấp hơn cả matched U2 và G3. Cả hai CAL thấp hơn G3. G9 trộn frozen CharSiU CTC features với W2V+gap_X; không phải same-encoder ablation.

#### Direct contiguous predicted-boundary test — G6-PREDICTED

- Input: 17 contiguous W2V `hidden_states[6]` frames quanh mỗi predicted exact-gap center, cộng hai intended-phone 39D vectors; Conv1d model, 173,250 parameters; seed101; 8 epochs, batch128, 480 steps/epoch.
- Center: archived CharSiU CTC logits aligned tới intended `target_phone`; map `round(CTC center × N_W2V / N_CTC)`, clip rounded endpoint vào W2V frame valid; pad zero ngoài audio. CTC/W2V stride lần lượt 160/320 samples.
- Validation: 2,699 WAV/CTC source records; 89,360 phone identity checks; 2,699 frame-clock checks; first utterance `ABA_arctic_a0003`, CTC 490 frames → W2V 245, 78,720 samples; reproduced H-002 target crop cache với max abs delta **.0009765625** (limit 2e-3). Exact target/gap IDs passed.
- Features chỉ dùng frozen full-wave W2V, archived CTC posterior và intended phones; không gold timing, realized phone hoặc correctness/insertion labels.
- AP-I TRAIN/CAL/SEARCH = **.832383/.102291/.080151**. SEARCH Macro-3 AP **.212276** vs U2 .224890. CAL20 Macro-3 F1 **.130327** vs U2 .125421; Macro4 **.334386**, FWR **.193656**; exact-gap TP/FP/FN/TN **13/27/112/20,202**.
- Đây là kết quả âm tính cho cấu hình contiguous center-window cụ thể này; không chứng minh mọi center/width/representation đều sai.

## 5. Target model và U2/U3 matched-seed comparison

### 5.1 U2 target scores vs H-002

| Target branch | AP-S | AP-O | Ý nghĩa |
|---|---:|---:|---|
| H-002 | .377810 | .167294 | Better S ranking |
| U2-S101 | .371360 | .185318 | Better O ranking |
| U2−H-002 | −.006451 | +.018024 | Trade-off; không có universal U2 target advantage |

### 5.2 U2 flat vs U3 hierarchical, SEARCH AP

| Seed | Model | AP-S | AP-O | AP-I | Macro-3 AP | CAL20 Macro-3 F1 |
|---:|---|---:|---:|---:|---:|---:|
| 101 | U2 flat | .371360 | .185318 | .117992 | .224890 | .125421 |
| 101 | U3 hierarchy | .362413 | .176117 | .123822 | .220784 | .07234 |
| 202 | U2 flat | .362558 | .170159 | .137553 | .223423 | .119090 |
| 202 | U3 hierarchy | .358654 | .172383 | .143724 | .224920 | .074159 |
| 303 | U2 flat | .371064 | .173591 | .161451 | .235369 | .097422 |
| 303 | U3 hierarchy | .364913 | .176963 | .115738 | .219205 | .071525 |
| Mean | U2 flat | .368327 | .176356 | .138999 | .227894 | .113978 |
| Mean | U3 hierarchy | .361993 | .175154 | .127761 | .221637 | .072675 |

Paired U3−U2 Macro-3 AP deltas **−.004106/+.001497/−.016164**. CAL20 Macro-3 F1 thấp hơn cho U3 ở cả 3 seed; mean delta **−.041303**. Kết quả không ủng hộ cấu hình hierarchy/loss đã thử; không loại trừ mọi hierarchical model.

### 5.3 Optimization equivalence / shared gradients

`UnifiedTwoHead` có target/gap trainable modules độc lập, shared trainable parameter count **0**. CPU synthetic check trên module thật, cùng initialization, per-head losses, batches, AdamW state, dropout disabled: joint-vs-separate updates cho parameter max delta **0** qua 4 steps. Điều này bác bỏ classic shared-parameter gradient conflict khi schedule/data trùng nhau; không phải acoustic-data training và không loại trừ schedule/batch/upstream coupling.

## 6. Threshold policy, confusion và Precision@K

### 6.1 U2 và direct boundary G6 — SEARCH CAL20 policy

| Arm | Class | TP | FP | FN | TN | Precision | Recall | F1 | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U2-S101 | S | 22 | 24 | 1,746 | 17,963 | .4783 | .0124 | .0243 | .371360 |
| U2-S101 | O | 32 | 65 | 177 | 19,481 | .3299 | .1531 | .2092 | .185318 |
| U2-S101 | I gap | 11 | 18 | 114 | 20,211 | .3793 | .0880 | .1429 | .117992 |
| G6-PREDICTED | S | 22 | 24 | 1,746 | 17,963 | .4783 | .0124 | .0243 | .371360 |
| G6-PREDICTED | O | 32 | 65 | 177 | 19,481 | .3299 | .1531 | .2092 | .185318 |
| G6-PREDICTED | I gap | 13 | 27 | 112 | 20,202 | .3250 | .1040 | .1576 | .080151 |

G6 fixes U2-S101 target score arrays. At CAL20, gap branch finds 2 more insertion TPs but adds 9 FPs; threshold F1 rises modestly while raw AP-I drops. This is why policy F1 cannot replace raw AP. G6 EVAL-V3.1 CAL20: Macro-3 error F1 **.130327**, Macro4 **.334386**, Binary MD AP/F1 **.363315/.057547**, FWR **.193656**. (Binary-MD score is source-provided and separate from classwise error AP.)

### 6.2 U2 unrestricted vs CAL20

| Policy | Macro-3 error F1 | FWR | S recall | O recall | I recall |
|---|---:|---:|---:|---:|---:|
| U2 unrestricted diagnostic | .280346 | 4.654424 | .589932 | .540670 | .248000 |
| U2 CAL20 | .125421 | .178631 | .012443 | .153110 | .088000 |

U2 unrestricted target confusion (gold rows→predicted C/S/O):

| Gold \ Pred | C | S | O |
|---|---:|---:|---:|
| C | 15,300 | 1,846 | 632 |
| S | 634 | 1,043 | 91 |
| O | 66 | 30 | 113 |

Gap unrestricted TP/FP/FN/TN = **31/189/94/20,040**.

U2 CAL20 target confusion:

| Gold \ Pred | C | S | O |
|---|---:|---:|---:|
| C | 17,696 | 22 | 60 |
| S | 1,741 | 22 | 5 |
| O | 175 | 2 | 32 |

Gap CAL20 TP/FP/FN/TN = **11/18/114/20,211**. Substitution recall chỉ **22/1,768=.0124** dù AP-S=.371360; đây là threshold operating point chứ không phải không có ranking signal.

### 6.3 Precision@support trên SEARCH

K bằng số positives thật: S=1,768, O=209, I=125. Recall@K equals TP@K/support.

| Arm | P@support S | P@support O | P@support I | I TP trong top-125 |
|---|---:|---:|---:|---:|
| Gold-Only | .4089 | .2105 | .2960 | 37 |
| H-002 target + GoldOnly gap | .4417 | .2392 | .2960 | 37 |
| U2 | .4395 | .2584 | .1920 | 24 |
| U3-S101 | .4338 | .2536 | .2240 | 28 |
| G1 | .4395 | .2584 | .2080 | 26 |
| Balanced | .4395 | .2584 | .2160 | 27 |
| G2 6D-only | .4395 | .2584 | .0960 | 12 |
| G6 crop sequence | .4395 | .2584 | .1760 | 22 |
| G6-REF | .4395 | .2584 | .1920 | 24 |
| G6-PREDICTED | .4395 | .2584 | .1280 | 16 |
| G8 broad | .4395 | .2584 | .1920 | 24 |
| G9-S101 | .4395 | .2584 | .2480 | 31 |
| G9-S202 | .4344 | .2249 | .2240 | 28 |

## 7. H-002 acoustic controls, G3 perturbations, HNM

### 7.1 H-002 controls

| Control / branch | AP-S | AP-O | Macro-3 AP | Ý nghĩa |
|---|---:|---:|---:|---|
| H-002 full acoustic + reference | .377810 | .167294 | .243940 | Baseline; gap scores Gold-Only |
| Reference-only retrained | .293833 | .024503 | .168350 | Retrained reference control |
| Acoustic shuffle inference perturbation | .239753 | .023376 | .149948 | OOD inference perturbation |
| Zero-reference inference perturbation | .095798 | .011834 | .098116 | OOD inference perturbation |

Các controls cho thấy target branch phụ thuộc acoustic/reference evidence; vì H-002 giữ Gold-Only gap scores, đây không phải kiểm định insertion acoustics.

### 7.2 G3-S101 inference perturbations (OOD)

Target U2 scores cố định; gap AP-I baseline **.134162**.

| Perturbation | AP-I | Macro-3 AP | CAL20 gap TP/FP/FN/TN |
|---|---:|---:|---|
| G3 baseline | .134162 | .230280 | 11/20/114/20,209 |
| Zero W2V blocks | .046932 | .201203 | 2/2/123/20,227 |
| Shuffle W2V blocks | .011221 | .189299 | 0/0/125/20,229 |
| Zero 6D gap_X | .083611 | .213430 | 7/11/118/20,218 |
| Reference-only | .019925 | .192201 | 0/0/125/20,229 |

Các perturbations cho thấy learned gap scorer nhạy với acoustic/6D inputs nhưng phân phối bị perturb ngoài support; không phải retrained modality ablations.

### 7.3 Historical hard-negative mining

Historical GRU64: top-128 TRAIN gold-negative penalty λ=.1, batch32, 12 epochs, 51D/6D input (khác U2). SEARCH AP S/O/I/Macro3 **.368680/.159243/.185065/.237663**. So với Gold-Only reference .364602/.159433/.186715/.236917: Macro3 AP chỉ **+.000746**, AP-I **−.001650**. CAL_MAX Macro3 F1 **.292242** vs Gold-Only .30014; Macro4 .435748; FWR **6.716194**. Historical candidate không phải matched U2 recipe, và không vượt H-002 M3 AP .243940.

## 8. Learned checkpoints vs score compositions / expert frontier

| Arm | Type | AP-S | AP-O | AP-I | Macro-3 AP | Diễn giải |
|---|---|---:|---:|---:|---:|---|
| Gold-Only | Neural baseline | .364602 | .159433 | .186715 | .236917 | Baseline |
| H-002 | Neural target + GoldOnly gap | .377810 | .167294 | .186715 | .243940 | Unified reference system |
| U2-S101 | Unified neural | .371360 | .185318 | .117992 | .224890 | Insertion gap weak |
| HNM-GRU64 | Unified neural, historical | .368680 | .159243 | .185065 | .237663 | +.000746 M3 vs GoldOnly, <H-002 |
| EXP24 | Standalone learned checkpoint replay | .357737 | .225092 | .247628 | .276819 | Not a unified RQ1-02 checkpoint |
| EXP29 | Fixed score composition | .378062 | .196539 | .272068 | .282223 | Composition, not trained model |
| EXP36 | Fixed score composition | .387088 | .204687 | .275924 | .289233 | Composition, not trained model |
| EXP43 | Fixed score composition | .363453 | .205771 | .285056 | .284760 | Composition, not trained model |
| Phase4 omission fusion | Fixed score composition, 5-seed mean | — | — | — | .301800 | Not neural checkpoint |
| Phase4 headwise blend | Fixed score composition, 5-seed mean | — | — | — | .295500 | Not neural checkpoint |

Historical HNM Macro-3 AP gaps vs EXP24/EXP36/EXP43/Phase4 omission fusion are −.039156/−.051570/−.047097/−.064137. OOF teacher audit did not find valid speaker-held-out TRAIN predictions for EXP24/36/43; therefore Phase-4 specialist distillation was not tested. DEV predictions are not used as teacher targets; no OOF scores were fabricated.

## 9. Hypothesis verdicts

| Q | Verdict | Summary evidence |
|---|---|---|
| Q1 gap branch attribution | **Supported** for U2-S101 Macro-3 AP regression | Frozen swap restores M3 AP +.022908; causal mechanism remains unknown. |
| Q2 dimensionality | **Inconclusive** | Train-transfer gap is severe; G1 halves parameters but CAL AP falls, SEARCH gain is one seed; G3 changes feature content and fails SEARCH mean. |
| Q3 localization/region | **Inconclusive** | Feature geometry is edge pooling; G6-PREDICTED continuous window negative, but no systematic region/width sweep. Borderline positives too few. |
| Q4 rare positives | **Partially supported as a condition, not dominant cause** | TRAIN prevalence .9126%; balancing hurts CAL and only modestly improves exposed SEARCH; HNM gain small. |
| Q5 joint training | **Refuted** for shared-parameter gradient conflict; schedules inconclusive | No shared trainable parameters; synthetic matched update delta=0. |
| Q6 U2 vs H-002 target | **Partially supported trade-off; cause inconclusive** | U2 improves AP-O +.018024, loses AP-S −.006451. |
| Q7 hierarchy U3 | **Not supported for this configuration** | Three-seed M3 mean −.006257; CAL20 F1 lower all three. |
| Q8 specialist transfer | **Not tested / inconclusive** | Missing valid EXP24/36/43 TRAIN OOF teacher scores. |
| Q9 AP vs F1 | **Supported distinction** | Raw AP and CAL20 decisions diverge; report separately. |
| Q10 acoustic dependence | **Supported** for tested U2/G3 score sensitivity; not a full retrained insertion modality ablation | OOD perturbation lowers AP-I; G9 auxiliary CTC mixed. |

## 10. Reproducibility, hashes and artifact map

### 10.1 Required reports and machine-readable artifacts

- Hypothesis verdicts: `reports/RQ1_UNIFIED_02_HYPOTHESIS_VERDICTS.md`
- Root-cause analysis: `reports/RQ1_UNIFIED_02_ROOT_CAUSE_ANALYSIS.md`
- Experiment leaderboard: `reports/RQ1_UNIFIED_02_EXPERIMENT_LEADERBOARD.csv` (32 rows; 27 columns)
- Ablation matrix: `reports/RQ1_UNIFIED_02_ABLATION_MATRIX.csv` (67 rows; 11 columns)
- Confusion analysis: `reports/RQ1_UNIFIED_02_CONFUSION_ANALYSIS.md`
- Scientific final report: `reports/RQ1_UNIFIED_02_FINAL_SCIENTIFIC_REPORT.md`
- Reproduction guide: `reports/RQ1_UNIFIED_02_REPRODUCTION.md`
- Per-class threshold/precision/recall/AP curve points: `experiments/rq1_unified_02/supervisor/RQ1_UNIFIED_02_CLASSWISE_PR_CURVES.csv.gz` — SHA256 `cd8c128fe1473138879d9949080c3527088f92e42ef4abb6e0f1f82013d22ca1`.
- Precision@K/Recall@K: `experiments/rq1_unified_02/supervisor/RQ1_UNIFIED_02_PRECISION_AT_K.csv` — SHA256 `1fc656f5e13821f99e9c20d3ab0007b75bfe9edf385489fbb19361ec4e5f510a`.
- PR export script: `experiments/rq1_unified_02/supervisor/export_ranking_artifacts.py` — SHA256 `c4669b2f0c91ddb3a154105bbcd624b201b37b559da4ee4bc508ab9c16e979ec`.

The export validates target/gap IDs against U2-S101 on DEV_SEARCH and records 17 arms × 3 classes. DEV_SEARCH remains exposed/descriptive.

### 10.2 G6-PREDICTED receipt and source files

- Training script SHA256: `ec815a432e52fe7b799ba1f7ac78d4e4966920612265c1d722056b54e1803668`.
- CPU-only receipt finalizer SHA256: `8198795ccc5303c9f8a0c0e1cd12a461e8f4c8d3069dca4e0452910ccf3f2d18`.
- Receipt: `experiments/rq1_unified_02/agent_a/RQ1-U02-G6-PREDICTED-BOUNDARY-W2V17-REF-S101.json`, SHA256 `b451a7e752311eaf09862706863e5073ad7cd41ed9f6b80cf90b2fce55392871`.
- Checkpoint SHA256 `f9dc8c58ddae34df3bf7e81d6b921c6a9f1815ce4fb4c7c93291dbda3f273582`.
- TRAIN/CAL/SEARCH predictions SHA256: `87566a02135102e124b927bd018abe4ee2754e36f67452ca799c222dfd8044f0` / `33c0a3306dbd5b07094c3a899a9a862f2c832edf7534e870e7b00bf7d7178ad1` / `1dcb6d4bee1946b9eaf82217cdf7eb25558d1c91fd496c04e52a64b217add76b`.
- Data, audio and model provenance, per-split window hashes, frame mapping checks, thresholds, confusions and limitations are enumerated in the JSON receipt.

The G6 lease ended with process return code 1 **only because JSON serialization failed after training and predictions completed**. Checkpoint and predictions were preserved; EVAL-V3.1 and receipt were rebuilt from saved arrays on CPU. No CUDA rerun or additional model forward was made. The attempt logs are preserved under `experiments/rq1_unified_02/agent_a/RQ1-U02-G6-PREDICTED-BOUNDARY-*-FAILED*.log` and `...R3-RECEIPT-FAILURE.log`.

### 10.3 Additional artifact families

- Frozen branch substitutions: `experiments/rq1_unified_02/agent_a/RQ1-U02_FROZEN_SUBSTITUTIONS.{csv,json}`.
- G1/G2/G3/G6/G6-REF/G8/G9 model artifacts, scripts and receipts: `experiments/rq1_unified_02/agent_a/` and `supervisor/`.
- U2/U3 matched seeds202/303, diagnostics and receipts: `experiments/rq1_unified_02/agent_b/`.
- G3 inference perturbations/acoustic controls: `experiments/rq1_unified_02/agent_c/`.
- Historical HNM: `experiments/hard_negative_mining/results.json`; no restart.

## 11. Giới hạn và hành động tiếp theo

- DEV_SEARCH đã exposed; các SEARCH contrasts chỉ là descriptive/exploratory.
- G1, balanced, crop-sequence G6, G6-REF, G6-PREDICTED, G8 là single-seed screens; G3 có 3 seeds nhưng không thắng trung bình SEARCH; G9 chỉ có 2 seeds và hỗn hợp encoder.
- G6-PREDICTED direct test kiểm tra một center mapping/window width (17 frames) và seed101; kết quả âm tính không bác bỏ mọi localization hypothesis.
- Waveform replay receipt ghi independent canonical-reference lineage là unverified; intended-phone sources và limitations được giữ rõ trong receipt.
- Không có valid speaker-held-out TRAIN OOF teacher scores đúng cho EXP24/36/43; specialist distillation vẫn chưa được kiểm định.
- Root cause được xác lập ở cấp score branch; cơ chế cause tiếp tục **inconclusive**. Không đề xuất thêm training trong campaign này nếu không có hypothesis/preregistration mới, và không dùng SEARCH để chọn cấu hình.

## 12. Bảng kiểm cuối

- [x] TRAIN/CAL/SEARCH roles giữ nguyên; không holdout.
- [x] Exact N+1 gap IDs và EVAL-V3.1; full event coverage cho các output đã báo cáo.
- [x] Raw AP và threshold-policy F1/confusions tách biệt.
- [x] Score compositions tách khỏi trained unified models.
- [x] Continuous predicted-boundary sequence test hoàn tất; không gold timing.
- [x] G6 checkpoint/predictions/receipt hashes xác minh; finalizer CPU-only; GPU idle.
- [x] Leaderboard/ablation CSV schema hợp lệ; PR curves và P@K exports đã tạo.
- [x] Không có cải thiện AP ổn định được claim; uncertainty và exposed SEARCH được công bố.
