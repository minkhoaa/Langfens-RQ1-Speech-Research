# LANGFENS RQ1-UNIFIED-02 — Báo cáo khoa học cuối

Bản tổng hợp toàn bộ số liệu, ablation, threshold/confusion, hashes và reproduction: [`RQ1_UNIFIED_02_BAO_CAO_TONG_HOP_CHI_TIET.md`](RQ1_UNIFIED_02_BAO_CAO_TONG_HOP_CHI_TIET.md).

## Kết luận

Kết quả attribution chắc nhất là **nhánh insertion/gap score mới của U2 seed101 làm giảm Macro-3 AP**: thay riêng gap scores U2 bằng Gold-Only scores làm AP-I tăng .068723 và Macro-3 AP tăng .022908, từ .224890 lên .247798 trên đúng event IDs và EVAL-V3.1. Không kết quả nào xác định được nguyên nhân cơ học của gap weakness. Thử bottleneck giảm một nửa số parameter cho AP-I SEARCH +.00794 một seed nhưng AP-I CAL −.00170; balanced minibatches cho AP-I SEARCH +.00770 nhưng CAL −.04407. Không intervention nào đã được lặp lại đủ seed để tuyên bố lợi ích ổn định.

Do đó **U2 không vượt Gold-Only hoặc H-002 về Macro-3 AP**, và kém rõ insertion AP; **không unified neural candidate mới trong RQ1-02 vượt H-002**. Có một historical hard-negative GRU64 (không cùng U2 feature/recipe) đạt Macro3 AP=.237663, hơn Gold-Only .236917 chỉ +.000746, nhưng vẫn dưới H-002 .243940. EXP24/36/43 và Phase 4 fusion vẫn là comparator điểm số, phần lớn là composition; không được trình bày như unified learned checkpoints.

## Bằng chứng chính

### 1. Branch substitution (Q1)

| Nhánh target / gap | AP-S | AP-O | AP-I | Macro3 AP |
|---|---:|---:|---:|---:|
| U2 / U2 | .371360 | .185318 | .117992 | .224890 |
| U2 / Gold-Only | .371360 | .185318 | .186715 | .247798 |
| H-002 / U2 | .377810 | .167294 | .117992 | .221032 |
| H-002 / Gold-Only | .377810 | .167294 | .186715 | .243940 |
| U2 / EXP43 (chẩn đoán composition) | .371360 | .185318 | .285056 | .280578 |

Event IDs và labels target/gap được so khớp chính xác; 19,755 target/20,354 gap, coverage đầy đủ. Gap branch substitution phục hồi nhiều hơn net regression của U2-vs-GoldOnly vì U2 target branch cải thiện AP-O. EXP43 row là **score composition diagnostic**, không phải mô hình neural đã train.

### 2. Gap head representation và rare-positive screen (Q2/Q4)

| Arm | Params gap | TRAIN AP-I (in-sample) | CAL AP-I | SEARCH AP-I* | SEARCH Macro3 AP* | CAL20 Macro3 F1 | GPU time / VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 U2 1614→256→64 | 430,018 | .97698 | .13811 | .11799 | .22489 | .12542 | run gốc: 24.88s / 41.33 MiB |
| G1 bottleneck128, giữ input 1614D | 215,106 | .97851 | .13641 | .12593 | .22754 | .14018 | 7.86s / 27.90 MiB |
| Balanced 64 pos + 64 neg | 430,018 | .99256 | .09404 | .12569 | .22746 | .14923 | 6.84s / 36.81 MiB |
| G6 paired crop-edge sequence, Conv1d | 168,258 | — | .06280 | .10900 | .22189 | .11067 | 32.11s / 83.96 MiB |
| G6-REF same sequence + phone inputs | 173,250 | — | .10078 | .09967 | .21878 | .12000 | 35.80s / 84.13 MiB |
| G6 continuous predicted-boundary W2V17 + phone inputs | 173,250 | .83238 | .10229 | .08015 | .21228 | .13033 | 193.92s lease / 1,136 MiB monitored |
| G8 all 17 frames per phone crop + reference | 173,250 | — | .07922 | .11651 | .22439 | .14608 | 54.81s / 140.16 MiB |
| G9 G3 + local CharSiU CTC6 auxiliary | 433,090 | .99673 | .15978 | .15596 | .23755 | .11834 | 10.27s / 30.10 MiB |
| G2 legacy 6D gap_X only | 9,282 | — | .03634 | .02594 | .19421 | .08761 | 7.20s / 18.17 MiB |
| G3 acoustic 1614D + legacy 6D | 431,554 | .97624 | .17427 | .13416 | .23028 | .12481 | 10.34s / 30.06 MiB |

G3 matched paired AP-I deltas vs U2 across seeds 101/202/303: CAL **+.03616/+.00945/+.01770**, SEARCH **+.01617/−.00183/−.03332**. Mean G3 SEARCH AP-I=.13267 vs U2=.13900 (paired Δ−.00633); mean Macro3 AP=.22579 vs .22789 (Δ−.00211). Mean CAL AP-I=.14903 vs .12793 (+.02110), but this CAL gain did not transfer to SEARCH. Do not select/integrate G3 from exposed SEARCH nor call the result a repeatable improvement.

`*` DEV_SEARCH exposed, chỉ exploratory. Cả G1 và balanced chỉ một seed, chưa đủ tuyên bố cải thiện lặp lại. G1 giảm dung lượng nhưng CAL AP không tăng; balanced có SEARCH uplift nhỏ song CAL AP giảm. U2 train AP-I/AUC=.97698/.99986 đối lập CAL=.13811/.81530 và SEARCH=.11799/.78343; positive score median .99764 TRAIN nhưng .000527 CAL/.000833 SEARCH: có bằng chứng mạnh overfit/poor transfer, nhưng chưa xác định nguyên nhân dimensionality. Effective covariance rank khoảng 236/768 cho mỗi acoustic block trên sample TRAIN, biến thiên không suy biến nhưng có redundancy. TRAIN có 560/61,362 insertion positives (0.9126%). Historical hard-negative GRU64 top-128 TRAIN negative penalty đạt SEARCH AP-I=.185065/Macro3=.237663 so Gold-Only .186715/.236917; AP-I vẫn −.00165, Macro3 chỉ +.00075, CAL_MAX Macro3 F1 .29224 vs .30014 và FWR=6.716. Đây là một-seed historical comparison trên GRU64/51D+6D, không phải matched U2 input. Cùng với balanced-screen CAL decline và SEARCH gain nhỏ, imbalance nghiêm trọng nhưng chưa là giới hạn thống trị.

### 3. Giải phẫu features/optimization

Implementation `experiments/rq1_unified_01/agent_a/train_twohead.py` tạo mỗi internal gap từ mean 5 frame cuối crop phone trái (`11:16`) và 5 frame đầu crop phone phải (`1:6`), crops lấy quanh predicted phone centers trên frozen normalized Wav2Vec2 `hidden_states[6]`; nối hai 39D intended-phone vectors thành 1,614D. Các vectors gần rìa hai predicted-center crops, không phải sequence liên tục của toàn interval. Cạnh utterance dùng zero sentinel. Không dùng gold boundary, realized phone hoặc insertion labels làm input.

Trên SEARCH có 124/125 positives internal; leading slots 0/599 positives, trailing 1/599 (AP .3333 từ một positive không ổn định). Probe RC2-H003 trước đó dùng 6D static gap logistic so với local 11-frame CTC posterior pooling (encoder khác) cho SEARCH AP-I .04002→.01132, không cải thiện; đây không phải matched neural W2V context intervention cho U2. G6 dùng W2V layer-6 cache và paired sequence [left crop 8:17, right crop 0:9], Conv1d; SEARCH AP-I=.10900 dưới U2 .11799, CAL AP-I=.06280, CAL20 Macro3 F1=.11067. G6 bỏ intended-phone vectors; G6-REF phục hồi chúng nhưng SEARCH AP-I=.09967, CAL=.10078, CAL20 Macro3 F1=.12000. Reference removal góp phần vào CAL degradation, nhưng paired sequence vẫn không tăng SEARCH AP. G8 all-17-frame-per-neighbor crop (34×768)+phone references có AP-I=.07922 CAL/.11651 SEARCH, Macro3=.22439, không cải thiện U2. G6-PREDICTED-BOUNDARY là can thiệp contiguous thật: full-wave normalized W2V hidden_states[6], 17 frame quanh từng predicted CTC gap center sau CTC→W2V proportional frame-clock mapping; intended phones giữ nguyên, không dùng gold timing. Target-crop cache parity max error <2e-3 và exact gap/target ID checks passed. Seed101 cho TRAIN/CAL/SEARCH AP-I=.83238/.10229/.08015, Macro3 SEARCH AP=.21228 (U2=.22489), CAL20 Macro3 F1=.13033 (U2=.12542). Kết quả âm tính ở cả CAL và SEARCH; continuous window cụ thể này không giải quyết U2 gap ranking. G9 thêm G9 thêm local CharSiU CTC posterior/uncertainty 6D vào G3. Seeds101/202 CAL AP-I=.15978/.13422 (cả hai cao hơn U2 .13811/.12857 nhưng thấp hơn G3 .17427/.13801); SEARCH=.15596/.13226 (deltas vs U2 +.03797/−.00530, vs G3 +.02180/−.00347). Đây là auxiliary second-encoder signal; uplift SEARCH không nhất quán. Reference features retained. Phép thử bottleneck giữ nguyên input, seed, 8 epoch và weighted CE. Vì vậy “1,614D gây overfit” và “localization sai” **chưa được chứng minh**. Hai module target/gap có tham số trainable độc lập, không có shared trainable parameters; gradient conflict cổ điển do shared weights không thể là giải thích. CPU equivalence check với actual U2 module, cùng initialization/loss/batches/AdamW states và dropout tắt trên synthetic frozen features cho parameter delta 0 qua 4 bước. Đây kiểm chứng tương đương khi schedules giống nhau, không phải real acoustic fit hay different schedules.

### 4. Target / hierarchy / score interpretation

U2 target AP-S/O=.371360/.185318; H-002=.377810/.167294. U2 tốt hơn O (+.018024), kém S (−.006451); không có cơ sở gọi target head U2 tốt hơn nói chung. U3 matched seeds101/202/303 có per-seed U3−U2 AP-S/O/I/Macro3: (−.00895,−.00920,+.00583,−.00411), (−.00390,+.00222,+.00617,+.00150), (−.00615,+.00337,−.04571,−.01616). Three-seed mean Macro3 AP=.22164 vs U2=.22789 (Δ−.00626); AP-I=.12776 vs .13900 (Δ−.01124). CAL20 Macro3 F1 thấp hơn cả ba seed, mean .07268 vs .11398 (Δ−.04130). Kết quả không ủng hộ lợi ích ổn định của hierarchy/objective đã thử, nhưng không loại trừ mọi hierarchical architecture.

AP và policy khác nhau. U2 CAL20 class P/R/F1: S .4783/.0124/.0243, O .3299/.1531/.2092, I .3793/.0880/.1429; Macro3 F1=.12542, Macro4=.33071, binary MD AP/F1=.36332/.05755, FWR=.17863. Unrestricted U2 Macro3 F1=.28035 nhưng FWR=4.654. Low substitution recall tại CAL20 là policy/threshold issue trên một ranking có AP-S=.37136; không nên coi hai chỉ số là cùng năng lực. Không áp FWR≤.20 làm success gate.

### 5. Acoustic dependence / experts

H-002 target arm dùng cùng evaluation contracts: full audio+reference AP S/O/M3=.377810/.167294/.243940; retrained reference-only=.293833/.024503/.168350; acoustic-shuffle inference perturbation=.239753/.023376/.149948; zero-reference perturbation=.095798/.011834/.098116. H002 controls vẫn dùng GoldOnly gap scores. G3-S101 gap-head inference perturbation, fixed U2 target scores: AP-I baseline .13416, zero W2V acoustics .04693, shuffle W2V blocks .01122, zero 6D gap_X .08361, reference-only .01992. Điều này cho thấy gap scorer này nhạy với âm thanh/6D evidence, nhưng perturbation OOD không thay thế retrained modality ablation và chỉ kiểm tra một seed.

EXP24 (standalone checkpoint replay) AP S/O/I/Macro3=.357737/.225092/.247628/.276819; EXP29 composition=.378062/.196539/.272068/.282223; EXP36 composition=.387088/.204687/.275924/.289233; EXP43 composition=.363453/.205771/.285056/.284760. Phase4 omission-specialist composition 5-seed mean Macro3 AP≈.3018; fixed headwise blend≈.2955. Specialist leaders khác nhau theo class. Audit OOF bổ sung tìm thấy raw speaker-held-out scores hợp lệ cho Phase-3 teacher replicas (not EXP24/36/43); các scores đó không thể làm proxy cho Phase-4 expert gains. Không transfer/distill các EXP24/36/43 vì không có đúng TRAIN OOF targets. Không distill từ DEV predictions và không fabricate OOF.

## Trả lời câu hỏi bắt buộc

1. **Gap head có thật sự là lý do chính U2 kém?** Có, ở attribution điểm số trên seed101; không đồng nghĩa đã giải thích vì sao gap head yếu.
2. **High dimensionality có trách nhiệm?** Chưa biết. Overfit/poor transfer được hỗ trợ mạnh bởi train-vs-CAL/SEARCH AP và AUC; bottleneck có SEARCH uplift nhỏ nhưng CAL không tăng. G3 +6D features có CAL lift nhưng SEARCH mean giảm, nên không chỉ ra high dimensionality; covariance effective rank không thấp. G6 temporal sequence screen kém hơn U2; G6-REF cũng kém SEARCH.
3. **Gap localization sai/thiếu?** Feature hiện tại là các rìa crop quanh predicted centers; H003 local posterior pooling (khác encoder/learner) giảm AP-I .04002→.01132. U2 W2V G6/G6-REF AP-I=.10900/.09967; G8 full 17-frame-per-crop context AP-I=.11651; all are crop-relative, not continuous interval tests and none improve U2 .11799. G3 6D summary fusion has mixed seed generalization and does not test localization. Leading/trailing positives quá ít để chẩn đoán.
4. **Imbalance là giới hạn chính?** Rare positives là có thật; balanced screen CAL AP giảm và SEARCH tăng nhẹ. Historical hard-negative GRU64 insertion AP gần Gold-Only nhưng Macro3 gain chỉ .00075, một seed/khác input. Dominant cause chưa được chứng minh.
5. **Multi-task interference?** Shared trainable gradient conflict bị bác bỏ; CPU synthetic 4-step AdamW replay cho joint/separate updates identical (max parameter delta 0) khi per-head data/schedule giống nhau. Schedule khác nhau chưa kiểm định.
6. **U2 so H-002?** U2 có AP-O cao hơn nhưng AP-S thấp hơn; nguyên nhân riêng (checkpoint, objective, duration, selection) chưa được tách bằng ablation.
7. **U3 thất bại?** Ba-seed matched evidence cho Macro3 AP mean −.00626 và insertion AP mean −.01124; CAL20 Macro3 F1 thấp hơn ở cả ba, mean −.04130. Không tiếp tục tune cấu hình U3 này.
8. **Transfer specialist fusion?** Chưa được xác định. Raw TRAIN OOF cho Phase-3 replicas có tồn tại nhưng không cho EXP24/36/43; vì vậy Phase-4 fusion transfer vẫn thiếu OOF teacher. Fusion numbers là score compositions.
9. **AP và F1 khác?** Đúng: raw ranking và threshold policy khác; substitution recall CAL20 thấp dù AP-S đáng kể. G9 seed101 SEARCH AP-I=.15596 nhưng CAL20 insertion F1=.12162/M3 F1=.11834; seed202 SEARCH AP-I=.13226, confirming no stable G9 policy gain.
10. **Ứng viên mạnh dùng âm thanh?** G3-S101 inference perturbation giảm AP-I .13416→.04693 khi zero W2V, .01122 khi shuffle, .08361 khi zero 6D gap_X. G9 local CTC auxiliary AP-I CAL/SEARCH=.15978/.15596 (S101), .13422/.13226 (S202), both second-encoder; not a same-encoder-only or retrained ablation. G3 perturbations are OOD inference tests.
11. **Giải thích trước nào sai?** Chẩn đoán dimensionality, localization sai, imbalance dominant, joint gradient interference không được xác nhận; shared-parameter interference là sai với code. Gap-score attribution thì đúng nhưng nguyên nhân chưa rõ.
12. **Can thiệp nào cải thiện repeatable lớn nhất?** Không có cải thiện AP SEARCH lặp lại. G3 CAL AP-I tăng cả ba seeds nhưng SEARCH mean giảm và một seed đảo mạnh; đây là CAL/SEARCH shift, không generalizable win. Score swap về GoldOnly phục hồi Macro3 +.02291 (diagnostic composition, không train).
13. **Có neural unified model đã train nào vượt baseline?** Historical hard-negative GRU64 đạt Macro3 AP .237663, hơn Gold-Only .236917 rất nhẹ (+.000746), nhưng kém H-002 .243940. Trong các ứng viên RQ1-02 mới, không model nào vượt cả Gold-Only và H-002; G1/balanced/G3 exploratory remain below or are not repeatable.
14. **Khoảng cách mô hình unified với experts/compositions?** Historical unified HNM GRU64 Macro3 AP .23766 còn kém EXP24 standalone learned checkpoint .27682 (−.03916), EXP36 .28923 (−.05157), EXP43 .28476 (−.04710), Phase4 omission fusion .3018 (−.06414); AP-I .18507 còn kém EXP43 .28506 (−.09999). EXP24 là checkpoint neural standalone; EXP29/36/43 và Phase4 fusion là score compositions, không được trình bày như neural checkpoints.
15. **Thí nghiệm tiếp hợp lý?** Không cần thêm training để kết luận chiến dịch RQ1-02: G6-PREDICTED-BOUNDARY kiểm tra trực tiếp contiguous full-wave W2V sequence quanh predicted CTC gap centers và kết quả CAL/SEARCH đều âm tính. Root cause vẫn không xác định. Một nghiên cứu kế tiếp riêng có thể preregister systematic center/width/context comparison, CAL-only selection và matched seeds, nhưng không nên tiếp tục sweep dựa trên SEARCH.

## Trạng thái tiến hành

Đã hoàn tất frozen-score substitution, geometry audit, G1/G2/G3/G6/G6-REF/G6-PREDICTED-BOUNDARY/G8/G9 screens, G3 three-seed replication, U2/U3 matched three-seed comparisons, CPU optimizer equivalence, target/acoustic controls, G3 inference perturbation. G9 auxiliary CTC feature uses a second frozen CharSiU model; it is not a same-W2V-only result. G6-PREDICTED-BOUNDARY dùng predicted CTC timing, không gold timing; feature mapping/checkpoint parity và EVAL-V3.1 raw ranking/threshold metrics đã kiểm tra. DEV_SEARCH vẫn exposed, chỉ exploratory. Không có unified candidate mới với improvement ổn định.

## Artifacts

Chi tiết: `RQ1_UNIFIED_02_HYPOTHESIS_VERDICTS.md`, `RQ1_UNIFIED_02_ROOT_CAUSE_ANALYSIS.md`, `RQ1_UNIFIED_02_CONFUSION_ANALYSIS.md`, leaderboard/ablation CSV, reproduction instructions. Raw classwise PR curve points and support-sized Precision@K/Recall@K are in `experiments/rq1_unified_02/supervisor/RQ1_UNIFIED_02_CLASSWISE_PR_CURVES.csv.gz` and `RQ1_UNIFIED_02_PRECISION_AT_K.csv`. CUDA checkpoints/predictions and scripts are preserved under `experiments/rq1_unified_02/`; DEV_SEARCH remains exploratory.
