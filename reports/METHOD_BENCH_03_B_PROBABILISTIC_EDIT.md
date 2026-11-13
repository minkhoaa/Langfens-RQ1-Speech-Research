# METHOD-BENCH-03 — Probabilistic edit-path localization

## Tóm tắt

Đã xây dựng một bộ chấm event từ posterior CTC thật, giữ đủ `N` target và `N+1` exact-gap event cho từng utterance. Marginal hóa edit path làm AP tăng so với một edit path cố định. Encoder Wav2Vec2 frozen cải thiện rõ so với log-mel BiGRU (DEV_SEARCH E3 Macro-3 AP `.10226` so với `.05557`), nhưng vẫn dưới Gold-Only `.23692` và specialist frontier. Candidate beam chỉ chứa một lượng cực nhỏ xác suất CTC tuyệt đối: trung vị tổng khối lượng của top-10 sequence cộng canonical reference là `3.54e-14` trên DEV_CAL và `3.15e-14` trên DEV_SEARCH của log-mel CTC. Vì vậy E2/E3 là posterior **có điều kiện trên tập ứng viên bị cắt cụt**, không phải posterior đầy đủ của chuỗi phát âm.

Kết quả DEV_SEARCH mạnh nhất là W2V2-H6 E3 beam-10. Qua seed 101/202/303, Macro-3 AP trung bình là `.10424` `[.10198,.10847]`, AP-S/O/I trung bình `.23775/.04095/.03401`; E3 vượt matched E1 ở mọi lớp và cả ba seed. Tuy vậy, Macro-3 AP còn thấp hơn một nửa Gold-Only (`.236917`); AP-O/I vẫn thấp xa Gold-Only (`.159433/.186715`) và các scorer insertion chuyên biệt. Đây là cải thiện sequence evidence lặp lại trong campaign này, nhưng chưa phải MDD frontier.

## Định nghĩa các phương pháp

- **E0 — tham chiếu lịch sử:** H05 frame-argmax + Levenshtein đạt AP-I `0.02825` trên DEV_SEARCH trong METHOD-BENCH-02. Đây là frame-classifier, không phải CTC và không được chạy lại.
- **E1 — greedy CTC:** collapse blank và lặp CTC; căn chỉnh chuỗi kết quả với canonical sequence bằng unit-cost Levenshtein và một tie-break xác định. Mỗi target/gap nhận điểm event 0/1.
- **E2 — N-best, một đường:** lấy top-k prefix-beam candidates từ Agent A; tính lại xác suất từng sequence bằng CTC forward algorithm trên log posterior frame đã xuất; thêm canonical sequence; chuẩn hóa trọng số chính xác trên tập hữu hạn này. Với mỗi sequence dùng một đường Levenshtein tối ưu xác định rồi lấy kỳ vọng event có trọng số.
- **E3 — N-best, marginal paths:** dùng đúng tập sequence và đúng trọng số E2. Với từng sequence, phân phối đều trên mọi đường Levenshtein unit-cost tối ưu; lấy marginal event theo đường, rồi kỳ vọng theo trọng số sequence.

**E4 constrained likelihood-ratio scoring chưa chạy.** Dữ liệu CTC hiện là utterance-level; lấy likelihood của mọi local phone alternative cho mọi target/gap tạo hàng triệu CTC forward passes và vẫn không cho biết exact temporal gap nếu không thêm quy tắc align. Sau khi kiểm tra candidate posterior gần bằng không và PER của recognizer yếu, một lưới likelihood-ratio lớn không có cơ sở để cải thiện suy luận; ưu tiên xác minh encoder/posterior trước.

Với reference `r_1...r_N`, mỗi sequence candidate sinh `N` xác suất substitution, `N` xác suất omission và `N+1` xác suất insertion. Substitution/omission là xác suất cạnh diagonal mismatch hoặc deletion trong DAG các đường edit tối ưu. Exact-gap insertion là xác suất có ít nhất một cạnh insertion tại cursor gap đó; được tính bằng tỷ lệ số đường tối ưu có insertion tại gap, không phải tổng số phone thừa. Các gap biên 0 và N được giữ nguyên. Target/gap event ID được lưu cùng utterance và slot trong prediction JSONL/NPZ.

CTC forward probability cho mỗi sequence candidate là chính xác theo log posterior frame được lưu; nhưng do beam pruning, tập sequence hữu hạn không phủ toàn bộ transcript posterior. E3 còn dùng giả định phân phối đều trên các edit path tối ưu. Vì vậy đây là marginal có điều kiện trên candidate set với quy tắc tie rõ ràng, **không** phải posterior chung chính xác của một CTC/edit transducer đã huấn luyện.

Gold S/O/I label chỉ được dùng ở EVAL-V3.1; không dùng để chọn đường edit, insertion gap, hay candidate sequence. Mọi event đều so với canonical reference đã chuẩn hóa và có sẵn như inference input.

## Kết quả EVAL-V3.1

### Log-mel BiGRU CTC, seed 101

| Split | Beam k | Arm | AP-S | AP-O | AP-I exact-gap | Macro-3 AP | Macro-3 F1 (argmax / gap .5) | Macro-4 F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| DEV_CAL | 1 | E1 greedy hard edit | 0.189118 | 0.048579 | 0.009862 | 0.082519 | 0.094670 | 0.166437 |
| DEV_CAL | 1 | E2 weighted hard path | 0.198579 | 0.056551 | 0.009862 | 0.088331 | 0.119151 | 0.213865 |
| DEV_CAL | 1 | E3 weighted path marginal | 0.251462 | 0.070760 | 0.010203 | 0.110808 | 0.117222 | 0.215684 |
| DEV_CAL | 3 | E1 greedy hard edit | 0.189118 | 0.048579 | 0.009862 | 0.082519 | 0.094670 | 0.166437 |
| DEV_CAL | 3 | E2 weighted hard path | 0.205050 | 0.056149 | 0.009862 | 0.090354 | 0.120100 | 0.214482 |
| DEV_CAL | 3 | E3 weighted path marginal | 0.260925 | 0.070582 | 0.010065 | 0.113857 | 0.112443 | 0.211648 |
| DEV_CAL | 10 | E1 greedy hard edit | 0.189118 | 0.048579 | 0.009862 | 0.082519 | 0.094670 | 0.166437 |
| DEV_CAL | 10 | E2 weighted hard path | 0.210145 | 0.056873 | 0.009862 | 0.092293 | 0.120063 | 0.214482 |
| DEV_CAL | 10 | E3 weighted path marginal | 0.267359 | 0.074858 | 0.011803 | 0.118007 | 0.109675 | 0.211648 |
| DEV_SEARCH | 3 | E1 greedy hard edit | 0.100426 | 0.012480 | 0.006141 | 0.039682 | 0.063926 | 0.144925 |
| DEV_SEARCH | 3 | E2 weighted hard path | 0.110646 | 0.015310 | 0.006141 | 0.044032 | 0.073860 | 0.177659 |
| DEV_SEARCH | 3 | E3 weighted path marginal | 0.131065 | 0.018158 | 0.012179 | 0.053801 | 0.068283 | 0.176442 |
| DEV_SEARCH | 10 | E1 greedy hard edit | 0.100426 | 0.012480 | 0.006141 | 0.039682 | 0.063926 | 0.144925 |
| DEV_SEARCH | 10 | E2 weighted hard path | 0.114718 | 0.015378 | 0.006964 | 0.045687 | 0.073219 | 0.176718 |
| DEV_SEARCH | 10 | E3 weighted path marginal | 0.135783 | 0.017343 | 0.013578 | 0.055568 | 0.067899 | 0.175532 |

DEV_CAL topology là 300 utterance / 10,043 target / 10,343 gap. DEV_SEARCH là 599 utterance / 19,755 target / 20,354 gap; tất cả predictions bao phủ đủ event ID, không thiếu hoặc trùng event. DEV_SEARCH class support là S=1,768, O=209, I=125. DEV_SEARCH dùng để đánh giá mô tả; không chọn beam depth hay phương pháp từ nhãn SEARCH.

E3 cải thiện SEARCH AP so với matched E1 thêm `+0.03536` S, `+0.00568` O, `+0.00744` I ở beam-10. Nhưng insertion AP `0.01358` vẫn thấp hơn H05 lịch sử `0.02825`, Gold-Only `0.186715`, EXP43 `0.285056`; Macro-3 AP `0.05557` thấp xa Gold-Only `0.236917`, H-002 `0.243940`, EXP24 `0.276819`, EXP36 `0.289233`, EXP43 `0.284760` và Phase 4 omission-specialist fusion `≈0.3018`. Trong log-mel arm, E3 vượt E1 nhưng còn yếu; kết quả này không phủ nhận W2V2 arm độc lập ở phần sau.

### Sensitivity theo k và candidate-mass coverage

Khối lượng được đo là tổng xác suất CTC forward tuyệt đối của top-k candidate cùng canonical sequence, trước chuẩn hóa hữu hạn.

| Split | k | Mean mass | Median mass | P90 mass | Mean posterior weight nằm ở candidate có nhiều edit path tối ưu | Utterances có candidate đa đường |
|---|---:|---:|---:|---:|---:|---:|
| DEV_CAL | 1 | 3.57e-7 | 4.00e-15 | 1.49e-9 | 0.9867 | 296/300 |
| DEV_CAL | 3 | 9.54e-7 | 1.13e-14 | 4.11e-9 | 0.9877 | 299/300 |
| DEV_CAL | 10 | 2.42e-6 | 3.54e-14 | 1.08e-8 | 0.9872 | 300/300 |
| DEV_SEARCH | 3 | 1.41e-6 | 1.24e-14 | 9.48e-9 | 0.9862 | 598/599 |
| DEV_SEARCH | 10 | 3.30e-6 | 3.15e-14 | 2.40e-8 | 0.9836 | 599/599 |

Tăng k từ 3 lên 10 chỉ đổi SEARCH E3 AP-S/O/I từ `0.13107/0.01816/0.01218` thành `0.13578/0.01734/0.01358`; Macro-3 AP đổi `0.05380 → 0.05557`. Trung vị candidate mass vẫn ở cỡ `1e-14`. Đây là độ phủ gần bằng không theo xác suất tuyệt đối, nên các trọng số `w(y)=P_CTC(y|x)/sum_{z∈Q}P_CTC(z|x)` chỉ diễn giải được **có điều kiện** trên một phần rất nhỏ posterior.

Đã kiểm tra thêm prefix beam không giới hạn token shortlist trên 30 utterance đầu DEV_CAL (không dùng nhãn), với active beam 10/50/100. Tổng exact probability của top-10 sequence có mean `2.32e-6/3.00e-6/3.02e-6`, median `5.60e-13/8.93e-13/9.27e-13`; 83–87% utterance vẫn dưới `1e-6`. Beam 100 gần như không tăng so với 50. Mở rộng beam hiện tại không sửa được tính khuếch tán candidate posterior. Run mất `85.54 s` CPU. Đây là diagnostic subset, không phải MDD benchmark.

## Encoder thứ hai: frozen Wav2Vec2 hidden-state CTC

Agent A cung cấp experiment độc lập `RQ1-MB03-A-CTC-W2V2H6-LIN-S101`; Agent B chạy cùng top-10, E1/E2/E3 và EVAL-V3.1, với input prefix `w2v2_` và output riêng. Mô hình này nhận frozen Wav2Vec2 hidden layer 6, trainable CTC head, cùng realized-phone supervision và blank inventory. Kết quả PER ở DEV_SEARCH cải thiện rõ so với log-mel BiGRU: realized-reference greedy/beam PER `0.3931/0.3208` so với `0.7613/0.6802`; canonical-reference greedy/beam PER `0.4071/0.3368` so với `0.7753/0.6978`. Đây là cải thiện nhận dạng; MDD vẫn phải được đánh giá riêng.

| DEV_SEARCH, beam 10 | AP-S | AP-O | AP-I exact-gap | Macro-3 AP | Macro-3 F1 (argmax / gap .5) | Macro-4 F1 |
|---|---:|---:|---:|---:|---:|---:|
| W2V2 E1 greedy hard edit | 0.146598 | 0.022569 | 0.010498 | 0.059889 | 0.136532 | 0.295967 |
| W2V2 E2 weighted hard path | 0.201183 | 0.035341 | 0.018073 | 0.084866 | 0.165517 | 0.332127 |
| W2V2 E3 weighted path marginal | 0.235737 | 0.041842 | 0.029214 | 0.102264 | 0.174375 | 0.340762 |

DEV_CAL W2V2 E1/E2/E3 AP-S/O/I và Macro-3 AP lần lượt là `.257759/.091816/.020295/.123290`, `.313988/.150536/.031029/.165185`, và `.376301/.210078/.058808/.215063`. Argmax/gap-.5 Macro-3 F1 là `.218913/.271994/.307002`; Macro-4 F1 là `.355234/.407826/.436529`. Đó là CAL assessment trước policy selection; ngưỡng vẫn được fit riêng bằng hàm EVAL-V3.1.

So với log-mel E3 beam-10, W2V2 seed 101 tăng SEARCH AP-S `+0.09995`, AP-O `+0.02450`, AP-I `+0.01564`, Macro-3 AP `+0.04670`. Seed 101 là lần đầu benchmark; phần lặp ba seed dưới đây xác nhận hướng tăng E3 so matched E1 trên cả ba seed. AP-I vẫn chỉ nhỉnh hơn H05 `.02825` một lượng nhỏ, và mean Macro-3 AP `.10424` còn xa Gold-Only `.23692`, EXP24/36/43 hoặc Phase 4 fusion. Đây không phải model frontier.

W2V2 candidate mass trên top-10 + canonical vẫn nhỏ: DEV_CAL mean `6.51e-6`, median `6.69e-14`; DEV_SEARCH mean `1.31e-5`, median `3.03e-13`. Tỷ trọng sequence mass trên candidates nhiều đường tối ưu là `.9066` CAL / `.8462` SEARCH; có ambiguity ở 293/300 CAL và 569/599 SEARCH utterance. Vì vậy cũng không được gọi là posterior đầy đủ. Search depth/k=10 không đổi; không chọn hệ số blend hoặc tối ưu hóa trên SEARCH.

### Lặp ba seed W2V2: 101/202/303

E1/E2/E3 đều được chạy lại với cùng beam-10 candidate protocol trên mỗi seed; mỗi seed có input, receipt, prediction vectors, EVAL JSON và CAL-policy receipt riêng. Bảng dưới là DEV_SEARCH mô tả, không dùng để chọn seed hoặc phương pháp.

| Seed | Arm | AP-S | AP-O | AP-I | Macro-3 AP | Macro-3 F1 (argmax/.5) |
|---:|---|---:|---:|---:|---:|---:|
| 101 | E1 | 0.146598 | 0.022569 | 0.010498 | 0.059889 | 0.136532 |
| 101 | E2 | 0.201183 | 0.035341 | 0.018073 | 0.084866 | 0.165517 |
| 101 | E3 | 0.235737 | 0.041842 | 0.029214 | 0.102264 | 0.174375 |
| 202 | E1 | 0.146554 | 0.023043 | 0.008359 | 0.059319 | 0.130633 |
| 202 | E2 | 0.207887 | 0.035211 | 0.026903 | 0.090000 | 0.171980 |
| 202 | E3 | 0.241050 | 0.042150 | 0.042219 | 0.108473 | 0.180736 |
| 303 | E1 | 0.141102 | 0.021821 | 0.010124 | 0.057682 | 0.130118 |
| 303 | E2 | 0.200160 | 0.033782 | 0.018636 | 0.084193 | 0.163330 |
| 303 | E3 | 0.236475 | 0.038843 | 0.030607 | 0.101975 | 0.172124 |

| Arm | AP-S mean [min,max] | AP-O mean [min,max] | AP-I mean [min,max] | Macro-3 AP mean [min,max] |
|---|---|---|---|---|
| E1 | 0.144751 [0.141102, 0.146598] | 0.022478 [0.021821, 0.023043] | 0.009661 [0.008359, 0.010498] | 0.058963 [0.057682, 0.059889] |
| E2 | 0.203077 [0.200160, 0.207887] | 0.034778 [0.033782, 0.035341] | 0.021204 [0.018073, 0.026903] | 0.086353 [0.084193, 0.090000] |
| E3 | 0.237754 [0.235737, 0.241050] | 0.040945 [0.038843, 0.042150] | 0.034013 [0.029214, 0.042219] | 0.104238 [0.101975, 0.108473] |

E3 exceeds E1 in S/O/I AP and Macro-3 AP for all three seeds. The exact-gap insertion gain is therefore repeatable across these recognizer seeds, but remains small against prevalence and specialists: the three-seed mean AP-I `.03401` is about 5.5× DEV_SEARCH insertion prevalence `.00614`, but only 18% of Gold-Only AP-I `.18672`; mean Macro-3 AP `.10424` is less than half Gold-Only `.23692`.

E3's CAL_MAX_ERROR_MACRO_F1 thresholds were fit separately per seed and frozen on SEARCH. SEARCH F1/FWR: seed 101 `.18087/7.684`, seed 202 `.18833/6.631`, seed 303 `.18191/7.409`; mean F1 `.18371`, range `[.18087,.18833]`. These operating results remain distinct from AP ranking and use high-warning macro-F1 policies.

| Seed | CAL mass mean / median | SEARCH mass mean / median | SEARCH posterior weight on multi-path candidates | SEARCH utterances with ambiguous candidates |
|---:|---|---|---:|---:|
| 101 | `6.51e-6 / 6.69e-14` | `1.31e-5 / 3.03e-13` | 0.8462 | 569/599 |
| 202 | `2.48e-5 / 8.58e-13` | `3.82e-5 / 2.44e-12` | 0.8131 | 567/599 |
| 303 | `7.05e-6 / 1.45e-13` | `1.43e-5 / 4.51e-13` | 0.8545 | 576/599 |

### Per-speaker DEV_SEARCH E3 AP and support

Các speaker dưới đây đều không thấy trong TRAIN. Không dùng kết quả per-speaker để chọn phương pháp. Support gold giống nhau qua ba seed; AP là trung bình và khoảng min–max qua seed 101/202/303.

| Speaker | Utterances / targets / gaps | S support (prevalence) | AP-S mean [range] | O support (prevalence) | AP-O mean [range] | I support (prevalence) | AP-I mean [range] |
|---|---:|---:|---:|---:|---:|---:|---:|
| NCC | 150 / 4,947 / 5,097 | 536 (0.1083) | 0.2799 [0.2762, 0.2846] | 55 (0.0111) | 0.0348 [0.0342, 0.0357] | 29 (0.0057) | 0.0443 [0.0343, 0.0540] |
| RRBI | 150 / 4,945 / 5,095 | 451 (0.0912) | 0.2053 [0.2030, 0.2073] | 65 (0.0131) | 0.0667 [0.0637, 0.0738] | 44 (0.0086) | 0.0279 [0.0249, 0.0289] |
| YBAA | 149 / 4,934 / 5,083 | 331 (0.0671) | 0.1209 [0.1149, 0.1291] | 31 (0.0063) | 0.0365 [0.0334, 0.0393] | 30 (0.0059) | 0.0715 [0.0542, 0.0903] |
| YDCK | 150 / 4,929 / 5,079 | 450 (0.0913) | 0.3273 [0.3238, 0.3322] | 58 (0.0118) | 0.0408 [0.0400, 0.0419] | 22 (0.0043) | 0.0494 [0.0414, 0.0630] |

Across these four held-out speakers and three seeds, E3 AP is above that speaker's class prevalence for each S/O/I cell. This is descriptive evidence that the ranking signal is not confined to one DEV_SEARCH speaker. Absolute omission and insertion AP remain low; insertion has only 22–44 positives per speaker, and omission has 31–65, so the per-speaker estimates are noisy. S varies materially by speaker (mean AP `.1209` to `.3273`), which signals remaining domain variation.

### F1 policy: tách khỏi AP

Argmax/gap `p≥0.5` trong bảng trên là policy chẩn đoán cố định. Ngoài ra đã chạy đúng `select_cal_max_error_macro_f1` của EVAL-V3.1 trên full DEV_CAL (finite predeclared grid, không tuyên bố continuous optimum), rồi áp ngưỡng đóng băng lên DEV_SEARCH. Đây không phải cùng policy với argmax.

| k=10 arm | Ngưỡng CAL (S/O/I) | CAL Macro-3 F1 | SEARCH Macro-3 F1 | SEARCH joint FWR |
|---|---|---:|---:|---:|
| E1 | 0 / 1.0000000000000002 / 0 | 0.10693 | 0.05883 | 63.800 |
| E2 | 0.13720 / 0.85342 / 0 | 0.13514 | 0.08071 | 55.050 |
| E3 | 0.12519 / 0.92931 / 0.02659 | 0.17682 | 0.10621 | 18.992 |

| W2V2 arm | Ngưỡng CAL (S/O/I) | CAL Macro-3 F1 | SEARCH Macro-3 F1 | SEARCH joint FWR |
|---|---|---:|---:|---:|
| E1 | 1 / 1 / 1 | 0.21891 | 0.13653 | 12.247 |
| E2 | 0.49483 / 1 / 1 | 0.28215 | 0.17172 | 7.047 |
| E3 | 0.21320 / 1 / 0.4 | 0.31378 | 0.18087 | 7.684 |

FWR lớn phản ánh tối ưu macro-F1 không giới hạn cảnh báo; đây là chỉ số được báo cáo, không phải cổng thành công. F1 policy không thay đổi ranking AP.

## So sánh diễn giải

- CTC sequence scores tạo tín hiệu MDD vượt greedy edit trên một số lớp, nhất là S và I ở E3, nhưng biên absolute và transfer CAL→SEARCH rất yếu.
- Marginal path có ý nghĩa thực nghiệm: trên SEARCH E3 vượt E2 ở AP-S `+0.0211`, AP-I `+0.0066` với k=10, còn AP-O E3 cao hơn E2 `+0.0020`. Kết quả không ổn định hoàn toàn theo split (CAL O tốt hơn, CAL I vẫn gần prevalence).
- Posterior alignment ambiguity cao: với log-mel k=10, `1 - 1/count` trung bình khoảng `.86`; W2V2 seed 101/202/303 là khoảng `.61–.62`. Tỷ trọng sequence mass nằm trên candidate có nhiều đường là `.81–.91` giữa các seed W2V2 và xấp xỉ `.98` trong log-mel. Repeated phones/tie-equivalent edits vì vậy không hiếm.
- Candidate mass thấp hơn nhiều mức cần thiết để coi top-10 normalized weights là posterior của transcript. E2/E3 ranking là exploratory conditional score. Beam width 100 trên subset xác nhận widening đơn thuần không khắc phục vấn đề.
- Exact-gap score vẫn yếu: E3 SEARCH AP-I `.01358`, gần gấp đôi E1 nhưng vẫn thấp Gold-Only `.18672` hơn 13 lần và EXP43 `.28506` hơn 20 lần.

## Kiểm chứng thuật toán và artifacts

Synthetic CPU fixture so sánh (1) CTC forward probability với enumeration mọi frame path cho 7 sequence; (2) wide-beam prefix search score với exact CTC forward cho mọi candidate của fixture; (3) substitution/deletion/gap-event marginals với enumeration tất cả edit paths tối ưu cho 8 trường hợp gồm boundary insert, omission cuối, repeated phones và nhiều đường tương đương. Tất cả pass. Đây là test toán học, không phải kết quả speech.

Artifacts chính:

- `experiments/rq1_method_bench_03/agent_b/probabilistic_edit.py` — CTC forward, prefix beam search, all-optimal edit-path marginal, finite-set aggregation.
- `experiments/rq1_method_bench_03/agent_b/evaluate_ctc_edit.py` — E1/E2/E3, full event mapping, EVAL-V3.1, receipts, JSONL/NPZ.
- `experiments/rq1_method_bench_03/agent_b/apply_cal_policy.py` — policy CAL-fitted and frozen SEARCH.
- `experiments/rq1_method_bench_03/agent_b/synthetic_fixtures.py` and `synthetic_fixture_receipt.json` — exhaustive CPU correctness fixtures.
- `experiments/rq1_method_bench_03/agent_b/wide_beam_coverage.py` and `wide_beam_coverage_receipt.json` — width diagnostic.
- `experiments/rq1_method_bench_03/agent_b/ctc_edit_run/logmel_ctc/beam_10/` — paired k=10 results, evaluator JSON, event prediction JSONL, per-event score NPZ, hashes, and CAL policy receipt.
- `experiments/rq1_method_bench_03/agent_b/ctc_edit_run/beam_3/` — k=3 paired results and policy run.
- `experiments/rq1_method_bench_03/agent_b/ctc_edit_run/frozen_w2v2_h6/beam_10/` — second-encoder paired results and separately frozen CAL policy.
- `experiments/rq1_method_bench_03/agent_b/ctc_edit_run/frozen_w2v2_h6_s202/beam_10/` and `.../frozen_w2v2_h6_s303/beam_10/` — seed repeat receipts, full event outputs and per-seed CAL policy evaluations.
- `experiments/rq1_method_bench_03/agent_b/speaker_ap_by_seed.py` and `w2v2_e3_dev_search_speaker_ap.csv` — per-speaker supports and AP computed only from saved event predictions.
- `experiments/rq1_method_bench_03/agent_b/build_summary.py` and `ctc_edit_summary.csv` — flattened per-seed/split/method leaderboard.

Reproduction commands:

```sh
.venv/bin/python experiments/rq1_method_bench_03/agent_b/synthetic_fixtures.py
.venv/bin/python experiments/rq1_method_bench_03/agent_b/evaluate_ctc_edit.py --splits dev_cal dev_search --k 3
.venv/bin/python experiments/rq1_method_bench_03/agent_b/evaluate_ctc_edit.py --splits dev_cal dev_search --k 10 --run-tag logmel_ctc
.venv/bin/python experiments/rq1_method_bench_03/agent_b/apply_cal_policy.py --run-dir experiments/rq1_method_bench_03/agent_b/ctc_edit_run/logmel_ctc/beam_10 --beam-k 10
.venv/bin/python experiments/rq1_method_bench_03/agent_b/evaluate_ctc_edit.py --splits dev_cal dev_search --k 10 --input-dir experiments/rq1_method_bench_03/agent_a --file-prefix w2v2_ --run-tag frozen_w2v2_h6
.venv/bin/python experiments/rq1_method_bench_03/agent_b/apply_cal_policy.py --run-dir experiments/rq1_method_bench_03/agent_b/ctc_edit_run/frozen_w2v2_h6/beam_10 --beam-k 10
.venv/bin/python experiments/rq1_method_bench_03/agent_b/evaluate_ctc_edit.py --splits dev_cal dev_search --k 10 --input-dir experiments/rq1_method_bench_03/agent_a --file-prefix w2v2_s202_ --run-tag frozen_w2v2_h6_s202
.venv/bin/python experiments/rq1_method_bench_03/agent_b/apply_cal_policy.py --run-dir experiments/rq1_method_bench_03/agent_b/ctc_edit_run/frozen_w2v2_h6_s202/beam_10 --beam-k 10
.venv/bin/python experiments/rq1_method_bench_03/agent_b/evaluate_ctc_edit.py --splits dev_cal dev_search --k 10 --input-dir experiments/rq1_method_bench_03/agent_a --file-prefix w2v2_s303_ --run-tag frozen_w2v2_h6_s303
.venv/bin/python experiments/rq1_method_bench_03/agent_b/apply_cal_policy.py --run-dir experiments/rq1_method_bench_03/agent_b/ctc_edit_run/frozen_w2v2_h6_s303/beam_10 --beam-k 10
.venv/bin/python experiments/rq1_method_bench_03/agent_b/speaker_ap_by_seed.py
.venv/bin/python experiments/rq1_method_bench_03/agent_b/build_summary.py
.venv/bin/python experiments/rq1_method_bench_03/agent_b/wide_beam_coverage.py --n-utterances 30 --widths 10 50 100
```

Paired EVAL run wall time: log-mel k=3 `55.27 s`, k=10 `158.32 s`; W2V2 k=10 `190.46 s` CPU. Evaluation consumes Agent A's float16 exported log posterior arrays; DP accumulates in float64. Token map is Agent A's 42-phone inventory plus `<blank>` ID 42, and canonical aliases use its documented `ER)→ER`, `D_→D`, `Z_→Z`, `Y_→Y`, `AH)→AH` normalization. Candidate set consists of Agent A's prefix-beam output and canonical sequence. Each encoder/seed uses a separate `--run-tag`/input directory, preserving independent experiment identities.

## Kết luận hiện tại

Marginalizing equivalent edit paths reduces arbitrary alignment tie-breaking and improves AP over E1 in all three W2V2 seeds and error classes. Wav2Vec2 CTC is a promising repeatable sequence-evidence improvement over log-mel, including a small AP-I edge over the historical H05 frame-argmax mapping. It remains far below trained gap heads and expert compositions, and its finite beam mass remains tiny. Do not integrate current E3 scores into U2 or describe them as a full sequence posterior.
