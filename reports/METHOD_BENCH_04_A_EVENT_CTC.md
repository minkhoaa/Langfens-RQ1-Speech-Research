# METHOD-BENCH-04 A — E4: event-conditioned CTC likelihood

## Kết luận

E4 đã được triển khai bằng forward likelihood CTC chính xác trên posterior của ba recognizer Wav2Vec2-CTC đã đóng băng từ METHOD-BENCH-03. Trên hai speaker TRAIN holdout, E4 tăng AP của S/O/I so với E1 và E3 trên cả ba seed. Trên DEV_CAL, E4 tăng AP-S/AP-O nhưng giảm AP-I so với E3 trên cả ba seed. Vì vậy E4 là cải thiện rõ cho xếp hạng substitution và omission trong thiết lập này, nhưng chưa phải bộ chấm S/O/I cân bằng hoặc đã được hiệu chuẩn.

Thiết kế này **không** tính posterior lỗi chính xác. Nó so sánh likelihood của một reference với tập đối thủ hữu hạn được xây từ TRAIN. Các tỷ số likelihood là điểm xếp hạng, không phải xác suất sự kiện đã hiệu chuẩn.

## Phương pháp đã thực thi

Tôi dùng cùng ba recognizer, posterior, phone tokenizer, blank ID và clock 20 ms cho E1/E3/E4. Không huấn luyện lại recognizer. `blank_id=42`; 42 phone không-blank. Posterior `log_posteriors` và `posteriors` trong các NPZ seed 101/202/303 có offset và frame metadata theo từng utterance.

Với reference `q`, E4 tính `log P_CTC(q_alt | x) - log P_CTC(q | x)` bằng dynamic programming CTC forward trong log-space:

- **Omission:** một sequence xóa đúng `q[i]` cho mỗi target.
- **Substitution:** tối đa bốn phone đối thủ cho mỗi target, theo confusion counts của greedy CTC trên 1,500 utterance TRAIN-fit; đối thủ bổ sung lấy từ phone inventory khi confusion count thiếu.
- **Insertion:** tối đa tám phone có tần suất chèn cao nhất trên TRAIN-fit; chấm riêng từng gap trong đủ `N+1` gap. Bao gồm chèn phone trùng phone kề, vì CTC vẫn có thể xuất hai phone giống nhau nếu có blank ở giữa.
- Với nhiều đối thủ, điểm chính là `logmeanexp(log P(alternative)) - log P(canonical)`. Phép lấy trung bình log loại hiệu ứng số lượng đối thủ. Omission có một đối thủ nên dùng trực tiếp likelihood ratio.

Lựa chọn đối thủ chỉ dùng TRAIN-fit. Không dùng gold timing, realized phone hoặc error label làm đặc trưng inference. E4 dùng posterior được tính từ WAV và reference được đưa vào khi tạo giả thuyết. Điểm S/O/I được giữ tách biệt theo target và gap.

## Xác minh xác suất và candidate mass

Mã C++ của E4 được so khớp với hàm Python CTC forward đã dùng ở MB03 trên 8 chuỗi đầu ra, với T từ 0 đến 6: mọi trường hợp khớp trong dung sai `1e-11`. Fixture độc lập duyệt toàn bộ `3^3` frame paths, gộp các path thành 9 chuỗi đầu ra CTC duy nhất và tổng xác suất sequence đúng bằng `1.0`.

Các posterior FP16 lưu trữ có sai lệch nhỏ: trị tuyệt đối log-sum-exp mỗi frame tối đa `5.05e-4` và trung vị khoảng `2–3e-5`. E4 chuẩn hóa lại từng hàng trong log-space; sai lệch còn lại tối đa `8.9e-16`. Đây không giải thích candidate mass thấp.

Kiểm tra candidate mass của E3 cộng xác suất CTC chính xác của các chuỗi beam-10 duy nhất cộng canonical sequence, sau khi chuẩn hóa posterior frame. Đây là tổng của một tập sequence rời nhau và phép tính hợp lệ, nhưng tập N-best rất không đầy đủ. Seed 101 có candidate mass trung bình/trung vị:

| Phần dữ liệu | Trung bình | Trung vị | p10 | p90 |
|---|---:|---:|---:|---:|
| TRAIN speaker holdout | 1.32e-5 | 8.13e-12 | 2.25e-18 | 8.05e-7 |
| DEV_CAL | 6.51e-6 | 6.69e-14 | 1.36e-21 | 1.70e-7 |
| DEV_SEARCH, thăm dò | 1.31e-5 | 3.03e-13 | 2.38e-20 | 2.04e-7 |

Diễn giải: phép tính mass được xác minh; mass nhỏ phản ánh sự thiếu bao phủ của tập beam/canonical hữu hạn đối với phân phối sequence, chứ không phải lỗi chuẩn hóa posterior. Đây vẫn là chẩn đoán phụ thuộc candidate construction, không phải mass của E4. Không cộng xác suất qua event E4, vì cùng một output sequence có thể phù hợp nhiều edit event.

## Kết quả đối chiếu cùng recognizer

Mỗi ô là AP. E1 là greedy hard edit, E3 là finite beam-10 CTC-weighted optimal-path marginal của MB03 được tính lại cùng manifest/evaluator, E4 là điểm likelihood event có đối thủ TRAIN-fit. E1/E3/E4 dùng cùng seed recognizer.

### TRAIN speaker holdout — đánh giá chính

| Seed | Hệ thống | AP-S | AP-O | AP-I exact-gap | Macro-3 Error AP |
|---:|---|---:|---:|---:|---:|
| 101 | E1 | .11970 | .01944 | .00581 | .04832 |
| 101 | E3 | .17852 | .05217 | .00997 | .08022 |
| 101 | **E4** | **.28233** | **.11844** | **.03872** | **.14650** |
| 202 | E1 | .11239 | .01880 | .00587 | .04568 |
| 202 | E3 | .17565 | .05671 | .01640 | .08292 |
| 202 | **E4** | **.28952** | **.12144** | **.03404** | **.14833** |
| 303 | E1 | .11214 | .01742 | .00485 | .04480 |
| 303 | E3 | .17409 | .04936 | .01743 | .08029 |
| 303 | **E4** | **.28737** | **.12549** | **.04480** | **.15255** |

So với E3, E4 tăng cả ba AP trên cả ba seed. Macro-3 tăng `+.0654`, `+.0654`, `+.0723`; trung bình E3 `.08114`, E4 `.14913`. AP-S ổn định trong khoảng `.282–.290`, AP-O `.118–.125`, AP-I `.034–.045`. Cả ba seed cho cùng chiều cải thiện. Speaker-specific AP vẫn biến động: seed 101, HJK có O-support 4 và I-support 7 (AP-O `.0179`, AP-I `.0027`), HKK có O-support 103, I-support 39 (AP-O `.1606`, AP-I `.0627`). AP theo speaker có trong CSV để thấy rõ giới hạn do support nhỏ.

### DEV_CAL — lựa chọn phát triển

| Seed | Hệ thống | AP-S | AP-O | AP-I exact-gap | Macro-3 Error AP |
|---:|---|---:|---:|---:|---:|
| 101 | E1 | .25776 | .09182 | .02029 | .12329 |
| 101 | E3 | .37671 | .21035 | **.06120** | .21608 |
| 101 | E4 | **.45752** | **.40937** | .03299 | **.29996** |
| 202 | E1 | .25946 | .09240 | .02198 | .12461 |
| 202 | E3 | .38499 | .20276 | **.05733** | .21503 |
| 202 | E4 | **.45841** | **.41100** | .03352 | **.30098** |
| 303 | E1 | .25295 | .09040 | .01874 | .12070 |
| 303 | E3 | .37061 | .19435 | **.06568** | .21021 |
| 303 | E4 | **.46037** | **.39847** | .04520 | **.30135** |

Trên CAL, E4 tăng S/O và Macro-3 so với E3 cả ba seed; E3 có AP-I cao hơn cả ba seed. Macro-3 của E4 ổn định gần `.301` qua seed. Vì insertion giảm trên CAL, không kết luận E4 đã cải thiện mọi lớp ngoài TRAIN holdout.

### DEV_SEARCH — chỉ thăm dò

Seed 101 E1/E3/E4 lần lượt đạt:

| Hệ thống | AP-S | AP-O | AP-I exact-gap | Macro-3 Error AP |
|---|---:|---:|---:|---:|
| E1 | .14660 | .02257 | .01050 | .05989 |
| E3 | .23547 | .04186 | **.02871** | .10202 |
| E4 | **.31685** | **.15774** | .02357 | **.16606** |

Đây là tập phát triển đã được xem trước và không được dùng để chọn K hay phép gộp; các con số chỉ là kiểm tra thăm dò sau khi chốt kết quả TRAIN holdout và CAL.

## PR, top-K dan threshold metric

Đường cong precision/recall/threshold đầy đủ cho S/O/I được lưu trong từng tệp JSON, cùng AP của likelihood ratio thô. Với seed 101 trên TRAIN holdout, P@10/P@25/P@50/P@100 là S `.80/.64/.52/.40`, O `.30/.24/.18/.18`, I `.10/.12/.10/.08`. Trên DEV_CAL: S `.70/.64/.64/.67`, O `.40/.72/.64/.56`, I `.00/.00/.02/.04`. Điều này cho thấy khả năng xếp hạng insertion còn yếu trên CAL dù AP cao hơn prevalence.

Evaluator cũng báo cáo F1 từ pseudo-probability softmax của điểm S/O và sigmoid cho I ở ngưỡng 0.5 (tương đương likelihood ratio bằng 1). Với E4 seed 101 trên TRAIN holdout, precision/recall/F1 lần lượt là S `.314/.253/.280`, O `.032/.785/.061`, I `.054/.217/.086`; trên DEV_CAL: S `.563/.241/.338`, O `.145/.913/.251`, I `.049/.186/.078`. Đây là các chỉ số F1 theo ngưỡng, tách biệt với AP; pseudo-probability chưa được hiệu chuẩn và không phải posterior.

## Ablation dan kontrol

- **Logmeanexp so với lấy max (seed 101):** trên TRAIN holdout, logmeanexp đạt AP `.28233/.11844/.03872` (Macro-3 `.14650`), còn max đạt `.27977/.11844/.03668` (Macro-3 `.14496`). Trên CAL, mean đạt `.45752/.40937/.03299` (Macro-3 `.29996`), còn max đạt `.46289/.40937/.02986` (Macro-3 `.30071`). Không có cách gộp nào thắng ổn định; mean nhỉnh hơn nhẹ với insertion và Macro-3 trên grouped holdout, còn chênh lệch tổng thể nhỏ.
- **Likelihood ratio chuẩn hóa theo số frame:** AP trên TRAIN holdout là `.28142/.06430/.03251`; trên CAL là `.45305/.31476/.03093`. Chuẩn hóa này làm giảm AP omission so với điểm LR thô, đặc biệt trên CAL. Điểm thô vẫn là cấu hình chính.
- **Chỉ dùng prior:** độ hiếm phone tham chiếu từ TRAIN-fit đạt AP S/O/I `.06586/.01088/.00454` trên TRAIN holdout và `.17734/.04600/.00986` trên CAL. Điểm S/O bằng prevalence lớp tương ứng. Prior-only thấp hơn E4 đáng kể, cho thấy phần cải thiện chủ yếu đến từ bằng chứng âm học có điều kiện.
- **Prevalence sự kiện:** điểm hằng số cho AP bằng prevalence: `.06586/.01088/.00454` trên TRAIN holdout và `.17734/.04600/.00986` trên CAL.
- **PER và AP MDD:** PER realized-phone beam-10 trung bình từ MB03 là `.3303` trên TRAIN holdout, `.3458` trên CAL và `.3216` trên SEARCH. E4 không huấn luyện mô hình mới. AP ranking tăng đáng kể từ E1 sang E3/E4, nên PER riêng lẻ không đủ để dự đoán khả năng xếp hạng sự kiện. Chỉ có ba seed và ba tập đánh giá, vì vậy chưa thể kết luận có tương quan nhân quả giữa PER và AP MDD.

## Biaya dan artefak

Tất cả điểm số được tính trên **CPU**; CUDA có sẵn nhưng không dùng để tránh tranh chấp với lượt huấn luyện của Agent C. Không cần GTX 1650 cho thí nghiệm posterior cố định này. E4 logmeanexp mất khoảng 30–32 giây cho 300 utterance TRAIN holdout, 36–39 giây cho 300 utterance CAL và 65 giây cho 599 utterance SEARCH. Số sequence candidate được chấm là 130,607 trên grouped holdout, 133,259 trên CAL và 262,206 trên SEARCH. Không ghi lại peak memory. Không tạo checkpoint mới; đầu vào là linear head CTC seed 101/202/303 cùng posterior W2V2 của MB03.

Artefak utama:

- Implementasi: `experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py`, kernel `ctc_batch.cpp`.
- Baseline matched: `matched_baselines.py`.
- Validasi: `validate_ctc_kernel.py`, kết quả `ctc_kernel_validation.json`.
- Phiếu kết quả theo split/seed: `s{101,202,303}_{train_holdout,dev_cal}_e4.json`; tệp `.scores.jsonl` đi kèm lưu điểm S/O/I theo từng utterance.
- AP và support theo speaker: `e4_per_speaker_ap.csv`.
- Posterior đầu vào: `experiments/rq1_method_bench_03/agent_a/w2v2_{,s202_,s303_}{train_speaker_holdout,dev_cal}_posteriors.npz`; tokenizer, blank và frame clock được mô tả trong phiếu kết quả CTC MB03.

## Reproduksi

```bash
.venv/bin/python experiments/rq1_method_bench_04/agent_a/validate_ctc_kernel.py
.venv/bin/python experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py --prefix w2v2_ --split train_speaker_holdout --output experiments/rq1_method_bench_04/agent_a/s101_train_holdout_e4.json
.venv/bin/python experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py --prefix w2v2_ --split dev_cal --output experiments/rq1_method_bench_04/agent_a/s101_dev_cal_e4.json
.venv/bin/python experiments/rq1_method_bench_04/agent_a/matched_baselines.py --prefix w2v2_ --split train_speaker_holdout
```

Untuk seed 202/303 gunakan `--prefix w2v2_s202_` atau `--prefix w2v2_s303_`; SEARCH seed 101 menggunakan `--split dev_search`. Max ablation memakai `--agg max`. Semua output baru berada di namespace METHOD-BENCH-04; tidak ada artifact historis MB03 ditimpa.

## Batasan dan keputusan

E4 đáng được tiếp tục như một nguồn bằng chứng âm học cho S/O: mức tăng trên grouped holdout đáng kể và ổn định qua seed; AP-O trên CAL gần gấp đôi E3. Kết quả insertion trái chiều hơn: tăng trên TRAIN holdout nhưng giảm trên CAL và SEARCH so với E3; P@10 insertion trên CAL bằng không. Việc giới hạn tám phone chèn theo TRAIN-fit làm giảm độ bao phủ đối thủ. Khác biệt phân phối sự kiện giữa speaker cùng support thấp ở một số lớp cũng có thể ảnh hưởng kết quả.

Bước tiếp theo hợp lý là mở rộng tập đối thủ insertion/substitution một cách có kiểm soát hoặc dùng bằng chứng âm học cục bộ theo thời gian để chấm gap, với quyết định dựa trên TRAIN grouped/CAL. Không diễn giải E4 thành posterior đầy đủ và không chọn cấu hình bằng SEARCH.
