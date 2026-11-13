# RQ1-METHOD-BENCH-04 — Thử nghiệm phương pháp non-CTC độc lập

**Phạm vi:** tìm phương pháp bằng chứng âm học không cần giải mã chuỗi phoneme; chạy trên TRAIN có speaker-grouped OOF, DEV_CAL và DEV_SEARCH theo cùng định nghĩa event EVAL-V3.1. DEV_SEARCH đã exposed nên mọi số SEARCH bên dưới chỉ mang tính thăm dò. **Ngày:** 2026-11-08.

## Tóm tắt

Tôi chọn một adaptation nhỏ theo hướng phát hiện bất thường từ mẫu phát âm đúng, lấy cảm hứng trực tiếp từ SylPh (Interspeech 2025) và các mô hình one-class theo phoneme trước đó. Với mỗi target phone, chỉ các token CORRECT từ phần TRAIN-fit tạo centroid âm học. Điểm lỗi chính là cosine similarity đến centroid của phone cạnh tranh tốt nhất trừ similarity đến centroid của phone dự định. Không có CTC, ASR chuỗi realized phone, hay nhãn lỗi dùng để fit centroid. Phương pháp được chạy với 4 fold speaker-held-out trên TRAIN và chấm riêng hai lớp target S/O.

Kết quả grouped TRAIN OOF của cosine margin là AP-S **0.211316**, AP-O **0.165356**; thấp hơn plain neural head đã có trên cùng thiết kế grouped OOF (0.423209/0.276706). Trên DEV_SEARCH, exploratory AP-S/O là **0.164551/0.060511**, thấp hơn các neural baseline có sẵn. Margin tốt hơn rõ rệt so với score chỉ dùng khoảng cách bất thường dương tính, nhưng không vượt baseline liên quan. Không đưa ra điểm insertion hoặc Macro-3 vì phương pháp hiện không định vị được insertion ở từng gap.

## Nguồn nghiên cứu và lựa chọn phương pháp

| Nguồn gốc | Ý tưởng mà bài báo hỗ trợ | Áp dụng trong LANGFENS |
|---|---|---|
| Jin et al., Interspeech 2025, [SylPh: Contrastive Learning-based Syllable-Level Mispronunciation Detection and Diagnosis for Speech Audiometry](https://www.isca-archive.org/interspeech_2025/jin25c_interspeech.pdf), DOI 10.21437/Interspeech.2025-1285 | Tạo positive bag từ các mẫu âm học bình thường của stimulus; pseudo-negative bag từ mẫu bình thường của stimulus khác; so độ tương đồng cosine để tạo ID/OOD features; học quyết định bằng contrastive objective và text prompts. Tác giả báo cáo syllable-level detection và phoneme-level localization trên dữ liệu speech-audiometry tiếng Hàn. | Chúng tôi giữ cốt lõi so sánh positive với pseudo-negative prototype, nhưng thực hiện adaptation ở mức target phone: một centroid đúng cho mỗi phone và score cosine margin trực tiếp. Không tái tạo syllable bags, Hubert Hàn, text encoder, projection head, hay contrastive training. Kết quả của paper không được coi là kết quả LANGFENS. |
| Shahin & Ahmed, Speech Communication 2019, [Anomaly detection based pronunciation verification approach using speech attribute features](https://doi.org/10.1016/j.specom.2019.06.003) | Dùng One-Class SVM riêng cho từng phoneme, fit bằng phát âm đúng và speech attributes; bài báo báo cáo cải thiện so với GOP trên các corpus khác nhau. | Củng cố giả thuyết chỉ dùng mẫu đúng có thể cung cấp acoustic normality model. Thử nghiệm này thay OCSVM/attribute-bank bằng một diagonal-density control và một cosine centroid margin trên frozen Wav2Vec2 features. Không tuyên bố tái tạo paper 2019. |

SylPh là lựa chọn phù hợp vì cơ chế của nó không yêu cầu nhận dạng realized-phone sequence trước khi đánh giá abnormality. Tuy nhiên, task/corpus của paper là từ đơn hoặc hai âm tiết tiếng Hàn trong speech audiometry, không phải L2-ARCTIC và không dùng S/O/I edit-event metrics. Chưa xác minh được implementation/checkpoint công khai tương thích LANGFENS; do đó đã thực hiện adaptation trực tiếp thay vì tuyên bố reproduction.

## Phương pháp và giao thức

Với vector âm học `x` cho target event và phone dự định `q`, mô hình lấy các mẫu CORRECT của các speaker trong TRAIN-fit fold để tính centroid `μ_p` cho 39 phone. Vector được chuẩn hóa L2 trước khi tính cosine. Score chính:

`score_margin(x,q) = max_{p != q} cos(x, μ_p) - cos(x, μ_q)`

Score cao nghĩa là crop âm học gần một phone cạnh tranh hơn phone dự định. Đây là score xếp hạng, không phải posterior hay xác suất đã calibration. Control one-class là mean standardized squared distance tới normal distribution của phone dự định, với variance diagonal co về variance toàn cục theo `n/(n+64)`.

- Input: frozen H-002/Wav2Vec2 layer-6 target crop mean tại offsets 3:14 (768 chiều) và reference one-hot 39 chiều để lấy phone dự định.
- Supervision: nhãn CORRECT của TRAIN-fit events chỉ dùng để chọn normal samples; lỗi ở held-out fold chỉ dùng để chấm AP. Mỗi fold fit lại toàn bộ centroid/density từ 9 speaker, rồi score 3 speaker bị giữ lại. Không dùng DEV labels để fit hay chọn tham số.
- Chấm lớp: target substitution (S) và omission (O), qua target labels EVAL-V3.1; báo AP riêng theo lớp.
- Không có inference trên gap, vì vậy không có I, exact-gap coverage, Macro-3, Macro-4 hay F1 được báo cáo.
- Đặc tính inference: không có ASR/CTC decode. Dùng intended reference và crop âm học. Crop lấy từ cache aligned H-002 hiện hữu; nhóm này không xây dựng lại bộ aligner/crop online hoàn chỉnh, nên không xem đây là một pipeline triển khai độc lập hoàn chỉnh.

## Kết quả thực nghiệm

### Speaker-grouped TRAIN OOF

| Fold speaker bị giữ lại | AP-S | AP-O | Support S/O |
|---|---:|---:|---:|
| ABA, ASI, BWC | .193759 | .133469 | 1,859 / 357 |
| EBVS, HJK, HKK | .181074 | .097110 | 1,524 / 301 |
| HQTV, LXC, NJS | .246310 | .270123 | 2,079 / 773 |
| PNV, SKA, SVBI | .222492 | .138264 | 1,919 / 394 |
| **Concatenated OOF** | **.211316** | **.165356** | **7,381 / 1,825** |

Grouped OOF comparator từ METHOD-BENCH-03 là plain neural head AP-S/O **.423209/.276706**; GRL là **.423371/.276634**. Cả hai đều cao hơn prototype margin. Fold OOF macro trung bình target S/O của margin là 0.1883; không phải Macro-3 do thiếu I. Kết quả này cho thấy normal prototypes có một phần tín hiệu acoustic transfer, nhưng chưa thay được supervised classifier trên các feature tương tự.

### DEV_CAL và DEV_SEARCH — chỉ exploratory

| Score | Split | AP-S | AP-O | Support S/O |
|---|---|---:|---:|---:|
| Mahalanobis positive-only | DEV_CAL | .226582 | .064455 | 1,781 / 462 |
| Cosine positive-vs-negative margin | DEV_CAL | .277618 | .250111 | 1,781 / 462 |
| Event-conditioned CTC E4 S101 | DEV_CAL | .457520 | .409375 | 1,781 / 462 |
| Mahalanobis positive-only | DEV_SEARCH | .101093 | .014765 | 1,768 / 209 |
| **Cosine positive-vs-negative margin** | **DEV_SEARCH** | **.164551** | **.060511** | **1,768 / 209** |

Margin cải thiện so với positive-only density trên cả S/O của TRAIN OOF, CAL và SEARCH. Trên CAL, E4 AP-S/O cao hơn margin. Mức giảm CAL→SEARCH rõ rệt, đặc biệt ở omission; hai split có speaker khác nhau và DEV_SEARCH chỉ có 209 omission positives. Không diễn giải chênh lệch này như bằng chứng nhân quả về speaker identity.

### Đối chiếu các neural score DEV_SEARCH cùng event population

Bảng dưới so AP target S/O trên 19,755 targets với supports S=1,768/O=209. Các historical values lấy từ báo cáo METHOD-BENCH-03; CTC E3 dùng cùng Wav2Vec2 recognizers và các target event outputs đã lưu. Đây là so sánh score trên cùng event population, nhưng SEARCH đã exposed nên chỉ exploratory.

| Hệ thống | AP-S | AP-O | Nguồn / ghi chú |
|---|---:|---:|---|
| Gold-Only S101 | .364602 | .159433 | Neural reference score |
| H-002 S101 | .377810 | .167294 | Neural target score |
| U2 S101 | .371360 | .185318 | Unified neural checkpoint |
| EXP24 | .357737 | .225092 | Learned checkpoint |
| EXP36 | .387088 | .204687 | Score composition |
| EXP43 | .363453 | .205771 | Score composition |
| CTC E3, mean 3 seeds | .237754 | .040945 | CTC event/path score; recognition-first |
| Normal-phone cosine margin | .164551 | .060511 | Non-CTC, target-only; adaptation này |

Cosine margin thấp hơn cả historical neural systems ở S/O và cũng thấp hơn E3 ở S, dù nhỉnh hơn E3 ở O. Vì không có gap score, không thể kết luận về insertion hoặc Macro-3. Candidate không nên được tích hợp/fuse dựa trên các số này.

## Đã thực thi và tái lập

- Source: `experiments/rq1_method_bench_04/agent_c/normal_phone_anomaly.py`
- Receipt/metrics/hash/supports: `experiments/rq1_method_bench_04/agent_c/normal_phone_anomaly_receipt.json`
- OOF scores: `normal_phone_anomaly_train_oof.npy` và `normal_phone_anomaly_contrastive_margin_train_oof.npy`
- Split scores: `normal_phone_anomaly_{dev_cal,dev_search}.npy` và `normal_phone_anomaly_contrastive_margin_{dev_cal,dev_search}.npy`
- Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_04/agent_c/normal_phone_anomaly.py`
- Compute: CPU, 3.31 giây cho 4-fold OOF cùng scoring CAL/SEARCH; peak memory không đo; không có GPU lease, checkpoint hay model weights.
- Script cũng lưu positive-only Mahalanobis control. Kết quả score margin được định nghĩa trước theo phép đối chiếu cosine positive/pseudo-negative trong SylPh; không tìm fusion weights trên DEV_SEARCH.

Một script `train_linear_margin.py` cũng xuất hiện trong namespace và đang được chạy bởi một process không do phiên thực hiện adaptation ini khởi chạy. File/scores của process đó không dùng trong các kết luận trên; provenance và receipt đầy đủ cần được xác minh riêng trước khi nhập vào leaderboard.

## Kết luận / falsification

Giả thuyết yếu rằng “mẫu phát âm đúng cùng với prototype phone cạnh tranh tạo thêm ranking signal non-CTC” được ủng hộ bởi grouped OOF AP vượt prevalence và margin tốt hơn positive-only distance. Giả thuyết mạnh hơn rằng adaptation này vượt recognizer-first hoặc unified neural baselines bị bác bỏ trong phép thử hiện tại: S/O grouped OOF thấp hơn supervised neural head, DEV_SEARCH thấp hơn tất cả historical target baselines, và không hề có insertion evidence. Việc tăng từ positive-only sang cosine margin chưa khắc phục độ nhạy theo speaker/split.

Hướng tiếp theo chỉ hợp lý nếu có aligned frame/phone evidence được tạo hoàn toàn từ WAV+reference ở inference và nếu training trên TRAIN speakers có thể học đối chiếu phone cạnh tranh mà vẫn giữ grouped OOF. Nếu thử tiếp, cần bổ sung exact-gap representation có căn cứ để chấm I; không nên suy diễn insertion từ score target.
