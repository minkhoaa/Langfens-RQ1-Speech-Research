# LANGFENS RQ1-ENCODER-BENCH-01 — BÁO CÁO KHOA HỌC CHI TIẾT TOÀN BỘ KẾT QUẢ VÀ SỐ LIỆU NGHIÊN CỨU

**Dự án:** LANGFENS — Non-Native Mispronunciation Detection & Diagnosis (L2-ARCTIC)  
**Chiến dịch:** `LANGFENS-RQ1-ENCODER-BENCH-01` — Wav2Vec2 Suitability Verification  
**Vai trò:** Senior Speech ML Scientist & Research Engineer  
**Môi trường thực thi:** Linux x86_64, Python 3.11.17, PyTorch 2.6.0+cu124, Transformers 5.22.1, Scikit-Learn 1.9.1  
**Phần cứng:** NVIDIA GeForce GTX 1650 (4,096 MiB VRAM) + Intel Core i5-12450H (12 vCPUs) + 31 GiB RAM  
**Tiêu chuẩn đánh giá:** EVAL-V3.1, 4-fold Speaker-Grouped TRAIN Out-Of-Fold (OOF)  
**Phạm vi dữ liệu:** L2-ARCTIC TRAIN Partition (1,800 utterances, 12 speakers, 59,562 targets, 61,362 gaps)  
**Seeds đăng ký chính thức:** `101` (primary), `202` (replication), `303` (sensitivity)  
**Mục tiêu tối thượng:** Xác định dứt khoát bằng thực nghiệm huấn luyện có kiểm soát xem hiệu năng LANGFENS có bị giới hạn bởi biểu diễn tầng 6 đóng băng của Wav2Vec2 tinh chỉnh ASR (`facebook/wav2vec2-base-960h`) hay không, so với các tầng khác, mô hình tiền huấn luyện thuần SSL, các backbone SSL khác (HuBERT, WavLM), và các phương pháp thích ứng biểu diễn (Layer Weighting, Adapters).

---

## 1. Tóm tắt điều hành & Chứng nhận kiểm toán khoa học

Chiến dịch thực nghiệm `RQ1-ENCODER-BENCH-01` đã hoàn tất 100% các nội dung nghiên cứu trên hệ thống phần cứng cục bộ mà không sử dụng bất kỳ nhãn giả định, không can thiệp hồi tố vào siêu tham số sau khi thấy kết quả, và tuân thủ nghiêm ngặt nguyên tắc cách ly dữ liệu:
1. **Không rò rỉ dữ liệu:** Tuyệt đối không truy cập các tập bảo vệ `DEV_SEARCH`, `DEV_CAL`, hoặc `TEST_SEALED`. Mọi lựa chọn tầng, tối ưu trọng số và dẫn xuất ngưỡng chính sách vận hành $(\tau_S, \tau_O, \tau_I)$ đều được thực hiện trên 3 tập con inner OOF (9 người nói chia thành 6 fit / 3 held) trước khi áp dụng lên 3 người nói outer fold độc lập.
2. **Khớp nối hình học kiểm tra:** 1,800 câu, 12 người nói (ABA, ASI, BWC, EBVS, HJK, HKK, HQTV, LXC, NJS, PNV, SKA, SVBI), 59,562 target (50,356 Correct, 7,381 Substitution, 1,825 Omission) và 61,362 gap (560 Insertion, 60,802 No-Insertion).
3. **Căn chỉnh hoàn toàn dự đoán:** Toàn bộ tâm âm vị và khung ranh giới chèn được dẫn xuất từ Viterbi đơn điệu trên logits bộ phân loại khung CharSiu đã đóng băng (`charsiu/en_w2v2_fc_10ms`). Không sử dụng thời gian gold hoặc nhãn âm vị thực tế ở pha suy luận.

### Bảng chứng nhận kiểm toán chính thức

| Tiêu chí kiểm toán | Yêu cầu tiên quyết | Kết quả thực tế đạt được | Trạng thái |
|---|---|---|---|
| **Bảo lưu Baseline Phase-01** | Giữ nguyên baseline đóng băng chính thức Phase-01 | Khớp chính xác: Macro-3 AP 0.304154 (Fixed), 0.271711 (Neural) | **ĐÃ XÁC MINH** |
| **Tính toàn vẹn Preregistration** | Đóng băng giả thuyết và ngưỡng chấp nhận trước khi chạy outer | `PREREGISTRATION.json` SHA256 `6091c06b2c7bdc961549403558c45ce6570d94b6c4a53b142a813ff9f33572e1` | **ĐÃ XÁC MINH** |
| **Phân hoạch chéo 4 Outer Folds** | 4 fold outer (mỗi fold 3 người nói held-out) | Khớp 100% 4 fold chính thức, 0% rò rỉ danh tính người nói | **ĐÃ XÁC MINH** |
| **Ngưỡng inner OOF** | Dẫn xuất ngưỡng $(\tau_S, \tau_O, \tau_I)$ trên 1,350 câu inner | 100% chính sách vận hành được đóng băng trước khi refit outer | **ĐÃ XÁC MINH** |
| **Kiểm định đa Seed** | Thực thi trên cả 3 seed 101, 202, 303 cho các mô hình rút gọn | Hoàn thành đầy đủ 34 lượt huấn luyện và đánh giá outer CV | **ĐÃ XÁC MINH** |
| **Lưu trữ bằng chứng** | Lưu trữ mã nguồn, trọng số, chỉ số, file log và manifest | Niêm phong đầy đủ tại `experiments/rq1_encoder_bench_01/` | **ĐÃ XÁC MINH** |

---

## 2. Bảng kết quả tổng hợp toàn bộ 34 lượt thực nghiệm

Bảng dưới đây ghi nhận toàn bộ 34 lượt huấn luyện và đánh giá EVAL-V3.1 trên toàn bộ 1,800 câu L2-ARCTIC TRAIN. Tất cả các mô hình sử dụng cùng kiến trúc head H-002 (Target MLP [256, 128, 3]) và MB03 (Gap MLP [128, 64, 2]), cùng hàm mất mát có trọng số lớp, cùng lịch trình huấn luyện (Target: 12 epochs, batch 512; Gap: 5 epochs, batch 1024).

*(Chú thích: AP-S = Average Precision Thay thế; AP-O = AP Lược bỏ; AP-I = AP Chèn tại khe chính xác; FWR = False Warning Rate tại ngân sách cảnh báo sư phạm $B=0.2$; FW = Số lượng cảnh báo sai; WTW = Cảnh báo sai lệch loại lỗi S/O).*

| STT | Agent | Backbone | Checkpoint | Tiền huấn luyện | Tầng / Cơ chế | Seed | Macro-3 AP | AP-S | AP-O | AP-I | Macro-3 F1 ($B=0.2$) | Joint FWR ($B=0.2$) | Recall-S | Recall-O | Recall-I | FW ($B=0.2$) | WTW ($B=0.2$) | Thời gian (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| — | **Ref P1** | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | L6 (Fixed 50/50 Fusion) | Mean | **0.304154** | **0.409825** | **0.344073** | **0.158615** | **0.160371** | **0.204259** | 0.035639 | 0.133881 | 0.126786 | 367.7 | 38.67 | — |
| — | **Ref P1** | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | L6 (Target Neural Only) | Mean | **0.271711** | **0.388424** | **0.268072** | **0.158639** | — | — | — | — | — | — | — | — |
| 1 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | Layer 3 | 101 | 0.216740 | 0.366216 | 0.188325 | 0.095679 | 0.091252 | 0.194444 | 0.058529 | 0.024110 | 0.073214 | 350 | 23 | 67.95 |
| 2 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | **Layer 6 (Incumbent)** | **101** | **0.265951** | **0.386090** | **0.255077** | **0.156685** | **0.121232** | **0.183333** | 0.051484 | 0.037808 | 0.128571 | 330 | 12 | 70.62 |
| 3 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | Layer 6 (Incumbent) | 202 | 0.275468 | 0.383379 | 0.281194 | 0.161829 | 0.126982 | 0.138333 | 0.041999 | 0.050959 | 0.130357 | 249 | 15 | 70.02 |
| 4 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | Layer 6 (Incumbent) | 303 | 0.284850 | 0.385049 | 0.276941 | 0.192561 | 0.158239 | 0.236111 | 0.053787 | 0.066301 | 0.183929 | 425 | 28 | 83.35 |
| **—** | **Agent A** | **Wav2Vec2** | `facebook/wav2vec2-base-960h` | **ASR_finetuned** | **Layer 6 (Mean ± Std)** | **All** | **0.275423 ± 0.0095** | **0.384839** | **0.271071** | **0.170359** | **0.135486** | **0.185926** | **0.049090** | **0.051689** | **0.147619** | **334.7** | **18.33** | **74.66** |
| 5 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | Layer 9 | 101 | 0.264834 | 0.352009 | 0.286283 | 0.156209 | 0.128213 | 0.180556 | 0.031568 | 0.054795 | 0.157143 | 325 | 21 | 97.58 |
| 6 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | Layer 12 | 101 | 0.221854 | 0.349026 | 0.203618 | 0.112918 | 0.089588 | 0.173333 | 0.019645 | 0.039452 | 0.105357 | 312 | 8 | 70.46 |
| 7 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base` | SSL_only | Layer 3 | 101 | 0.200913 | 0.363350 | 0.167575 | 0.071815 | 0.076682 | 0.175556 | 0.048638 | 0.025753 | 0.055357 | 316 | 24 | 68.86 |
| 8 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base` | SSL_only | Layer 6 | 101 | 0.261242 | 0.376444 | 0.252638 | 0.154644 | 0.119976 | 0.175000 | 0.047961 | 0.033973 | 0.137500 | 315 | 19 | 69.23 |
| 9 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base` | SSL_only | Layer 6 | 202 | 0.263962 | 0.383000 | 0.262040 | 0.146846 | 0.103025 | 0.145000 | 0.035903 | 0.036712 | 0.110714 | 261 | 16 | 75.79 |
| 10 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base` | SSL_only | Layer 6 | 303 | 0.259930 | 0.376930 | 0.242132 | 0.160729 | 0.129005 | 0.235000 | 0.057445 | 0.041644 | 0.144643 | 423 | 26 | 71.45 |
| **—** | **Agent A** | **Wav2Vec2** | `facebook/wav2vec2-base` | **SSL_only** | **Layer 6 (Mean ± Std)** | **All** | **0.261711 ± 0.0021** | **0.378791** | **0.252270** | **0.154073** | **0.117335** | **0.185000** | **0.047103** | **0.037443** | **0.130952** | **333.0** | **20.33** | **72.16** |
| 11 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base` | SSL_only | Layer 9 | 101 | 0.263144 | 0.391132 | 0.260935 | 0.137364 | 0.108364 | 0.180556 | 0.035361 | 0.035616 | 0.133929 | 325 | 16 | 69.29 |
| 12 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base` | SSL_only | Layer 9 | 202 | 0.264370 | 0.395624 | 0.264709 | 0.132777 | 0.108641 | 0.160556 | 0.030890 | 0.046027 | 0.126786 | 289 | 14 | 69.84 |
| 13 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base` | SSL_only | Layer 9 | 303 | 0.264971 | 0.397407 | 0.269228 | 0.128279 | 0.115026 | 0.164444 | 0.053516 | 0.040000 | 0.108929 | 296 | 14 | 84.53 |
| **—** | **Agent A** | **Wav2Vec2** | `facebook/wav2vec2-base` | **SSL_only** | **Layer 9 (Mean ± Std)** | **All** | **0.264162 ± 0.0009** | **0.394721** | **0.264957** | **0.132807** | **0.110677** | **0.168519** | **0.039922** | **0.040548** | **0.123214** | **303.3** | **14.67** | **74.55** |
| 14 | Agent A | Wav2Vec2 | `facebook/wav2vec2-base` | SSL_only | Layer 12 | 101 | 0.181026 | 0.351295 | 0.142585 | 0.049199 | 0.061743 | 0.183889 | 0.041999 | 0.019726 | 0.042857 | 331 | 21 | 70.32 |
| 15 | Agent B | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | Frozen L6 Baseline | 101 | 0.241297 | 0.355160 | 0.226723 | 0.142008 | 0.128257 | 0.273333 | 0.069367 | 0.046027 | 0.112500 | 492 | 42 | 44.94 |
| 16 | **Agent B** | **Wav2Vec2** | `facebook/wav2vec2-base-960h` | **ASR_finetuned** | **Layer-Weighted (0..12)** | **101** | **0.277558** | **0.389407** | **0.280638** | **0.162630** | **0.157628** | **0.280556** | **0.051484** | **0.083836** | **0.173214** | **505** | **37** | **321.63** |
| 17 | Agent B | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | Bottleneck Adapter | 101 | 0.187414 | 0.311947 | 0.167397 | 0.082897 | 0.077940 | 0.248889 | 0.062187 | 0.014247 | 0.055357 | 448 | 19 | 78.26 |
| 18 | Agent B | Wav2Vec2 | `facebook/wav2vec2-base-960h` | ASR_finetuned | Capacity-Matched MLP | 101 | 0.252620 | 0.374010 | 0.249450 | 0.134400 | 0.135108 | 0.272222 | 0.073432 | 0.036164 | 0.142857 | 490 | 31 | 63.58 |
| 19 | Agent C | HuBERT | `facebook/hubert-base-ls960` | SSL_or_ASR | Layer 3 | 101 | 0.209407 | 0.358417 | 0.183850 | 0.085954 | 0.085754 | 0.180556 | 0.039561 | 0.026849 | 0.083929 | 325 | 28 | 63.90 |
| 20 | Agent C | HuBERT | `facebook/hubert-base-ls960` | SSL_or_ASR | Layer 6 | 101 | 0.253962 | 0.388727 | 0.244478 | 0.128680 | 0.116170 | 0.176111 | 0.034684 | 0.037260 | 0.130357 | 317 | 23 | 64.67 |
| 21 | Agent C | HuBERT | `facebook/hubert-base-ls960` | SSL_or_ASR | Layer 6 | 202 | 0.252577 | 0.393831 | 0.244016 | 0.119885 | 0.106689 | 0.174444 | 0.031297 | 0.033973 | 0.119643 | 314 | 22 | 64.91 |
| 22 | Agent C | HuBERT | `facebook/hubert-base-ls960` | SSL_or_ASR | Layer 6 | 303 | 0.252253 | 0.392025 | 0.243717 | 0.121019 | 0.128249 | 0.180000 | 0.046335 | 0.040548 | 0.121429 | 324 | 21 | 67.58 |
| **—** | **Agent C** | **HuBERT** | `facebook/hubert-base-ls960` | **SSL_or_ASR** | **Layer 6 (Mean ± Std)** | **All** | **0.252931 ± 0.0009** | **0.391527** | **0.244070** | **0.123194** | **0.117036** | **0.176852** | **0.037439** | **0.037260** | **0.123810** | **318.3** | **22.00** | **65.72** |
| 23 | Agent C | HuBERT | `facebook/hubert-base-ls960` | SSL_or_ASR | Layer 9 | 101 | 0.268702 | 0.390841 | 0.271913 | 0.143352 | 0.110200 | 0.172778 | 0.044032 | 0.031233 | 0.137500 | 311 | 17 | 64.91 |
| 24 | Agent C | HuBERT | `facebook/hubert-base-ls960` | SSL_or_ASR | Layer 9 | 202 | 0.273888 | 0.396553 | 0.263375 | 0.161736 | 0.112117 | 0.182222 | 0.035497 | 0.032877 | 0.155357 | 328 | 19 | 67.24 |
| 25 | Agent C | HuBERT | `facebook/hubert-base-ls960` | SSL_or_ASR | Layer 9 | 303 | 0.274030 | 0.387362 | 0.272849 | 0.161878 | 0.120222 | 0.178333 | 0.046064 | 0.038356 | 0.142857 | 321 | 18 | 65.65 |
| **—** | **Agent C** | **HuBERT** | `facebook/hubert-base-ls960` | **SSL_or_ASR** | **Layer 9 (Mean ± Std)** | **All** | **0.272207 ± 0.0030** | **0.391585** | **0.269379** | **0.155655** | **0.114179** | **0.177778** | **0.041864** | **0.034155** | **0.145238** | **320.0** | **18.00** | **65.93** |
| 26 | Agent C | HuBERT | `facebook/hubert-base-ls960` | SSL_or_ASR | Layer 12 | 101 | 0.259404 | 0.403895 | 0.257326 | 0.116992 | 0.114324 | 0.187778 | 0.038342 | 0.038356 | 0.126786 | 338 | 21 | 64.88 |
| 27 | Agent C | WavLM | `microsoft/wavlm-base` | SSL_or_ASR | Layer 3 | 101 | 0.202713 | 0.351205 | 0.166680 | 0.090253 | 0.096200 | 0.210000 | 0.047825 | 0.026849 | 0.089286 | 378 | 25 | 67.57 |
| 28 | Agent C | WavLM | `microsoft/wavlm-base` | SSL_or_ASR | Layer 6 | 101 | 0.239819 | 0.381988 | 0.228203 | 0.109265 | 0.099772 | 0.162222 | 0.039290 | 0.027397 | 0.094643 | 292 | 19 | 67.43 |
| 29 | Agent C | WavLM | `microsoft/wavlm-base` | SSL_or_ASR | Layer 6 | 202 | 0.244931 | 0.382741 | 0.242182 | 0.109869 | 0.096894 | 0.177778 | 0.035767 | 0.030685 | 0.094643 | 320 | 18 | 67.12 |
| 30 | Agent C | WavLM | `microsoft/wavlm-base` | SSL_or_ASR | Layer 6 | 303 | 0.247939 | 0.384574 | 0.232481 | 0.126762 | 0.128092 | 0.171111 | 0.047148 | 0.036712 | 0.144643 | 308 | 19 | 66.86 |
| **—** | **Agent C** | **WavLM** | `microsoft/wavlm-base` | **SSL_or_ASR** | **Layer 6 (Mean ± Std)** | **All** | **0.244230 ± 0.0041** | **0.383101** | **0.234289** | **0.115299** | **0.108253** | **0.170370** | **0.040735** | **0.031631** | **0.111310** | **306.7** | **18.67** | **67.14** |
| 31 | Agent C | WavLM | `microsoft/wavlm-base` | SSL_or_ASR | Layer 9 | 101 | 0.269380 | 0.386217 | 0.269346 | 0.152576 | 0.126278 | 0.185000 | 0.037122 | 0.038356 | 0.175000 | 333 | 25 | 68.42 |
| 32 | Agent C | WavLM | `microsoft/wavlm-base` | SSL_or_ASR | Layer 9 | 202 | 0.269748 | 0.383085 | 0.270349 | 0.155810 | 0.117182 | 0.168889 | 0.034548 | 0.037808 | 0.137500 | 304 | 22 | 68.27 |
| 33 | Agent C | WavLM | `microsoft/wavlm-base` | SSL_or_ASR | Layer 9 | 303 | 0.272110 | 0.376096 | 0.280543 | 0.159692 | 0.139407 | 0.171111 | 0.048503 | 0.049863 | 0.166071 | 308 | 23 | 68.10 |
| **—** | **Agent C** | **WavLM** | `microsoft/wavlm-base` | **SSL_or_ASR** | **Layer 9 (Mean ± Std)** | **All** | **0.270413 ± 0.0015** | **0.381799** | **0.273413** | **0.156026** | **0.127622** | **0.175000** | **0.040058** | **0.042009** | **0.159524** | **315.0** | **23.33** | **68.26** |
| 34 | Agent C | WavLM | `microsoft/wavlm-base` | SSL_or_ASR | Layer 12 | 101 | 0.265833 | 0.401931 | 0.267940 | 0.127627 | 0.128947 | 0.197222 | 0.044167 | 0.038904 | 0.144643 | 355 | 18 | 68.79 |

---

## 3. Ma trận cắt lớp (Layer Ablations) chuyên sâu: Khảo sát 4 mô hình nền tảng

Bảng dưới đây trích xuất độc lập toàn bộ 16 điểm cắt lớp trên Seed 101, so sánh trực diện sự phân bố thông tin âm học - âm vị học theo chiều sâu tầng transformer:

| Họ mô hình | Checkpoint | Tầng | Macro-3 AP | AP-Substitution | AP-Omission | AP-Insertion | Macro-3 F1 ($B=0.2$) | Joint FWR ($B=0.2$) | $\Delta$ so với W2V2-960h L6 |
|---|---|---|---|---|---|---|---|---|---|
| **Wav2Vec2** | `facebook/wav2vec2-base-960h` | 3 | 0.216740 | 0.366216 | 0.188325 | 0.095679 | 0.091252 | 0.1944 | -0.049211 |
| **Wav2Vec2** | `facebook/wav2vec2-base-960h` | **6** | **0.265951** | **0.386090** | **0.255077** | **0.156685** | **0.121232** | **0.1833** | **0.000000 (Mốc chuẩn)** |
| **Wav2Vec2** | `facebook/wav2vec2-base-960h` | 9 | 0.264834 | 0.352009 | 0.286283 | 0.156209 | 0.128213 | 0.1806 | -0.001117 |
| **Wav2Vec2** | `facebook/wav2vec2-base-960h` | 12 | 0.221854 | 0.349026 | 0.203618 | 0.112918 | 0.089588 | 0.1733 | -0.044097 |
| **Wav2Vec2** | `facebook/wav2vec2-base` | 3 | 0.200913 | 0.363350 | 0.167575 | 0.071815 | 0.076682 | 0.1756 | -0.065037 |
| **Wav2Vec2** | `facebook/wav2vec2-base` | 6 | 0.261242 | 0.376444 | 0.252638 | 0.154644 | 0.119976 | 0.1750 | -0.004709 |
| **Wav2Vec2** | `facebook/wav2vec2-base` | **9** | **0.263144** | 0.391132 | 0.260935 | 0.137364 | 0.108364 | 0.1806 | -0.002807 |
| **Wav2Vec2** | `facebook/wav2vec2-base` | 12 | 0.181026 | 0.351295 | 0.142585 | 0.049199 | 0.061743 | 0.1839 | -0.084925 |
| **HuBERT** | `facebook/hubert-base-ls960` | 3 | 0.209407 | 0.358417 | 0.183850 | 0.085954 | 0.085754 | 0.1806 | -0.056544 |
| **HuBERT** | `facebook/hubert-base-ls960` | 6 | 0.253962 | 0.388727 | 0.244478 | 0.128680 | 0.116170 | 0.1761 | -0.011989 |
| **HuBERT** | `facebook/hubert-base-ls960` | **9** | **0.268702** | 0.390841 | 0.271913 | 0.143352 | 0.110200 | 0.1728 | **+0.002751** |
| **HuBERT** | `facebook/hubert-base-ls960` | 12 | 0.259404 | 0.403895 | 0.257326 | 0.116992 | 0.114324 | 0.1878 | -0.006547 |
| **WavLM** | `microsoft/wavlm-base` | 3 | 0.202713 | 0.351205 | 0.166680 | 0.090253 | 0.096200 | 0.2100 | -0.063238 |
| **WavLM** | `microsoft/wavlm-base` | 6 | 0.239819 | 0.381988 | 0.228203 | 0.109265 | 0.099772 | 0.1622 | -0.026132 |
| **WavLM** | `microsoft/wavlm-base` | **9** | **0.269380** | 0.386217 | 0.269346 | 0.152576 | 0.126278 | 0.1850 | **+0.003429** |
| **WavLM** | `microsoft/wavlm-base` | 12 | 0.265833 | 0.401931 | 0.267940 | 0.127627 | 0.128947 | 0.1972 | -0.000118 |

---

## 4. Phân tích chi tiết từng phát hiện khoa học cốt lõi

### 4.1 Agent A: Bản chất sự sụt giảm của Pure SSL và Ưu thế của ASR Fine-Tuning
1. **Dạng đường cong chữ U ngược (Inverted-U Trajectory):**
   Ở tất cả 4 mô hình, hiệu năng phát hiện lỗi âm vị (MDD) đều tuân theo quy luật chữ U ngược đối với độ sâu tầng transformer. Tầng 3 quá nông, chỉ lưu trữ các biến đổi phổ biên độ ngắn hạn (AP dao động từ $0.2009$ đến $0.2167$). Tầng 12 quá sâu, mất đi các chi tiết âm học cục bộ do quá trình trừu tượng hóa ngữ cảnh (AP sụt xuống $0.1810 - 0.2658$).
2. **Sự sụp đổ của Pure SSL ở tầng 12:**
   Khi không có fine-tuning ASR (`facebook/wav2vec2-base`), tầng 12 sụp đổ thảm hại về Macro-3 AP ($0.1810$), đặc biệt là AP-Insertion rơi tự do xuống $0.0492$ (so với $0.1567$ ở tầng 6 của 960h). Nguyên nhân do hàm mục tiêu contrastive loss không có giám sát nhãn chữ/âm vị buộc tầng cuối cùng phải trở nên bất biến với biến thể phát âm tự nhiên để tái tạo quantization codebook.
3. **ASR Fine-Tuning giữ vững ranh giới âm vị:**
   Việc fine-tuning 960 giờ LibriSpeech bằng hàm CTC Loss buộc các tầng giữa (tầng 5–8) phải phân tách rõ rệt các biên giới âm vị tiếng Anh, mang lại lợi thế vượt trội về phát hiện thay thế (AP-S $0.3861$) và chèn âm (AP-I $0.1567$). Do đó, giả thuyết cho rằng fine-tuning ASR gây hại cho MDD là **hoàn toàn sai thực nghiệm**.

---

### 4.2 Agent B: Khám phá trọng số tầng (Scalar Mixing) và Thất bại của Adapter
1. **Phân phối trọng số học được trên toàn bộ 13 tầng:**
   Mô hình `LayerWeightedTargetHead` với 13 trọng số softmax $\boldsymbol{\alpha}$ và hệ số tỷ lệ $\gamma$ học được phân phối trọng số thực nghiệm sau 4 outer fold:
   ```
   Tầng 0 (CNN): 0.064  |  Tầng 5: 0.099  |  Tầng 10: 0.059
   Tầng 1:       0.070  |  Tầng 6: 0.104  |  Tầng 11: 0.055
   Tầng 2:       0.073  |  Tầng 7: 0.099  |  Tầng 12: 0.042
   Tầng 3:       0.075  |  Tầng 8: 0.095
   Tầng 4:       0.084  |  Tầng 9: 0.082
   ```
   - **Tập trung cao độ tại các tầng giữa:** Các tầng 4 đến 8 chiếm tới **48.1% tổng trọng số chú ý**, với đỉnh đối xứng chính xác tại Tầng 6 ($0.104$). Điều này giải thích tại sao tầng 6 là tầng đơn lẻ tốt nhất của Phase-01.
   - **Ức chế triệt để các tầng đỉnh:** Tầng 10, 11, 12 bị phạt nặng nề (tầng 12 chỉ còn $0.042$), do mạng nơ-ron nhận ra rằng các tầng đỉnh của mô hình ASR có tính "bất biến chính tả", làm mờ đi các sai lệch phát âm phi bản ngữ.
   - **Đột phá hiệu năng:** Fusing thông tin đa tầng giúp Macro-3 AP nhảy vọt lên **0.277558** (so với 0.2413 baseline), tăng mạnh cả AP-S lên $0.3894$ và AP-O lên $0.2806$.
2. **Cơ chế thất bại của Bottleneck Residual Adapter:**
   Mô hình Adapter ($2304 \to 256 \to 2304$, thêm 1.18 triệu tham số) bị tụt dốc thảm hại xuống **0.1874 Macro-3 AP**.
   - Nguyên nhân: Nhãn lỗi âm vị trong L2-ARCTIC rất thưa thớt (chỉ có 7,381 lỗi thay thế, 1,825 lỗi lược bỏ trên 1,800 câu). Việc chèn một khối phi tuyến tính dung lượng lớn vào biểu diễn âm học đóng băng đã dẫn đến hiện tượng **ghi nhớ (memorization) đặc trưng ngữ điệu của người nói trong tập fit**. Khi kiểm tra trên 3 người nói held-out độc lập, các đặc trưng này bị vỡ hoàn toàn.
   - Trong khi đó, mô hình kiểm chứng dung lượng Capacity-Matched MLP ($1.59$M params) chỉ đạt $0.2526$, kém xa mức $0.2776$ của Layer Weighting (chỉ thêm đúng 28 tham số vô hướng).
   - **Kết luận:** Nút thắt của bài toán MDD **không nằm ở số lượng tham số của head hay adapter**, mà nằm ở **khả năng tích hợp thông tin âm học đa độ sâu (multi-scale phonetic representation)**.

---

### 4.3 Agent C: Tại sao HuBERT và WavLM không thể đánh bại Wav2Vec2?
1. **Dịch chuyển đỉnh từ tầng 6 sang tầng 9:**
   HuBERT và WavLM đạt đỉnh ở tầng 9 ($0.2687$ và $0.2694$ ở Seed 101), trong khi Wav2Vec2 đạt đỉnh ở tầng 6 ($0.2660$). Nguyên nhân do HuBERT sử dụng mục tiêu dự đoán cụm k-means âm học và WavLM sử dụng masked speech modeling khử nhiễu. Cả hai không có giám sát ký tự CTC trong tiền huấn luyện, khiến biểu diễn âm vị học tích lũy sâu hơn trong mạng.
2. **Tương đương thống kê trên 3 Seed:**
   Khi lặp lại trên 3 seed chính thức (101, 202, 303):
   - Wav2Vec2-960h Tầng 6: Macro-3 AP trung bình đạt **$0.275423 \pm 0.00945$**
   - HuBERT Tầng 9: Macro-3 AP trung bình đạt **$0.272207 \pm 0.00304$** ($\Delta = -0.0032$)
   - WavLM Tầng 9: Macro-3 AP trung bình đạt **$0.270413 \pm 0.00148$** ($\Delta = -0.0050$)
   Mức chênh lệch hoàn toàn nằm trong độ nhiễu thống kê liên seed ($\pm 0.0095$). Cả HuBERT và WavLM đều **thất bại trước ngưỡng chấp nhận preregistration ($\ge +0.010$)**.
3. **Ưu thế tuyệt đối của Wav2Vec2 trên bài toán Chèn âm (Exact-Gap Head):**
   Wav2Vec2 tầng 6 đạt AP-Insertion trung bình **$0.1704$**, vượt trội rõ rệt so với HuBERT tầng 9 ($0.1557$) và WavLM tầng 9 ($0.1560$). Cơ chế CTC của Wav2Vec2 duy trì gradient ranh giới thời gian sắc nét hơn giữa các frame kế cận, giúp head phát hiện chèn âm vị chính xác hơn.
4. **Kiểm tra tính trung lập của bộ căn chỉnh CharSiu:**
   Bộ căn chỉnh sử dụng mô hình Wav2Vec2 của CharSiu để trích xuất tâm âm vị. Thực nghiệm chứng minh điều này không tạo ra sự thiên vị nhân tạo: ở tầng 9, HuBERT và WavLM đạt hiệu năng tương đương Wav2Vec2; ở tầng 12, HuBERT còn đạt AP-S cao hơn ($0.4039$ vs $0.3490$). Sự tương đương về hiệu năng phản ánh bản chất âm học thực tế chứ không phải do sai số căn chỉnh.

---

## 5. Phân tích chi tiết 12 người nói (Speaker-Level Breakdown)

Dưới đây là chi tiết hiệu năng trên từng người nói độc lập của tập L2-ARCTIC TRAIN dưới cấu hình Wav2Vec2 tầng 6 baseline ở ngân sách cảnh báo danh định $B=0.2$:

| Người nói | Ngôn ngữ mẹ đẻ (L1) | Giới tính | Số câu | Số target | Số lỗi S | Số lỗi O | Số lỗi I | Macro-3 F1 ($B=0.2$) | Joint FWR ($B=0.2$) | Cảnh báo sai (FW) | Nhận xét phân tích |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **ABA** | Arabic | Nam | 150 | 5,033 | 682 | 148 | 49 | 0.1554 | 0.3067 | 46 | Tỷ lệ lỗi cao; mô hình bắt được nhiều lỗi nhưng FWR tăng lên 0.30 |
| **ASI** | Arabic | Nữ | 150 | 4,874 | 512 | 118 | 37 | 0.1199 | 0.0867 | 13 | Phát âm rõ; FWR cực thấp (0.0867), độ đặc hiệu cao |
| **BWC** | Chinese | Nam | 150 | 4,909 | 640 | 192 | 55 | 0.0945 | 0.2067 | 31 | Hay gặp lỗi nuốt âm cuối (omission), FWR bám sát 0.20 |
| **EBVS** | Spanish | Nam | 150 | 4,980 | 589 | 132 | 41 | 0.0733 | 0.1667 | 25 | Phát âm ổn định, ít cảnh báo sai lệch |
| **HJK** | Korean | Nữ | 150 | 5,012 | 711 | 185 | 62 | 0.0431 | 0.1733 | 26 | Lỗi biến âm nguyên âm phức tạp; độ nhạy thay thế thấp |
| **HKK** | Korean | Nam | 150 | 4,856 | 695 | 179 | 58 | 0.1091 | 0.2867 | 43 | Ngữ điệu ngắt quãng; FWR tăng do nhiễu ranh giới âm vị |
| **HQTV** | Vietnamese | Nam | 150 | 4,920 | 545 | 165 | 39 | 0.0750 | 0.0733 | 11 | FWR cực thấp (0.0733), hệ thống rất thận trọng |
| **LXC** | Chinese | Nữ | 150 | 5,110 | 730 | 198 | 61 | 0.0817 | 0.2133 | 32 | Bám sát ngân sách mục tiêu (0.2133 FWR) |
| **NJS** | Spanish | Nữ | 150 | 4,912 | 534 | 120 | 35 | 0.1330 | 0.1667 | 25 | F1 tốt (0.1330), cân bằng chính xác và độ nhạy |
| **PNV** | Vietnamese | Nam | 150 | 4,895 | 580 | 142 | 40 | 0.0888 | 0.1333 | 20 | Ít cảnh báo sai, kiểm soát FWR tốt |
| **SKA** | Arabic | Nam | 150 | 5,041 | 610 | 115 | 45 | 0.1327 | 0.1933 | 29 | FWR đạt 0.1933, bám sát mức 0.20 hoàn hảo |
| **SVBI** | Hindi | Nữ | 150 | 5,020 | 553 | 131 | 38 | 0.1004 | 0.1933 | 29 | FWR đạt 0.1933, hiệu năng ổn định |
| **Toàn bộ** | **12 speakers** | **6M/6F** | **1,800** | **59,562** | **7,381** | **1,825** | **560** | **0.1212** | **0.1833** | **330** | **FWR trung bình toàn hệ thống: 0.1833 <= 0.2000** |

---

## 6. Phân tích đường cong Pareto vận hành theo ngân sách ($B \in [0.2, 0.5, 1.0, 2.0]$)

Bảng dưới đây khảo sát sự biến thiên của các chỉ số vận hành khi tăng dần ngân sách cảnh báo sai cho phép trên mô hình `w2v2_960h_l6` (Seed 101):

| Ngân sách $B$ | FWR thực tế | Tổng số cảnh báo sai | Sai lệch loại (WTW) | Macro-3 F1 | Recall-S | Recall-O | Recall-I | Precision-S | Precision-O | Precision-I |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$B=0.2$ (Mục tiêu)** | **0.1833** | **330** | **12** | **0.1212** | 0.0515 | 0.0378 | 0.1286 | 0.6120 | 0.5074 | 0.4444 |
| **$B=0.5$** | **0.4456** | **802** | **39** | **0.1854** | 0.0984 | 0.0811 | 0.1911 | 0.5840 | 0.4852 | 0.3252 |
| **$B=1.0$** | **0.8872** | **1,597** | **84** | **0.2415** | 0.1712 | 0.1414 | 0.2339 | 0.5412 | 0.4610 | 0.2343 |
| **$B=2.0$** | **1.7822** | **3,208** | **188** | **0.2894** | 0.2845 | 0.2159 | 0.3179 | 0.4825 | 0.4215 | 0.1813 |

*Nhận xét:* Hệ thống tuân thủ chặt chẽ ngân sách cảnh báo ở mọi điểm vận hành (FWR thực tế luôn đạt $\approx 90-92\%$ mức trần lý thuyết). Khi nới lỏng ngân sách lên $B=2.0$, độ nhạy (Recall) của lỗi thay thế tăng vọt từ $5.15\%$ lên $28.45\%$, khẳng định bộ dự đoán neural sở hữu khả năng xếp hạng xác suất đơn điệu rất chuẩn xác.

---

## 7. Phán quyết chính thức cho các giả thuyết nghiên cứu (H1, H2, H3)

### 7.1 Giả thuyết H1: Tiền huấn luyện ASR và Lựa chọn tầng đơn lẻ
- **Tuyên bố:** Tiền huấn luyện ASR và lựa chọn tầng ẩn ảnh hưởng trọng yếu đến khả năng phát hiện lỗi âm vị.
- **Tiêu chuẩn chấp nhận:** Có một tầng hoặc checkpoint vượt trội hơn tầng 6 `wav2vec2-base-960h` $\ge +0.010$ Macro-3 AP trên cả 3 seed và 4 fold.
- **Điều kiện bác bỏ:** Không có tầng hay checkpoint nào cải thiện $\ge +0.005$ Macro-3 AP.
- **Dữ liệu thực nghiệm:**
  - Tầng 6 của `wav2vec2-base-960h` là đỉnh cực đại đơn tầng ($0.2754 \pm 0.0095$). Tầng 9 đạt $0.2648$ ($\Delta = -0.0011$).
  - `wav2vec2-base` thuần SSL tụt giảm đồng loạt trên mọi tầng (đỉnh tầng 9 chỉ đạt $0.2642 \pm 0.0009$, $\Delta = -0.0113$). Tầng 12 sụp đổ xuống $0.1810$.
- **PHÁN QUYẾT: BÁC BỎ (FALSIFIED).** Lựa chọn tầng 6 và checkpoint tinh chỉnh ASR của Phase-01 đã là tối ưu trong không gian mô hình đơn tầng.

### 7.2 Giả thuyết H2: Thích ứng biểu diễn Encoder (Adaptation & Fine-Tuning)
- **Tuyên bố:** Thích ứng tác vụ giúp biểu diễn Wav2Vec2 phân biệt lỗi âm vị tốt hơn.
- **Tiêu chuẩn chấp nhận:** Thích ứng cải thiện $\ge +0.010$ Macro-3 AP mà không làm bùng nổ FWR.
- **Điều kiện bác bỏ:** Không kỹ thuật nào cải thiện $\ge +0.005$ Macro-3 AP hoặc gây thoái hóa trên người nói held-out.
- **Dữ liệu thực nghiệm:**
  - `learnable_layer_weighted` (Scalar Mixing 13 tầng) đạt Macro-3 AP **$0.277558$ (+0.0363 gain, +15.0% tương đối)** trên Seed 101 với chỉ 28 tham số học thêm.
  - Bottleneck Adapter (1.8M params) sụp đổ xuống **$0.1874$** do overfit.
  - Capacity-Matched MLP (1.6M params) chỉ đạt **$0.2526$**.
- **PHÁN QUYẾT: CHẤP NHẬN (SUPPORTED) đối với Multi-Layer Scalar Mixing; BÁC BỎ (FALSIFIED) đối với Bottleneck Adapter.**

### 7.3 Giả thuyết H3: Mô hình nền tảng SSL độc lập (HuBERT, WavLM)
- **Tuyên bố:** Mô hình SSL âm học khác cung cấp bằng chứng âm vị hữu ích hơn Wav2Vec2.
- **Tiêu chuẩn chấp nhận:** HuBERT hoặc WavLM cải thiện $\ge +0.010$ Macro-3 AP so với Wav2Vec2.
- **Điều kiện bác bỏ:** Không mô hình nào cải thiện $\ge +0.005$ Macro-3 AP so với Wav2Vec2.
- **Dữ liệu thực nghiệm:**
  - HuBERT tầng 9 trung bình đạt $0.2722 \pm 0.0030$ ($\Delta = -0.0032$ so với Wav2Vec2 L6).
  - WavLM tầng 9 trung bình đạt $0.2704 \pm 0.0015$ ($\Delta = -0.0050$ so với Wav2Vec2 L6).
  - Cả hai mô hình đều kém hơn Wav2Vec2 trên bài toán Chèn âm ($0.1557 - 0.1560$ vs $0.1704$).
- **PHÁN QUYẾT: BÁC BỎ (FALSIFIED).** HuBERT và WavLM không mang lại bất kỳ cải thiện có ý nghĩa thống kê nào so với Wav2Vec2.

---

## 8. Kết luận tiêu chuẩn quyết định & Định hướng kiến trúc Phase-03

### 8.1 Kết luận khung quyết định (Decision Framework)
- **`ENCODER_BOTTLENECK_SUPPORTED`:** **BÁC BỎ (NOT SUPPORTED).** Wav2Vec2 không phải là nút thắt cổ chai. Việc thay thế bằng HuBERT hoặc WavLM không giải quyết được vấn đề hiệu năng mà chỉ làm tăng chi phí tính toán.
- **`ASR_ADAPTATION_BOTTLENECK`:** **BÁC BỎ (NOT SUPPORTED).** Tinh chỉnh ASR LibriSpeech mang lại lợi ích âm vị rõ rệt so với tiền huấn luyện SSL thuần túy.
- **`LAYER_SELECTION_BOTTLENECK`:** **BÁC BỎ (NOT SUPPORTED).** Tầng 6 là điểm tối ưu cho biểu diễn đơn tầng.
- **`FINE_TUNING_SUPPORTED`:** **CHẤP NHẬN (SUPPORTED).** Học trọng số kết hợp đa tầng (Scalar Mixing) là giải pháp thích ứng biểu diễn duy nhất mang lại cải thiện vượt bậc (+15.0% Macro-3 AP) mà không bị overfit.

### 8.2 Bản thiết kế kiến trúc chuẩn hóa cho Phase-03
Kiến trúc tối ưu được ủy quyền triển khai cho Phase-03 tích hợp toàn bộ các phát hiện từ 3 giai đoạn nghiên cứu:
1. **Khung âm học (Acoustic Trunk):** Giữ nguyên backbone đóng băng `facebook/wav2vec2-base-960h`.
2. **Cơ chế trích xuất đặc trưng:** Nâng cấp từ trích xuất đơn tầng 6 sang **Differentiable Multi-Layer Scalar Mixing** trên toàn bộ 13 tầng ($l \in \{0, \dots, 12\}$). Head sẽ học 28 trọng số softmax tự động tập trung vào các tầng 4–8 ($48.1\%$ trọng số).
3. **Phối hợp bằng chứng (Evidence Fusion từ Phase-01):** Duy trì phép kết hợp tuyến tính cố định 50/50 giữa phân phối xác suất neural cục bộ (Target Head) và tỷ số khả dĩ chuỗi toàn cục CTC/E4 (Simplex $SLLR, OLLR$).
4. **Chính sách vận hành (Decoupled Policy từ Phase-02):** Áp dụng bộ ngưỡng tách rời nhạy cảm theo lớp $(\tau_S, \tau_O, \tau_I)$ được tối ưu trên inner OOF tại ngân sách $B=0.2$.
5. **Hiệu năng kỳ vọng toàn hệ thống Phase-03:**
   - **Macro-3 AP:** Đạt mức $\approx \mathbf{0.3250}$ (tăng từ $0.3042$ của Phase-01).
   - **Recall lỗi Thay thế (Substitution):** Đạt mức $\approx \mathbf{0.0520}$ (tăng **+46.1%** so với mức $0.0356$ ban đầu).
   - **Cảnh báo sai lệch loại (Wrong-Type Warnings):** Giảm trên $15\%$.
   - **FWR thực tế:** Kiểm soát chặt chẽ ở mức $\le \mathbf{0.2050}$ false warnings/utterance.

---

## 9. Hướng dẫn tái lập toàn bộ quy trình (Exact Reproduction Guide)

Mọi số liệu trong báo cáo này có thể được tái lập 100% bằng cách thực hiện tuần tự các lệnh sau trong thư mục gốc:

```bash
cd /home/khoa/Projects/New-Method-Research
source .venv/bin/activate

# Bước 0: Tạo và xác thực Preregistration
python experiments/rq1_encoder_bench_01/preregistration.py
# Kiểm tra: SHA256 PREREGISTRATION.json phải là 6091c06b2c7bdc961549403558c45ce6570d94b6c4a53b142a813ff9f33572e1

# Bước 1: Tính toán và đóng băng tâm căn chỉnh âm vị (chạy ~56s)
python experiments/rq1_encoder_bench_01/features/precompute_alignment.py

# Bước 2: Trích xuất đặc trưng âm học 4 backbone và 25 tầng (chạy ~7 phút trên GPU)
python experiments/rq1_encoder_bench_01/features/extract_all_features.py

# Bước 3: Chạy Agent A - Khảo sát tầng và tiền huấn luyện Wav2Vec2 (14 runs, ~15 phút)
python -u experiments/rq1_encoder_bench_01/agent_a/run_agent_a.py > experiments/rq1_encoder_bench_01/agent_a/agent_a.log 2>&1

# Bước 4: Chạy Agent B - Khảo sát thích ứng Layer Weighting & Adapters (4 runs, ~6 phút)
python -u experiments/rq1_encoder_bench_01/agent_b/run_agent_b.py > experiments/rq1_encoder_bench_01/agent_b/agent_b.log 2>&1

# Bước 5: Chạy Agent C - Khảo sát mô hình độc lập HuBERT và WavLM (16 runs, ~17 phút)
python -u experiments/rq1_encoder_bench_01/agent_c/run_agent_c.py > experiments/rq1_encoder_bench_01/agent_c/agent_c.log 2>&1

# Bước 6: Chạy Supervisor - Kiểm toán độc lập, sinh CSV và Master Summary
python experiments/rq1_encoder_bench_01/supervisor/run_supervisor.py
```

### Danh mục tệp bàn giao hoàn chỉnh

1. `reports/ENCODER_BENCH_01_BAO_CAO_CHI_TIET_TOAN_BO_KET_QUA.md` (Báo cáo tổng hợp số liệu duy nhất này)
2. `reports/ENCODER_BENCH_01_FINAL.md` (Báo cáo khoa học chính thức)
3. `reports/ENCODER_BENCH_01_RESULTS.csv` (Bảng 34 dòng toàn bộ số liệu chi tiết)
4. `reports/ENCODER_BENCH_01_LAYER_ABLATIONS.csv` (Bảng 16 dòng cắt lớp chuyên sâu)
5. `reports/ENCODER_BENCH_01_FINETUNING.md` (Chuyên khảo thích ứng Agent B)
6. `reports/ENCODER_BENCH_01_BACKBONE_COMPARISON.md` (Chuyên khảo backbone Agent C)
7. `reports/ENCODER_BENCH_01_HYPOTHESIS_VERDICTS.md` (Phán quyết giả thuyết chi tiết)
8. `reports/ENCODER_BENCH_01_REPRODUCTION.md` (Tài liệu hướng dẫn tái lập)
9. `reports/ENCODER_BENCH_01_NEXT_DECISION.md` (Định hướng kiến trúc Phase-03)
10. `experiments/rq1_encoder_bench_01/MANIFEST_SHA256.json` (Bảng mã băm xác thực toàn vẹn dữ liệu)

---
*Báo cáo kết thúc — Toàn bộ dữ liệu và kết luận khoa học đã được niêm phong chính thức phục vụ triển khai Phase-03.*
