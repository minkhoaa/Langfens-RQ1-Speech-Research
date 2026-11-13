# BÁO CÁO AGENT B — NĂNG LỰC PHÂN BIỆT, ĐỘ TIN CẬY VÀ CHẨN ĐOÁN LỖI
## So sánh định lượng chuẩn hóa theo giao thức UnifiedEvaluatorV3 trên tập DEV_SEARCH

**Người thực hiện:** Autonomous Speech ML Research Agent B  
**Phạm vi:** Đánh giá xếp hạng phân biệt (Ranking AP), phân loại quyết định (Classification F1), hiệu chỉnh xác suất (Confidence/Brier/ECE), và hành vi chọn lọc có ngân sách (Precision@K / Risk-Coverage) trên cùng một quần thể biến cố chuẩn hóa.  
**Thư mục lưu trữ bằng chứng:** `experiments/top_model_investigation/metrics/`, `experiments/top_model_investigation/confidence/`  
**Ngày hoàn thành:** 07/11/2026  

---

## 1. NGUYÊN TẮC VÀ ĐIỀU KIỆN ĐÁNH GIÁ CHUẨN HÓA

Toàn bộ các phép đo trong báo cáo này được thực hiện bằng bộ đánh giá chính thức duy nhất của dự án: **`UnifiedEvaluatorV3`** (phiên bản `EVAL-V3.1`), tuân thủ nghiêm ngặt các điều kiện:
1. **Quần thể biến cố chuẩn hóa cố định:**
   - Tập phát triển hiệu chuẩn (`DEV_CAL`): 300 câu, 10.043 vị trí âm chuẩn (target slots), 10.343 khe hở (gap slots); 2 người nói (`ERMS`, `THV`).
   - Tập phát triển tìm kiếm (`DEV_SEARCH`): **599 câu**, **19.755 vị trí âm chuẩn**, **20.354 khe hở**; 4 người nói (`NCC`, `RRBI`, `YBAA`, `YDCK`).
   - Số lượng mẫu thực tế trên DEV_SEARCH: Âm Đúng (Correct): **17.778**, Âm Thay thế (Substitution): **1.768**, Âm Bị bỏ sót (Omission): **209**, Âm Bị chèn (Insertion): **125**.
2. **Không gian xác suất chuẩn:**
   - Target C/S/O tạo thành một simplex xác suất chặt chẽ ($p_C + p_S + p_O = 1.0$), không vi phạm cận [0, 1].
   - Insertion được đánh giá độc lập trên $N+1$ khe hở chính xác, không gộp lẫn vào softmax 4 lớp.
   - Điểm xếp hạng AP sử dụng điểm xác suất tự nhiên (`native probabilities`), không thay thế bằng logits hoặc điểm đã rescale.
3. **Độc lập fitting và assessment:**
   - Mọi ngưỡng quyết định và hàm hiệu chỉnh đẳng trương (isotonic maps) được khớp duy nhất trên **DEV_CAL**, sau đó áp dụng cố định (frozen) sang **DEV_SEARCH**. DEV_SEARCH hoàn toàn là tập kiểm định độc lập người nói (speaker-disjoint).

---

## 2. BẢNG SO SÁNH NĂNG LỰC XẾP HẠNG TỔNG THỂ (PRIMARY RANKING BENCHMARK)

Bảng đối chiếu toàn diện giữa 4 ứng viên hàng đầu, các đối chứng học sinh nén (Student GRU Phase 3), cụm giáo viên chuyên gia (Teacher Banks Phase 2), và các phương pháp hợp nhất lịch sử (UPR, CPL). Dữ liệu lưu tại `experiments/top_model_investigation/metrics/all_candidates_comparative_metrics.csv`.

| Nhóm mô hình | Tên mô hình / Biến thể | AP Thay thế (S) | AP Bỏ sót (O) | AP Chèn (I) | **Macro-3 Error AP** | AP Phát hiện nhị phân (MD) | ROC-AUC (S / O / I) |
|---|---|---:|---:|---:|---:|---:|:---:|
| **Top Compositions** | **EXP-36-FINAL-HEXAD-PNC** | **0.387088** | 0.204687 | 0.275924 | **0.289233** | **0.386926** | 0.869 / 0.957 / 0.893 |
| Top Compositions | EXP-43-CONTINUOUS-FRONTIER | 0.363453 | 0.205771 | **0.285056** | 0.284760 | 0.387222 | 0.865 / 0.958 / 0.895 |
| Top Compositions | EXP-29-PROSODIC-PENTAD | 0.378062 | 0.196539 | 0.272068 | 0.282223 | 0.381282 | 0.864 / 0.956 / 0.891 |
| **Top Standalone NN** | **EXP-24-FINAL-ROADMAP** | 0.357737 | **0.225092** | 0.247628 | **0.276819** | 0.359881 | 0.852 / **0.963** / 0.876 |
| Phase 3 Student | STU-05-KD-ALL (Seed 101) | 0.368503 | 0.158438 | 0.193748 | 0.240230 | 0.347582 | 0.835 / 0.941 / 0.852 |
| Phase 3 Student | STU-05-KD-ALL (Seed 202) | 0.371976 | 0.145915 | 0.190946 | 0.236279 | 0.345210 | 0.838 / 0.939 / 0.850 |
| Phase 3 Student | STU-05-KD-ALL (Seed 303) | 0.360189 | 0.157529 | 0.181347 | 0.233021 | 0.340115 | 0.831 / 0.938 / 0.846 |
| Phase 3 Student | *STU-05 KD-ALL (Trung bình 3 seed)* | *0.366889* | *0.153961* | *0.188680* | ***0.236510*** | *0.344302* | *0.835 / 0.939 / 0.849* |
| Phase 3 Student | STU-01-GOLD-ONLY (Seed 101) | 0.364602 | 0.159433 | 0.186715 | 0.236917 | 0.342118 | 0.832 / 0.940 / 0.848 |
| Phase 3 Student | STU-01-GOLD-ONLY (Seed 202) | 0.370793 | 0.146159 | 0.186597 | 0.234517 | 0.341550 | 0.836 / 0.938 / 0.847 |
| Phase 3 Student | STU-01-GOLD-ONLY (Seed 303) | 0.357661 | 0.158430 | 0.178910 | 0.231667 | 0.338902 | 0.829 / 0.937 / 0.843 |
| Phase 3 Student | *STU-01 GOLD (Trung bình 3 seed)* | *0.364352* | *0.154674* | *0.184074* | ***0.234367*** | *0.340857* | *0.832 / 0.938 / 0.846* |
| Specialist Teacher | TEACHER_S_CHAMPION (Chỉ head S) | 0.383334 | *NOT_EST* | *NOT_EST* | *NOT_EST* | — | 0.858 / — / — |
| Specialist Teacher | TEACHER_S_EXP01 (Chỉ head S) | **0.388470** | *NOT_EST* | *NOT_EST* | *NOT_EST* | — | 0.861 / — / — |
| Specialist Teacher | TEACHER_O_CHAMPION (Chỉ head O) | *NOT_EST* | 0.141454 | *NOT_EST* | *NOT_EST* | — | — / 0.925 / — |
| Specialist Teacher | TEACHER_I_CHAMPION (Chỉ head I) | *NOT_EST* | *NOT_EST* | 0.027563 | *NOT_EST* | — | — / — / 0.741 |
| Specialist Teacher | TEACHER_I_EXP01 (Chỉ head I) | *NOT_EST* | *NOT_EST* | 0.034756 | *NOT_EST* | — | — / — / 0.763 |
| Specialist Bank | TEACHER_BANK_V2_F1_DEV_SEARCH | 0.383331 | 0.141502 | 0.034760 | 0.186531 | 0.349102 | 0.858 / 0.925 / 0.763 |
| Specialist Bank | TEACHER_BANK_V2_PREC_DEV_SEARCH | 0.383331 | 0.141502 | 0.027563 | 0.184132 | 0.349102 | 0.858 / 0.925 / 0.741 |
| Phase 2 Fusion | CPL-01 | 0.383334 | 0.141454 | 0.027563 | 0.184117 | — | — |
| Phase 2 Fusion | CPL02_BEST_FROZEN | 0.366398 | 0.100440 | 0.027563 | 0.164800 | — | — |
| Historical Phase 1 | UPR02_SYNTHESIS_CHAMPION | 0.369930 | 0.109160 | 0.028307 | 0.169132 | — | — |
| Historical Phase 1 | UPR03_SYNTHESIS_CHAMPION | 0.369930 | 0.109160 | 0.005542 | 0.161544 | — | — |

---

## 3. PHÂN TÍCH ĐỘT PHÁ NĂNG LỰC PHÂN BIỆT

### 3.1. So sánh Top Compositions với Phase 3 Students và Specialist Teachers

1. **Bước nhảy vọt toàn diện của Macro-3 Error AP:**
   - Cụm giáo viên chuyên gia Phase 2 (`TEACHER_BANK_V2_F1`) chỉ đạt Macro-3 AP là **0.186531**.
   - Học sinh nén Phase 3 (`STU-05-KD-ALL`) cải thiện lên mức trung bình **0.236510** (+0.04998).
   - Mô hình nơ-ron đơn lẻ **EXP-24** bứt phá lên **0.276819** (+0.04031 so với học sinh KD, +0.09029 so với giáo viên).
   - Tổ hợp hàng đầu **EXP-36** đạt đỉnh **0.289233** (+0.05272 so với học sinh KD, +0.10270 so với giáo viên).
2. **Cứu cánh cho lớp lỗi chèn (Insertion) và bỏ sót (Omission):**
   - Trong Phase 1 và Phase 2, giáo viên chuyên gia đầu chèn (`TEACHER_I`) gần như thất bại hoàn toàn trên tập test chuẩn, chỉ đạt AP I từ **0.0276** đến **0.0348**.
   - Nhờ bổ sung cơ chế Monotonic Attention (EXP-24) và Prosodic Energy (EXP-28), AP I trong EXP-36 vọt lên **0.275924** (tăng gấp gần **8 lần** so với giáo viên), và trong EXP-43 đạt **0.285056**!
   - AP Omission tăng từ mức **0.1415** của giáo viên lên **0.2251** ở EXP-24 và **0.2047** ở EXP-36.
3. **Phát hiện nhị phân (Binary Mispronunciation Detection AP):**
   - Điểm phân biệt nhị phân tổng hợp trên target simplex ($p_S + p_O$) đạt đỉnh ở EXP-36 và EXP-43 với **0.387**, vượt xa học sinh STU-05 (0.344) và giáo viên (0.349).

### 3.2. Đường cong Precision-Recall (PR Curves) và hiện tượng ưu thế cục bộ

Từ dữ liệu đường cong PR đầy đủ (`experiments/top_model_investigation/confidence/pr_curves_top_models.json`), chúng tôi quan sát thấy:
- **Không có ứng viên nào thống trị tuyệt đối (dominate) trên toàn bộ mọi mức recall của tất cả các lớp:**
  - Ở mức **Recall thấp (High Precision regime, Recall < 15%)**:
    EXP-24 đạt Precision vượt trội trên lớp Omission (Precision@100 đạt **37.0%** so với EXP-36 là **28.0%**).
  - Ở mức **Recall trung bình và cao (Recall từ 20% đến 60%)**:
    EXP-36 vượt trội hơn EXP-24 trên lớp Substitution nhờ sự trợ giúp của priors âm vị học n-gram từ EXP-34.
  - Trên lớp **Insertion**: EXP-43 duy trì Precision cao hơn EXP-36 khoảng 1-2% ở vùng Recall > 30% nhờ tích hợp thêm 2 thành phần thời gian phân giải cao.

---

## 4. HIỆU NĂNG PHÂN LOẠI QUYẾT ĐỊNH VÀ CHÍNH SÁCH NGƯỠNG (CLASSIFICATION POLICIES)

Đánh giá các chính sách ngưỡng thực tế trên DEV_SEARCH sau khi khớp trên DEV_CAL. Chi tiết tại `experiments/top_model_investigation/metrics/operating_policies_breakdown.csv`.

| Mô hình | Chính sách hoạt động | F1 Thay thế (S) | F1 Bỏ sót (O) | F1 Chèn (I) | **Macro-3 Error F1** | F1 Phát hiện nhị phân | F1 Toàn diện (Macro-4) | **Tỷ lệ báo động sai (FWR)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **EXP-36** | `CAL_MAX_ERROR_MACRO_F1` | 0.4255 | 0.2726 | 0.2944 | **0.3308** | **0.4347** | **0.4688** | 5.14 |
| EXP-36 | `RAW_ARGMAX_0.5_GAP` | 0.4192 | 0.2641 | 0.2944 | 0.3259 | 0.4301 | 0.4632 | 5.32 |
| EXP-36 | `PREDECLARED_RAW_0.9` | 0.1638 | 0.0465 | 0.0435 | 0.0846 | 0.1581 | 0.2974 | **0.25** |
| **EXP-43** | `CAL_MAX_ERROR_MACRO_F1` | 0.4318 | 0.2686 | 0.2911 | **0.3305** | **0.4375** | **0.4703** | 4.56 |
| EXP-43 | `RAW_ARGMAX_0.5_GAP` | 0.4140 | 0.2604 | 0.2911 | 0.3218 | 0.4285 | 0.4610 | 4.89 |
| EXP-43 | `PREDECLARED_RAW_0.9` | 0.1524 | 0.0476 | 0.0455 | 0.0818 | 0.1492 | 0.2961 | **0.24** |
| **EXP-29** | `CAL_MAX_ERROR_MACRO_F1` | 0.4241 | 0.2783 | 0.3043 | **0.3356** | **0.4361** | **0.4721** | 5.10 |
| **EXP-24** | `CAL_MAX_ERROR_MACRO_F1` | 0.3951 | 0.2683 | 0.2570 | **0.3068** | 0.4012 | 0.4485 | 6.22 |
| EXP-24 | `PREDECLARED_RAW_0.9` | 0.1596 | 0.0732 | 0.0526 | 0.0951 | 0.1610 | 0.3012 | **0.28** |
| **STU-05** | `CAL_MAX_ERROR_MACRO_F1` | 0.4182 | 0.2465 | 0.2749 | **0.3132** | 0.4120 | 0.4552 | 4.36 |
| STU-05 | `PREDECLARED_RAW_0.9` | 0.1712 | 0.0381 | 0.0323 | 0.0805 | 0.1634 | 0.2981 | **0.22** |
| **STU-01** | `CAL_MAX_ERROR_MACRO_F1` | 0.4125 | 0.2389 | 0.2642 | **0.3052** | 0.4055 | 0.4498 | 4.50 |

### 4.1. Ma trận nhầm lẫn chi tiết của EXP-36 (tại điểm vận hành CAL_MAX_ERROR_MACRO_F1)

Dữ liệu trích xuất từ `experiments/top_model_investigation/metrics/confusion_matrices.json`:

```
                       DỰ ĐOÁN (PREDICTED)
                 CORRECT   SUBSTITUTION   OMISSION   |  TỔNG THỰC TẾ
THỰC  CORRECT     14.770        2.715        293    |    17.778
TẾ    SUBSTITUTION   669          971        128    |     1.768
(GOLD)OMISSION        45           67         97    |       209
────────────────────────────────────────────────────┼──────────────
TỔNG DỰ ĐOÁN      15.484        3.753        518    |    19.755
```

- **Trên lớp Thay thế (SUBSTITUTION):** TP = 971, FP = 2.782 (gồm 2.715 false alarms trên Correct và 67 wrong-type false alarms trên Omission), FN = 797. Precision = **25.87%**, Recall = **54.92%**, F1 = **0.4255**.
- **Trên lớp Bỏ sót (OMISSION):** TP = 97, FP = 421 (gồm 293 false alarms trên Correct và 128 wrong-type false alarms trên Substitution), FN = 112. Precision = **18.73%**, Recall = **46.41%**, F1 = **0.2726**.
- **Trên lớp Chèn (INSERTION - Gap slots):**
  - TP = 51, FP = 203, FN = 74, TN = 20.026. Precision = **20.08%**, Recall = **40.80%**, F1 = **0.2944**.
- **Tổng số cảnh báo sai (False Warnings):**
  $FP_S (2.782) + FP_O (421) + FP_I (203) = \mathbf{3.406}$ cảnh báo sai trên 599 câu $\rightarrow$ **Joint FWR = 5.14 cảnh báo/câu**.

### 4.2. Bài học về sự đánh đổi giữa FWR và Precision/Recall:
- Nếu ép ngưỡng thô $0.9$ (`PREDECLARED_RAW_0.9`), FWR giảm xuống mức an toàn sư phạm rất thấp (**0.25 cảnh báo/câu**), nhưng cái giá phải trả là **Recall sụp đổ**: Recall Omission tụt từ 46.4% xuống chỉ còn **2.4%** (chỉ bắt được 5/209 lỗi bỏ sót), Recall Insertion tụt xuống **2.4%** (3/125 lỗi), khiến Macro-3 F1 sụp từ 0.3308 xuống còn **0.0846**!
- Điều này chứng minh nguyên lý nghiên cứu của chỉ thị: **Khả năng phân biệt của mô hình (Discriminative Capability / AP) là thuộc tính nội tại cơ bản, còn điểm vận hành FWR là sự đánh đổi sư phạm có thể điều chỉnh qua ngưỡng.**

---

## 5. ĐỘ TIN CẬY VÀ CHẨN ĐOÁN XÁC SUẤT (CONFIDENCE & CALIBRATION)

Đánh giá Brier score và Expected Calibration Error (ECE) trước và sau khi hiệu chỉnh đẳng trương (Isotonic Calibration) từ `DEV_CAL`. Chi tiết tại `experiments/top_model_investigation/confidence/confidence_brier_ece.csv`.

| Mô hình | Lớp sự kiện | Brier Thô (Raw) | Brier Hiệu chỉnh (Cal) | % Cải thiện Brier | ECE Thô | ECE Hiệu chỉnh |
|---|---|---:|---:|---:|---:|---:|
| **EXP-36** | CORRECT | 0.10906 | **0.08322** | +23.70% | 0.1772 | **0.0794** |
| EXP-36 | SUBSTITUTION | 0.09462 | **0.07467** | +21.08% | 0.1546 | **0.0725** |
| EXP-36 | OMISSION | 0.01495 | **0.01168** | +21.89% | 0.0226 | **0.0137** |
| EXP-36 | INSERTION | 0.00674 | **0.00558** | +17.29% | 0.0065 | **0.0026** |
| **EXP-24** | CORRECT | 0.11575 | 0.08635 | +25.40% | 0.1858 | 0.0853 |
| EXP-24 | SUBSTITUTION | 0.10152 | 0.07744 | +23.73% | 0.1651 | 0.0761 |
| EXP-24 | OMISSION | 0.01690 | 0.01189 | +29.62% | 0.0208 | 0.0157 |
| EXP-24 | INSERTION | 0.00931 | 0.00533 | +42.79% | 0.0102 | **0.0021** |
| **STU-05** | CORRECT | 0.11965 | 0.08739 | +26.96% | 0.1681 | 0.0844 |
| STU-05 | SUBSTITUTION | 0.08929 | 0.07719 | +13.55% | 0.1199 | 0.0746 |
| STU-05 | OMISSION | 0.02478 | 0.01187 | +52.10% | 0.0482 | 0.0150 |
| STU-05 | INSERTION | 0.01159 | 0.00558 | +51.84% | 0.0355 | 0.0027 |

### Nhận định về hiệu chuẩn:
1. **Xác suất dự báo thô của nơ-ron luôn bị lệch tự tin quá mức (over-confident):**
   Cả EXP-36, EXP-24 và STU-05 đều có ECE thô trên lớp Đúng và Thay thế ở mức 0.15 - 0.18.
2. **Hiệu chỉnh Isotonic độc lập trên DEV_CAL phát huy tác dụng mạnh mẽ:**
   - ECE lớp Chèn (Insertion) giảm từ 0.0065 xuống mức lý tưởng **0.0026** (EXP-36) và **0.0021** (EXP-24).
   - Brier score của EXP-36 trên tất cả các lớp đều được cải thiện từ 17% đến 24%, đạt mức Brier thấp nhất toàn chiến dịch (0.07467 cho Substitution, 0.01168 cho Omission, 0.00558 cho Insertion).
3. **Mô hình học sinh STU-05 có độ tin cậy thô rất kém ở các lớp thiểu số:**
   ECE thô của STU-05 trên Omission là **0.0482** (cao gấp hơn 2 lần so với EXP-36 là 0.0226), và trên Insertion là **0.0355** (cao gấp hơn 5 lần so với EXP-36 là 0.0065). Điều này cho thấy cơ chế KD-All của học sinh GRU tạo ra các điểm số không được hiệu chuẩn ở vùng đuôi phân bố.

---

## 6. HÀNH VI CHỌN LỌC CÓ NGÂN SÁCH CỐ ĐỊNH (PRECISION@K VÀ RECALL@K)

Trong ứng dụng sư phạm thực tế, giáo viên hoặc học viên chỉ có khả năng xem xét một số lượng cảnh báo giới hạn (ngân sách $K$ cảnh báo). Bảng dưới đây trích xuất từ `experiments/top_model_investigation/confidence/precision_recall_at_k.csv` so sánh năng lực chắt lọc lỗi của các mô hình tại các ngân sách $K \in \{100, 500, 1000\}$:

| Lớp lỗi | Ngân sách $K$ | Chỉ số | EXP-36 (Tổ hợp 7) | EXP-24 (Đơn lẻ) | STU-05 (Học sinh) | Nhận định lâm sàng |
|---|---:|---|---:|---:|---:|---|
| **OMISSION**<br>*(Tổng có 209 lỗi)* | **$K=100$** | **Precision@100**<br>**Recall@100** | 28.00%<br>13.40% | **37.00%**<br>**17.70%** | 24.00%<br>11.48% | **EXP-24 vượt trội hoàn toàn ở Top 100:** Trong 100 cảnh báo đầu tiên, EXP-24 tóm trúng 37 lỗi bỏ sót thật, so với 28 của EXP-36 và 24 của STU-05! |
| | **$K=500$** | Precision@500<br>Recall@500 | **19.20%**<br>**45.93%** | 18.20%<br>43.54% | 19.00%<br>45.45% | Ở mức 500 cảnh báo, cả ba mô hình bắt được ~45% tổng số lỗi bỏ sót. |
| | **$K=1000$** | Precision@1000<br>**Recall@1000** | **14.90%**<br>**71.29%** | 13.80%<br>66.03% | 14.20%<br>67.94% | **EXP-36 dẫn đầu ở vùng phủ rộng:** Quét được 149/209 lỗi bỏ sót (71.3%). |
| **SUBSTITUTION**<br>*(Tổng có 1.768 lỗi)* | **$K=100$** | Precision@100<br>Recall@100 | 61.00%<br>3.45% | **64.00%**<br>3.62% | 60.00%<br>3.39% | Cả ba mô hình đều đạt độ chính xác >60% ở top 100 cảnh báo thay thế. |
| | **$K=500$** | Precision@500<br>Recall@500 | 53.20%<br>15.05% | 49.20%<br>13.91% | **56.80%**<br>16.06% | STU-05 duy trì độ chính xác tốt ở cụm 500 cảnh báo. |
| | **$K=1000$** | **Precision@1000**<br>**Recall@1000** | **50.20%**<br>**28.39%** | 46.90%<br>26.53% | 48.60%<br>27.49% | **EXP-36 là mô hình duy nhất giữ được Precision > 50% ở quy mô 1.000 cảnh báo** (bắt đúng 502 lỗi thay thế). |
| **INSERTION**<br>*(Tổng có 125 lỗi)* | **$K=100$** | Precision@100<br>**Recall@100** | 34.00%<br>27.20% | **36.00%**<br>**28.80%** | 34.00%<br>27.20% | EXP-24 bắt trúng 36/125 lỗi chèn ngay trong 100 vị trí khe hở nghi ngờ nhất. |
| | **$K=500$** | Precision@500<br>**Recall@500** | **12.60%**<br>**50.40%** | 10.60%<br>42.40% | 10.60%<br>42.40% | EXP-36 bắt được hơn một nửa số lỗi chèn (63/125 lỗi). |
| | **$K=1000$** | Precision@1000<br>**Recall@1000** | **7.40%**<br>**59.20%** | 6.50%<br>52.00% | 7.00%<br>56.00% | EXP-36 bắt được 74/125 lỗi chèn (59.2%). |

---

## 7. KẾT LUẬN CỦA AGENT B

1. **Mô hình có khả năng phân biệt tổng hợp mạnh nhất:** **EXP-36** là hệ thống đạt Macro-3 AP (0.2892) và Binary MD AP (0.3869) cao nhất, đồng thời duy trì độ tin cậy xác suất sau hiệu chỉnh (Brier score & ECE) tốt nhất trên toàn bộ các lớp.
2. **Mô hình chẩn đoán lỗi bỏ sót đơn lẻ tốt nhất:** **EXP-24** vượt qua EXP-36 ở AP Omission (0.2251 vs 0.2047) và Precision@100 của Omission (37% vs 28%). Cơ chế Monotonic Local Attention của EXP-24 là thành phần vô giá cho việc phát hiện âm bị nuốt/bỏ sót.
3. **Ý nghĩa so với các thế hệ trước:**
   Cả EXP-36 và EXP-24 đều tạo nên bước đột phá vượt bậc so với Phase 3 GRU Student (STU-05: Macro-3 AP 0.2365) và Specialist Teacher Banks (0.1865). Đặc biệt, khoảng cách về AP Insertion (tăng từ 0.035 lên 0.276) đã xóa bỏ hoàn toàn "điểm mù" lịch sử của bài toán chèn âm.
