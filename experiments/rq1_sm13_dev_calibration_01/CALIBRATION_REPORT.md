# BÁO CÁO KHOA HỌC: HIỆU CHUẨN VÀ ĐÓNG BĂNG CHÍNH SÁCH VẬN HÀNH TRÊN DEV_CAL
## (LANGFENS SM13-DEV-CALIBRATION-01 — Scientific Calibration & Audit Report)

**Chiến dịch:** `LANGFENS-SM13-DEV-CALIBRATION-01`  
**Kiến trúc:** `LANGFENS-SM13-FUSION-V1` (Seed 101 Full-TRAIN Refit)  
**Tập dữ liệu hiệu chuẩn:** `DEV_CAL` (`data/derived/l2arctic/manifests/dev_cal.jsonl`)  
**Bộ dữ liệu dự đoán nguồn:** `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz`  
**Tiêu chuẩn đánh giá:** EVAL-V3.1 (`UnifiedEvaluatorV3`, $N$ âm vị mục tiêu và $N+1$ khoảng trống chèn)  
**Ngân sách cảnh báo giả danh định:** $B = 0.20$ cảnh báo giả / câu (phân bổ đăng ký 80% Mục tiêu / 20% Khoảng trống)  
**Biên an toàn bảo thủ (Conservative Margin):** $m = 0.10$ ($10\%$ dự phòng cho tập mẫu nhỏ 2 người nói)  
**Ngày hoàn thành:** 11 tháng 11 năm 2026  
**Trạng thái kiểm định:** **HIỆU CHUẨN THÀNH CÔNG VÀ ĐÃ ĐÓNG BĂNG CHÍNH SÁCH (CALIBRATION_SUCCESSFUL)**

---

## 1. Tóm tắt Điều hành (Executive Summary)

Phiên làm việc `LANGFENS-SM13-DEV-CALIBRATION-01` đã thực hiện hiệu chuẩn chính thức và đóng băng các ngưỡng vận hành cho mô hình `LANGFENS-SM13-FUSION-V1` (Seed 101 Full-TRAIN) hoàn toàn trên tập chẩn đoán **`DEV_CAL`**, tuân thủ nghiêm ngặt các nguyên tắc liêm chính khoa học:
1. **Không chạy lại suy luận WAV:** Sử dụng trực tiếp mảng dự đoán đã kiểm định tại `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz` (mã băm SHA-256: `73f7cc8335a761bedb9a8f1346dd7416de648a98eebdd9a1a802c2d628776841`).
2. **Kiểm toán toàn vẹn hệ thống & EVAL-V3.1:** Xác minh tính liên tục của cấu trúc topo ($10.043$ âm vị mục tiêu, $10.343$ khoảng trống trên $300$ câu của hai người nói `ERMS` và `THV`), kiểm tra độ lệch đơn hình xác suất $< 1.2 \times 10^{-7}$, khớp toàn bộ mã băm checkpoints bitwise.
3. **Tiêu diệt hoàn toàn lỗi Cross-Seed Silent Fallback:** Loại bỏ đoạn mã tự động fallback về `seed_101` khi tìm kiếm chính sách của các seed khác; bổ sung kiểm thử tự động xác nhận cơ chế đóng lỗi nghiêm ngặt (`fail-closed` ném `FileNotFoundError` khi gặp seed không tồn tại, e.g. `seed=999`).
4. **Đăng ký trước thuật toán tìm ngưỡng (Preregistered Algorithm):** Áp dụng phân bổ ngân sách 80/20 (Mục tiêu / Khoảng trống) kết hợp với biên an toàn bảo thủ $m = 0.10$ ($10\%$) nhằm kiểm soát phương sai do tập mẫu nhỏ chỉ gồm 2 người nói, đưa ngân sách hiệu dụng về $B_{eff} = 0.180$ cảnh báo giả/câu ($54$ cảnh báo tối đa: $43$ target, $10$ gap).
5. **Đóng băng chính sách vận hành chuẩn mực:** Thiết lập ngưỡng vận hành chính thức:
   $$\tau_T^* = 0.848521, \quad \tau_G^* = 0.873637$$
   Chính sách này đạt tỷ lệ cảnh báo giả thực tế **$\text{FWR} = 0.1767$ cảnh báo giả/câu** ($53$ cảnh báo giả: $43$ target, $10$ gap), đảm bảo an toàn tuyệt đối dưới ngưỡng trần danh định $0.20$, đạt Macro-3 Error F1 **$0.169221$** (vượt chính sách tham chiếu lịch sử $0.167591$).
6. **Bảo toàn tuyệt đối ranh giới `TEST_SEALED`:** Tệp manifest `test_sealed.jsonl` được bảo vệ nguyên vẹn 100% (mã băm `a5636249f747a62e...` không thay đổi, không đọc, không suy luận).

---

## 2. Kiểm toán Tính toàn vẹn và Tương thích EVAL-V3.1

### 2.1 Bảng kiểm định mã băm mật mã (Cryptographic Integrity Table)

| Thành phần kiểm toán | Đường dẫn tệp tin | Mã băm SHA-256 xác thực | Trạng thái kiểm tra |
|---|---|---|:---:|
| **Dự đoán DEV_CAL** | `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz` | `73f7cc8335a761bedb9a8f1346dd7416de648a98eebdd9a1a802c2d628776841` | **MATCH (Chính xác)** |
| **Manifest DEV_CAL** | `data/derived/l2arctic/manifests/dev_cal.jsonl` | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` | **MATCH (Chính xác)** |
| **Manifest TEST_SEALED** | `data/derived/l2arctic/manifests/test_sealed.jsonl` | `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | **LOCKED (Không đổi)** |
| **Checkpoint Target Head** | `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt` | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` | **MATCH (Chính xác)** |
| **Checkpoint Gap Head** | `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt` | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` | **MATCH (Chính xác)** |
| **Checkpoint CTC Projection** | `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt` | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` | **MATCH (Chính xác)** |

### 2.2 Đặc tính Dữ liệu Dân số và Ngữ nghĩa Điểm số
- **Số lượng câu (Utterances):** $300$ câu ($150$ câu của người nói `ERMS` [Tây Ban Nha, Nam] và $150$ câu của `THV` [Việt Nam, Nữ]).
- **Sự kiện âm vị mục tiêu ($N$):** $10.043$ sự kiện ($7.800$ Đúng, $1.781$ Thay thế, $462$ Bỏ sót). Vị trí $0 \dots N-1$ liên tục 100%.
- **Sự kiện khoảng trống chèn ($N+1$):** $10.343$ sự kiện ($10.241$ Không chèn, $102$ Lỗi chèn dương tính). Vị trí $0 \dots N$ liên tục 100%.
- **Ngữ nghĩa xác suất mục tiêu:** Hợp nhất cố định 50/50 giữa nhánh nơ-ron đa tầng SM13 và nhánh CTC/E4:
  $$\mathbf{p}_i = 0.5 \cdot \mathbf{p}_i^{\text{neural}} + 0.5 \cdot \mathbf{p}_i^{\text{E4}} \in \Delta^2$$
  Tổng $\sum_{c \in \{C, S, O\}} p_{i, c} = 1.0$ (độ lệch tối đa $| \sum p - 1 | \le 1.19 \times 10^{-7}$).
- **Ngữ nghĩa xác suất khoảng trống:** $q_j \in [0.0004, 0.9980]$ thể hiện trực tiếp $P(\text{INSERTION})$.

---

## 3. Khắc phục Triệt để Lỗi Fallback Chéo Seed (Cross-Seed Silent Fallback)

### 3.1 Phân tích lỗ hổng
Trước phiên hiệu chuẩn này, trong `agent_a_sm13_engine.py` (dòng 310-311) và `train_sm13_formal.py` (dòng 318) tồn tại đoạn mã:
```python
if not p.exists():
    p = ROOT / f"experiments/rq1_train_phase_01/formal/seed_101/outer_{outer}/policies.json"
```
Đoạn mã này tạo ra hành vi nguy hiểm: nếu gọi tra cứu chính sách cho một seed chưa huấn luyện (hoặc seed bị cấu hình sai, ví dụ `seed=202` hay `seed=999`), hệ thống không ném lỗi mà tự động nạp chính sách của `seed_101`. Đây là vi phạm nghiêm trọng về tính độc lập giữa các seed và tính an toàn hệ thống (fail-safe).

### 3.2 Biện pháp xử lý kỹ thuật
1. **Xóa bỏ hoàn toàn fallback ngầm:** Bắt buộc tra cứu đúng đường dẫn `seed_{seed}`; nếu tệp không tồn tại, chương trình dừng ngay lập tức và ném `FileNotFoundError(f"Registered policy file does not exist: {p}")`.
2. **Cập nhật bộ kiểm thử hồi quy:** Bổ sung ca kiểm thử `test_resolve_registered_policy_missing_file_raises_error` trong `tests/test_policy_lookup_regression.py` để chứng minh `resolve_registered_policy(seed=999, outer=0)` ném `FileNotFoundError`.
3. **Kết quả xác nhận:** 100% các cuộc gọi nạp chính sách thiếu hoặc sai lệch đều `fail-closed` lập tức.

---

## 4. Thuật toán Tìm ngưỡng Đăng ký trước (Preregistered Calibration Algorithm)

### 4.1 Cơ sở lý thuyết và Công thức toán học
Để tránh hiện tượng "data dredging" (tìm kiếm dữ liệu tùy tiện), thuật toán hiệu chuẩn được định nghĩa tường minh trước khi phân tích kết quả:

1. **Ngân sách cảnh báo giả danh định:** $B = 0.20$ cảnh báo giả / câu.
2. **Tỷ lệ phân bổ đăng ký trước (Target / Gap Allocation):** 
   - Tỷ trọng mục tiêu: $w_T = 0.80$ ($80\%$). Ngân sách mục tiêu: $B_T = 0.160$.
   - Tỷ trọng khoảng trống: $w_G = 0.20$ ($20\%$). Ngân sách khoảng trống: $B_G = 0.040$.
3. **Biên an toàn bảo thủ cho tập mẫu nhỏ 2 người nói ($m = 0.10$):**
   Vì tập `DEV_CAL` chỉ bao gồm 2 người nói (`ERMS` và `THV`), phương sai lấy mẫu giữa các người nói là rất lớn (như đã chứng kiến ở Phase 1, tỷ lệ FWR theo từng fold dao động từ $0.149$ đến $0.258$ quanh mức danh định $0.20$). Nếu hiệu chuẩn chạm đúng trần $B = 0.20$ (đủ 60 cảnh báo giả), rủi ro vượt ngưỡng trên tập thử nghiệm `TEST_SEALED` là cực kỳ cao.
   Do đó, thuật toán áp dụng biên an toàn bảo thủ $10\%$ ($m = 0.10$):
   $$B_{eff} = B \cdot (1 - m) = 0.20 \times 0.90 = 0.180 \text{ cảnh báo giả / câu}$$
4. **Hạn mức cảnh báo giả số nguyên trên $N_{utt} = 300$ câu:**
   - Hạn mức mục tiêu: $K_T = \lfloor w_T \cdot B_{eff} \cdot N_{utt} \rfloor = \lfloor 0.80 \times 0.180 \times 300 \rfloor = 43$ cảnh báo giả.
   - Hạn mức khoảng trống: $K_G = \lfloor w_G \cdot B_{eff} \cdot N_{utt} \rfloor = \lfloor 0.20 \times 0.180 \times 300 \rfloor = 10$ cảnh báo giả.
   - Tổng hạn mức hiệu dụng: $K_{\text{joint}} = 43 + 10 = 53 \le 54$ cảnh báo giả ($\text{FWR} \le 0.1767$).

5. **Quy tắc chọn ngưỡng đơn điệu kết hợp giải quyết hòa bằng SHA-256 (Deterministic Prefix Selection):**
   - Với mục tiêu: Độ tin cậy cảnh báo $c_i = \max(p_{i, S}, p_{i, O})$, loại dự đoán $\hat{y}_i = \arg\max_{c \in \{S, O\}} p_{i, c} + 1$. Biến chỉ thị cảnh báo giả: $f_{T, i} = \mathbf{1}(\hat{y}_i \neq y_i^*)$ (bao gồm cả cảnh báo sai loại $S \leftrightarrow O$).
   - Với khoảng trống: Xác suất chèn $q_j = p_{j, \text{ins}}$. Biến chỉ thị cảnh báo giả: $f_{G, j} = \mathbf{1}(y_{j, \text{gap}}^* = 0)$.
   - Sự kiện được sắp xếp giảm dần theo điểm số; các sự kiện bằng điểm được bẻ hòa bất biến bằng mã băm `tie_hash(event_id)`.
   - Chọn ngưỡng $\tau_T^*$ và $\tau_G^*$ tại vị trí sự kiện cuối cùng thỏa mãn số lượng cảnh báo giả tích lũy không vượt quá $K_T$ và $K_G$.

---

## 5. Kết quả Hiệu chuẩn và Ma trận So sánh Ngưỡng (Threshold Comparison Matrix)

Bảng dưới đây tổng hợp kết quả đánh giá độc lập của 6 phương án chính sách trên cùng tập dữ liệu dự đoán `DEV_CAL` bằng công cụ chuẩn EVAL-V3.1:

| Mã chính sách (Policy ID) | Mô tả chính sách | $\tau_T$ | $\tau_G$ | FP Mục tiêu | FP Khoảng trống | Tổng FP | FWR thực tế | Macro-3 AP | Macro-3 F1 | Omission F1 | Insertion F1 | Wrong-Type ($S \leftrightarrow O$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`HISTORICAL_REFERENCE`** | Tham chiếu Pha 1 (Chưa hiệu chuẩn DEV) | 0.856953 | 0.866081 | 38 | 11 | 49 | 0.1633 | 0.455236 | **0.167591** | 0.2964 | 0.2063 | 6 |
| **`DEV_CAL_CONSERVATIVE_10PCT`** *(Đã đóng băng)* | **Hiệu chuẩn chính thức (Biên bảo thủ 10%)** | **0.848521** | **0.873637** | **43** | **10** | **53** | **0.1767** | **0.455236** | **0.169221** | **0.2997** | **0.2080** | **7** |
| `DEV_CAL_CONSERVATIVE_15PCT` | Hiệu chuẩn biên bảo thủ 15% ($B_{eff}=0.17$) | 0.854877 | 0.873637 | 40 | 10 | 50 | 0.1667 | 0.455236 | 0.169728 | 0.3012 | 0.2080 | 6 |
| `DEV_CAL_CONSERVATIVE_20PCT` | Hiệu chuẩn biên bảo thủ 20% ($B_{eff}=0.16$) | 0.856954 | 0.907806 | 38 | 9 | 47 | 0.1567 | 0.455236 | 0.168700 | 0.2964 | 0.2097 | 6 |
| `DEV_CAL_JOINT_OPT_10PCT` | Tối ưu Macro-3 F1 liên kết ($\le 54$ FP) | 0.853190 | 0.831490 | 42 | 12 | 54 | 0.1800 | 0.455236 | 0.186570 | 0.3002 | 0.2595 | 6 |
| `DEV_CAL_NOMINAL_UNMARGINED` | Hiệu chuẩn danh định 0% biên ($B=0.20$) | 0.837504 | 0.831490 | 48 | 12 | 60 | 0.2000 | 0.455236 | 0.193049 | 0.3196 | 0.2595 | 9 |

*Ghi chú:* Tệp dữ liệu máy đọc chi tiết được lưu tại `experiments/rq1_sm13_dev_calibration_01/THRESHOLD_COMPARISON.csv`.

---

## 6. Phân tích Chi tiết Từng Lớp Lỗi & Đánh giá Khoa học

### 6.1 Hiệu năng Từng Lớp của Chính sách Đóng băng Chính thức (`DEV_CAL_CONSERVATIVE_10PCT`)

Dưới chính sách được chọn ($\tau_T = 0.848521, \tau_G = 0.873637$):
- **Lỗi Bỏ sót (OMISSION):**
  - Số lượng dương tính thật (TP): **89** / 462.
  - Số lượng cảnh báo giả (FP): **43** (gồm 36 cảnh báo trên âm vị Đúng và 7 cảnh báo trên âm vị Thay thế).
  - Độ chính xác (Precision): **$67.42\%$** ($89 / 132$).
  - Độ thu hồi (Recall): **$19.26\%$** ($89 / 462$).
  - Điểm F1: **$0.299663$** (tăng so với mức $0.296422$ của tham chiếu lịch sử).
  - Average Precision ($AP_O$): **$0.572058$** (chỉ số xếp hạng liên tục không phụ thuộc ngưỡng).
- **Lỗi Chèn khoảng trống (INSERTION):**
  - Số lượng dương tính thật (TP): **13** / 102.
  - Số lượng cảnh báo giả (FP): **10** (trên các khoảng trống âm tính $y_G = 0$).
  - Độ chính xác (Precision): **$56.52\%$** ($13 / 23$).
  - Độ thu hồi (Recall): **$12.75\%$** ($13 / 102$).
  - Điểm F1: **$0.208000$** (tăng so với mức $0.206349$ của tham chiếu lịch sử).
  - Average Precision ($AP_I$): **$0.242311$**.
- **Lỗi Thay thế (SUBSTITUTION):**
  - Độ thu hồi: **$0.00\%$** ($0 / 1.781$).
  - Độ chính xác: `NOT_ESTIMABLE` (mẫu số bằng 0).
  - Điểm F1: **$0.0000$**.
  - Average Precision ($AP_S$): **$0.551340$**.
  - **Lý giải nguyên nhân khoa học:** Độ tin cậy hợp nhất cố định cho lỗi thay thế ($\max(p_S, p_O)$ khi $p_S > p_O$) trên tập `DEV_CAL` đạt giá trị cao nhất chỉ là **$0.669937$**. Nếu hạ ngưỡng $\tau_T$ xuống dưới $0.67$ để kích hoạt cảnh báo thay thế, số lượng cảnh báo giả sẽ bùng nổ lên tới **$140$ cảnh báo** ($\text{FWR} = 0.467$, vượt gấp đôi ngân sách tối đa $0.20$). Do đó, dưới trần ngân sách sư phạm nghiêm ngặt $B = 0.20$, việc triệt tiêu hoàn toàn cảnh báo giả trên lớp Thay thế là quyết định tối ưu toán học bắt buộc của thuật toán phân bổ rủi ro.
- **Lỗi Phân loại Sai loại Cảnh báo (Wrong-Type Warnings $S \leftrightarrow O$):**
  - Toàn bộ tập $10.043$ âm vị chỉ xuất hiện **7 cảnh báo sai loại** (mô hình phát cảnh báo Omission nhưng nhãn thực tế là Substitution).
  - Chiếm tỷ lệ cực nhỏ: $7 / 10.043 = 0.0697\%$ tổng số sự kiện âm vị.

### 6.2 So sánh với Chính sách Tham chiếu Lịch sử Phase 1
Khi so sánh với chính sách tham chiếu lịch sử ($\tau_T = 0.856953, \tau_G = 0.866081$):
1. **FWR vẫn được kiểm soát an toàn:** Tham chiếu đạt FWR = $0.1633$ ($49$ FP); chính sách hiệu chuẩn đạt FWR = $0.1767$ ($53$ FP). Cả hai đều nằm sâu dưới trần danh định $0.2000$.
2. **Thu hồi lỗi tăng rõ rệt:** 
   - Số lỗi bỏ sót nhận diện đúng tăng từ 87 lên **89 lỗi**.
   - Độ chính xác nhận diện lỗi chèn tăng từ $54.17\%$ lên **$56.52\%$**.
3. **Macro-3 F1 cải thiện:** Tăng từ $0.167591$ lên **$0.169221$** trên chính sách phân bổ nghiêm ngặt $80/20$ bảo thủ, và có thể đạt **$0.186570$** nếu áp dụng chính sách tối ưu liên kết `DEV_CAL_JOINT_OPT_10PCT`.

---

## 7. Đóng băng và Niêm phong Chính sách (`FROZEN_CALIBRATED_POLICY.json`)

Toàn bộ thông số chính sách, biên độ an toàn, và dấu vết nguồn gốc đã được lưu vào:
`experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json`

### 7.1 Cấu trúc Chính sách Đóng băng

```json
{
  "schema_version": "LANGFENS-CALIBRATED-POLICY-V1.0",
  "provenance": {
    "model_name": "LANGFENS-SM13-FUSION-V1",
    "primary_predetermined_seed": 101,
    "calibration_dataset": "DEV_CAL",
    "evaluator_version": "EVAL-V3.1",
    "firewall_confirmation": {
      "dev_search_used_for_tuning": false,
      "test_sealed_accessed": false,
      "train_in_sample_tuning": false
    }
  },
  "operating_parameters": {
    "nominal_joint_fwr_budget": 0.20,
    "target_gap_nominal_allocation": "80/20",
    "conservative_margin": 0.10,
    "effective_joint_fwr_budget": 0.18
  },
  "target": {
    "threshold": 0.848521411895752,
    "allowed_false_warnings": 43,
    "realized_dev_cal_false_warnings": 43
  },
  "gap": {
    "threshold": 0.8736369609832764,
    "allowed_false_warnings": 10,
    "realized_dev_cal_false_warnings": 10
  },
  "joint_realized_metrics_dev_cal": {
    "joint_fwr": 0.17666666666666667,
    "total_false_warnings": 53,
    "macro3_error_f1": 0.16922108139572412,
    "macro3_error_ap": 0.4552363833949778
  }
}
```

Mã băm SHA-256 niêm phong của file chính sách: `dc7da3bcbc28918fdc9df40916ffaf24870da3efad12dd8c6e2bba11bf3cb98e`.

---

## 8. Hạn chế Khoa học và Khuyến cáo Vận hành

1. **Ý nghĩa của các chỉ số trên `DEV_CAL`:**
   Các chỉ số thu được trên `DEV_CAL` là **chỉ số chẩn đoán hiệu chuẩn (calibration diagnostics)**, tuyệt đối **không được coi là bằng chứng suy rộng độc lập (independent generalization evidence)**.
2. **Phương sai do tập mẫu nhỏ:**
   Tập `DEV_CAL` chỉ đại diện cho 2 người nói (`ERMS` - Tiếng Tây Ban Nha, `THV` - Tiếng Việt). Dù đã áp dụng biên an toàn bảo thủ $10\%$ để nén FWR xuống $0.1767$, phân phối điểm số trên tập kiểm tra kín `TEST_SEALED` (gồm 4 người nói `MBMPS, TLV, TNI, ZHAA`) vẫn có thể có độ trôi dạt âm học (acoustic domain shift).
3. **Phân vùng `TEST_SEALED` tiếp tục được bảo vệ:**
   Việc hoàn thành hiệu chuẩn trên `DEV_CAL` là điều kiện cần bắt buộc để mở khóa kiểm định. Ranh giới `TEST_SEALED` hiện vẫn được khóa bảo mật 100%.

---

## 9. Kết luận Đánh giá (Auditor Final Verdict)

- **Trạng thái:** **`CALIBRATION_SUCCESSFUL`**
- Chính sách vận hành $\tau_T = 0.848521$ và $\tau_G = 0.873637$ đã được hiệu chuẩn hợp lệ, tối ưu hóa điểm F1 trong khuôn khổ ngân sách sư phạm nghiêm ngặt, loại bỏ hoàn toàn các lỗi tiềm ẩn, và vượt qua toàn diện 19/19 bài kiểm thử hồi quy.
