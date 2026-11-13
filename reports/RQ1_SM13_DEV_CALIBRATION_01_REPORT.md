# BÁO CÁO TOÀN DIỆN VÀ DUY NHẤT: HIỆU CHUẨN VÀ ĐÓNG BĂNG CHÍNH SÁCH VẬN HÀNH TRÊN DEV_CAL
## (LANGFENS RQ1-SM13-DEV-CALIBRATION-01 — Master Scientific Calibration & Audit Report)

**Chiến dịch nghiên cứu:** `LANGFENS-SM13-DEV-CALIBRATION-01`  
**Vai trò:** Kỹ sư Trưởng Speech ML & Kiểm toán viên Khoa học Độc lập (Senior Speech ML Engineer & Scientific Auditor)  
**Mô hình mục tiêu:** `LANGFENS-SM13-FUSION-V1` (Seed 101 Full-TRAIN Refit)  
**Tiêu chuẩn đánh giá:** EVAL-V3.1 (`UnifiedEvaluatorV3`, $N$ âm vị mục tiêu và $N+1$ khoảng trống chèn)  
**Tập dữ liệu hiệu chuẩn duy nhất:** `DEV_CAL` (`data/derived/l2arctic/manifests/dev_cal.jsonl`)  
**Tập dữ liệu dự đoán nguồn:** `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz`  
**Môi trường thực thi:** Host Linux x86_64, GPU NVIDIA GeForce GTX 1650 (4096 MiB VRAM), Python 3.11.17, PyTorch 2.6.0+cu124, pytest-9.2.0  
**Ngày hoàn thành:** 11 tháng 11 năm 2026  
**Trạng thái kiểm định:** **`CALIBRATION_SUCCESSFUL` (100% KIỂM THỬ ĐẠT & ĐÃ ĐÓNG BĂNG CHÍNH SÁCH)**  
**Trạng thái Ranh giới Bảo mật:** **`TEST_SEALED` TIẾP TỤC ĐƯỢC KHÓA CHẶT 100% (MÃ BĂM SHA-256 BẤT BIẾN)**

---

## 1. Tóm tắt Điều hành (Executive Summary)

Báo cáo này là tài liệu tổng hợp toàn diện duy nhất ghi lại toàn bộ quá trình, kết quả toán học, ma trận kiểm định, và việc đóng băng chính thức các ngưỡng vận hành cho kiến trúc mô hình `LANGFENS-SM13-FUSION-V1` (Seed 101 Full-TRAIN) trên tập hiệu chuẩn chẩn đoán **`DEV_CAL`**. 

Phiên làm việc đã giải quyết dứt điểm 10 nhiệm vụ trọng yếu theo các tiêu chuẩn khoa học cao nhất:
1. **Tuân thủ quy tắc Không chạy lại suy luận WAV:** Tận dụng 100% mảng dự đoán xác suất và log-likelihood đã được kiểm định từ chiến dịch trước tại `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz` (mã băm SHA-256: `73f7cc8335a761bedb9a8f1346dd7416de648a98eebdd9a1a802c2d628776841`). Tuyệt đối không chạy lại mô hình nhận dạng âm học WAV.
2. **Kiểm toán dữ liệu và tính tương thích EVAL-V3.1:** Xác minh toàn vẹn cấu trúc liên tục của $10.043$ sự kiện âm vị mục tiêu và $10.343$ khoảng trống chèn trên $300$ câu của hai người nói `ERMS` (Tây Ban Nha, Nam) và `THV` (Việt Nam, Nữ). Đơn hình xác suất hợp nhất 50/50 có độ lệch tối đa $| \sum p - 1 | \le 1.19 \times 10^{-7}$. Toàn bộ các checkpoint đều khớp mã băm bitwise.
3. **Tiêu diệt hoàn toàn lỗi Fallback ngầm chéo seed (Cross-Seed Silent Fallback):** Tìm và sửa tận gốc lỗi fallback tự động về `seed_101` khi gọi hàm nạp chính sách với seed khác; thiết lập cơ chế ném ngoại lệ rõ ràng (`FileNotFoundError`, `ValueError`); bổ sung kiểm thử tự động xác nhận cơ chế đóng lỗi nghiêm ngặt (`fail-closed` ném lỗi khi gặp `seed=999`).
4. **Đăng ký trước thuật toán tìm ngưỡng (Preregistered Algorithm):** Áp dụng phân bổ ngân sách danh định $B = 0.20$ theo tỷ lệ cố định 80% Mục tiêu ($B_T = 0.160$) / 20% Khoảng trống ($B_G = 0.040$). Bổ sung biên an toàn bảo thủ $m = 0.10$ ($10\%$) để kiểm soát phương sai lấy mẫu của tập chỉ có 2 người nói, đưa ngân sách hiệu dụng về $B_{eff} = 0.180$ cảnh báo giả/câu ($54$ cảnh báo tối đa: $43$ target, $10$ gap).
5. **Tối ưu hóa và Đóng băng Chính sách Vận hành Chính thức:**
   $$\mathbf{\tau_T^* = 0.848521}, \quad \mathbf{\tau_G^* = 0.873637}$$
   Đạt tỷ lệ cảnh báo giả thực tế **$\text{FWR} = 0.1767$ cảnh báo giả/câu** ($53$ cảnh báo giả: $43$ target, $10$ gap), đảm bảo nằm sâu dưới trần an toàn sư phạm $0.2000$. Macro-3 Error F1 đạt **$0.169221$** (cải thiện so với chính sách tham chiếu lịch sử Phase 1 là $0.167591$).
6. **Bảo toàn tuyệt đối Ranh giới Tập kín `TEST_SEALED`:** Mã băm manifest `test_sealed.jsonl` được bảo toàn nguyên vẹn (`a5636249f747a62e...`), không đọc nhãn, không truy cập tệp, giữ kín 100% cho đến khi có lệnh kiểm tra độc lập từ Operator.
7. **Vượt qua 100% Bộ kiểm thử Hồi quy:** Hoàn thành xuất sắc 19/19 ca kiểm thử tự động trên repository (`pytest tests/`).

---

## 2. Phân tích Nguyên nhân Gốc rễ & Sửa lỗi Cross-Seed Silent Fallback

### 2.1 Bối cảnh lỗi (Forensic Diagnosis)
Trong các phiên bản code trước đây, tại hàm `resolve_registered_policy` (`experiments/rq1_sm13_formal_train_01/agent_a_sm13_engine.py`) và trong tệp huấn luyện `train_sm13_formal.py`, đoạn mã nạp ngưỡng cảnh báo được viết như sau:

```python
# ĐOẠN CODE CŨ CHỨA LỖ HỔNG FALLBACK THẦM LẶNG:
p = ROOT / f"experiments/rq1_train_phase_01/formal/seed_{seed}/outer_{outer}/policies.json"
if not p.exists():
    p = ROOT / f"experiments/rq1_train_phase_01/formal/seed_101/outer_{outer}/policies.json"

if not p.exists():
    raise FileNotFoundError(f"Registered policy file does not exist: {p}")
```

**Nguy cơ tiềm ẩn:**
- Nếu một tiến trình gọi hàm với `seed=202`, `seed=303`, hoặc bất kỳ seed thử nghiệm nào khác (`seed=999`), nếu thư mục của seed đó chưa có `policies.json`, đoạn mã trên sẽ **âm thầm nạp file của `seed_101`** mà không hề đưa ra bất kỳ cảnh báo hay ngoại lệ nào.
- Điều này vi phạm nghiêm trọng tính độc lập giữa các seed và che giấu lỗi cấu hình (silent cross-seed contamination).

### 2.2 Giải pháp Khắc phục Kỹ thuật
1. **Loại bỏ hoàn toàn fallback ngầm:**
   - Trong `experiments/rq1_sm13_formal_train_01/agent_a_sm13_engine.py`: Xóa bỏ dòng `if not p.exists(): p = ROOT / ... / seed_101 / ...`.
   - Trong `experiments/rq1_sm13_formal_train_01/train_sm13_formal.py`: Xóa bỏ dòng fallback tương tự ở cấp độ outer fold.
2. **Thực thi nguyên tắc Fail-Closed:**
   Nếu chính sách tương ứng với `seed` được yêu cầu không tồn tại trên đĩa, chương trình phải lập tức dừng lại và ném `FileNotFoundError(f"Registered policy file does not exist: {p}")`.
3. **Bổ sung Kiểm thử Tự động:**
   Cập nhật ca kiểm thử `test_resolve_registered_policy_missing_file_raises_error` trong `tests/test_policy_lookup_regression.py`:
   ```python
   # Cross-seed missing policy must also fail closed with FileNotFoundError, never fallback to seed 101
   with pytest.raises(FileNotFoundError, match="does not exist"):
       resolve_registered_policy(seed=999, outer=0)
   ```
4. **Kết quả kiểm chứng:** Chạy thử nghiệm với `seed=999` lập tức ném `FileNotFoundError` chính xác như thiết kế.

---

## 3. Kiểm toán Toàn vẹn Hệ thống, Checkpoints & Ranh giới Dữ liệu

### 3.1 Bảng Kiểm toán Mã băm Mật mã (Bitwise Hash Audit Table)

Tất cả các tệp trọng số mô hình, manifest dữ liệu và tệp dự đoán đã được kiểm toán băm bitwise SHA-256:

| Thành phần kiểm toán | Đường dẫn tệp tin | Kích thước | Mã băm SHA-256 xác thực | Kết quả kiểm toán |
|---|---|:---:|---|:---:|
| **Dự đoán DEV_CAL** | `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz` | 614 KB | `73f7cc8335a761bedb9a8f1346dd7416de648a98eebdd9a1a802c2d628776841` | **MATCH (Nguyên vẹn)** |
| **Manifest DEV_CAL** | `data/derived/l2arctic/manifests/dev_cal.jsonl` | 1.79 MB | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` | **MATCH (Nguyên vẹn)** |
| **Manifest TEST_SEALED** | `data/derived/l2arctic/manifests/test_sealed.jsonl` | 3.55 MB | `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | **LOCKED (Không chạm vào)** |
| **Target Head Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt` | 10.7 KB | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` | **MATCH (Đóng băng)** |
| **Gap Head Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt` | 10.7 KB | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` | **MATCH (Đóng băng)** |
| **CTC Head Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt` | 129 KB | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` | **MATCH (Đóng băng)** |
| **Chính sách Đóng băng** | `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json` | 5.08 KB | `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1` | **SEALED (Niêm phong)** |

### 3.2 Phân bố Dân số và Kiểm tra Ngữ nghĩa Điểm số
- **Dân số DEV_CAL:** $300$ câu, gồm $150$ câu của người nói `ERMS` (Tây Ban Nha, Nam) và $150$ câu của người nói `THV` (Việt Nam, Nữ).
- **Phân bố thực tế lớp Mục tiêu ($10.043$ sự kiện):**
  - Đúng (CORRECT): **7.800** sự kiện ($77.67\%$).
  - Thay thế (SUBSTITUTION): **1.781** sự kiện ($17.73\%$).
  - Bỏ sót (OMISSION): **462** sự kiện ($4.60\%$).
- **Phân bố thực tế lớp Khoảng trống ($10.343$ sự kiện):**
  - Không chèn (NO_INSERTION): **10.241** khoảng trống ($99.01\%$).
  - Chèn (INSERTION): **102** khoảng trống dương tính ($0.99\%$).
- **Tính toán Đơn hình Xác suất:**
  - Xác suất mục tiêu hợp nhất $\mathbf{p}_i = 0.5 \cdot \mathbf{p}_i^{\text{neural}} + 0.5 \cdot \mathbf{p}_i^{\text{E4}} \in \Delta^2$ có tổng 3 lớp là $1.0$. Độ lệch tối đa: $| \sum p - 1 | = 1.19 \times 10^{-7}$, hoàn toàn hợp lệ dưới ngưỡng kiểm tra $10^{-4}$.
  - Xác suất khoảng trống chèn $q_j \in [0.0004, 0.9980]$, thuộc đoạn $[0, 1]$ chuẩn mực.

---

## 4. Đặc tả Thuật toán Tìm ngưỡng Đăng ký trước (Preregistered Algorithm Specification)

Để tuân thủ tiêu chuẩn liêm chính khoa học của các thử nghiệm lâm sàng và kiểm toán Speech ML, thuật toán tìm ngưỡng được mô tả toán học tường minh **trước khi khảo sát các điểm vận hành**:

### 4.1 Quy tắc Phân bổ Ngân sách
1. **Ngân sách danh định toàn cục (Nominal Joint Budget):**
   $$B = 0.20 \text{ cảnh báo giả / câu}$$
2. **Tỷ lệ phân bổ danh định đăng ký trước (Registered Target / Gap Allocation):**
   - Phân bổ cho Âm vị Mục tiêu (Target Domain): $w_T = 0.80$ ($80\%$). Ngân sách danh định mục tiêu:
     $$B_T = w_T \cdot B = 0.80 \times 0.20 = 0.160 \text{ cảnh báo giả / câu}$$
   - Phân bổ cho Khoảng trống Chèn (Gap Insertion Domain): $w_G = 0.20$ ($20\%$). Ngân sách danh định khoảng trống:
     $$B_G = w_G \cdot B = 0.20 \times 0.20 = 0.040 \text{ cảnh báo giả / câu}$$

### 4.2 Biên an toàn Bảo thủ cho Tập Mẫu Nhỏ (Conservative Cohort Safety Margin)
- **Vấn đề phương sai:** Tập `DEV_CAL` chỉ có $N_{\text{speakers}} = 2$ người nói. Khi kích thước mẫu người nói nhỏ, tỷ lệ cảnh báo giả thực tế có độ lệch chuẩn cao giữa các người nói. Nếu chọn ngưỡng chạm đúng $100\%$ ngân sách danh định ($60$ cảnh báo giả trên $300$ câu), khi triển khai sang tập người nói mới (`TEST_SEALED`), hệ thống sẽ có xác suất rất cao bị vượt trần $0.20$.
- **Định lượng biên bảo thủ:** Áp dụng hệ số an toàn bảo thủ $m = 0.10$ ($10\%$ biên dự phòng an toàn):
  $$B_{eff} = B \cdot (1 - m) = 0.20 \times (1 - 0.10) = 0.180 \text{ cảnh báo giả / câu}$$
- **Hạn mức cảnh báo giả số nguyên trên $N_{utt} = 300$ câu:**
  - Hạn mức Mục tiêu hiệu dụng:
    $$K_T = \lfloor w_T \cdot B_{eff} \cdot N_{utt} \rfloor = \lfloor 0.80 \times 0.180 \times 300 \rfloor = 43 \text{ cảnh báo giả}$$
  - Hạn mức Khoảng trống hiệu dụng:
    $$K_G = \lfloor w_G \cdot B_{eff} \cdot N_{utt} \rfloor = \lfloor 0.20 \times 0.180 \times 300 \rfloor = 10 \text{ cảnh báo giả}$$
  - Tổng số cảnh báo giả tối đa cho phép:
    $$K_{\text{joint}} = K_T + K_G = 43 + 10 = 53 \le 54 \text{ cảnh báo giả} \quad (\text{FWR} \le 0.1767)$$

### 4.3 Quy tắc Quyết định và Chọn Ngưỡng Bất biến (Deterministic Prefix Selection)
- Với mỗi sự kiện âm vị mục tiêu $i \in \{1, \dots, N_T\}$:
  - Độ tin cậy cảnh báo: $c_i = \max(p_{i, S}, p_{i, O})$.
  - Loại cảnh báo ứng viên: $\hat{k}_i = \arg\max_{c \in \{S, O\}} p_{i, c} + 1 \in \{1 \text{ (SUB)}, 2 \text{ (OMI)}\}$.
  - Chỉ thị cảnh báo giả (False Alarm Indicator): $f_{T, i} = \mathbf{1}(\hat{k}_i \neq y_i^*)$, trong đó $y_i^* \in \{0, 1, 2\}$ là nhãn thực tế. Lưu ý: Trường hợp $\hat{k}_i = 1$ nhưng $y_i^* = 2$ hoặc ngược lại (cảnh báo sai loại $S \leftrightarrow O$) đều được tính là cảnh báo giả theo chuẩn EVAL-V3.1.
- Với mỗi khoảng trống $j \in \{1, \dots, N_G\}$:
  - Xác suất chèn: $q_j = p_{j, \text{ins}}$.
  - Chỉ thị cảnh báo giả: $f_{G, j} = \mathbf{1}(y_{j, \text{gap}}^* == 0)$.
- **Thuật toán chọn ngưỡng:** Sắp xếp các sự kiện giảm dần theo điểm số xác suất. Trong trường hợp bằng điểm số, bẻ hòa xác định bằng hàm băm mật mã:
  $$\text{key}(i) = (-s_i, -\text{tie\_hash}(\text{event\_id}_i))$$
  Chọn ngưỡng $\tau_T^*$ và $\tau_G^*$ là giá trị điểm số của sự kiện cuối cùng thỏa mãn số lượng cảnh báo giả tích lũy $\sum f \le K$.

---

## 5. Bảng So sánh Toàn diện Các Chính sách Vận hành trên DEV_CAL

Bảng ma trận so sánh chi tiết tất cả 6 phương án chính sách được xuất ra tệp `experiments/rq1_sm13_dev_calibration_01/THRESHOLD_COMPARISON.csv`:

| Mã Chính sách (Policy ID) | Mô tả Chiến lược | $\tau_T$ | $\tau_G$ | FP Mục tiêu | FP Khoảng trống | Tổng FP | FWR Thực tế | Macro-3 AP | Macro-3 F1 | Omission F1 | Insertion F1 | Wrong-Type ($S \leftrightarrow O$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`HISTORICAL_REFERENCE`** | Tham chiếu Pha 1 (Chưa hiệu chuẩn DEV) | 0.856953 | 0.866081 | 38 | 11 | 49 | 0.1633 | 0.455236 | **0.167591** | 0.2964 | 0.2063 | 6 |
| **`DEV_CAL_CONSERVATIVE_10PCT`** *(Chính thức đóng băng)* | **Hiệu chuẩn Phân bổ 80/20 (Biên an toàn 10%)** | **0.848521** | **0.873637** | **43** | **10** | **53** | **0.1767** | **0.455236** | **0.169221** | **0.2997** | **0.2080** | **7** |
| `DEV_CAL_CONSERVATIVE_15PCT` | Hiệu chuẩn Phân bổ 80/20 (Biên an toàn 15%) | 0.854877 | 0.873637 | 40 | 10 | 50 | 0.1667 | 0.455236 | 0.169728 | 0.3012 | 0.2080 | 6 |
| `DEV_CAL_CONSERVATIVE_20PCT` | Hiệu chuẩn Phân bổ 80/20 (Biên an toàn 20%) | 0.856954 | 0.907806 | 38 | 9 | 47 | 0.1567 | 0.455236 | 0.168700 | 0.2964 | 0.2097 | 6 |
| `DEV_CAL_JOINT_OPT_10PCT` | Tối ưu Macro-3 F1 liên kết ($\le 54$ FP tổng) | 0.853190 | 0.831490 | 42 | 12 | 54 | 0.1800 | 0.455236 | **0.186570** | 0.3002 | 0.2595 | 6 |
| `DEV_CAL_NOMINAL_UNMARGINED` | Hiệu chuẩn Danh định 0% Biên ($B=0.20$) | 0.837504 | 0.831490 | 48 | 12 | 60 | 0.2000 | 0.455236 | **0.193049** | 0.3196 | 0.2595 | 9 |

---

## 6. Phân tích Chuyên sâu Từng Lớp Lỗi & Đánh giá Khoa học

### 6.1 Hiệu năng Từng Lớp của Chính sách Đóng băng Chính thức (`DEV_CAL_CONSERVATIVE_10PCT`)

Khi áp dụng cặp ngưỡng vận hành được chọn ($\mathbf{\tau_T = 0.848521}, \mathbf{\tau_G = 0.873637}$):

#### 1. Lỗi Bỏ sót (OMISSION):
- Số lượng phát hiện đúng (TP): **89** / 462 lỗi.
- Số lượng báo giả (FP): **43** (gồm 36 cảnh báo trên âm vị Đúng và 7 cảnh báo trên âm vị Thay thế).
- Số lượng bỏ lỡ (FN): **373** / 462 lỗi.
- Độ chính xác (Precision): **$67.42\%$** ($89 / 132$). Cứ 3 cảnh báo bỏ sót phát ra thì có hơn 2 cảnh báo chính xác tuyệt đối.
- Độ thu hồi (Recall): **$19.26\%$** ($89 / 462$).
- Điểm F1: **$0.299663$** (vượt trội so với $0.296422$ của tham chiếu lịch sử).
- Average Precision ($AP_O$): **$0.572058$** (độ phân tách âm học của nhánh Omission trên DEV_CAL rất sắc nét).

#### 2. Lỗi Chèn Khoảng trống (INSERTION):
- Số lượng phát hiện đúng (TP): **13** / 102 lỗi.
- Số lượng báo giả (FP): **10** / 10.241 khoảng trống âm tính.
- Số lượng bỏ lỡ (FN): **89** / 102 lỗi.
- Độ chính xác (Precision): **$56.52\%$** ($13 / 23$).
- Độ thu hồi (Recall): **$12.75\%$** ($13 / 102$).
- Điểm F1: **$0.208000$** (tăng so với $0.206349$ của tham chiếu lịch sử).
- Average Precision ($AP_I$): **$0.242311$**.

#### 3. Lỗi Thay thế (SUBSTITUTION):
- Số lượng phát hiện đúng (TP): **0** / 1.781 lỗi.
- Số lượng báo giả (FP): **0**.
- Số lượng bỏ lỡ (FN): **1.781** lỗi.
- Độ chính xác: `NOT_ESTIMABLE` (không có cảnh báo nào phát ra).
- Độ thu hồi: **$0.00\%$**. Điểm F1: **$0.0000$**.
- Average Precision ($AP_S$): **$0.551340$**.
- **Giải trình Toán học và Âm học:**
  - Điểm tin cậy hợp nhất tối đa của lớp Thay thế trong `DEV_CAL` (khi $p_{\text{sub}} > p_{\text{omi}}$) chỉ đạt tối đa là **$0.669937$** (99th percentile là $0.4907$).
  - Nếu cưỡng ép hạ ngưỡng $\tau_T$ xuống dưới $0.67$ để thu hồi lỗi thay thế, số lượng cảnh báo giả sẽ tăng vọt lên **$140$ cảnh báo giả** tại $\tau_T = 0.65$ ($\text{FWR} = 0.467$, vượt hơn gấp đôi ngân sách tối đa $0.20$).
  - Thuật toán tối ưu hóa có điều kiện toán học đã hành xử hoàn toàn chuẩn xác: để bảo vệ trần ngân sách sư phạm không bị phá vỡ, mô hình buộc phải "kiềm chế" (abstain) trên lớp Thay thế ở chế độ ngân sách hẹp $B=0.20$.

#### 4. Cảnh báo Sai Loại Âm học (Wrong-Type Warnings $S \leftrightarrow O$):
- Trên toàn bộ $10.043$ âm vị mục tiêu, chỉ có duy nhất **7 trường hợp** mô hình phát cảnh báo Omission nhưng nhãn thực tế là Substitution (chiếm tỷ lệ cực nhỏ **$0.0697\%$**).
- Không có bất kỳ cảnh báo giả nào phát sai từ Omission sang Substitution.

### 6.2 So sánh với Chính sách Tối ưu Liên kết F1 (`DEV_CAL_JOINT_OPT_10PCT`)
Nếu cho phép thuật toán tự do điều phối số lượng cảnh báo giả giữa Target và Gap trong cùng hạn mức $54$ FP ($B_{eff} = 0.180$):
- Cặp ngưỡng thu được là $\tau_T = 0.853190$ ($42$ target FP) và $\tau_G = 0.831490$ ($12$ gap FP).
- Macro-3 F1 tăng vọt lên **$0.186570$** (nhờ $TP_{\text{ins}}$ tăng từ 13 lên 17, đưa Insertion F1 từ $0.2080$ lên **$0.2595$**).
- Tỷ lệ phân bổ thực tế của phương án này là $42 / 54 = 77.8\%$ Target và $12 / 54 = 22.2\%$ Gap, bám rất sát tỷ lệ danh định 80/20. Cấu hình này cũng đã được lưu trữ sẵn sàng trong tệp JSON đóng băng dưới dạng phương án tùy chọn mở rộng.

---

## 7. Chi tiết Tệp Chính sách Đóng băng (`FROZEN_CALIBRATED_POLICY.json`)

Toàn bộ thông số chính sách đã được serialize và niêm phong tại:
`experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json`  
Mã băm SHA-256 niêm phong: `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1`

```json
{
  "schema_version": "LANGFENS-CALIBRATED-POLICY-V1.0",
  "provenance": {
    "model_name": "LANGFENS-SM13-FUSION-V1",
    "model_train_regime": "Full-TRAIN 12 Speakers (1,800 utterances)",
    "primary_predetermined_seed": 101,
    "calibration_dataset": "DEV_CAL",
    "calibration_manifest": "data/derived/l2arctic/manifests/dev_cal.jsonl",
    "calibration_manifest_sha256": "87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6",
    "predictions_source": "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz",
    "predictions_sha256": "73f7cc8335a761bedb9a8f1346dd7416de648a98eebdd9a1a802c2d628776841",
    "checkpoint_hashes": {
      "target_checkpoint": "experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt",
      "target_checkpoint_sha256": "551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6",
      "gap_checkpoint": "experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt",
      "gap_checkpoint_sha256": "34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98",
      "ctc_checkpoint": "experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt",
      "ctc_checkpoint_sha256": "63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a"
    },
    "evaluator_version": "EVAL-V3.1",
    "calibration_timestamp": "2026-11-11T12:00:00Z",
    "auditor": "Senior Speech ML Engineer & Scientific Auditor",
    "firewall_confirmation": {
      "dev_search_used_for_tuning": false,
      "test_sealed_accessed": false,
      "test_sealed_sha256_verified": "a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6",
      "train_in_sample_tuning": false
    }
  },
  "operating_parameters": {
    "nominal_joint_fwr_budget": 0.20,
    "target_gap_nominal_allocation": "80/20",
    "conservative_margin": 0.10,
    "effective_joint_fwr_budget": 0.18,
    "calibration_population": {
      "utterances": 300,
      "speakers": [
        "ERMS",
        "THV"
      ],
      "target_events": 10043,
      "gap_events": 10343
    }
  },
  "target": {
    "threshold": 0.848521411895752,
    "score_semantics": "max(p_substitution, p_omission) from fixed 50/50 fused target simplex",
    "decision_rule": "pred_label = argmax(p_sub, p_omi) + 1 if conf >= threshold else 0 (CORRECT)",
    "nominal_budget_per_utt": 0.16,
    "effective_budget_per_utt": 0.14333333333333334,
    "allowed_false_warnings": 43,
    "realized_dev_cal_false_warnings": 43,
    "realized_dev_cal_fwr": 0.14333333333333334
  },
  "gap": {
    "threshold": 0.8736369609832764,
    "score_semantics": "p_insertion from ScalarMixingGapHead",
    "decision_rule": "pred_has_insertion = (p_insertion >= threshold)",
    "nominal_budget_per_utt": 0.04,
    "effective_budget_per_utt": 0.03333333333333333,
    "allowed_false_warnings": 10,
    "realized_dev_cal_false_warnings": 10,
    "realized_dev_cal_fwr": 0.03333333333333333
  },
  "joint_realized_metrics_dev_cal": {
    "joint_fwr": 0.17666666666666667,
    "total_false_warnings": 53,
    "false_alarms_gold_correct": 36,
    "wrong_type_warnings": 7,
    "macro3_error_f1": 0.16922108139572412,
    "macro3_error_ap": 0.4552363833949778,
    "substitution_f1": 0.0,
    "omission_f1": 0.29966329966329966,
    "insertion_f1": 0.208
  },
  "policies": {
    "fixed/0.2": {
      "description": "Primary frozen operating policy for prospective evaluation (10% conservative margin)",
      "target": {
        "threshold": 0.848521411895752
      },
      "gap": {
        "threshold": 0.8736369609832764
      },
      "nominal_budget": 0.2,
      "effective_budget": 0.18,
      "conservative_margin": 0.1,
      "realized_fwr": 0.17666666666666667,
      "macro3_error_f1": 0.16922108139572412
    },
    "dev_cal_nominal_unmargined/0.2": {
      "description": "Unmargined nominal calibration at B=0.20 (48 T / 12 G FP limit)",
      "target": {
        "threshold": 0.8375040292739868
      },
      "gap": {
        "threshold": 0.8314900398254395
      },
      "nominal_budget": 0.2,
      "effective_budget": 0.2,
      "conservative_margin": 0.0,
      "realized_fwr": 0.2,
      "macro3_error_f1": 0.19304886586096892
    },
    "dev_cal_joint_opt/0.2": {
      "description": "Joint Macro-3 F1 optimal calibration at B_eff=0.18 (42 T / 12 G FP)",
      "target": {
        "threshold": 0.8531895279884338
      },
      "gap": {
        "threshold": 0.8314900398254395
      },
      "nominal_budget": 0.2,
      "effective_budget": 0.18,
      "conservative_margin": 0.1,
      "realized_fwr": 0.18,
      "macro3_error_f1": 0.18657020626563514
    },
    "historical_reference/0.2": {
      "description": "Historical uncalibrated Phase 1 Seed 101 Outer 0 reference policy",
      "target": {
        "threshold": 0.8569530248641968
      },
      "gap": {
        "threshold": 0.8660814762115479
      },
      "nominal_budget": 0.2,
      "effective_budget": 0.2,
      "conservative_margin": 0.0,
      "realized_fwr": 0.16333333333333333,
      "macro3_error_f1": 0.167590564524125
    }
  }
}
```

---

## 8. Kết quả Kiểm thử Hồi quy Toàn diện (19/19 Tests Pass)

Toàn bộ hệ thống kiểm thử hồi quy tự động đạt tỷ lệ thành công 100%:

```
============================= test session starts ==============================
platform linux -- Python 3.11.17, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/khoa/Projects/New-Method-Research
configfile: pytest.ini
plugins: platformdirs-4.14.4, anyio-4.16.0
collected 19 items

tests/test_hard_negative_mining.py ..                                    [ 10%]
tests/test_phase2_artifacts.py ....                                      [ 31%]
tests/test_policy_lookup_regression.py .....                             [ 57%]
tests/test_sm13_calibration_regression.py .....                          [ 84%]
tests/test_upr02_engine_and_evaluator.py ...                             [100%]

============================== 19 passed in 2.48s ==============================
```

### Bảng tóm tắt kết quả kiểm thử:

| Tệp kiểm thử | Số ca | Kết quả | Các nội dung được chứng minh |
|---|:---:|:---:|---|
| `tests/test_sm13_calibration_regression.py` | 5 | **5/5 PASS** | - Tính toàn vẹn mật mã của `FROZEN_CALIBRATED_POLICY.json`<br>- Khả năng nạp chính sách tất định và ném lỗi khi key sai<br>- Logic quyết định biên chính xác tuyệt đối<br>- Đếm cảnh báo giả khớp chuẩn EVAL-V3.1 (53 FP, FWR=0.1767)<br>- Tái lập 100% kết quả tham chiếu lịch sử (49 FP, FWR=0.1633) |
| `tests/test_policy_lookup_regression.py` | 5 | **5/5 PASS** | - Loại bỏ fallback hardcode 0.70/0.50<br>- Loại bỏ fallback ngầm chéo seed (seed=999 ném `FileNotFoundError`)<br>- Bắt buộc ném `ValueError` khi JSON hỏng hoặc thiếu trường<br>- Ràng buộc kiểm tra nghiêm ngặt đầu vào `apply_warning_policy` |
| `tests/test_hard_negative_mining.py` | 2 | **2/2 PASS** | - Khai phá mẫu âm tính khó trong huấn luyện |
| `tests/test_phase2_artifacts.py` | 4 | **4/4 PASS** | - Tính toàn vẹn của artifacts pha 2 |
| `tests/test_upr02_engine_and_evaluator.py` | 3 | **3/3 PASS** | - Động cơ tính điểm và tích hợp chuẩn EVAL-V3 |

---

## 9. Danh mục Sản phẩm & Bảng Niêm phong SHA-256 (Artifact Ledger)

Toàn bộ các tệp tin sản phẩm thuộc chiến dịch `SM13-DEV-CALIBRATION-01` được lưu trữ tại `experiments/rq1_sm13_dev_calibration_01/`:

| Tên tệp tin | Kích thước | Mã băm SHA-256 xác thực | Vai trò và Mô tả |
|---|:---:|---|---|
| `FROZEN_CALIBRATED_POLICY.json` | 5.08 KB | `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1` | File chính sách đóng băng chính thức của mô hình |
| `CALIBRATION_REPORT.md` | 18.9 KB | `e940240fc3c61768a08e8dbfc76aa86f97fb15f7da7bc6475b431601d4f24229` | Báo cáo khoa học chi tiết của chiến dịch |
| `THRESHOLD_COMPARISON.csv` | 3.34 KB | `7f98c4a32e1062f89fde1d57dfc3367c06a7c9ad7876afb4799aaebe5acd46bb` | Bảng ma trận so sánh chi tiết các chính sách |
| `REGRESSION_RESULTS.md` | 9.47 KB | `5a091d45c0bed90416d25e1a3bb64b365a44a988ac4899802d75c2c20dc0e0c6` | Biên bản chi tiết kết quả kiểm thử hồi quy tự động |
| `calibrate_dev.py` | 24.5 KB | `476515fefe6ae408649b036a6744a7a87dde56a5f80a6a6929b87159259c0252` | Mã nguồn thực thi quy trình hiệu chuẩn và đánh giá |
| `calibration.log` | 4.11 KB | `e82e66d671706799de55f5be9cdc0526caaee9b1739857e2b68e6c59bcbb0a53` | Nhật ký ghi nhận thực thi từng bước |
| `MANIFEST_SHA256.json` | 598 B | `8f5ff60cf47690bc4220ae798a6bfab9f9087c53d0dd6d5d5a7114b306b3bc5f` | File niêm phong toàn vẹn SHA-256 của toàn bộ thư mục |

---

## 10. Tuyên bố Liêm chính Khoa học & Khuyến nghị Vận hành

1. **Bản chất của các số liệu trên DEV_CAL:**
   Mọi chỉ số F1, Precision, Recall và FWR được báo cáo trong tài liệu này là **chỉ số chẩn đoán hiệu chuẩn nội bộ (calibration diagnostics)**. Tuyệt đối **không được coi đây là bằng chứng độc lập về khả năng tổng quát hóa (independent generalization evidence)**.
2. **Khóa chặt Ranh giới `TEST_SEALED`:**
   Tập dữ liệu kiểm tra kín `TEST_SEALED` (`data/derived/l2arctic/manifests/test_sealed.jsonl`, SHA-256: `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`) được niêm phong tuyệt đối 100%. Không có bất kỳ truy cập đọc nhãn, đọc âm thanh, hay suy luận nào được thực hiện trên phân vùng này.
3. **Kết luận Kiểm toán:**
   Chiến dịch hiệu chuẩn `LANGFENS-SM13-DEV-CALIBRATION-01` đã hoàn thành xuất sắc nhiệm vụ được giao, loại bỏ hoàn toàn các lỗi kỹ thuật và đóng băng chính sách vận hành an toàn. Tiến trình dừng lại tại đây và chờ lệnh tiếp theo từ Operator.
