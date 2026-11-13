# HỢP ĐỒNG BÀN GIAO CHUYỂN GIAO GIAI ĐOẠN 3: CHƯNG CẤT MÔ HÌNH HỌC SINH THỐNG NHẤT
## LANGFENS PHASE 3 HANDOFF CONTRACT: UNIFIED MULTI-TASK STUDENT DISTILLATION

- **Đơn vị phát hành:** Autonomous Scientific Research Director
- **Giai đoạn chuyển giao:** Hoàn tất Giai đoạn 2 $\to$ Sẵn sàng khởi động Giai đoạn 3
- **Ngày phát hành:** 2026-11-06
- **Git Commit Provenance:** `41f02ad9ad4ea2df4734ebc38bc2052503586fb6`
- **Mục tiêu Giai đoạn 3:** Huấn luyện một mô hình học sinh duy nhất (Single Unified Student Model), nhận âm thanh và chuỗi âm vị tham chiếu, xuất đồng thời nhãn Target Phone (CORRECT, SUBSTITUTION, OMISSION) và Inter-phone Gap (NO_INSERTION, INSERTION), chưng cất tri thức từ 3 chuyên gia đã đóng băng.

---

## 1. MÃ BĂM VÀ ĐỊA CHỈ CHECKPOINT CỦA CÁC CHUYÊN GIA ĐÓNG BĂNG (FROZEN TEACHER CHECKSUMS)

| Chuyên gia | File Checkpoint | Checkpoint SHA-256 | Sổ cái dự đoán Dev-Search SHA-256 |
|---|---|---|---|
| **Teacher S Champion** | `models/teachers/TEACHER_S_CHAMPION_checkpoint.pt` | `47e804011f6fa744bf09dbd13896cfcd9ef874534d9eaa9f403e37265b000510` | `6816101dafb88268f3c725e74ab4407f01ec063e5811f4ebb5ddb90f455082e4` |
| **Teacher O Champion** | `models/teachers/TEACHER_O_CHAMPION_checkpoint.pt` | `80618baaefe59eff0da3db0a9a64941fb2bdb1a10146c60ee93b286f9634db76` | `2a1f1f31ee4df8c9fac33890b0d7027851f73184659a7e89bee5e7b766f1ba66` |
| **Teacher I Champion** | `models/teachers/TEACHER_I_CHAMPION_checkpoint.pt` | `301a3fc7e8f627a058c165be53cd024a9aaa3e60b0f598e36170322e1d0f31bd` | `a1de03b87f528a1b1cf68c77b3ca5f4194562ed8c5cabaa8c9a260d6ef030e53` |
| **Teacher I EXP01** | `models/teachers/TEACHER_I_EXP01_checkpoint.pt` | `98da8bb5ff756df026e7ec17e17b4c5c60b8149c6cdba8c269005831238c6220` | `0621a6d4a0cd55b69844f7901b06bd91554f22fd9b4d9788281704deaba8bcd2` |

---

## 2. HỢP ĐỒNG SUY LUẬN CHUYÊN GIA (TEACHER INFERENCE CONTRACTS)

### 2.1. Teacher S Champion
- **Kiến trúc:** `TeacherS_ChampionNet` (Bilinear Prototype Contrast + Adaptive Temperature Scaling, $in\_dim=57, hidden\_dim=256, prototypes=39$).
- **Đầu vào:** Vector 57 chiều gồm 39 logit âm vị, 12 đặc trưng wav2vec2 layer 6, và 6 đặc trưng đối kháng cấu âm (Margin, Ratio, Voicing, Manner, Duration, Acoustic Presence).
- **Chuẩn hóa:** Chỉ chuẩn hóa 18 chiều cuối: `X[:, 39:] = (X[:, 39:] - mean) / std` với `mean, std` lưu trong checkpoint.
- **Đầu ra:** Tuple `(logits, features)`, xác suất thay thế $p_{sub} = \text{Softmax}(logits)[:, 1]$.

### 2.2. Teacher O Champion
- **Kiến trúc:** `TeacherO_ChampionNet` (Acoustic Presence-Consistency Net + Multi-Scale Temporal Context, $in\_dim=60, hidden\_dim=256$).
- **Đầu vào:** Vector 60 chiều gồm 39 logit âm vị, 12 đặc trưng layer 6, và 9 đặc trưng hiện diện âm học (Manner flags, Acoustic Presence, Silence Dominance, Short/Zero Duration flags).
- **Chuẩn hóa:** Chỉ chuẩn hóa 21 chiều cuối: `X[:, 39:] = (X[:, 39:] - mean) / std`.
- **Đầu ra:** Logit 2 lớp, xác suất mất âm $p_{omis} = \text{Softmax}(logits)[:, 1]$.

### 2.3. Teacher I Champion
- **Kiến trúc:** `TeacherI_ChampionNet` (Boundary-Aware Epenthesis Gated Residual Net, $in\_dim=25, hidden\_dim=128$).
- **Đầu vào:** Vector 25 chiều gồm 23 đặc trưng biên ngữ cảnh khe hở và 2 đặc trưng tương tác mật độ năng lượng chèn âm.
- **Chuẩn hóa:** Chuẩn hóa toàn bộ vector: $X_{norm} = \text{clip}((X - mean) / std, -10.0, 10.0)$.
- **Đầu ra:** Logit 2 lớp, xác suất chèn âm $p_{ins} = \text{Softmax}(logits)[:, 1]$.

---

## 3. ĐỊNH DẠNG ĐẦU RA CHUẨN TẮC (CANONICAL SCORE SCHEMAS)

Mọi dự đoán của giáo viên và học sinh phải bảo toàn schema:
```json
{
  "event_id": "L2_ARCTIC:{speaker_id}:{utterance_id}:{target|gap}:{slot_idx}",
  "dataset_id": "L2_ARCTIC",
  "speaker_id": "string",
  "utterance_id": "string",
  "phone_index": "int (hoặc gap_index)",
  "reference_phone": "string (ví dụ: 'DH', 'AA', 'T')",
  "score": "float [0.0, 1.0]",
  "uncertainty": "float [0.0, 1.0]",
  "pred_label": "string ('CORRECT', 'SUBSTITUTION', 'OMISSION', 'INSERTION')"
}
```

---

## 4. CÁC ĐỐI CHỨNG HỢP NHẤT ĐƯỢC CHỌN (SELECTED FUSION BASELINES)

Giai đoạn 3 sẽ so sánh trực tiếp mô hình học sinh với 3 đối chứng chuẩn mực đã đóng băng:
1. **`UPR03_SYNTHESIS_CHAMPION`:** Đối chứng lịch sử ($\text{Macro-4} = 0.3069, \text{FWR} = 0.1886, \text{Sub Recall} = 5.94\%, \text{Omis Recall} = 4.31\%, \text{Ins Recall} = 5.60\%$).
2. **`F5-RISK-B030`:** Chính sách hợp nhất tối ưu rủi ro tại Giai đoạn 2 ($\text{Macro-4} = 0.3030, \text{FWR} = 0.1836, \text{Sub Recall} = 4.75\%, \text{Omis Recall} = 3.83\%, \text{Ins Recall} = 7.20\%$).
3. **`F3-CALIB-FAMILY`:** Chính sách phân tầng họ âm vị an toàn ($\text{Macro-4} = 0.3019, \text{FWR} = 0.1970, \text{Sub Recall} = 9.45\%$).

---

## 5. ĐỊNH NGHĨA GIÁM SÁT NHÃN GOLD (GOLD-LABEL SUPERVISION)

- **Target Head (3 lớp rời rạc):**
  - Nhãn 0: `CORRECT` (Âm vị được phát âm đúng theo chuẩn CMU).
  - Nhãn 1: `SUBSTITUTION` (Âm vị bị phát âm thay thế bằng âm vị khác).
  - Nhãn 2: `OMISSION` (Âm vị tham chiếu bị nuốt/bỏ qua hoàn toàn trong âm thanh).
- **Gap Head (2 lớp nhị phân):**
  - Nhãn 0: `NO_INSERTION` (Khe hở bình thường, không có âm chèn).
  - Nhãn 1: `INSERTION` (Khe hở xuất hiện âm phát âm thừa ngoài từ điển).
- **Cam kết Clean-Room:** Tuyệt đối không dùng thời gian TextGrid của realized-phone hoặc nhãn gold khi chạy forward pass suy luận.

---

## 6. THIẾT KẾ KIẾN TRÚC MÔ HÌNH HỌC SINH (PROPOSED STUDENT ARCHITECTURE)

- **Tên mô hình:** `UnifiedPronunciationConformerStudent`
- **Bộ mã hóa âm học (Acoustic Encoder):**
  - 3 tầng Conformer khối (Dim=256, Feed-Forward=1024, Attention Heads=4, Depthwise Conv Kernel=31).
- **Cơ chế Cross-Attention với chuỗi tham chiếu:**
  - Chuỗi âm vị tham chiếu được nhúng qua ma trận Embedding ($39 \to 256$) kết hợp Sinusoidal Positional Encoding.
  - Cross-attention định vị biểu diễn âm học vào từng slot âm vị tham chiếu.
- **Hai đầu ra đa nhiệm chuyên biệt (Multi-Task Heads):**
  - `target_head`: Mạng MLP 2 tầng ($256 \to 128 \to 3$) với Softmax simplex.
  - `gap_head`: Mạng MLP 2 tầng ($256 \to 64 \to 2$) trên biểu diễn ghép nối biên liên âm $(h_i, h_{i+1})$.

---

## 7. CÔNG THỨC HÀM MẤT MÁT CHƯNG CẤT & CÁC NHÁNH ABLATION (KD LOSSES & ABLATIONS)

### 7.1. Hàm mất mát tổng hợp
$$\mathcal{L}_{Total} = \mathcal{L}_{Gold\_Target} + 0.5 \cdot \mathcal{L}_{Gold\_Gap} + 0.4 \cdot \mathcal{L}_{KD\_S} + 0.35 \cdot \mathcal{L}_{KD\_O} + 0.25 \cdot \mathcal{L}_{KD\_I} + 0.10 \cdot \mathcal{L}_{Consistency}$$
- Nhiệt độ chưng cất: $T = 2.0$.
- Trọng số độ tin cậy: Áp dụng $(1.0 - u_{teacher})$ làm trọng số phạt mẫu cho từng âm vị.

### 7.2. Bảy nhánh thực nghiệm đối chứng bắt buộc (Control Experiments)
1. `STU-01-GOLD-ONLY`: Học sinh chỉ học từ nhãn Gold (không có KD).
2. `STU-02-KD-S-ONLY`: Học sinh học Gold + Teacher S KD.
3. `STU-03-KD-O-ONLY`: Học sinh học Gold + Teacher O KD.
4. `STU-04-KD-I-ONLY`: Học sinh học Gold + Teacher I KD.
5. `STU-05-KD-ALL`: Học sinh học đầy đủ cả 3 chuyên gia.
6. `STU-06-CONF-GATED`: Học sinh học có cổng lọc độ bất định ($u < 0.30$).
7. `BASELINE-UPR03`: So sánh đối chứng chuẩn mực lịch sử.

---

## 8. KẾ HOẠCH DỮ LIỆU & CHỐNG RÒ RỈ (TRAINING DATA & LEAKAGE DEFENSE)

- **Tập huấn luyện học sinh:** Chỉ sử dụng tập `TRAIN` (1.800 câu, 12 người nói: ABA, ASI, BWC, EBVS, HJK, HKK, HQTV, LXC, NJS, PNV, SKA, SVBI).
- **Tập hiệu chuẩn ngưỡng & dừng sớm (Validation):** Chỉ sử dụng `DEV_CAL` (300 câu, 2 người nói ERMS, THV).
- **Tập kiểm chứng chốt hạ (Sealed Test Evaluation):** Đánh giá mù trên `DEV_SEARCH` (599 câu, 4 người nói NCC, RRBI, YBAA, YDCK).
- Tuyệt đối không chạm vào `test_sealed.jsonl` hoặc kho nén GPG.

---

## 9. GIAO THỨC CHẤM ĐIỂM & HIỆU CHUẨN (EVALUATION PROTOCOL)

- Sử dụng trực tiếp `UnifiedEvaluator` kế thừa từ `evaluator/unified_evaluator.py`.
- **Tiêu chuẩn nghiệm thu sư phạm:** $\text{FWR} \le 0.20$ (tương đương $\le 119$ cảnh báo giả trên 599 câu).
- Báo cáo đầy đủ Macro-3, Macro-4, và chỉ số Recall từng lớp lỗi (Sub Recall, Omis Recall, Ins Recall).

---

## 10. KẾ HOẠCH TÀI NGUYÊN GPU (GPU RESOURCE PLAN)

- **Thiết bị:** NVIDIA GeForce GTX 1650 4GB VRAM.
- **Tải bộ nhớ ước tính:** ~1.1 GB VRAM (với PyTorch AMP FP16, batch size = 32).
- **Thời lượng ước tính:** ~12 - 15 phút cho mỗi nhánh huấn luyện 30 epochs.
- Toàn bộ 7 nhánh thực nghiệm đối chứng có thể hoàn thành trong vòng 2.5 giờ máy chạy.

---

## 11. CÁC BẤT ĐỊNH KHOA HỌC ĐÃ BIẾT (KNOWN SCIENTIFIC UNCERTAINTIES)

1. **Độ lệch phân phối nuốt âm:** Tỷ lệ Nuốt âm trên `DEV_CAL` là $4.6\%$, trong khi trên `DEV_SEARCH` chỉ là $1.06\%$. Khi hiệu chuẩn ngưỡng trên `DEV_CAL`, cần áp dụng hệ số bù trừ mật độ tiên nghiệm (Prior Odds Adjustment).
2. **Nhiễu nhãn giáo viên ở biên khe hở:** Một số nhãn giả của Teacher I-F1 có thể gây nhiễu cho học sinh. Khắc phục bằng cách lọc cứng thời lượng khe hở $\ge 2.5$ frames trước khi tính hàm mất mát KD.

---

## 12. LỆNH TÁI LẬP TOÀN BỘ KẾT QUẢ GIAI ĐOẠN 2 (REPRODUCTION COMMANDS)

```bash
cd /home/khoa/Projects/New-Method-Research

# 1. Sinh dự đoán dev_cal cho các chuyên gia đóng băng
.venv/bin/python3 research/complementarity/generate_dev_cal_teacher_predictions.py

# 2. Chạy toàn bộ phân tích tính bổ trợ CPL-01 (Workstream A)
.venv/bin/python3 research/complementarity/run_cpl01_joint_analysis.py

# 3. Chạy toàn bộ 26 chính sách hợp nhất quyết định CPL-02 (Workstream B)
.venv/bin/python3 research/complementarity/run_cpl02_joint_fusion.py

# 4. Chạy toàn bộ kiểm định tính khả thi chưng cất CPL-03 (Workstream C)
.venv/bin/python3 research/complementarity/run_cpl03_distillation_feasibility.py
```
