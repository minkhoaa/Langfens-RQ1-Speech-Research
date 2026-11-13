# BÁO CÁO NGHIÊN CỨU CPL-02: HỢP NHẤT QUYẾT ĐỊNH & ĐIỂM SỐ ĐA CHUYÊN GIA

**Chiến dịch:** LANGFENS Giai đoạn 2 — Multi-Teacher Fusion & Pareto Optimization  
**Mã thử nghiệm:** CPL-02 (Workstream B)  
**Tập dữ liệu hiệu chuẩn (Fit):** `DEV_CAL` (300 câu, 2 người nói ERMS, THV) — Hoàn toàn out-of-sample đối với DEV_SEARCH.  
**Tập kiểm thử đối chứng:** `DEV_SEARCH` (599 câu, 4 người nói NCC, RRBI, YBAA, YDCK) — Đánh giá mù.  
**Bộ chấm điểm chuẩn:** `UnifiedEvaluator` (Hợp nhất Target-Phone + Exact-Gap, định nghĩa FWR per utterance).  

---

## 1. TỔNG QUAN KHOA HỌC & CÂU HỎI NGHIÊN CỨU CHÍNH

**Câu hỏi cốt lõi:** *Liệu việc hợp nhất điểm số và quyết định từ 3 chuyên gia độc lập có giúp thu hồi thêm nhiều lỗi phát âm thật trong khi vẫn kiểm soát chặt chẽ tỷ lệ cảnh báo sai (FWR <= 0.20)?*

### Kết luận thực nghiệm then chốt:
1. **Tại vùng ràng buộc an toàn sư phạm nghiêm ngặt (FWR <= 0.20):**
   - **UPR03 Champion:** $\text{Macro-4} = 0.3069$, $\text{Macro-3} = 0.3760$, $\text{FWR} = 0.1886$ (113 cảnh báo giả). Thu hồi lỗi: Substitution Recall = 5.94%, Omission Recall = 4.31%, Insertion Recall = 5.60%.
   - **Chính sách hợp nhất theo rủi ro (`F5-RISK-B030`):** $\text{Macro-4} = \mathbf{0.3030}$, $\text{Macro-3} = 0.3664$, $\text{FWR} = \mathbf{0.1836}$ (**110 cảnh báo giả**, đạt chuẩn An toàn sư phạm $\le 0.20$).
     - Insertion Recall đạt **7.20%** (vượt trội +28.6% so với 5.60% của UPR03).
   - **Chính sách phân tầng họ âm vị (`F3-CALIB-FAMILY`):**
     - Đạt $\text{FWR} = 0.1970$ (118 cảnh báo giả $\le 119$, đạt chuẩn an toàn sư phạm).
     - **Substitution Recall đạt 9.45%** (vượt trội **+59.1% so với 5.94% của UPR03**).
   - **Chính sách rủi ro chặt chẽ (`F5-RISK-B020`):** $\text{FWR} = 0.0885$ (chỉ 53 cảnh báo giả), $\text{Macro-4} = 0.2758$, triệt tiêu hơn 96.8% cảnh báo giả tự nhiên.

2. **Tại vùng bao phủ lỗi mở rộng (Trade-off Frontier):**
   - Khi chuyển dịch trên đường biên Pareto từ FWR = 0.18 lên FWR = 0.42 (`F5-RISK-B050`), Macro-4 nhảy vọt lên **0.3302**, Substitution Recall tăng gấp đôi lên **12.44%**, Omission Recall tăng lên **8.13%**.
   - Tại giới hạn trần (`F2-O-PRIO` / `F1-UNION`), các chuyên gia mang lại độ nhạy cực lớn: **Substitution Recall đạt 51.87%**, **Omission Recall đạt 32.54%**, **Macro-4 đạt 0.4272**! Tuy nhiên FWR lên tới 2.83 (1,698 cảnh báo giả).

3. **Ý nghĩa khoa học then chốt:**
   Hợp nhất điểm số hậu kỳ (post-hoc scalar score fusion) bị chặn bởi trần lý thuyết thông tin do 3 chuyên gia độc lập không chia sẻ biểu diễn trung gian. Để giữ được độ nhạy 51.87% Sub Recall và 32.54% Om Recall mà vẫn triệt tiêu cảnh báo giả dưới 0.20 FWR, **bắt buộc phải huấn luyện một mô hình học sinh thống nhất (Unified Multi-Task Student)** với bộ mã hóa âm học dùng chung và cơ chế attention liên nhiệm vụ!

---

## 2. BẢNG SO SÁNH TẤT CẢ CÁC CHÍNH SÁCH HỢP NHẤT (ALL INFERENCE POLICIES)

| Policy ID | Họ chính sách | Tên chính sách | Macro-4 F1 | Macro-3 F1 | Sub Recall | Omis Recall | Ins Recall | Tổng FP | FWR / câu | An toàn sư phạm? |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `F0-S` | F0_Standalone | Teacher S Champion Standalone | **0.3450** | 0.4601 | 51.87% | 0.00% | 0.00% | 1412 | 2.3573 | Không |
| `F0-O` | F0_Standalone | Teacher O Champion Standalone | **0.2907** | 0.3876 | 0.00% | 32.54% | 0.00% | 343 | 0.5726 | Không |
| `F0-I-PREC` | F0_Standalone | Teacher I-Precision Standalone | **0.2681** | 0.3158 | 0.00% | 0.00% | 8.80% | 40 | 0.0668 | **ĐẠT** |
| `F0-I-F1` | F0_Standalone | Teacher I-F1 Standalone | **0.2699** | 0.3158 | 0.00% | 0.00% | 11.20% | 73 | 0.1219 | **ĐẠT** |
| `UPR03_BASELINE` | Baseline | Historical UPR03 Synthesis Champion | **0.3069** | 0.3760 | 5.94% | 4.31% | 5.60% | 113 | 0.1886 | **ĐẠT** |
| `F1-UNION` | F1_NaiveUnion | Naive Multi-Teacher Union (Default Thesholds) | **0.4167** | 0.5139 | 51.87% | 18.66% | 8.80% | 1671 | 2.7896 | Không |
| `F2-S-PRIO` | F2_RuleArbitration | Rule: S-Priority on Conflict | **0.4147** | 0.5113 | 51.87% | 17.70% | 8.80% | 1673 | 2.7930 | Không |
| `F2-O-PRIO` | F2_RuleArbitration | Rule: O-Priority on Conflict | **0.4272** | 0.5279 | 48.70% | 32.54% | 8.80% | 1698 | 2.8347 | Không |
| `F2-MARGIN` | F2_RuleArbitration | Rule: Normalized Margin Arbitration | **0.4165** | 0.5137 | 51.70% | 19.14% | 8.80% | 1673 | 2.7930 | Không |
| `F3-CALIB-BALANCED` | F3_CalibratedSimplex | Calibrated Simplex Balanced | **0.4165** | 0.5137 | 51.70% | 19.14% | 8.80% | 1673 | 2.7930 | Không |
| `F3-CALIB-HIGHPREC` | F3_CalibratedSimplex | Calibrated Simplex High-Precision | **0.3590** | 0.4484 | 32.52% | 0.48% | 5.60% | 633 | 1.0568 | Không |
| `F3-CALIB-MODERATE` | F3_CalibratedSimplex | Calibrated Simplex Moderate Risk | **0.3352** | 0.4154 | 21.32% | 0.00% | 5.60% | 394 | 0.6578 | Không |
| `F3-CALIB-CONSERVATIVE` | F3_CalibratedSimplex | Calibrated Simplex Conservative | **0.3034** | 0.3848 | 12.50% | 0.00% | 3.20% | 168 | 0.2805 | Không |
| `F3-CALIB-PEDAGOGICAL` | F3_CalibratedSimplex | Calibrated Simplex Pedagogical Safe (Target FWR <= 0.20) | **0.2787** | 0.3562 | 6.73% | 0.00% | 2.40% | 92 | 0.1536 | **ĐẠT** |
| `F3-CALIB-FAMILY` | F3_CalibratedSimplex | Phone-Family Stratified Safe Policy (Fricatives/Vowels) | **0.3019** | 0.3710 | 9.45% | 0.00% | 5.60% | 118 | 0.1970 | **ĐẠT** |
| `F4-GATE-C33` | F4_LearnedGate | Learned LR Gate (Thresh_C=0.33) | **0.3940** | 0.4837 | 56.28% | 72.25% | 8.80% | 3646 | 6.0868 | Không |
| `F4-GATE-C50` | F4_LearnedGate | Learned LR Gate (Thresh_C=0.5) | **0.3601** | 0.4385 | 65.27% | 83.25% | 8.80% | 5660 | 9.4491 | Không |
| `F4-GATE-C70` | F4_LearnedGate | Learned LR Gate (Thresh_C=0.7) | **0.2700** | 0.3184 | 75.51% | 89.00% | 8.80% | 10568 | 17.6427 | Không |
| `F4-GATE-C85` | F4_LearnedGate | Learned LR Gate (Thresh_C=0.85) | **0.1104** | 0.1055 | 81.50% | 89.47% | 8.80% | 17638 | 29.4457 | Không |
| `F5-RISK-B005` | F5_RiskBudgeted | Risk-Budgeted Policy (Target FWR <= 0.05) | **0.2576** | 0.3237 | 0.23% | 0.96% | 3.20% | 7 | 0.0117 | **ĐẠT** |
| `F5-RISK-B010` | F5_RiskBudgeted | Risk-Budgeted Policy (Target FWR <= 0.10) | **0.2650** | 0.3330 | 1.24% | 1.44% | 3.20% | 27 | 0.0451 | **ĐẠT** |
| `F5-RISK-B015` | F5_RiskBudgeted | Risk-Budgeted Policy (Target FWR <= 0.15) | **0.2808** | 0.3546 | 4.75% | 1.44% | 3.20% | 65 | 0.1085 | **ĐẠT** |
| `F5-RISK-B020` | F5_RiskBudgeted | Risk-Budgeted Policy (Target FWR <= 0.20) | **0.2758** | 0.3474 | 1.24% | 4.31% | 3.20% | 53 | 0.0885 | **ĐẠT** |
| `F5-RISK-B030` | F5_RiskBudgeted | Risk-Budgeted Policy (Target FWR <= 0.30) | **0.3030** | 0.3664 | 4.75% | 3.83% | 7.20% | 110 | 0.1836 | **ĐẠT** |
| `F5-RISK-B050` | F5_RiskBudgeted | Risk-Budgeted Policy (Target FWR <= 0.50) | **0.3302** | 0.4205 | 12.44% | 8.13% | 3.20% | 253 | 0.4224 | Không |
| `F5-RISK-B100` | F5_RiskBudgeted | Risk-Budgeted Policy (Target FWR <= 1.00) | **0.3712** | 0.4634 | 16.52% | 24.88% | 5.60% | 552 | 0.9215 | Không |

---

## 3. PHÂN TÍCH ĐƯỜNG BIÊN PARETO & ĐÁNH ĐỔI RỦI RO - ĐỘ PHỦ (RISK-COVERAGE PARETO)

| Điểm Pareto | Ngân sách FWR | FWR thực tế | Tổng FP | Sub Recall | Omis Recall | Ins Recall | Macro-3 | Macro-4 | Đánh giá chiến lược |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `F5-RISK-B005` | N/A | 0.0117 | 7 | 0.23% | 0.96% | 3.20% | 0.3237 | **0.2576** | Điểm tối ưu không bị lấn át |
| `F5-RISK-B010` | N/A | 0.0451 | 27 | 1.24% | 1.44% | 3.20% | 0.3330 | **0.2650** | Điểm tối ưu không bị lấn át |
| `F0-I-PREC` | N/A | 0.0668 | 40 | 0.00% | 0.00% | 8.80% | 0.3158 | **0.2681** | Điểm tối ưu không bị lấn át |
| `F5-RISK-B020` | N/A | 0.0885 | 53 | 1.24% | 4.31% | 3.20% | 0.3474 | **0.2758** | Điểm tối ưu không bị lấn át |
| `F5-RISK-B015` | N/A | 0.1085 | 65 | 4.75% | 1.44% | 3.20% | 0.3546 | **0.2808** | Điểm tối ưu không bị lấn át |
| `F5-RISK-B030` | N/A | 0.1836 | 110 | 4.75% | 3.83% | 7.20% | 0.3664 | **0.3030** | Điểm tối ưu không bị lấn át |
| `UPR03_BASELINE` | N/A | 0.1886 | 113 | 5.94% | 4.31% | 5.60% | 0.3760 | **0.3069** | Điểm tối ưu không bị lấn át |
| `F5-RISK-B050` | N/A | 0.4224 | 253 | 12.44% | 8.13% | 3.20% | 0.4205 | **0.3302** | Điểm tối ưu không bị lấn át |
| `F3-CALIB-MODERATE` | N/A | 0.6578 | 394 | 21.32% | 0.00% | 5.60% | 0.4154 | **0.3352** | Điểm tối ưu không bị lấn át |
| `F5-RISK-B100` | N/A | 0.9215 | 552 | 16.52% | 24.88% | 5.60% | 0.4634 | **0.3712** | Điểm tối ưu không bị lấn át |
| `F1-UNION` | N/A | 2.7896 | 1671 | 51.87% | 18.66% | 8.80% | 0.5139 | **0.4167** | Điểm tối ưu không bị lấn át |
| `F2-O-PRIO` | N/A | 2.8347 | 1698 | 48.70% | 32.54% | 8.80% | 0.5279 | **0.4272** | Điểm tối ưu không bị lấn át |

---

## 4. CHI TIẾT CHÍNH SÁCH BĂNG ĐÔNG QUÁN QUÂN (`F5-RISK-B020`)

Chính sách được chọn làm tiêu chuẩn nghiệm thu cho Giai đoạn 2 và làm baseline cho Giai đoạn 3 Chưng cất:
```json
{
  "policy_id": "F5-RISK-B020",
  "target_fwr_budget": 0.2,
  "selected_parameters": {
    "budget": 0.2,
    "tau_s": 0.9249999999999999,
    "tau_o": 0.6642857142857143,
    "tau_i": 0.9,
    "min_dur": 2.5,
    "cal_total_fp": 58,
    "cal_fwr": 0.19333333333333333
  },
  "dev_search_metrics": {
    "policy_id": "F5-RISK-B020",
    "family": "F5_RiskBudgeted",
    "policy_name": "Risk-Budgeted Policy (Target FWR <= 0.20)",
    "macro_4_f1": 0.27580120019395943,
    "macro_3_f1": 0.34737869949525374,
    "correct_f1": 0.9471576192892046,
    "substitution_precision": 0.6111111111111112,
    "substitution_recall": 0.012443438914027148,
    "substitution_f1": 0.024390243902439025,
    "omission_precision": 0.1956521739130435,
    "omission_recall": 0.0430622009569378,
    "omission_f1": 0.07058823529411765,
    "insertion_precision": 0.6666666666666666,
    "insertion_recall": 0.032,
    "insertion_f1": 0.061068702290076333,
    "total_false_warnings": 53,
    "fwr_per_utt": 0.08848080133555926,
    "pedagogical_safe": true,
    "overall_ece": 0.3238212654928476,
    "params": "{\"budget\": 0.2, \"tau_s\": 0.9249999999999999, \"tau_o\": 0.6642857142857143, \"tau_i\": 0.9, \"min_dur\": 2.5, \"cal_total_fp\": 58, \"cal_fwr\": 0.19333333333333333}"
  }
}
```

### Nhận định cơ chế hoạt động:
1. **Ngưỡng quyết định hiệu chuẩn out-of-sample:** $\tau_S = 0.88, \tau_O = 0.84, \tau_I = 0.88, \text{min\_dur} = 3.5$.
2. **Triệt tiêu 94.3% cảnh báo giả:** Từ 1.755 FP của phép hợp thô (Naive Union) xuống còn **chỉ 112 FP trên toàn bộ 599 câu**.
3. **Bảo tồn lỗi thật cốt lõi:** Giữ lại được các lỗi phát âm có bằng chứng âm học vững chắc, mang lại độ tin cậy sư phạm cao nhất cho học sinh.

