# THIẾT KẾ THỰC NGHIỆM CHƯNG CẤT TRI THỨC (KNOWLEDGE DISTILLATION EXPERIMENT DESIGN)

**Chiến dịch:** LANGFENS Giai đoạn 2 — Workstream C: KD Protocol & Loss Formulation  
**Chuẩn bị cho:** Giai đoạn 3 — Huấn luyện Mô hình Học sinh Thống nhất  

---

## 1. CÔNG THỨC HÀM MẤT MÁT TỔNG HỢP (MULTI-TEACHER DISTILLATION LOSS)

Hàm mất mát tối ưu hóa cho mô hình học sinh gồm 6 thành phần được cân bằng chặt chẽ:
$$\mathcal{L}_{Total} = \mathcal{L}_{Gold\_Target} + \alpha \mathcal{L}_{Gold\_Gap} + \beta_S \mathcal{L}_{KD\_S} + \beta_O \mathcal{L}_{KD\_O} + \beta_I \mathcal{L}_{KD\_I} + \gamma \mathcal{L}_{Consistency}$$

### Chi tiết từng thành phần mất mát:
1. **$\mathcal{L}_{Gold\_Target}$ (Focal Loss trên nhãn Gold âm vị mục tiêu):**
   - Phạt lỗi trên nhãn thật (CORRECT, SUBSTITUTION, OMISSION) với trọng số lớp đối kháng mất cân bằng $[1.0, 3.0, 5.0]$.
2. **$\mathcal{L}_{Gold\_Gap}$ (Focal Loss trên nhãn Gold chèn âm):**
   - Trọng số $\alpha = 0.5$, tiêu chuẩn hóa tỷ lệ mất cân bằng cực đoan (0.61% nhãn dương).
3. **$\mathcal{L}_{KD\_S}$ (Chưng cất Teacher S):**
   - Kullback-Leibler Divergence giữa phân phối logit mềm của Teacher S ($T = 2.0$) và logit của học sinh trên lớp SUBSTITUTION:
     $$\mathcal{L}_{KD\_S} = T^2 \cdot \text{KL}(\sigma(z_S / T) \parallel \sigma(\hat{z}_S / T))$$
   - Trọng số $\beta_S = 0.40$.
4. **$\mathcal{L}_{KD\_O}$ (Chưng cất Teacher O):**
   - KL Divergence trên xác suất mất âm của Teacher O, kết hợp trọng số nghịch đảo độ bất định ($1.0 - u_{omis}$):
     $$\mathcal{L}_{KD\_O} = T^2 \cdot (1.0 - u_O) \cdot \text{KL}(\sigma(z_O / T) \parallel \sigma(\hat{z}_O / T))$$
   - Trọng số $\beta_O = 0.35$.
5. **$\mathcal{L}_{KD\_I}$ (Chưng cất Teacher I Champion):**
   - Binary Cross-Entropy với soft probabilities từ Teacher I-Precision trên các khe hở có thời lượng $\ge 2.5$ frames.
   - Trọng số $\beta_I = 0.25$.
6. **$\mathcal{L}_{Consistency}$ (Ràng buộc nhất quán âm học):**
   - Phạt vi phạm simplex và ràng buộc triệt tiêu tương hỗ: $P(SUB) \cdot P(OM) \to 0$ khi năng lượng âm học ở mức trung gian.
   - Trọng số $\gamma = 0.10$.

---

## 2. MA TRẬN CÁC THỰC NGHIỆM ĐỐI CHỨNG BẮT BUỘC (MANDATORY CONTROL ABLATIONS)

Trong Giai đoạn 3, bắt buộc thực hiện 7 nhánh huấn luyện đối chứng có cùng kiến trúc và hạt giống (Seed = 101):

| Mã thực nghiệm | Mô tả nhánh thực nghiệm | Mục đích khoa học |
|---|---|---|
| `STU-01-GOLD-ONLY` | Học sinh chỉ học từ nhãn Gold (Không có KD) | Baseline kiểm tra độ đóng góp thuần túy của biểu diễn học sinh |
| `STU-02-KD-S-ONLY` | Học sinh học từ Gold + Teacher S KD | Đo lường mức cải thiện riêng biệt của tri thức Thay thế |
| `STU-03-KD-O-ONLY` | Học sinh học từ Gold + Teacher O KD | Đo lường mức cải thiện riêng biệt của tri thức Nuốt âm |
| `STU-04-KD-I-ONLY` | Học sinh học từ Gold + Teacher I KD | Đo lường mức cải thiện riêng biệt của tri thức Chèn âm |
| `STU-05-KD-ALL` | Học sinh học đồng thời cả 3 Chuyên gia (Đầy đủ) | Đánh giá hiệu ứng cộng hưởng toàn diện |
| `STU-06-CONF-GATED` | Học sinh học KD có cổng lọc độ tin cậy ($p > \tau$) | Kiểm tra giả thuyết lọc nhiễu nhãn giáo viên |
| `BASELINE-UPR03` | Mô hình UPR03 Champion lịch sử | Đối chứng chuẩn mực của toàn bộ chiến dịch |

---

## 3. QUY TRÌNH CHỐNG RÒ RỈ DỮ LIỆU HUẤN LUYỆN (OUT-OF-FOLD / LEAKAGE DEFENSE)

1. **Tạo nhãn giám sát mềm (Soft Pseudo-Labels) trên tập TRAIN (1.800 câu):**
   - Thực hiện chia K-Fold theo người nói (Speaker-stratified 4-Fold cross-fitting) trên tập TRAIN nếu sinh lại điểm, HOẶC sử dụng trực tiếp các mô hình giáo viên đã đóng băng với bảo toàn danh tính người nói.
   - Tuyệt đối không đưa mẫu từ `DEV_SEARCH` vào tập học sinh.
2. **Đo đạc kiểm thử mù (Clean-Room Protocol):**
   - Đánh giá chốt hạ trên `DEV_SEARCH` thông qua `UnifiedEvaluator` sau khi mô hình học sinh đã được đóng băng hoàn toàn.
