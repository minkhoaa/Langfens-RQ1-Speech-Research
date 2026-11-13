# TÀI LIỆU BÀN GIAO NGHIÊN CỨU — THIẾT KẾ KIẾN TRÚC MÔ HÌNH THỐNG NHẤT TIẾP THEO
## Next Research Handoff: Unified Pronunciation Error Detector (UPED)

**Người bàn giao:** Autonomous Speech ML Research Directorate  
**Người tiếp nhận:** Đội ngũ phát triển kiến trúc thế hệ tiếp theo (Unified Model Architecture Team)  
**Cơ sở khoa học:** Kết luận từ chiến dịch TMCI sau phát hành EVAL-V3  
**Ngày bàn giao:** 07/11/2026  

---

## 1. MỤC TIÊU CHIẾN LƯỢC

Chiến dịch TMCI đã chứng minh rõ ràng:
1. Việc tiếp tục đi theo hướng **hòa trộn điểm số hậu kỳ (post-hoc score ensembling)** của nhiều mô hình độc lập (như EXP-36 gồm 7 mô hình, EXP-43 gồm 9 mô hình) đã chạm trần bão hòa toán học, đồng thời tạo ra gánh nặng tính toán khổng lồ (yêu cầu chạy 7 lượt forward nơ-ron song song, độ trễ cao, khó tối ưu hóa và không khả thi cho triển khai di động/edge).
2. Mô hình nơ-ron đơn lẻ **EXP-24 (JSAC-Net)** với cơ chế **Monotonic Local Attention** sở hữu tiềm năng vượt bậc (đơn lẻ đạt Macro-3 AP 0.2768, AP Omission 0.2251), và khi được bổ sung thông tin tiên nghiệm từ điển từ **EXP-34**, hiệu năng vọt lên **0.2920 Macro-3 AP**.

Mục tiêu của chiến dịch tiếp theo là: **Xây dựng một mô hình mạng nơ-ron thống nhất duy nhất (Unified Pronunciation Error Detector - UPED)**, tích hợp toàn bộ các ưu điểm cơ bản của Tam giác Trụ cột vào một đồ thị tính toán khả vi từ đầu đến cuối (end-to-end differentiable), loại bỏ hoàn toàn việc hòa trộn điểm số sau suy luận.

---

## 2. ĐẶC TẢ KIẾN TRÚC MÔ HÌNH THỐNG NHẤT (UPED ARCHITECTURE SPECIFICATION)

```
                            Learner WAV Audio (16kHz)
                                       │
                                       ▼
                       [Acoustic Frame Encoder (Wav2Vec2)]
                                       │
                Intended Text ─────────┼──► [Text-to-Phoneme G2P + CMUdict Resolver]
                                       │              │
                                       ▼              ▼
                     [Viterbi Soft Temporal Alignment Layer]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
       [Target Acoustic Stream]                 [Gap Acoustic Stream]
      - 39-dim phone one-hot                   - 6-dim base temporal features
      - 12-dim GOP & entropy                   - 3-dim prosodic energy dynamics
      - 4-dim tempo-normalized duration          (RMS ratio, dB energy, tilt - từ EXP-28)
      - 2-dim phonotactic n-gram priors        ───────────────┬───────────────────────
        (P(p_t|p_{t-1}), cluster flag - EXP-34)               │
      ───────────────────────────────────────                 │
             Tổng cộng: 57 chiều                              │ Tổng cộng: 9 chiều
                    │                                         │
                    ▼                                         │
   [Projection & LayerNorm (128 dims)]                        │
                    │                                         │
                    ▼                                         │
     [Bidirectional LSTM (2 layers, 192 dims)]                 │
                    │                                         │
                    ▼                                         │
  ┌────────────────────────────────────────────────────────┐  │
  │     Monotonic Local Gaussian Attention Block           │  │
  │     - Attention Type: Monotonic Local (sigma = 3.0)    │  │
  │     - Number of Heads: 4, Dim: 128                     │  │
  │     - Chức năng: Chống rung biên, gom tụ coarticulation│  │
  └──────────────────────────┬─────────────────────────────┘  │
                             │ (Attended Target Features)     │
                             │                                │
            ┌────────────────┴────────────────┐               │
            ▼                                 ▼               ▼
  [Multitask Articulatory Heads]    [Phonological Contrastive] [Context-Injected Gap Net]
  - Manner Head (8 classes)         [Margin Loss Projection]  - Gap 9-dim + Context Target (256)
  - Place Head (9 classes)          - Supervised Margin m=0.35 - 2-layer MLP (64 -> 32 -> 1)
  - Voicing Head (3 classes)        (Phân tách âm thay thế)   - Sigmoid Output
            │                                 │               └───────────────┬─────────────────
            └────────────────┬────────────────┘                               │
                             ▼                                                ▼
              [Unified Diagnosis Head (3 classes)]                    p_insertion [M]
              - Input: Hidden + Manner + Place + Voic (148 dims)
              - Softmax Output: [p_Correct, p_Substitution, p_Omission]
```

### 2.1. Các tham số kiến trúc cụ thể
- **Đầu vào Target (57 chiều):**
  - 39 chiều one-hot âm vị chuẩn.
  - 12 chiều âm học GOP truyền thống (log-likelihood, logit margin, silence probability, frame entropy).
  - 4 chiều chuẩn hóa nhịp độ nói (từ EXP-17): `dur_ratio`, `log_dur_ratio`, `is_abbrev`, `sil_contrast`.
  - 2 chiều tiên nghiệm âm vị học (từ EXP-34): `log_p_transition` ($P(p_t \mid p_{t-1})$) và `is_cluster_legal`.
- **Đầu vào Gap (9 chiều):**
  - 6 chiều cơ sở: thời lượng khung, log thời lượng, max logit, mean logit, sil prob, max gap prob.
  - 3 chiều động lực năng lượng dạng sóng (từ EXP-28): `rms_energy_ratio`, `relative_db`, `spectral_tilt`.
- **Lớp làm mượt biên giới:** Monotonic Local Gaussian Attention với $\sigma = 3.0$, nhead = 4, d_model = 128.
- **Tổng số tham số ước tính:** **~650.000 tham số** (nhỏ hơn 7 lần so với cụm 7 mô hình của EXP-36 là 4.4M tham số).

---

## 3. HÀM TỔN THẤT VÀ CHIẾN LƯỢC HUẤN LUYỆN (LOSS FUNCTIONS & TRAINING)

Mô hình thống nhất UPED sẽ được tối ưu hóa bằng hàm tổn thất liên hợp đa nhiệm:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diagnosis}} + \lambda_1 \mathcal{L}_{\text{gap}} + \lambda_2 \mathcal{L}_{\text{articulatory}} + \lambda_3 \mathcal{L}_{\text{contrastive}}$$

Trong đó:
1. $\mathcal{L}_{\text{diagnosis}}$: Weighted Cross-Entropy Loss trên 3 lớp Target [Correct, Substitution, Omission], với trọng số bù trừ mất cân bằng lớp ($w_C = 0.2, w_S = 1.0, w_O = 2.5$).
2. $\mathcal{L}_{\text{gap}}$: Binary Cross-Entropy Loss trên khe hở chèn ($w_I = 3.0$).
3. $\mathcal{L}_{\text{articulatory}}$: Tổng tổn thất Cross-Entropy trên 3 đầu phụ trợ đa nhiệm phát âm (Manner of articulation 8 lớp, Place of articulation 9 lớp, Voicing 3 lớp) giúp ràng buộc đặc trưng nơ-ron theo giải phẫu học phát âm.
4. $\mathcal{L}_{\text{contrastive}}$: Supervised Contrastive Margin Loss ($m = 0.35$) ép góc biểu diễn giữa âm chuẩn và allophone lỗi thay thế phải cách nhau tối thiểu một góc biên hình học.

---

## 4. GIẢI QUYẾT TÍNH ĐỘC LẬP THAM CHIẾU VÀ VẤN ĐỀ TỪ OOV

Chiến dịch TMCI đã chỉ ra rằng **6.5% số câu phát ngôn (39 câu)** trong L2-ARCTIC chứa từ ngoài từ điển CMUdict (như tên riêng *Sahara, Agassiz, Swedenborg*). Để đạt được chứng nhận suy luận độc lập không cần can thiệp con người:
1. **Xây dựng module Fallback G2P:**
   Tích hợp một mô hình transformer nơ-ron trọng số nhẹ (hoặc thư viện `g2p_en`) tự động sinh phiên âm ARPAbet cho các từ OOV khi từ điển tra cứu thất bại.
2. **Bộ giải quyết biến thể phát âm (Pronunciation Variant Resolver):**
   Với các từ có nhiều cách đọc (ví dụ: *THE* $\rightarrow$ /DH AH/ hoặc /DH IY/), module sẽ sử dụng quy tắc ngữ cảnh âm vị học (đứng trước phụ âm hay nguyên âm) để chọn biến thể dự định phù hợp, tuyệt đối không sử dụng nhãn âm thanh thực tế để chọn variant.

---

## 5. KẾ HOẠCH BÀN GIAO VÀ CÁC MỐC THẨM ĐỊNH (VERIFICATION GATES)

| Mốc thẩm định | Tiêu chí nghiệm thu bắt buộc (Pass Gate) | Rủi ro cần phòng tránh |
|---|---|---|
| **Gate 1: Pipeline Thống nhất** | Mô hình đơn lẻ UPED chạy được từ file WAV 16kHz và chuỗi text thuần túy; tỷ lệ từ chối OOV = 0%. | Không dùng bất kỳ nhãn gold hay realized phone nào trong tiền xử lý. |
| **Gate 2: Ngưỡng năng lực phân biệt** | DEV_SEARCH Macro-3 Error AP $\ge$ **0.290** (vượt qua EXP-36 chính thức); AP Omission $\ge$ **0.220**; AP Insertion $\ge$ **0.270**. | Không được để AP Omission bị hy sinh để đổi lấy AP Substitution. |
| **Gate 3: Độ bền vững đa người nói** | Đạt Macro-3 AP $\ge$ **0.240** trên cả 4 người nói (`NCC`, `RRBI`, `YBAA`, `YDCK`), không để thua EXP-24 trên Hindi và Korean. | Không để mô hình thiên lệch vào 1 nhóm ngôn ngữ mẹ đẻ (L1). |
| **Gate 4: Độ tin cậy & Độ trễ** | Brier score sau hiệu chỉnh $\le$ **0.075** (S), $\le$ **0.012** (O), $\le$ **0.006** (I); độ trễ suy luận $\le$ **25ms/câu** trên CPU/GPU. | Không đưa cấu trúc phân tầng phức tạp hai giai đoạn (EXP-38) trở lại. |

*Tài liệu này đánh dấu sự kết thúc thành công của chiến dịch điều tra hậu EVAL-V3 và mở ra lộ trình phát triển cho thế hệ mô hình thống nhất UPED.*
