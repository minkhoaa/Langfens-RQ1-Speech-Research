# THIẾT KẾ KIẾN TRÚC MÔ HÌNH HỌC SINH THỐNG NHẤT (UNIFIED MULTI-TASK STUDENT ARCHITECTURE)

**Chiến dịch:** LANGFENS Giai đoạn 2 — Workstream C: Student Contract  
**Mục tiêu:** Mô hình học sinh đơn lẻ (Single Model), đầu vào là âm thanh và âm vị tham chiếu, xuất đồng thời nhãn Target Phone (3 lớp) và Inter-phone Gap (2 lớp).  

---

## 1. SƠ ĐỒ KHỐI TỔNG THỂ (SYSTEM ARCHITECTURE DIAGRAM)

```text
Audio Waveform [16kHz] -------------> [ Frozen wav2vec2-Layer6 / Feature Extractor ]
                                                   |
Reference Phonemes [0..N-1] --------> [ Phone Embedding + Positional Encoding ]
                                                   |
                                                   v
                    +-----------------------------------------------------------+
                    |        SHARED MULTI-HEAD ACOUSTIC CROSS-ATTENTION         |
                    |      (Conformer / Transformer-Encoder: Dim=256, Heads=4)  |
                    +-----------------------------------------------------------+
                                          |                       |
              +---------------------------+                       +---------------------------+
              | Target Alignment Sequence                         | Boundary Gap Pooling
              v                                                   v
    +--------------------+                             +--------------------+
    |    TARGET HEAD     |                             |      GAP HEAD      |
    | (Phoneme Errors)   |                             |    (Insertions)    |
    | Linear(256 -> 128) |                             | Linear(256 -> 64)  |
    | GELU + Norm        |                             | GELU + Norm        |
    | Linear(128 -> 3)   |                             | Linear(64 -> 2)    |
    +--------------------+                             +--------------------+
              |                                                   |
              v                                                   v
     Softmax [0..N-1]:                                  Softmax [0..N]:
     - P(CORRECT)                                       - P(NO_INSERTION)
     - P(SUBSTITUTION)                                  - P(INSERTION)
     - P(OMISSION)
```

---

## 2. THIẾT KẾ CÁC KHỐI CHỨC NĂNG (SPECIFICATION OF COMPONENTS)

### 2.1. Bộ mã hóa dùng chung (Shared Cross-Attention Encoder)
- **Kích thước biểu diễn trung gian:** $d = 256$.
- **Số tầng Conformer:** 3 blocks, mỗi block gồm Feed-Forward, Multi-Head Self-Attention (4 heads), và Depthwise Convolution (kernel size = 31).
- **Tham số ước tính:** ~850K tham số (cực kỳ nhẹ, suy luận dưới 15ms trên CPU Intel i5/AMD Ryzen).

### 2.2. Đầu dự đoán âm vị mục tiêu (Target Prediction Head)
- Không gian dự đoán: $0..N-1$ vị trí âm vị tham chiếu.
- Đầu ra: Phân phối xác suất 3 lớp rời rạc chuẩn tắc (Standard Probability Simplex):
  $$\sum_{c \in \{CORR, SUB, OM\}} P(y_i = c) = 1.0$$
- Giải quyết triệt để vi phạm simplex và triệt tiêu xung đột S-O tại thời gian suy luận.

### 2.3. Đầu dự đoán chèn âm (Inter-Phone Gap Head)
- Không gian dự đoán: $N+1$ khe hở liên âm.
- Vector đặc trưng đầu vào: Biểu diễn ghép nối của âm vị trái $h_i$, âm vị phải $h_{i+1}$, và vector năng lượng trung gian trích xuất từ ranh giới thời gian.
- Đầu ra: Xác suất nhị phân $P(\text{INSERTION})$.

---

## 3. RÀNG BUỘC PHẦN CỨNG & THỜI GIAN THỰC (HARDWARE & LATENCY BUDGET)

- **VRAM huấn luyện (GTX 1650 4GB):** < 1.2 GB VRAM (với FP16 AMP và batch size = 32 câu).
- **Thời gian huấn luyện:** ~15 phút cho 30 epochs trên GTX 1650.
- **Kích thước mô hình đã đóng gói:** < 15 MB định dạng ONNX / PyTorch JIT.
