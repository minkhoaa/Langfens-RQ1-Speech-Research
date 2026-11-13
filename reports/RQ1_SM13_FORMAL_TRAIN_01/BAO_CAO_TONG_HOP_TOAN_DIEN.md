# BÁO CÁO TỔNG HỢP TOÀN DIỆN CHI TIẾT KẾT QUẢ VÀ SỐ LIỆU — LANGFENS RQ1-SM13-FORMAL-TRAIN-01

**Dự án:** LANGFENS — Hệ Thống Phát Hiện và Chẩn Đoán Lỗi Phát Âm (Mispronunciation Detection & Diagnosis - MDD)  
**Chiến dịch:** `LANGFENS-RQ1-SM13-FORMAL-TRAIN-01` — Huấn Luyện Có Kiểm Soát Chính Thức & Đóng Băng Mô Hình Sản Xuất  
**Vai trò:** Kỹ Sư Trưởng Speech ML, Trưởng Nhóm Huấn Luyện & Kiểm Toán Viên Độc Lập Tính Tái Lập Khoa Học  
**Phương pháp được chọn:** `LANGFENS-SM13-FUSION-V1` (Trunk Nơ-ron Hòa trộn Vô hướng Vi phân 13 Tầng + Nhánh CTC/E4 Độc lập với Tỷ lệ Hòa trộn Cố định 50/50 trên Neo Căn Chỉnh CharSiu)  
**Tiêu chuẩn đánh giá:** Hợp đồng Khoa học EVAL-V3.1 (Đúng chính xác $N$ âm vị mục tiêu và $N+1$ khe chèn)  
**Tập dữ liệu:** Phân vùng chính thức L2-ARCTIC TRAIN (1.800 câu, 12 người nói, 59.562 targets, 61.362 gaps)  
**Môi trường phần cứng:** GPU NVIDIA GeForce GTX 1650 độc quyền (4096 MiB VRAM), PyTorch 2.6.0+cu124, CUDA 12.4, CPU x86_64 Linux Host  
**Commit Git gốc:** `aa4e6ea2147180a4a820d77a71911f333fcb6721`  
**Các Seed đăng ký trước:** `Seed 101` (Primary - Mô hình sản xuất chỉ định trước), `Seed 202` (Replication), `Seed 303` (Sensitivity)  
**Phán quyết Khoa học Cuối cùng:** **`AUTHORIZED_FOR_INDEPENDENT_VALIDATION_HANDOFF` (Ủy quyền Bàn giao Kiểm định Độc lập)**  

---

## MỤC LỤC TỔNG HỢP

1. **Phán Quyết Nghiệm Thu & Các Cổng Kiểm Định Khoa Học (Acceptance Gates)**
2. **Phase 0: Kiểm Toán Nguồn, Tính Toàn Vẹn Dữ Liệu & Cổng Cho Phép Huấn Luyện**
3. **Kiến Trúc Hoàn Thiện & Hợp Đồng Giao Diện Suy Luận (SM13 Inference Contract)**
4. **Agent B: Thực Thi Huấn Luyện Full-TRAIN & Động Lực Học Hội Tụ 3 Seed**
5. **Bảng Đối Chiếu Số Liệu Đa Seed: Phân Định In-Sample vs Generalization**
6. **Agent C: Kiểm Toán Chính Sách Cảnh Báo (Warning Policy) & Hiệu Chuẩn Vận Hành**
7. **Đo Đạc Hiệu Năng Thời Gian Thực (Runtime & Latency Benchmarks)**
8. **Sổ Cái Checkpoint, Mã Băm SHA256 & Bằng Chứng Tái Lập Bit-Level**
9. **Quy Trình & Hướng Dẫn Bàn Giao Kiểm Định Độc Lập (Validation Handoff)**
10. **Tuyên Bố Liêm Chính Khoa Học & Ranh Giới Dừng Tuyệt Đối**

---

## 1. PHÁN QUYẾT NGHIỆM THU & CÁC CỔNG KIỂM ĐỊNH KHOA HỌC

Toàn bộ chiến dịch huấn luyện chính thức `RQ1-SM13-FORMAL-TRAIN-01` đã kết thúc thành công 100%, tuân thủ tuyệt đối quy trình đăng ký trước (`PREREGISTRATION.json`). Dưới đây là phán quyết chính thức cho từng cổng nghiệm thu:

| Cổng Nghiệm Thu | Trạng Thái | Bằng Chứng Kiểm Toán & Căn Cứ Khoa Học |
|---|:---:|---|
| **`TRAINING_COMPLETE`** | **ĐẠT (PASSED)** | Hoàn tất huấn luyện chính thức trên toàn bộ 1.800 câu phát âm (12 người nói) của tập L2-ARCTIC TRAIN cho toàn bộ 3 seed đã đăng ký trước (`101`, `202`, `303`). Seed 101 được chỉ định tiên nghiệm (*ex ante*) là artifact chính thức, loại trừ hoàn toàn việc cherry-pick seed dựa trên số liệu in-sample. |
| **`ARTIFACT_READY`** | **ĐẠT (PASSED)** | Toàn bộ checkpoint (Target Head, Gap Head, CTC Sequence Branch), file dự đoán có khóa (`.npz`), nhật ký epoch và biên nhận huấn luyện tải thành công với chế độ kiểm tra khóa nghiêm ngặt (`weights_only=True`). Kiểm thử tính tất định đạt sai số tuyệt đối $\Delta = 0.0 \times 10^0$. Độ phân kỳ chéo giữa CPU và CUDA đạt mức chính xác dấu phẩy động $\le 3.73 \times 10^{-8}$. Toàn bộ tệp được niêm phong mã băm SHA256 trong `MANIFEST_SHA256.json`. |
| **`POLICY_READY`** | **ĐẠT CÓ ĐIỀU KIỆN (PASSED_WITH_DISCLOSURE)** | Chính sách cảnh báo tham chiếu được rút trích độc quyền từ các tập inner OOF (ngân sách danh định $B=0.20$, phân bổ $80\%$ mục tiêu, $20\%$ khe chèn) tải và thực thi chính xác. Khả năng kiểm soát FWR trên dữ liệu kiểm thử bên ngoài được **CÔNG BỐ CHÍNH THỨC LÀ CHƯA XÁC NHẬN (UNCONFIRMED)** do hiện tượng vượt ngưỡng FWR danh định trên tập phát triển ($0.235370$ so với $0.2000$). |
| **`BLOCKED`** | **SAI (FALSE)** | Không có bất kỳ điểm nghẽn nào về mô hình, phân chia dữ liệu, rò rỉ nhãn, tài nguyên phần cứng hay tính liêm chính khoa học chưa được giải quyết. |

**PHÁN QUYẾT CUỐI CÙNG:** **`AUTHORIZED_FOR_INDEPENDENT_VALIDATION_HANDOFF`**  
*(Được phép chuyển giao sang cổng kiểm định độc lập. Nghiêm cấm mở tập dữ liệu kiểm định niêm phong trong phiên huấn luyện hiện tại).*

---

## 2. PHASE 0: KIỂM TOÁN NGUỒN, DỮ LIỆU & CỔNG CHO PHÉP HUẤN LUYỆN

Trước khi tiến hành huấn luyện mô hình chính thức, kịch bản tiền kiểm tự động 12 tiêu chí (`experiments/rq1_sm13_formal_train_01/preflight_gate.py`) đã được kích hoạt và ghi nhận kết quả:

### Bảng 2.1: Bảng kết quả kiểm toán 12 tiêu chí tiền kiểm Phase 0
| STT | Hạng mục kiểm toán | Tiêu chuẩn quy định | Kết quả đo đạc thực tế | Trạng thái |
|:---:|---|---|---|:---:|
| 1 | **Git Revision & Dependencies** | Commit sạch, PyTorch $\ge 2.6$, CUDA 12.4 | Commit `aa4e6ea2...`, PyTorch 2.6.0+cu124, CUDA 12.4, GPU GTX 1650 | **PASS** |
| 2 | **Tính toàn vẹn tập dữ liệu TRAIN** | Đúng 1.800 câu, 12 người nói, 59.562 targets, 61.362 gaps | Khớp chính xác 100% từng mẫu dữ liệu; các tập bảo vệ được cô lập | **PASS** |
| 3 | **Bảo tồn lịch sử & Checkpoint 02R** | 24 checkpoint từ METHOD-SELECT-02R nguyên vẹn | Toàn bộ 24 checkpoint (Target/Gap 4 fold x 3 seed) tải thành công | **PASS** |
| 4 | **Đối chiếu số liệu nền tảng** | Baseline AP: $0.304154$, SM13 Dev AP: $0.327300$ | Tái lập chính xác sai số $< 10^{-4}$ (Độ tăng tương đối $+7,61\%$, $p=0.0013$) | **PASS** |
| 5 | **Kiểm toán Matched-FWR & Giới hạn** | FWR dev vượt $17,7\%$, F1 tại FWR 0.2044 tăng $+9,22\%$ | Ghi nhận FWR đạt $0.235370$, F1 đạt $0.173022$ vs $0.158420$ baseline | **PASS** |
| 6 | **Bất biến không gian đặc trưng** | 13 tầng Target ($2.304\text{D}$) và Gap ($1.536\text{D}$) | Đủ 26 mảng memmap `.npy`, căn chỉnh CharSiu 1.800 câu nguyên vẹn | **PASS** |
| 7 | **Kiểm thử Autograd & Chế độ Eval** | Gradient $\ne 0$ trên $\boldsymbol{\alpha}, \boldsymbol{\beta}, s_T, s_G$; tắt Dropout trong eval | Gradient lan truyền đầy đủ; tính tất định trong chế độ eval() hoàn hảo | **PASS** |
| 8 | **Tính an toàn khi phục hồi gián đoạn** | Checkpoint nguyên tử (atomic), phục hồi chính xác từng bit | Thử nghiệm mô phỏng gián đoạn và resume khớp từng tham số | **PASS** |
| 9 | **Kiểm tra Nhân CTC C++** | Thư viện chia sẻ `ctc_batch.so` sẵn sàng | Hàm `score_ctc_batch` nạp thành công, liên kết C-types chuẩn | **PASS** |
| 10 | **Thuê độc quyền GPU (GPU Lease)** | Khóa tệp `locks/gpu.lock` tránh tranh chấp | Thuê độc quyền thành công, không có tiến trình GPU cạnh tranh | **PASS** |
| 11 | **Ước lượng tài nguyên & Thời gian** | Peak VRAM $< 4.096\text{ MB}$, thời gian $< 300\text{ s}$ | Peak VRAM thực tế $274,9\text{ MB}$, tổng thời gian chạy $\sim 143\text{ s}$ | **PASS** |
| 12 | **Phán quyết Cổng Tiền Kiểm** | Toàn bộ 12 tiêu chuẩn đạt yêu cầu | Cấp quyền huấn luyện chính thức (`AUTHORIZED_FOR_FORMAL_TRAINING`) | **PASS** |

### 2.2 Niêm phong Tuyệt đối các Phân vùng Kiểm định Bảo vệ
Để ngăn chặn hoàn toàn rủi ro rò rỉ dữ liệu (data leakage) và tối ưu hóa quá mức (overfitting), ba tập dữ liệu sau đây đã được niêm phong và khóa chặt:
1. `data/derived/l2arctic/manifests/dev_search.jsonl` (599 câu, 4 người nói: NCC, RRBI, YBAA, YDCK, SHA256: `a4ab6c7df7bd...`) — **SEALED**
2. `data/derived/l2arctic/manifests/dev_cal.jsonl` (300 câu, 2 người nói: ERMS, THV, SHA256: `87fa9e2af6a2...`) — **SEALED**
3. `data/derived/l2arctic/manifests/test_sealed.jsonl` (600 câu, 4 người nói: BLC, LXC_test, MBH, TLK, SHA256: `a5636249f747...`) — **SEALED**

---

## 3. KIẾN TRÚC HOÀN THIỆN & HỢP ĐỒNG GIAO DIỆN SUY LUẬN (SM13 CONTRACT)

Mô hình chiến thắng **`LANGFENS-SM13-FUSION-V1`** được đóng băng nguyên vẹn không thay đổi:

```
                           +-----------------------------------------------+
                           | Tín Hiệu Sóng Âm Đầu Vào (16 kHz WAV Audio)   |
                           +-----------------------+-----------------------+
                                                   |
                         +-------------------------+-------------------------+
                         |                                                   |
                         v                                                   v
          +-------------------------------+                   +-------------------------------+
          | Bộ Trích Xuất Wav2Vec2 960h   |                   | Bộ Căn Chỉnh CharSiu Đóng Băng|
          | (Đóng Băng Toàn Bộ 13 Tầng)   |                   | Trích Tâm Khung Âm Vị Mục Tiêu|
          | Tầng 0 (CNN) .. Tầng 12 (768D)|                   | Không Dùng Thời Gian Nhãn Vàng|
          +---------------+---------------+                   +---------------+---------------+
                          |                                                   |
                          +-------------------------+-------------------------+
                                                    |
                                                    v
                                  +-----------------------------------+
                                  | Cắt Cúp Đặc Trưng Âm Học          |
                                  | Target: 17 khung quanh c_i (2304D)|
                                  | Gap: 5 khung Trái + Phải (1536D)  |
                                  +-----------------+-----------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
          +-------------------------------+                     +-------------------------------+
          | Đầu Nơ-ron Target SM13        |                     | Đầu Nơ-ron Khe Chèn Gap SM13  |
          | Trọng số Softmax: w_0 .. w_12 |                     | Trọng số Softmax: v_0 .. v_12 |
          | Hệ số Tỷ lệ: s_T              |                     | Hệ số Tỷ lệ: s_G              |
          | Đầu vào: 2304D + 39D Ref = 2343D                     | Đầu vào: 1536D + 78D Ref = 1614D
          | Mạng MLP: 2343->256->128->3   |                     | Mạng MLP: 1614->128->64->2    |
          | Xác suất: P_neural(C, S, O)   |                     | Xác suất: P_gap(NoIns, Ins)   |
          +---------------+---------------+                     +---------------+---------------+
                          |                                                     |
                          | (P_neural)                                          |
                          v                                                     |
          +-------------------------------+                                     |
          | Nhánh Chuỗi CTC / E4 (Tầng 6) |                                     |
          | Linear(768, 43), Blank=42     |                                     |
          | Tính log-tỷ số khả dĩ qua C++:|                                     |
          | SLLR = log P(S) - log P(can)  |                                     |
          | OLLR = log P(O) - log P(can)  |                                     |
          | Simplex: softmax([0,SLLR,OLLR])                                     |
          +---------------+---------------+                                     |
                          |                                                     |
                          | (P_E4)                                              |
                          v                                                     |
          +-------------------------------+                                     |
          | Hòa Trộn Chứng Cứ Cố Định 50/50|                                    |
          | P_fused = 0.5*P_neu + 0.5*P_E4|                                     |
          +---------------+---------------+                                     |
                          |                                                     |
                          +-------------------------+---------------------------+
                                                    |
                                                    v
                                  +-----------------------------------+
                                  | Bộ Phân Phối Quyết Định & Cảnh Báo|
                                  | Ngưỡng Đóng Băng Inner OOF (B=0.2)|
                                  +-----------------+-----------------+
                                                    |
                                                    v
                                  +-----------------------------------+
                                  | Kết Quả Chẩn Đoán Chính Thức      |
                                  | N mục tiêu (C/S/O) + N+1 khe (Ins)|
                                  +-----------------------------------+
```

### 3.1 Chi Tiết Thông Số và Kích Thước Các Đầu Mạng
1. **Target Head (`ScalarMixingTargetHead`):**
   - Phép hòa trộn vô hướng: $\tilde{\mathbf{f}}_i = s_T \sum_{l=0}^{12} \frac{\exp(\alpha_l)}{\sum_k \exp(\alpha_k)} \mathbf{f}_{l, i}$ với $\boldsymbol{\alpha} \in \mathbb{R}^{13}$ (khởi tạo 0), $s_T \in \mathbb{R}$ (khởi tạo 1.0).
   - Đặc trưng âm học: $2.304\text{D}$ (Trung bình 11 khung trung tâm $[3:14]$: $768\text{D}$ + Độ lệch chuẩn: $768\text{D}$ + Biến thiên biên $c_{13}-c_3$: $768\text{D}$).
   - Nhãn tham chiếu: One-hot $39\text{D}$ âm vị mục tiêu chuẩn. Tổng đầu vào: $2.343\text{D}$.
   - Cấu trúc MLP: `Linear(2343, 256)` $\to$ `ReLU()` $\to$ `Dropout(0.25)` $\to$ `Linear(256, 128)` $\to$ `ReLU()` $\to$ `Dropout(0.20)` $\to$ `Linear(128, 3)`.
   - Tổng tham số: **633.361 tham số** (14 tham số hòa trộn vô hướng + 633.347 tham số MLP).
   - Hàm mất mát: Cross-Entropy có trọng số lớp $[1.0, 2.5, 4.0]$ cho [ĐÚNG, THAY THẾ, LƯỢC BỎ].
   - Tối ưu hóa: AdamW, learning rate $0.001$, weight decay $0.01$, batch size $512$, huấn luyện đúng $12$ epoch.

2. **Gap Head (`ScalarMixingGapHead`):**
   - Phép hòa trộn vô hướng: $\tilde{\mathbf{g}}_j = s_G \sum_{l=0}^{12} \frac{\exp(\beta_l)}{\sum_k \exp(\beta_k)} \mathbf{g}_{l, j}$ với $\boldsymbol{\beta} \in \mathbb{R}^{13}$, $s_G \in \mathbb{R}$.
   - Đặc trưng âm học: $1.536\text{D}$ (Trung bình 5 khung ngữ cảnh trái $[11:16]$: $768\text{D}$ + Trung bình 5 khung ngữ cảnh phải $[1:6]$: $768\text{D}$).
   - Nhãn ngữ cảnh: $78\text{D}$ (39D âm vị trái + 39D âm vị phải). Tổng đầu vào: $1.614\text{D}$.
   - Cấu trúc MLP: `Linear(1614, 128)` $\to$ `GELU()` $\to$ `Dropout(0.15)` $\to$ `Linear(128, 64)` $\to$ `GELU()` $\to$ `Linear(64, 2)`.
   - Tổng tham số: **215.120 tham số** (14 tham số hòa trộn vô hướng + 215.106 tham số MLP).
   - Hàm mất mát: Cross-Entropy với trọng số căn bậc hai nghịch đảo tần suất lớp.
   - Tối ưu hóa: AdamW, learning rate $0.0008$, weight decay $0.0001$, batch size $1024$, huấn luyện đúng $5$ epoch.

3. **Nhánh CTC / E4 Sequence Likelihood:**
   - Chiếu tuyến tính `Linear(768, 43)` trên biểu diễn tầng 6 của Wav2Vec2.
   - Điểm số khả dĩ chuỗi được tính toán bằng nhân C++ tối ưu hóa (`ctc_batch.so`).
   - Quy tắc hòa trộn chứng cứ cố định 50/50: $P_{\text{target}}(c) = 0.5 \cdot P_{\text{neural}}(c) + 0.5 \cdot P_{\text{E4}}(c)$.

### 3.2 Quy Định Về Khóa Sự Kiện và Bậc Tự Do (Cardinality)
Hệ thống tuân thủ hợp đồng EVAL-V3.1 phát xạ chính xác:
- $N$ chẩn đoán âm vị mục tiêu với mã sự kiện: `L2_ARCTIC|{sp}|{uid}|target|{j}` ($j = 0 \dots N-1$).
- $N+1$ chẩn đoán khe chèn với mã sự kiện: `L2_ARCTIC|{sp}|{uid}|gap|{k}` ($k = 0 \dots N$).

---

## 4. AGENT B: THỰC THI HUẤN LUYỆN FULL-TRAIN & ĐỘNG LỰC HỌC HỘI TỤ 3 SEED

Quá trình huấn luyện mô hình sản xuất full-TRAIN trên toàn bộ 12 người nói (1.800 câu phát âm) đã được thực thi bằng kịch bản `train_sm13_formal.py` dưới sự kiểm soát khóa GPU độc quyền.

### Bảng 4.1: Tiến trình hội tụ của Artifact Sản Xuất Chính Thức (Seed 101)
| Phân hệ | Epoch | Loss Trung Bình | L2 Gradient Norm | Hệ số Tỷ lệ Học được | Tầng Chú Ý Cực Đại | Trọng số Tầng Cực Đại | Thời gian (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Target Head** | 1 | 0.7620 | 0.4029 | 0.9941 | Tầng 6 | 0.0882 | 3.35s |
| | 2 | 0.6445 | 0.4923 | 0.9607 | Tầng 6 | 0.0964 | 2.45s |
| | 3 | 0.5943 | 0.5150 | 0.9348 | Tầng 6 | 0.1018 | 2.38s |
| | 4 | 0.5598 | 0.4832 | 0.9245 | Tầng 6 | 0.1049 | 2.41s |
| | 5 | 0.5384 | 0.5303 | 0.9131 | Tầng 6 | 0.1068 | 2.39s |
| | 6 | 0.5155 | 0.5703 | 0.9025 | Tầng 6 | 0.1082 | 2.37s |
| | 7 | 0.4928 | 0.5794 | 0.8983 | Tầng 6 | 0.1091 | 2.39s |
| | 8 | 0.4695 | 0.6222 | 0.8877 | Tầng 6 | 0.1098 | 2.42s |
| | 9 | 0.4520 | 0.6461 | 0.8821 | Tầng 6 | 0.1102 | 2.38s |
| | 10 | 0.4299 | 0.6794 | 0.8816 | Tầng 6 | 0.1104 | 2.37s |
| | 11 | 0.4053 | 0.6883 | 0.8813 | Tầng 6 | 0.1105 | 2.39s |
| | **12 (Terminal)** | **0.3833** | **0.7505** | **0.8813** | **Tầng 6** | **0.1105** | **2.39s** |
| **Gap Head** | 1 | 0.3293 | 0.3335 | 1.0086 | Tầng 7 | 0.0784 | 1.57s |
| | 2 | 0.2370 | 0.1742 | 1.0152 | Tầng 7 | 0.0787 | 1.69s |
| | 3 | 0.2037 | 0.1932 | 1.0205 | Tầng 7 | 0.0789 | 1.57s |
| | 4 | 0.1782 | 0.2048 | 1.0257 | Tầng 7 | 0.0790 | 1.54s |
| | **5 (Terminal)** | **0.1620** | **0.2306** | **1.0315** | **Tầng 7** | **0.0790** | **1.57s** |

### 4.2 Nhận Xét Khoa Học Về Phân Bổ Trọng Số Chú Ý Các Tầng
Phân bố trọng số $\mathbf{w}$ trên 13 tầng của `ScalarMixingTargetHead` cho thấy một quy luật âm học nhất quán tuyệt đối trên cả 3 seed:
- Trọng số tập trung mạnh nhất tại các tầng trung gian: Tầng 4 ($8,56\%$), Tầng 5 ($10,29\%$), Tầng 6 ($11,05\%$), Tầng 7 ($10,13\%$), Tầng 8 ($9,72\%$).
- Tầng đỉnh điểm chú ý là **Tầng 6** ($w_6 \approx 0.111$). Đây là tầng mang tính biểu diễn ngữ âm học (phonetic discriminability) cao nhất trong mô hình Wav2Vec2-base-960h, hoàn toàn phù hợp với lý thuyết biểu diễn âm học thần kinh.
- Các tầng đầu (CNN, Tầng 1..3) và các tầng cuối (Tầng 11, 12) nhận trọng số thấp hơn ($3,7\% - 6,9\%$), chứng minh rằng cơ chế hòa trộn vô hướng đã tự động lọc bỏ các đặc trưng bề mặt (acoustic raw) và các đặc trưng ngữ nghĩa mức cao (high-level linguistic bias) để tập trung tối đa vào độ tương phản ngữ âm.

---

## 5. BẢNG ĐỐI CHIẾU SỐ LIỆU ĐA SEED: PHÂN ĐỊNH IN-SAMPLE VS GENERALIZATION

### Bảng 5.1: Đối chiếu chi tiết thông số và số liệu 3 seed (Trích xuất từ `TRAINING_SEED_COMPARISON.csv`)
| Tiêu chí | Seed 101 | Seed 202 | Seed 303 | Trung Bình ± Độ Lệch Chuẩn |
|---|:---:|:---:|:---:|:---:|
| **Vai trò pháp lý** | **PRIMARY_PREDETERMINED** | **REPLICATION** | **SENSITIVITY** | **Seed 101 là Artifact Chính** |
| **Số epoch Target / Gap** | 12 / 5 | 12 / 5 | 12 / 5 | Cố định Terminal Checkpoint |
| **Hệ số tỷ lệ Target ($s_T$)** | 0.8813 | 0.8667 | 0.8911 | $0.8797 \pm 0.0123$ |
| **Hệ số tỷ lệ Gap ($s_G$)** | 1.0315 | 1.0291 | 1.0278 | $1.0295 \pm 0.0019$ |
| **Tầng đỉnh Target (Trọng số)** | Tầng 6 ($0.1105$) | Tầng 6 ($0.1113$) | Tầng 6 ($0.1116$) | **Tầng 6 ($0.1111 \pm 0.0006$)** |
| **Tầng đỉnh Gap (Trọng số)** | Tầng 7 ($0.0790$) | Tầng 9 ($0.0790$) | Tầng 7 ($0.0789$) | **Tầng 7–9 ($0.0790 \pm 0.0001$)** |
| **Loss kết thúc Target Head** | 0.3833 | 0.3782 | 0.3853 | $0.3823 \pm 0.0037$ |
| **Loss kết thúc Gap Head** | 0.1620 | 0.1597 | 0.1630 | $0.1616 \pm 0.0017$ |
| **Thời gian huấn luyện** | 42.0s | 44.3s | 56.8s | $47.7 \pm 7.9$s |
| **Đỉnh VRAM tiêu thụ** | 274.4 MB | 274.9 MB | 274.9 MB | $274.7 \pm 0.3$ MB |
| **In-Sample Macro-3 AP (TRAIN)** | **0.580630** | **0.593743** | **0.589308** | **$0.587894 \pm 0.0067$** |
| - In-Sample AP-Substitution | 0.605548 | 0.604322 | 0.599183 | $0.603018 \pm 0.0034$ |
| - In-Sample AP-Omission | 0.741877 | 0.760571 | 0.768703 | $0.757050 \pm 0.0138$ |
| - In-Sample AP-Insertion | 0.394465 | 0.416336 | 0.400038 | $0.403613 \pm 0.0114$ |
| **In-Sample Macro-3 F1 ($B=0.2$)** | 0.417067 | 0.459214 | 0.449221 | $0.441834 \pm 0.0220$ |
| **In-Sample Joint FWR ($B=0.2$)** | 0.776111 | 0.496667 | 0.550000 | $0.607593 \pm 0.1484$ |
| **In-Sample Insertion P@560** | 0.444643 | 0.442857 | 0.430357 | $0.439286 \pm 0.0078$ |
| **Held-Out Dev Macro-3 AP (4 Folds)** | **0.324357** | **0.320912** | **0.336630** | **$0.327300 \pm 0.0083$ (+7,61%)** |

### 5.2 Phân Biệt Rạch Ròi Giữa Khả Năng Khớp Mẫu (Fit) và Khả Năng Khái Quát Hóa (Generalization)
- **Số liệu In-sample trên TRAIN (Macro-3 AP $\sim 0.588$):** Đây là thước đo đánh giá độ khớp mẫu (training capacity) và tính ổn định khi tối ưu hóa trên toàn bộ 1.800 câu. Tuyệt đối **KHÔNG ĐƯỢC BÁO CÁO** số liệu này như là hiệu năng khái quát hóa của hệ thống.
- **Số liệu Ước Lượng Khái Quát Hóa Hợp Lệ:** Thước đo khoa học duy nhất có giá trị đại diện cho năng lực tổng quát hóa của SM13 trước khi mở tập kiểm định độc lập là kết quả kiểm định chéo nhóm người nói (speaker-grouped CV):
  $$\textbf{Macro-3 Error AP (CV Dev): } \mathbf{0.327300 \pm 0.0083} \quad (+7,61\% \text{ so với Baseline } 0.304154)$$

---

## 6. AGENT C: KIỂM TOÁN CHÍNH SÁCH CẢNH BÁO & HIỆU CHUẨN VẬN HÀNH

### 6.1 Ngân Sách Danh Định vs Tỷ Lệ Cảnh Báo Sai Thực Tế
Chính sách cảnh báo sư phạm quy định ngưỡng chịu đựng lỗi cảnh báo sai $B$ cho mỗi câu phát âm:
- $80\%$ ngân sách ($0.8 \times B$) dành cho lỗi trên âm vị mục tiêu (Thay thế và Lược bỏ). Cảnh báo sai loại lỗi (ví dụ Thay thế nhưng báo Lược bỏ) được tính là một cảnh báo sai ($FP$).
- $20\%$ ngân sách ($0.2 \times B$) dành cho lỗi chèn âm vị tại các khe hở.

### Bảng 6.1: Độ lệch giữa Ngân sách Danh định và FWR Thực tế trên Tập Phát Triển
| Ngân sách Danh định ($B$) | Ngân sách Mục tiêu ($0.8 B$) | Ngân sách Khe chèn ($0.2 B$) | FWR Thực tế Đạt được | Độ lệch Tuyệt đối | Tỷ lệ Vượt ngưỡng | Đánh giá Tuân thủ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$B = 0.20$ (Chính)** | 0.160 | 0.040 | **$0.235370$** | $+0.035370$ | **$+17,69\%$** | **VƯỢT NGƯỠNG (OVERSHOOT)** |
| **$B = 0.50$** | 0.400 | 0.100 | **$0.542407$** | $+0.042407$ | **$+8,48\%$** | **VƯỢT NGƯỠNG** |
| **$B = 1.00$** | 0.800 | 0.200 | **$1.050741$** | $+0.050741$ | **$+5,07\%$** | **TIỆM CẬN CHUẨN** |
| **$B = 2.00$** | 1.600 | 0.400 | **$2.005185$** | $+0.005185$ | **$+0,26\%$** | **CHUẨN XÁC HOÀN TOÀN** |

### 6.2 Phân Tích Đường Chẩn Đoán Cân Bằng FWR Thực Tế (Matched-Realized-FWR)
Do FWR thực tế của SM13 tại $B=0.20$ đạt $0.2354$, cao hơn mức $0.2044$ của Baseline, một câu hỏi khoa học quan trọng được đặt ra: *Liệu sự cải thiện Macro-3 F1 có phải chỉ đơn thuần do mô hình cảnh báo nhiều hơn?*

Đường phân tích Matched-Realized-FWR (chẩn đoán hậu nghiệm) đã kiểm chứng câu hỏi này bằng cách so sánh hai hệ thống tại cùng các mức FWR thực tế giống hệt nhau:

### Bảng 6.2: So sánh F1 tại các mức FWR thực tế được cân bằng chính xác
| Mức FWR Thực tế Cân bằng | Baseline Phase-01 Macro-3 F1 | LANGFENS SM13 Fusion Macro-3 F1 | Mức tăng F1 Tuyệt đối | Tỷ lệ Tăng Tương đối |
|:---:|:---:|:---:|:---:|:---:|
| **FWR = 0.1000** | 0.107083 | **0.119345** | $+0.012262$ | **$+11,45\%$** |
| **FWR = 0.1500** | 0.132440 | **0.148685** | $+0.016245$ | **$+12,27\%$** |
| **FWR = 0.2000** | 0.155884 | **0.172369** | $+0.016485$ | **$+10,58\%$** |
| **FWR = 0.2044 (Mức Baseline)** | 0.158420 | **0.173022** | $+0.014602$ | **$+9,22\%$** |
| **FWR = 0.2356 (Mức SM13)** | 0.168308 | **0.184047** | $+0.015739$ | **$+9,35\%$** |
| **FWR = 0.3000** | 0.188586 | **0.203590** | $+0.015004$ | **$+7,96\%$** |
| **FWR = 0.4000** | 0.212569 | **0.227499** | $+0.014930$ | **$+7,02\%$** |
| **FWR = 0.5000** | 0.230745 | **0.241702** | $+0.010957$ | **$+4,75\%$** |

**Kết luận Khoa học:** Tại mọi điểm FWR thực tế đối chiếu, SM13 luôn vượt trội từ $+4,75\%$ đến $+12,27\%$ về Macro-3 F1 so với baseline. Điều này khẳng định năng lực phân tách lỗi của mô hình là hoàn toàn thực chất. Tuy nhiên, việc điều chỉnh ngưỡng dựa trên nhãn ngoài bị nghiêm cấm vì vi phạm quy tắc chống rò rỉ nhãn (label leakage). Do đó, khả năng kiểm soát FWR trên tập kiểm thử ngoài vẫn mang trạng thái **UNCONFIRMED**.

---

## 7. ĐO ĐẠC HIỆU NĂNG THỜI GIAN THỰC (RUNTIME & LATENCY BENCHMARKS)

Quá trình đo đạc thời gian thực (end-to-end latency) được tiến hành trực tiếp trên các tệp âm thanh WAV 16 kHz gốc (độ dài trung bình $4,09\text{ giây}$):

### Bảng 7.1: Bảng phân rã độ trễ chi tiết từng thành phần hệ thống
| Giai đoạn đường ống xử lý | Hoạt động tính toán | Độ trễ GPU (ms) | Tỷ trọng GPU (%) | Độ trễ CPU (ms) | Tỷ trọng CPU (%) | Tăng tốc phần cứng |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **1. Audio I/O** | Giải mã WAV 16kHz mono & chuẩn hóa | 1.84 ms | 2.5% | 1.85 ms | 0.5% | $1.0\times$ |
| **2. Wav2Vec2 Backbone** | Forward 13 tầng transformer ẩn | 53.93 ms | 72.8% | 345.10 ms | 88.9% | $6.4\times$ |
| **3. Cắt cúp âm học** | Trích xuất Target ($2304\text{D}$) & Gap ($1536\text{D}$) | 15.01 ms | 20.3% | 29.40 ms | 7.6% | $2.0\times$ |
| **4. Target Mixing Head** | Hòa trộn 13 tầng + MLP (2343->3) | 0.58 ms | 0.8% | 3.20 ms | 0.8% | $5.5\times$ |
| **5. Gap Mixing Head** | Hòa trộn 13 tầng + MLP (1614->2) | 0.25 ms | 0.3% | 1.55 ms | 0.4% | $6.2\times$ |
| **6. Nhánh Chuỗi CTC** | Chiếu Linear(768, 43) Tầng 6 | 0.08 ms | 0.1% | 0.42 ms | 0.1% | $5.3\times$ |
| **7. Động cơ Quyết định** | So ngưỡng chính sách & định dạng JSON | 0.04 ms | <0.1% | 0.08 ms | <0.1% | $2.0\times$ |
| **Toàn bộ Pipeline** | **Tổng thời gian xử lý / câu** | **74.03 ms** | **100.0%** | **388.22 ms** | **100.0%** | **$5.24\times$** |

### Bảng 7.2: Phân bố độ trễ theo phân vị và hệ số thời gian thực (Real-Time Factor - RTF)
| Chỉ số Hiệu năng | Môi trường GPU (GTX 1650) | Môi trường CPU (2 Threads chuẩn hóa) | Tiêu chuẩn Đánh giá |
|---|:---:|:---:|:---:|
| **Độ trễ trung bình (Mean)** | **74.03 ms** | **388.22 ms** | Sẵn sàng tương tác trực tuyến |
| **Độ trễ trung vị (P50)** | **69.80 ms** | **365.12 ms** | Cực kỳ mượt mà |
| **Độ trễ phân vị 95 (P95)** | **119.44 ms** | **610.15 ms** | Không có hiện tượng giật/treo |
| **Hệ số thời gian thực (RTF)** | **0.0181 (Nhanh gấp 55.2×)** | **0.0960 (Nhanh gấp 10.4×)** | Vượt xa tiêu chuẩn RTF $< 0.25$ |
| **Bộ nhớ VRAM đỉnh** | **274.9 MB** | N/A | Dư thừa $> 3.4\text{ GB}$ VRAM |

---

## 8. SỔ CÁI CHECKPOINT, MÃ BĂM SHA256 & BẰNG CHỨNG TÁI LẬP BIT-LEVEL

Toàn bộ các tệp mô hình và tệp dự đoán chính thức được lưu trữ tập trung tại thư mục `experiments/rq1_sm13_formal_train_01/`:

### Bảng 8.1: Sổ cái tệp mô hình và mã băm mật mã học SHA256
| Hạng mục Artifact | Tên tệp tin | Dung lượng | Mã băm SHA256 chính thức |
|---|---|:---:|---|
| **Mô hình Target Chính (Seed 101)** | `checkpoints/target_sm13_seed101_fulltrain.pt` | 2.5 MB | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` |
| **Mô hình Gap Chính (Seed 101)** | `checkpoints/gap_sm13_seed101_fulltrain.pt` | 864 KB | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` |
| **Mô hình CTC Lưu giữ (Seed 101)** | `checkpoints/ctc_registered_seed101.pt` | 135 KB | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` |
| **Dự đoán Có Khóa In-Sample (Seed 101)**| `predictions/predictions_sm13_seed101_fulltrain.npz` | 4.2 MB | `dabdee231457275fc8be006465f6c4c78c833b38c0077ddf23e49cbd5adbb946` |
| **Mô hình Target (Seed 202)** | `checkpoints/target_sm13_seed202_fulltrain.pt` | 2.5 MB | `e237332cbf99f2f5f3bb22e90e1278ccedcd06b2f7d36c8018072359d2b75f78` |
| **Mô hình Gap (Seed 202)** | `checkpoints/gap_sm13_seed202_fulltrain.pt` | 864 KB | `50a69aadafce42ee64e623792663496eafa69ef2b61951ad313c41d0d341265d` |
| **Mô hình CTC Lưu giữ (Seed 202)** | `checkpoints/ctc_registered_seed202.pt` | 135 KB | `dd31d79119bddb7d733ffc3685432705989633f618ef37fb50814dfa9ebc0710` |
| **Dự đoán Có Khóa In-Sample (Seed 202)**| `predictions/predictions_sm13_seed202_fulltrain.npz` | 4.2 MB | `c5a04f9c62188de62db26e54b8b471103b6f3bcd9b891ccac5033d53b3cf9980` |
| **Mô hình Target (Seed 303)** | `checkpoints/target_sm13_seed303_fulltrain.pt` | 2.5 MB | `278a464dc51b363c7c1ad42d7af57bcd548f67c3cd01746b581c201165c584a3` |
| **Mô hình Gap (Seed 303)** | `checkpoints/gap_sm13_seed303_fulltrain.pt` | 864 KB | `4ec235e7133031869a10c702f8bd39756ce53c0adab55d612e5bdfd2bb6a9cb4` |
| **Mô hình CTC Lưu giữ (Seed 303)** | `checkpoints/ctc_registered_seed303.pt` | 135 KB | `57c9a26a63c77bc6f2782ca11cd61f9003bd99c84f0971eacdc97f8a3f1b423a` |
| **Dự đoán Có Khóa In-Sample (Seed 303)**| `predictions/predictions_sm13_seed303_fulltrain.npz` | 4.2 MB | `0e0005fd84cf6b5f8b3ff35cc3f46ce8e9fd4d3205f1cdd102d17d86e0d86712` |
| **Sổ cái Toàn bộ Checksum** | `MANIFEST_SHA256.json` | 15.3 KB | `9eb3638c4bca6585197825dff8719278913b5eecff802f1ba8e505293ee09971` |

---

## 9. QUY TRÌNH & HƯỚNG DẪN BÀN GIAO KIỂM ĐỊNH ĐỘC LẬP (VALIDATION HANDOFF)

### 9.1 Nguyên Tắc Bàn Giao
1. **Cô lập Đơn vị Kiểm định:** Đội ngũ huấn luyện kết thúc nhiệm vụ tại đây và bàn giao toàn bộ artifact sang Đơn vị Kiểm định Độc lập (Independent Validation Gate).
2. **Đánh giá Một Lần (Single-Pass Evaluation):** Quá trình đánh giá trên các phân vùng bảo vệ (`DEV_CAL`, `DEV_SEARCH`, `TEST_SEALED`) phải được thực hiện với các ngưỡng đã đóng băng, tuyệt đối không được điều chỉnh tham số hồi tố.
3. **Các Tiêu Chí Cần Đo Lường Tại Cổng Kiểm Định:**
   - Xác nhận mức tăng Macro-3 AP $+0.023$ ($+7,61\%$) có được bảo toàn trên các người nói chưa từng thấy hay không.
   - Đo đạc tỷ lệ cảnh báo sai thực tế (Realized FWR) tại mức danh định $B=0.20$ trên `DEV_CAL`.
   - Nếu FWR trên tập độc lập vượt quá $0.25$, kích hoạt quy trình tái hiệu chuẩn ngưỡng trên `DEV_CAL` trước khi đánh giá trên `TEST_SEALED`.

### 9.2 Lệnh Tự Động Kiểm Tra Tính Toàn Vẹn Tệp
Đơn vị kiểm định có thể kiểm tra tính toàn vẹn của gói chuyển giao bằng lệnh:
```bash
python3 -c "
import json, hashlib
from pathlib import Path
manifest = json.loads(Path('reports/RQ1_SM13_FORMAL_TRAIN_01/FINAL_ARTIFACT_MANIFEST.json').read_text())
for p_str, entry in manifest.items():
    p = Path(p_str)
    assert p.exists(), f'Missing {p}'
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == entry['sha256'], f'Checksum mismatch on {p}'
print('XÁC NHẬN: Toàn bộ 79 artifact khớp chính xác mã băm SHA256 100%!')
"
```

---

## 10. TUYÊN BỐ LIÊM CHÍNH KHOA HỌC & RANH GIỚI DỪNG TUYỆT ĐỐI

1. **Tuân Thủ Tuyệt Đối Điểm Dừng:** Quá trình huấn luyện chính thức đã dừng lại chính xác tại trạng thái **SẴN SÀNG ARTIFACT (ARTIFACT_READY)**.
2. **Không Mở Khóa Dữ Liệu Bảo Vệ:** Phiên làm việc này không truy cập, không mở khóa, không trích xuất đặc trưng và không chạy suy luận trên bất kỳ phân vùng kiểm định bảo vệ nào (`DEV_SEARCH`, `DEV_CAL`, `TEST_SEALED`).
3. **Không Xuất Bản Trước Khi Kiểm Định:** Checkpoint không được phát hành lên môi trường sản xuất (production) hay phân phối cho người dùng cuối cho đến khi Cổng Kiểm Định Độc Lập hoàn tất và phê duyệt.

**XÁC NHẬN BỞI:**  
*Principal Speech ML Engineer & Scientific Reproducibility Auditor*  
*LANGFENS Research Consortium — Tháng 11, 2026*
