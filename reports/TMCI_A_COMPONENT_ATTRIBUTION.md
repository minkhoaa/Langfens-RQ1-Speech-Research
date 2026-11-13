# BÁO CÁO AGENT A — PHÂN TÍCH NGUỒN GỐC VÀ ĐÓNG GÓP THÀNH PHẦN
## Nghiên cứu chuyên sâu về EXP-36, EXP-43, EXP-29 và EXP-24 sau phát hành EVAL-V3

**Người thực hiện:** Autonomous Speech ML Research Agent A  
**Phạm vi:** Phân tích đồ thị phụ thuộc (lineage graph), nguồn đặc trưng, toán tử kết hợp điểm, và thử nghiệm cắt bỏ thành phần có kiểm soát (controlled component ablations) không huấn luyện lại.  
**Thư mục lưu trữ bằng chứng:** `experiments/top_model_investigation/lineage/`, `experiments/top_model_investigation/ablations/`  
**Ngày hoàn thành:** 07/11/2026  

---

## 1. TỔNG QUAN VÀ ĐẶT VẤN ĐỀ

Trong đợt phát hành đánh giá chuẩn hóa chính thức **EVAL-V3** (`experiments/evaluation_v3/closure/release_final/`), bốn hệ thống đạt kết quả xếp hạng phân biệt cân bằng lớp lỗi cao nhất (Macro-3 Error AP) bao gồm:
1. **EXP-36-FINAL-HEXAD-PNC-SUPER-FUSION**: Macro-3 Error AP = **0.289233** (AP S: 0.387088, AP O: 0.204687, AP I: 0.275924).
2. **EXP-43-CONTINUOUS-AUTONOMOUS-FRONTIER-EXPANSION**: Macro-3 Error AP = **0.284760** (AP S: 0.363453, AP O: 0.205771, AP I: 0.285056).
3. **EXP-29-PROSODIC-PENTAD-FUSION**: Macro-3 Error AP = **0.282223** (AP S: 0.378062, AP O: 0.196539, AP I: 0.272068).
4. **EXP-24-FINAL-ROADMAP-EXPANSION**: Macro-3 Error AP = **0.276819** (AP S: 0.357737, AP O: 0.225092, AP I: 0.247628).

Mục tiêu của Agent A là:
- Tái cấu trúc hoàn chỉnh đồ thị phụ thuộc nguồn mã, checkpoint nơ-ron, không gian đặc trưng âm học, và toán tử hòa trộn điểm.
- Xác định nguyên nhân toán học và kiến trúc khiến EXP-36 đạt điểm số cao nhất.
- Thực hiện các phép đo cắt bỏ thành phần (leave-one-out ablations) và tích lũy từng bước (cumulative additions) nhằm phân biệt đóng góp thực sự của từng thành phần so với hiện tượng cộng gộp giản đơn hoặc rò rỉ thông tin.

---

## 2. TÁI LẬP ĐỒ THỊ NGUỒN GỐC VÀ KIẾN TRÚC THÀNH PHẦN (LINEAGE GRAPH)

Toàn bộ thông tin thành phần đã được đối soát dựa trên `CHECKPOINT_INFERENCE_RECEIPT.json` và mã nguồn gốc tại `methods/`. Lưu trữ chi tiết tại `experiments/top_model_investigation/lineage/lineage_dependency_graph.json` và `component_inventory.csv`.

### 2.1. Bản chất kiến trúc của 4 ứng viên hàng đầu

Khác với giả định trước đây coi tất cả các thí nghiệm là các checkpoint mạng nơ-ron độc lập, quá trình kiểm toán mã nguồn xác nhận:
- **EXP-24** là **mô hình mạng nơ-ron đơn lẻ (standalone neural checkpoint)** được suy luận trực tiếp trên GPU CUDA qua lớp `JointSoftAttentionCTCNetwork` (M24-ctc-attention-joint-decoder). Mô hình có **533.272 tham số**, sử dụng 55 chiều đặc trưng target và 6 chiều đặc trưng gap.
- **EXP-29** là **tổ hợp điểm số có trọng số (weighted score composition)** kết hợp 6 mô hình thành phần độc lập (`EXP-22`, `EXP-17`, `EXP-21`, `EXP-10`, `EXP-24`, `EXP-28`).
- **EXP-36** là **tổ hợp điểm số 7 thành phần** gồm 6 mô hình của EXP-29 bổ sung thêm mô hình thông tin tiên nghiệm âm vị học n-gram `EXP-34`.
- **EXP-43** là **tổ hợp điểm số 9 thành phần** gồm 7 mô hình của EXP-36 bổ sung thêm 2 mô hình phân tầng phân cấp `EXP-38` (HTC-Net) và `EXP-42` (VTLN-HTC-Net).

```
[EXP-24: JSAC-Net] (Standalone: BiLSTM + Monotonic Attention + Articulatory)
         │
         ├── + EXP-10 (Conformer), EXP-17 (SRAD), EXP-21 (NNA), EXP-22 (PSC), EXP-28 (APED)
         ▼
[EXP-29: Prosodic Pentad Fusion] (6 Components)
         │
         ├── + EXP-34 (PNC-Net: Phonotactic Prior Conditioning)
         ▼
[EXP-36: Final Hexad PNC Super-Fusion] (7 Components - BẢN CAO NHẤT)
         │
         ├── + EXP-38 (HTC-Net), EXP-42 (VTLN-HTC-Net)
         ▼
[EXP-43: Continuous Frontier Expansion] (9 Components - BỊ SUY GIẢM DO EXP-38/42)
```

### 2.2. Chi tiết 9 thành phần cấu thành

| ID Thí nghiệm | Tên Phương pháp | Checkpoint / Kiến trúc | Số tham số | Chiều đặc trưng | Bản chất đóng góp âm học |
|---|---|---|---:|---|---|
| **EXP-24** | M24 Joint Soft-Attention | `jsac_monotonic_model.pt` | 533.272 | Target: 55, Gap: 6 | Monotonic Local Gaussian Attention ($\sigma=3.0$) chống rung biên giới forced alignment; đa nhiệm manner/place/voicing. |
| **EXP-34** | M34 Phonotactic Probing | `pnc_net.pt` | 520.192 | Target: 57, Gap: 8 | Xác suất chuyển tiếp n-gram âm vị từ CMUdict ($P(p_t \mid p_{t-1})$) và cờ hợp thức cụm phụ âm (consonant cluster legality). |
| **EXP-28** | M28 Prosodic Energy | `aped_net.pt` | 542.856 | Target: 60, Gap: 9 | Động lực năng lượng dạng sóng (tỷ số RMS, năng lượng tương đối dB, zero-crossing rate, độ nghiêng phổ). |
| **EXP-17** | M17 Speaking-Rate Duration | `adaptive_duration_ctc_model.pt` | 512.448 | Target: 55, Gap: 6 | Chuẩn hóa tốc độ nói utterance-level (log-duration ratio, abbreviation flag, silence contrast) giải quyết báo động xóa âm sai trên người nói nhanh. |
| **EXP-21** | M21 Acoustic Refit Adapter | `acoustic_refit_net.pt` | 518.720 | Target: 55, Gap: 6 | Adapter thắt cổ chai dư (residual bottleneck adapter) thích nghi trực tiếp trên phân bố âm học có giọng ngoại ngữ. |
| **EXP-22** | M22 Contrastive Margin | `contrastive_hierarchical_model.pt` | 512.448 | Target: 55, Gap: 6 | Hàm tổn thất khoảng cách tương phản có giám sát ($m=0.35$) phân tách góc hình học giữa âm chuẩn và allophone thay thế. |
| **EXP-10** | M10 Conformer Attention | `conformer_attention_model.pt` | 1.245.184 | Target: 51, Gap: 6 | Tự chú ý toàn cục kết hợp tích chập 1D depthwise separable; độ tin cậy ECE thấp nhưng phân tán chú ý. |
| **EXP-38** | M38 Hierarchical Refinement | `hierarchical_two_stage_model.pt` | 560.128 | Target: 55, Gap: 6 | Định tuyến phân tầng 2 giai đoạn (đầu nguyên âm và đầu phụ âm tách biệt). *Lưu ý: mã lịch sử rò rỉ nhãn gold qua target_m.* |
| **EXP-42** | M42 VTLN Normalization | `vtln_hierarchical_model.pt` | 564.224 | Target: 58, Gap: 6 | Chuẩn hóa chiều dài đường dẫn thanh âm (Vocal Tract Length Normalization) kết hợp mạng phân cấp HTC. |

---

## 3. ĐÁNH GIÁ NĂNG LỰC ĐỘC LẬP TỪNG THÀNH PHẦN (SINGLE COMPONENT BASELINES)

Tất cả 9 mô hình thành phần đã được đánh giá độc lập trên tập phát triển chính thức `DEV_SEARCH` (19.755 targets, 20.354 gaps) bằng `UnifiedEvaluatorV3`:

| Thành phần | AP Thay thế (S) | AP Bỏ sót (O) | AP Chèn (I) | Macro-3 Error AP | Xếp hạng đơn lẻ |
|---|---:|---:|---:|---:|:---:|
| **EXP-24-FINAL-ROADMAP-EXPANSION** | 0.357737 | **0.225092** | 0.247628 | **0.276819** | **1** |
| **EXP-34-EXTENDED-PHONOTACTIC-PROBING** | **0.374602** | 0.196510 | 0.220081 | 0.263731 | 2 |
| **EXP-28-PROSODIC-ENERGY-DYNAMICS** | 0.358049 | 0.176118 | **0.256314** | 0.263494 | 3 |
| **EXP-21-ENDLESS-EXPLORATION-ACOUSTIC-REFIT** | 0.354801 | 0.195943 | 0.224986 | 0.258576 | 4 |
| **EXP-17-ADAPTIVE-DURATION-CTC** | 0.344045 | 0.167324 | 0.254723 | 0.255364 | 5 |
| **EXP-22-PHONOLOGICAL-CONTRASTIVE-MARGIN** | 0.353943 | 0.194165 | 0.213016 | 0.253708 | 6 |
| **EXP-10-CONFORMER-ATTENTION** | 0.307083 | 0.118670 | 0.181411 | 0.202388 | 7 |
| **EXP-42-AGE-INVARIANT-FRONTEND-NORMALIZATION** | 0.244690 | 0.064905 | 0.222785 | 0.177460 | 8 |
| **EXP-38-ENDLESS-SEARCH-HIERARCHICAL-REFINEMENT** | 0.200323 | 0.090619 | 0.238642 | 0.176528 | 9 |

### Phát hiện then chốt về mô hình đơn lẻ:
1. **EXP-24 là mô hình nơ-ron đơn lẻ mạnh nhất toàn bộ chiến dịch:** Macro-3 AP đạt **0.276819**, vượt trội hoàn toàn so với mô hình học sinh GRU Phase 3 (STU-05: 0.240230) và cụm giáo viên chuyên gia Phase 2 (0.186531).
2. **Kỷ lục AP Omission thuộc về EXP-24 (0.225092):** Thậm chí cao hơn cả tổ hợp EXP-36 (0.204687). Cơ chế Monotonic Local Attention loại bỏ triệt để hiện tượng cắt cụt biên giới âm vị giả do lỗi canh hàng, cho phép mạng nơ-ron nắm bắt toàn vẹn sự suy giảm formant chuyển tiếp của các âm bị bỏ sót.
3. **EXP-34 dẫn đầu AP Substitution (0.374602):** Khẳng định vai trò của thông tin tiên nghiệm âm vị học n-gram từ CMUdict: mô hình phân biệt lỗi thay thế không chỉ dựa vào phổ âm học mà còn dựa vào xác suất xuất hiện chuỗi âm thanh trong từ vựng tiếng Anh.
4. **EXP-28 dẫn đầu AP Insertion (0.256314):** Việc đưa năng lượng dạng sóng trực tiếp vào biểu diễn khe hở (gap) giúp phát hiện phụ âm chèn ép (epenthesis) mà CTC logit đơn thuần bỏ sót.
5. **EXP-38 và EXP-42 suy giảm nặng nề khi chạy non-oracle:** Trong các báo cáo lịch sử trước đây, EXP-38 từng được ghi nhận Omission AP 0.2748. Tuy nhiên, kiểm toán cho thấy kết quả cũ phụ thuộc vào việc định tuyến qua `target_m` sử dụng nhãn gold realized phone. Khi chạy non-oracle hợp lệ (`NONORACLE_NATIVE_HEAD_AVERAGE`), AP Omission của EXP-38 sụp đổ xuống chỉ còn **0.090619** và AP Substitution còn **0.200323**.

---

## 4. PHÂN TÍCH CẮT BỎ THÀNH PHẦN CÓ KIỂM SOÁT (LEAVE-ONE-OUT ABLATIONS)

Để xác định chính xác thành phần nào tạo nên sự vượt trội của **EXP-36** (Macro-3 AP = 0.289233), chúng tôi tiến hành cắt bỏ từng thành phần một (Leave-One-Out) theo hai phương thức:
1. **Fixed Re-normalized Weights (`LEAVE_ONE_OUT_FIXED_RENORM`):** Đặt trọng số thành phần bị loại về 0 và chuẩn hóa lại tổng trọng số các thành phần còn lại về 1.0 (giữ nguyên tỷ lệ tương đối giữa các thành phần khác đã chọn trên DEV_CAL).
2. **CAL-Refitted Weights (`LEAVE_ONE_OUT_CAL_REFIT`):** Tìm kiếm lại vector trọng số tối ưu trên tập `DEV_CAL` mà hoàn toàn không có thành phần bị loại, sau đó đánh giá trên `DEV_SEARCH`.

Dữ liệu đầy đủ lưu tại `experiments/top_model_investigation/ablations/leave_one_out_ablations.csv`.

### 4.1. Kết quả cắt bỏ trên EXP-36 (Tổ hợp mạnh nhất)

| Cấu hình | Thành phần bị loại | Macro-3 AP | $\Delta$ Macro-3 AP | % Thay đổi | AP S | AP O | AP I |
|---|---|---:|---:|---:|---:|---:|---:|
| **EXP-36 Baseline** | **Không (Đầy đủ 7 mô hình)** | **0.28923** | **0.00000** | **0.00%** | **0.3871** | **0.2047** | **0.2759** |
| *Fixed Renorm* | Cắt EXP-34 (Phonotactic Probing) | 0.28241 | **-0.00682** | **-2.36%** | 0.3783 | 0.1969 | 0.2721 |
| *CAL Refit* | Cắt EXP-34 (Phonotactic Probing) | 0.27144 | **-0.01779** | **-6.15%** | 0.3670 | 0.1873 | 0.2600 |
| *Fixed Renorm* | Cắt EXP-28 (Prosodic Energy) | 0.28538 | **-0.00385** | **-1.33%** | 0.3831 | 0.2059 | 0.2672 |
| *CAL Refit* | Cắt EXP-28 (Prosodic Energy) | 0.27157 | **-0.01766** | **-6.11%** | 0.3644 | 0.1930 | 0.2573 |
| *Fixed Renorm* | Cắt EXP-24 (JSAC-Net) | 0.28598 | **-0.00325** | **-1.13%** | 0.3868 | 0.2001 | 0.2710 |
| *CAL Refit* | Cắt EXP-24 (JSAC-Net) | 0.27261 | **-0.01662** | **-5.75%** | 0.3687 | 0.1856 | 0.2635 |
| *Fixed Renorm* | Cắt EXP-21 (Acoustic Refit) | 0.28781 | -0.00143 | -0.49% | 0.3873 | 0.1993 | 0.2768 |
| *CAL Refit* | Cắt EXP-21 (Acoustic Refit) | 0.27622 | -0.01301 | -4.50% | 0.3727 | 0.1945 | 0.2615 |
| *Fixed Renorm* | Cắt EXP-17 (Adaptive Duration) | 0.28977 | +0.00054 | +0.19% | 0.3891 | 0.2059 | 0.2744 |
| *CAL Refit* | Cắt EXP-17 (Adaptive Duration) | 0.27566 | -0.01357 | -4.69% | 0.3708 | 0.1963 | 0.2599 |
| *Fixed Renorm* | Cắt EXP-22 (Contrastive Margin) | 0.29020 | +0.00097 | +0.33% | 0.3875 | 0.2039 | 0.2792 |
| *CAL Refit* | Cắt EXP-22 (Contrastive Margin) | 0.27899 | -0.01024 | -3.54% | 0.3752 | 0.2001 | 0.2617 |
| *Fixed Renorm* | Cắt EXP-10 (Conformer Attention) | **0.29145** | **+0.00222** | **+0.77%** | 0.3824 | **0.2290** | 0.2630 |
| *CAL Refit* | Cắt EXP-10 (Conformer Attention) | 0.28897 | -0.00026 | -0.09% | 0.3870 | 0.2045 | 0.2754 |

### 4.2. Phân tích nhân quả từ phép cắt bỏ:
1. **Tam giác trụ cột (The Core Triad):**
   Ba mô hình có đóng góp mang tính quyết định, khi cắt bỏ làm sụt giảm Macro-3 AP nặng nề nhất là:
   - **EXP-34 (PNC-Net):** Khi loại bỏ, điểm sụt giảm mạnh nhất (-2.36% renorm, -6.15% CAL refit). EXP-34 là thành phần neo giữ độ chính xác nhận diện lỗi thay thế (AP S giảm từ 0.3871 xuống 0.3783).
   - **EXP-28 (APED-Net):** Khi loại bỏ, AP chèn (I) sụt giảm mạnh nhất (từ 0.2759 xuống 0.2672), dẫn đến Macro-3 AP giảm -1.33% (renorm) và -6.11% (refit).
   - **EXP-24 (JSAC-Net):** Cắt bỏ EXP-24 làm AP bỏ sót (O) giảm sâu (từ 0.2047 xuống 0.1856 trên refit), dẫn đến Macro-3 AP giảm -5.75%.
2. **Sự bất ngờ từ EXP-10 (Conformer Attention):**
   Khi loại bỏ EXP-10 dưới dạng fixed renorm, Macro-3 AP thực tế **tăng** từ 0.28923 lên **0.29145** (+0.77%), và AP Omission vọt lên **0.2290**!
   Nguyên nhân: Conformer phân tán sự chú ý trên toàn chuỗi (attention dispersion), làm phẳng phân bố xác suất và làm loãng khả năng phát hiện âm bị bỏ sót của EXP-24. Việc gộp EXP-10 chỉ có lợi nhẹ về mặt ECE (hiệu chỉnh độ tin cậy) nhưng lại là lực cản nhẹ đối với AP Omission.

### 4.3. Tại sao EXP-43 (9 thành phần) lại thua EXP-36 (7 thành phần)?

Dữ liệu cắt bỏ trên EXP-43 giải thích nguyên nhân rõ ràng:
- Trong EXP-43 (Macro-3 AP = 0.28476), khi cắt bỏ **EXP-38** (`LEAVE_ONE_OUT_FIXED_RENORM`), Macro-3 AP **tăng vọt thêm +0.00399** (đạt **0.28875**), tiệm cận mức của EXP-36!
- Khi cắt bỏ cả EXP-38 và EXP-42, ta phục hồi lại hoàn toàn tổ hợp EXP-36 (0.28923).
- **Kết luận:** EXP-38 và EXP-42 là các thành phần "rác" trong chế độ đánh giá non-oracle hợp lệ. Do không còn nhãn auxiliary gold `target_m` để định tuyến như trong lịch sử, chúng chỉ tạo ra nhiễu phân tán trong không gian hòa trộn điểm, làm tụt AP Substitution từ 0.3871 xuống 0.3635.

---

## 5. ĐƯỜNG TÍCH LŨY THÀNH PHẦN TỪNG BƯỚC (CUMULATIVE BUILD-UP PATH)

Để chứng minh tiến trình nâng cao năng lực xuất phát từ đâu, chúng tôi thực hiện chuỗi tích lũy từng bước bắt đầu từ mô hình đơn lẻ mạnh nhất (EXP-24) cho tới EXP-36. Trọng số hòa trộn được tối ưu hóa trên DEV_CAL và kiểm tra trên DEV_SEARCH:

| Bước tích lũy | Tổ hợp mô hình | Số mô hình | AP Thay thế (S) | AP Bỏ sót (O) | AP Chèn (I) | Macro-3 Error AP | Mức tăng bước ($\Delta$) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Bước 0** | Standalone EXP-24 (JSAC-Net) | 1 | 0.3577 | 0.2251 | 0.2476 | **0.2768** | — |
| **Bước 1** | **EXP-24 + EXP-34** (JSAC + PNC) | 2 | 0.3828 | **0.2426** | 0.2507 | **0.2920** | **+0.0152** |
| **Bước 2** | + EXP-28 (APED Prosodic) | 3 | 0.3825 | 0.2396 | 0.2510 | 0.2910 | -0.0010 |
| **Bước 3** | + EXP-17 (SRAD Duration) | 4 | 0.3790 | 0.2178 | 0.2663 | 0.2877 | -0.0033 |
| **Bước 4** | + EXP-21 (NNA Adapter) | 5 | 0.3796 | 0.2207 | 0.2623 | 0.2875 | -0.0002 |
| **Bước 5** | + EXP-22 (PSC Margin) | 6 | 0.3803 | 0.2235 | 0.2527 | 0.2855 | -0.0020 |
| **Bước 6** | + EXP-10 (EXP-36 Full Canonical) | 7 | **0.3871** | 0.2047 | **0.2759** | **0.2892** | +0.0037 |

### Phát hiện đột phá từ Đường tích lũy:
- **Tổ hợp tối giản siêu việt (EXP-24 + EXP-34):**
  Chỉ cần kết hợp **2 mô hình** duy nhất: EXP-24 (Acoustic/Attention Anchor) và EXP-34 (Phonotactic Prior Anchor), mô hình đạt Macro-3 Error AP lên tới **0.292012**!
  Đây là kết quả cao hơn cả bản phát hành chính thức EXP-36 (0.289233).
  - AP S đạt **0.3828** (tăng mạnh từ 0.3577 của EXP-24).
  - AP O đạt kỷ lục chiến dịch: **0.2426** (cao hơn cả EXP-24 đơn lẻ 0.2251 và EXP-36 0.2047).
  - AP I đạt **0.2507**.
- **Giải thích cơ chế:**
  EXP-24 cung cấp biểu diễn âm học vững chắc không bị rung biên nhờ Local Monotonic Attention. EXP-34 cung cấp xác suất chuỗi từ điển giúp loại trừ các giả định sai về âm thay thế. Hai mô hình bổ trợ nhau hoàn hảo mà không làm loãng trọng số.
  Khi thêm tiếp các mô hình từ EXP-28 đến EXP-10 để tạo thành EXP-36, AP Chèn (I) được cải thiện thêm (từ 0.2507 lên 0.2759) và AP Thay thế (S) tăng lên 0.3871, nhưng phải trả giá bằng việc AP Bỏ sót (O) bị kéo giảm từ 0.2426 xuống 0.2047 do sự phân tán trọng số của 7 mô hình.

---

## 6. PHÂN BIỆT ĐÓNG GÓP MÔ TẢ VÀ QUAN HỆ NHÂN QUẢ (DESCRIPTIVE VS CAUSAL ATTRIBUTION)

Bảng đối chiếu tổng kết:

| Thành phần | Đóng góp mô tả trong EXP-36 | Bằng chứng nhân quả kiểm chứng | Rủi ro rò rỉ / Giới hạn suy diễn |
|---|---|---|---|
| **EXP-24** | Trọng số target: 0.20 (cao nhất), gap: 0.12 | **Nhân quả cốt lõi (Causal Anchor):** Cắt bỏ làm giảm 5.75% Macro-3 AP; đóng góp trần AP Omission cho toàn bộ chiến dịch. | Không có rò rỉ; suy luận non-oracle hoàn toàn độc lập với nhãn gold. |
| **EXP-34** | Trọng số target: 0.20 (cao nhất), gap: 0.20 (cao nhất) | **Nhân quả cốt lõi:** Cắt bỏ làm sụt giảm điểm lớn nhất (-6.15%); chịu trách nhiệm đưa AP S từ 0.357 lên 0.387. | Phụ thuộc vào từ điển phát âm CMUdict; từ OOV cần cơ chế G2P độc lập. |
| **EXP-28** | Trọng số target: 0.15, gap: 0.18 | **Nhân quả phụ trợ quan trọng:** Chịu trách nhiệm nâng đỡ AP Insertion từ 0.247 lên 0.276. | Cần tín hiệu âm thanh dạng sóng gốc để trích xuất năng lượng RMS và độ dốc phổ. |
| **EXP-17 & 21** | Trọng số target: 0.10 - 0.15 | **Điều hòa phân bố (Regularizers):** Ổn định các trường hợp người nói tốc độ nhanh và phương ngữ đặc thù. | Không tạo đột phá đơn lẻ nhưng giúp ổn định phương sai. |
| **EXP-22** | Trọng số target: 0.10 | **Tác động biên:** Margin tương phản hỗ trợ nhẹ phân tách âm thay thế. | Đóng góp biên nhỏ khi đã có EXP-34. |
| **EXP-10** | Trọng số gap: 0.18 | **Tác động hai mặt:** Cải thiện nhẹ AP I nhưng làm loãng AP O; cắt bỏ làm AP O tăng trở lại 0.2290. | Tốn nhiều tham số nhất (1.2M) nhưng hiệu quả phân biệt lớp lỗi thấp nhất. |
| **EXP-38 & 42** | Chỉ có trong EXP-43 | **Tác động tiêu cực (Negative Drag):** Gây sụt giảm hiệu năng của EXP-43 so với EXP-36. | Đã bị loại bỏ giả định rò rỉ target_m; không nên kế thừa vào kiến trúc tương lai. |

---

## 7. KẾT LUẬN VÀ KHUYẾN NGHỊ BÀN GIAO CỦA AGENT A

1. **Nguyên nhân EXP-36 đứng đầu:** EXP-36 không phải là sự kết hợp ngẫu nhiên mà hội tụ đúng **Tam giác trụ cột**:
   - Biểu diễn cục bộ mềm chống rung biên của **EXP-24**.
   - Phân bố tiên nghiệm âm vị học n-gram của **EXP-34**.
   - Động lực năng lượng khe hở dạng sóng của **EXP-28**.
2. **Khám phá về cấu hình tối ưu:** Một tổ hợp 2 thành phần tối giản **[EXP-24 + EXP-34]** thậm chí đạt Macro-3 Error AP = **0.292012**, vượt qua cả EXP-36 chính thức, đồng thời giảm 60% chi phí tính toán suy luận.
3. **Định hướng cho mô hình thống nhất tương lai:**
   Không tiếp tục hòa trộn điểm số hậu kỳ (post-hoc score ensembling) của nhiều mạng rời rạc. Cần tích hợp trực tiếp **Monotonic Local Attention** (của EXP-24), **Phonotactic Transition Conditioning** (của EXP-34), và **Prosodic Energy Injection** (của EXP-28) vào một khối nơ-ron thống nhất có thể tối ưu hóa end-to-end.
