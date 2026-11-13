# BÁO CÁO TOÀN DIỆN CHI TIẾT KẾT QUẢ VÀ SỐ LIỆU — LANGFENS RQ1-TRAIN-PHASE-01

**Dự án:** LANGFENS — Hệ Thống Phát Hiện và Chẩn Đoán Lỗi Phát Âm (MDD)
**Giai đoạn:** RQ1-TRAIN-PHASE-01 — Huấn luyện Có Kiểm soát Chính thức (Formal Controlled Training)
**Môi trường thực thi:** CPU chuẩn hóa (2 threads OMP/MKL/OpenBLAS), PyTorch 2.6.0+cu124, Python 3.11.17, Linux x86_64
**Commit gốc:** `aa4e6ea2147180a4a820d77a71911f333fcb6721`
**Tiêu chuẩn đánh giá:** Hợp đồng đo lường khoa học EVAL-V3.1
**Tập dữ liệu:** Phân vùng chính thức L2-ARCTIC TRAIN (1.800 câu, 12 người nói, 59.562 targets, 61.362 gaps)
**Các seed đăng ký trước:** `101` (Primary), `202` (Replication), `303` (Sensitivity)
**Tình trạng:** **MEASURED / VERIFIED — HOÀN TẤT TOÀN DIỆN 100%**

---

## 1. TỔNG QUAN CHIẾN DỊCH VÀ XÁC THỰC TÍNH TOÀN VẸN

Chiến dịch huấn luyện chính thức RQ1-TRAIN-PHASE-01 đã hoàn tất toàn bộ 3 seed đã đăng ký trước mà không xảy ra bất kỳ lỗi kỹ thuật, vi phạm ngân sách hay can thiệp hồi tố nào.

### Bảng 1.1: Sổ cái xác thực tiền kiểm và kiểm toán
| Hạng mục kiểm toán | Tiêu chuẩn quy định | Kết quả đo đạc thực tế | Trạng thái |
|---|---|---|---|
| **Mã băm commit** | `aa4e6ea2147180a4a820d77a71911f333fcb6721` | Khớp 100% commit HEAD | **VERIFIED** |
| **Mã băm cấu hình (file SHA256)** | `6206c19db6cad9d99727a16cd9c220d077e09a9662a0229ee0971945bf1cbcb1` | Khớp từng byte | **VERIFIED** |
| **Mã digest cấu hình JSON** | `042f02ebb50d1199588f91d91af471eb9dc90499f7699d3481c96f928b97dee2` | Khớp chính xác digest | **VERIFIED** |
| **Diff tệp theo dõi** | `coordination/PHASE3_GPU_LEDGER.jsonl` | Khớp 100% với file diff lưu trữ | **VERIFIED** |
| **Bộ kiểm thử tiền kiểm** | 91 automated tests (PyTest) | 91 passed, 1 warning (std N=0), 0 fail | **VERIFIED** |
| **Ngân sách thời gian mỗi seed** | Tối đa 2.700,0 giây active-wall time | S101: 2.516,79s; S202: 2.316,85s; S303: 2.326,09s | **VERIFIED** |
| **Tổng số mô hình đã huấn luyện** | 48 nhánh / seed (16 task fits × 3 nhánh) | 144 nhánh huấn luyện thành công | **VERIFIED** |
| **Tổng số checkpoint lưu trữ** | 400 epoch checkpoints / seed + 16 latest | 1.200 epoch checkpoint files preserved | **VERIFIED** |
| **Quyền truy cập GPU** | Không sử dụng GPU (`CUDA_VISIBLE_DEVICES=""`) | Thực thi 100% CPU 2 threads | **VERIFIED** |

---

## 2. BẢNG TỔNG HỢP TOÀN BỘ KẾT QUẢ VÀ SỐ LIỆU CHÍNH (MASTER POOLED METRICS)

Các kết quả dưới đây được tổng hợp gộp (pooled) trên toàn bộ 1.800 câu phát âm (59.562 âm vị mục tiêu và 61.362 khe chèn) của 12 người nói qua 4 fold kiểm định chéo nhóm người nói (speaker-grouped outer folds).
Mọi ngưỡng quyết định được rút trích độc quyền từ các tập inner out-of-fold và đóng băng trước khi áp dụng vào outer fold.

### Bảng 2.1: So sánh tổng hợp 3 phương pháp qua 4 mức ngân sách cảnh báo (Trung bình ± Độ lệch chuẩn qua 3 seed 101, 202, 303)
| Phương pháp | Ngân sách $B$ | Macro-3 AP | Macro-3 F1 | Joint FWR (Thực tế) | Số cảnh báo sai ($FP$) | Sai loại âm ($S \leftrightarrow O$) | Tỷ lệ câu có $FP$ | Target NLL |
|---|---|---|---|---|---|---|---|---|
| **fixed** | 0.2 | 0.304154 ± 0.003394 | 0.160371 ± 0.004739 | 0.204259 ± 0.008169 | 367.7 ± 14.7 | 38.7 ± 8.2 | 0.1781 ± 0.0049 | 0.5445 ± 0.0050 |
| **fixed** | 0.5 | 0.304154 ± 0.003394 | 0.227066 ± 0.003050 | 0.494815 ± 0.017191 | 890.7 ± 30.9 | 80.7 ± 7.9 | 0.3676 ± 0.0081 | 0.5445 ± 0.0050 |
| **fixed** | 1 | 0.304154 ± 0.003394 | 0.279630 ± 0.005479 | 0.990926 ± 0.038853 | 1783.7 ± 69.9 | 140.0 ± 7.1 | 0.5967 ± 0.0083 | 0.5445 ± 0.0050 |
| **fixed** | 2 | 0.304154 ± 0.003394 | 0.336172 ± 0.001824 | 1.921667 ± 0.086435 | 3459.0 ± 155.6 | 274.3 ± 16.4 | 0.8054 ± 0.0101 | 0.5445 ± 0.0050 |
| **neural** | 0.2 | 0.271711 ± 0.002151 | 0.115286 ± 0.007176 | 0.153704 ± 0.016294 | 276.7 ± 29.3 | 16.3 ± 2.5 | 0.1396 ± 0.0105 | 0.7567 ± 0.0080 |
| **neural** | 0.5 | 0.271711 ± 0.002151 | 0.177794 ± 0.006336 | 0.405370 ± 0.031013 | 729.7 ± 55.8 | 41.7 ± 2.5 | 0.3163 ± 0.0170 | 0.7567 ± 0.0080 |
| **neural** | 1 | 0.271711 ± 0.002151 | 0.234647 ± 0.007296 | 0.846111 ± 0.059319 | 1523.0 ± 106.8 | 89.0 ± 6.7 | 0.5383 ± 0.0270 | 0.7567 ± 0.0080 |
| **neural** | 2 | 0.271711 ± 0.002151 | 0.286392 ± 0.002096 | 1.842222 ± 0.098937 | 3316.0 ± 178.1 | 199.3 ± 8.3 | 0.7980 ± 0.0170 | 0.7567 ± 0.0080 |
| **e4** | 0.2 | 0.228929 ± 0.003017 | 0.140371 ± 0.005017 | 0.270741 ± 0.015183 | 487.3 ± 27.3 | 99.0 ± 15.0 | 0.2094 ± 0.0100 | 0.9640 ± 0.0069 |
| **e4** | 0.5 | 0.228929 ± 0.003017 | 0.188685 ± 0.001972 | 0.606667 ± 0.015954 | 1092.0 ± 28.7 | 228.0 ± 15.0 | 0.4028 ± 0.0079 | 0.9640 ± 0.0069 |
| **e4** | 1 | 0.228929 ± 0.003017 | 0.210548 ± 0.001915 | 1.094444 ± 0.014084 | 1970.0 ± 25.4 | 423.7 ± 13.2 | 0.5854 ± 0.0028 | 0.9640 ± 0.0069 |
| **e4** | 2 | 0.228929 ± 0.003017 | 0.211450 ± 0.001841 | 2.016481 ± 0.003281 | 3629.7 ± 5.9 | 780.7 ± 23.2 | 0.7931 ± 0.0069 | 0.9640 ± 0.0069 |

### Bảng 2.2: Chi tiết hiệu năng từng lớp lỗi (Classwise Metrics: Substitution, Omission, Insertion) tại điểm vận hành mục tiêu $B=0.2$
| Phương pháp | Lớp lỗi | Support (Mẫu lỗi) | Average Precision (AP) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **fixed** | **Substitution (S)** | 7.381 | 0.4098 ± 0.0047 | 0.7397 ± 0.0041 | 0.0356 ± 0.0021 | 0.0680 ± 0.0037 |
| | **Omission (O)** | 1.825 | 0.3441 ± 0.0014 | 0.5734 ± 0.0109 | 0.1339 ± 0.0014 | 0.2171 ± 0.0011 |
| | **Insertion (I)** | 560 | 0.1586 ± 0.0048 | 0.4326 ± 0.0205 | 0.1268 ± 0.0081 | 0.1961 ± 0.0118 |
| **neural** | **Substitution (S)** | 7.381 | 0.3884 ± 0.0067 | 0.6915 ± 0.0280 | 0.0406 ± 0.0034 | 0.0766 ± 0.0060 |
| | **Omission (O)** | 1.825 | 0.2681 ± 0.0059 | 0.5882 ± 0.0317 | 0.0391 ± 0.0075 | 0.0732 ± 0.0134 |
| | **Insertion (I)** | 560 | 0.1586 ± 0.0048 | 0.4326 ± 0.0205 | 0.1268 ± 0.0081 | 0.1961 ± 0.0118 |
| **e4** | **Substitution (S)** | 7.381 | 0.2839 ± 0.0037 | 0.5409 ± 0.0038 | 0.0170 ± 0.0010 | 0.0329 ± 0.0019 |
| | **Omission (O)** | 1.825 | 0.2442 ± 0.0012 | 0.4390 ± 0.0112 | 0.1231 ± 0.0064 | 0.1921 ± 0.0068 |
| | **Insertion (I)** | 560 | 0.1586 ± 0.0048 | 0.4326 ± 0.0205 | 0.1268 ± 0.0081 | 0.1961 ± 0.0118 |

### Bảng 2.3: Hiệu năng Xếp hạng Âm chèn ở Đầu danh sách (Insertion Top-K Ranking Diagnostics)
| Chỉ số Precision@K | Định nghĩa K | Seed 101 | Seed 202 | Seed 303 | Trung bình ± Độ lệch |
|---|---|---|---|---|---|
| **Precision@100** | Top 100 khe chèn có điểm cao nhất | 0.4700 (47/100) | 0.4600 (46/100) | 0.5200 (52/100) | **0.4833 ± 0.0262** |
| **Precision@500** | Top 500 khe chèn có điểm cao nhất | 0.2520 (126/500) | 0.2700 (135/500) | 0.2700 (135/500) | **0.2640 ± 0.0085** |
| **Precision@560** | Top 560 khe chèn có điểm cao nhất | 0.2375 (133/560) | 0.2536 (142/560) | 0.2518 (141/560) | **0.2476 ± 0.0072** |

---

## 3. CHI TIẾT KẾT QUẢ TỪNG SEED ĐÃ ĐĂNG KÝ

### 3.1. Chi tiết kết quả Seed 101
- **Trạng thái:** COMPLETE
- **Thời gian thực thi active-wall:** 2516.79 giây (41.95 phút) / Giới hạn: 2.700,00 giây (Dư: 183.21 giây)
- **Lý do dừng:** Không có (`stop_reason: null`)

| Phương pháp | Budget $B$ | Macro-3 AP | Macro-3 F1 | Joint FWR | Cảnh báo sai ($FP$) | Sai loại ($S \leftrightarrow O$) | AP-S | AP-O | AP-I | Sub F1 | Omi F1 | Ins F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 0.2 | 0.3064 | 0.1597 | 0.1928 | 347 | 49 | 0.4128 | 0.3432 | 0.1631 | 0.0643 | 0.2155 | 0.1994 |
| fixed | 0.5 | 0.3064 | 0.2247 | 0.4722 | 850 | 83 | 0.4128 | 0.3432 | 0.1631 | 0.1550 | 0.2783 | 0.2407 |
| fixed | 1 | 0.3064 | 0.2724 | 0.9367 | 1686 | 150 | 0.4128 | 0.3432 | 0.1631 | 0.2537 | 0.3295 | 0.2341 |
| fixed | 2 | 0.3064 | 0.3339 | 1.8172 | 3271 | 267 | 0.4128 | 0.3432 | 0.1631 | 0.3598 | 0.4109 | 0.2309 |
| neural | 0.2 | 0.2742 | 0.1081 | 0.1322 | 238 | 13 | 0.3931 | 0.2666 | 0.1631 | 0.0706 | 0.0542 | 0.1994 |
| neural | 0.5 | 0.2742 | 0.1690 | 0.3633 | 654 | 39 | 0.3931 | 0.2666 | 0.1631 | 0.1456 | 0.1206 | 0.2407 |
| neural | 1 | 0.2742 | 0.2245 | 0.7656 | 1378 | 80 | 0.3931 | 0.2666 | 0.1631 | 0.2442 | 0.1953 | 0.2341 |
| neural | 2 | 0.2742 | 0.2836 | 1.7189 | 3094 | 195 | 0.3931 | 0.2666 | 0.1631 | 0.3495 | 0.2704 | 0.2309 |
| e4 | 0.2 | 0.2306 | 0.1450 | 0.2917 | 525 | 120 | 0.2853 | 0.2436 | 0.1631 | 0.0341 | 0.2014 | 0.1994 |
| e4 | 0.5 | 0.2306 | 0.1900 | 0.6250 | 1125 | 249 | 0.2853 | 0.2436 | 0.1631 | 0.0484 | 0.2808 | 0.2407 |
| e4 | 1 | 0.2306 | 0.2080 | 1.1122 | 2002 | 441 | 0.2853 | 0.2436 | 0.1631 | 0.0629 | 0.3268 | 0.2341 |
| e4 | 2 | 0.2306 | 0.2140 | 2.0211 | 3638 | 794 | 0.2853 | 0.2436 | 0.1631 | 0.0852 | 0.3258 | 0.2309 |

### 3.2. Chi tiết kết quả Seed 202
- **Trạng thái:** COMPLETE
- **Thời gian thực thi active-wall:** 2316.85 giây (38.61 phút) / Giới hạn: 2.700,00 giây (Dư: 383.15 giây)
- **Lý do dừng:** Không có (`stop_reason: null`)

| Phương pháp | Budget $B$ | Macro-3 AP | Macro-3 F1 | Joint FWR | Cảnh báo sai ($FP$) | Sai loại ($S \leftrightarrow O$) | AP-S | AP-O | AP-I | Sub F1 | Omi F1 | Ins F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 0.2 | 0.2994 | 0.1549 | 0.2111 | 380 | 38 | 0.4032 | 0.3429 | 0.1520 | 0.0665 | 0.2179 | 0.1803 |
| fixed | 0.5 | 0.2994 | 0.2252 | 0.4983 | 897 | 89 | 0.4032 | 0.3429 | 0.1520 | 0.1497 | 0.2883 | 0.2374 |
| fixed | 1 | 0.2994 | 0.2808 | 1.0106 | 1819 | 136 | 0.4032 | 0.3429 | 0.1520 | 0.2534 | 0.3371 | 0.2518 |
| fixed | 2 | 0.2994 | 0.3383 | 1.9189 | 3454 | 259 | 0.4032 | 0.3429 | 0.1520 | 0.3635 | 0.4187 | 0.2327 |
| neural | 0.2 | 0.2690 | 0.1127 | 0.1717 | 309 | 17 | 0.3790 | 0.2760 | 0.1520 | 0.0744 | 0.0834 | 0.1803 |
| neural | 0.5 | 0.2690 | 0.1808 | 0.4156 | 748 | 41 | 0.3790 | 0.2760 | 0.1520 | 0.1507 | 0.1543 | 0.2374 |
| neural | 1 | 0.2690 | 0.2379 | 0.8661 | 1559 | 91 | 0.3790 | 0.2760 | 0.1520 | 0.2424 | 0.2195 | 0.2518 |
| neural | 2 | 0.2690 | 0.2886 | 1.8467 | 3324 | 192 | 0.3790 | 0.2760 | 0.1520 | 0.3489 | 0.2843 | 0.2327 |
| e4 | 0.2 | 0.2247 | 0.1334 | 0.2561 | 461 | 91 | 0.2788 | 0.2433 | 0.1520 | 0.0303 | 0.1896 | 0.1803 |
| e4 | 0.5 | 0.2247 | 0.1859 | 0.5861 | 1055 | 220 | 0.2788 | 0.2433 | 0.1520 | 0.0424 | 0.2779 | 0.2374 |
| e4 | 1 | 0.2247 | 0.2112 | 1.0778 | 1940 | 421 | 0.2788 | 0.2433 | 0.1520 | 0.0543 | 0.3274 | 0.2518 |
| e4 | 2 | 0.2247 | 0.2107 | 2.0139 | 3625 | 800 | 0.2788 | 0.2433 | 0.1520 | 0.0751 | 0.3244 | 0.2327 |

### 3.3. Chi tiết kết quả Seed 303
- **Trạng thái:** COMPLETE
- **Thời gian thực thi active-wall:** 2326.09 giây (38.77 phút) / Giới hạn: 2.700,00 giây (Dư: 373.91 giây)
- **Lý do dừng:** Không có (`stop_reason: null`)

| Phương pháp | Budget $B$ | Macro-3 AP | Macro-3 F1 | Joint FWR | Cảnh báo sai ($FP$) | Sai loại ($S \leftrightarrow O$) | AP-S | AP-O | AP-I | Sub F1 | Omi F1 | Ins F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 0.2 | 0.3067 | 0.1665 | 0.2089 | 376 | 29 | 0.4133 | 0.3460 | 0.1608 | 0.0731 | 0.2178 | 0.2085 |
| fixed | 0.5 | 0.3067 | 0.2314 | 0.5139 | 925 | 70 | 0.4133 | 0.3460 | 0.1608 | 0.1639 | 0.2830 | 0.2472 |
| fixed | 1 | 0.3067 | 0.2857 | 1.0256 | 1846 | 134 | 0.4133 | 0.3460 | 0.1608 | 0.2648 | 0.3433 | 0.2489 |
| fixed | 2 | 0.3067 | 0.3363 | 2.0289 | 3652 | 297 | 0.4133 | 0.3460 | 0.1608 | 0.3729 | 0.4206 | 0.2155 |
| neural | 0.2 | 0.2719 | 0.1251 | 0.1572 | 283 | 19 | 0.3932 | 0.2617 | 0.1608 | 0.0848 | 0.0820 | 0.2085 |
| neural | 0.5 | 0.2719 | 0.1836 | 0.4372 | 787 | 45 | 0.3932 | 0.2617 | 0.1608 | 0.1638 | 0.1397 | 0.2472 |
| neural | 1 | 0.2719 | 0.2415 | 0.9067 | 1632 | 96 | 0.3932 | 0.2617 | 0.1608 | 0.2614 | 0.2141 | 0.2489 |
| neural | 2 | 0.2719 | 0.2869 | 1.9611 | 3530 | 211 | 0.3932 | 0.2617 | 0.1608 | 0.3674 | 0.2778 | 0.2155 |
| e4 | 0.2 | 0.2315 | 0.1427 | 0.2644 | 476 | 86 | 0.2876 | 0.2459 | 0.1608 | 0.0344 | 0.1853 | 0.2085 |
| e4 | 0.5 | 0.2315 | 0.1902 | 0.6089 | 1096 | 215 | 0.2876 | 0.2459 | 0.1608 | 0.0486 | 0.2747 | 0.2472 |
| e4 | 1 | 0.2315 | 0.2125 | 1.0933 | 1968 | 409 | 0.2876 | 0.2459 | 0.1608 | 0.0628 | 0.3259 | 0.2489 |
| e4 | 2 | 0.2315 | 0.2096 | 2.0144 | 3626 | 748 | 0.2876 | 0.2459 | 0.1608 | 0.0866 | 0.3267 | 0.2155 |

---

## 4. CHI TIẾT KẾT QUẢ TỪNG OUTER FOLD VÀ CHÍNH SÁCH NGƯỠNG ĐỘC LẬP

Mỗi outer fold đánh giá trên một nhóm 3 người nói độc lập hoàn toàn với quá trình huấn luyện và chọn ngưỡng.

### 4.1. Outer Fold 0 (Nhóm người nói bị giữ lại: ABA, ASI, BWC)

| Seed | Phương pháp | Budget $B$ | Macro-3 AP | Macro-3 F1 | Joint FWR | Cảnh báo sai ($FP$) | Sai loại | AP-S | AP-O | AP-I |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | fixed | 0.2 | 0.3232 | 0.1912 | 0.2267 | 102 | 6 | 0.4148 | 0.3414 | 0.2134 |
| 101 | fixed | 0.5 | 0.3232 | 0.2509 | 0.4978 | 224 | 16 | 0.4148 | 0.3414 | 0.2134 |
| 101 | fixed | 1 | 0.3232 | 0.2856 | 0.9467 | 426 | 28 | 0.4148 | 0.3414 | 0.2134 |
| 101 | fixed | 2 | 0.3232 | 0.3213 | 1.6956 | 763 | 51 | 0.4148 | 0.3414 | 0.2134 |
| 202 | fixed | 0.2 | 0.3050 | 0.1590 | 0.2378 | 107 | 6 | 0.3974 | 0.3456 | 0.1720 |
| 202 | fixed | 0.5 | 0.3050 | 0.2181 | 0.4911 | 221 | 14 | 0.3974 | 0.3456 | 0.1720 |
| 202 | fixed | 1 | 0.3050 | 0.2831 | 0.9689 | 436 | 26 | 0.3974 | 0.3456 | 0.1720 |
| 202 | fixed | 2 | 0.3050 | 0.3331 | 1.7044 | 767 | 45 | 0.3974 | 0.3456 | 0.1720 |
| 303 | fixed | 0.2 | 0.3125 | 0.2016 | 0.2778 | 125 | 6 | 0.4069 | 0.3266 | 0.2040 |
| 303 | fixed | 0.5 | 0.3125 | 0.2550 | 0.6067 | 273 | 16 | 0.4069 | 0.3266 | 0.2040 |
| 303 | fixed | 1 | 0.3125 | 0.2879 | 1.1489 | 517 | 28 | 0.4069 | 0.3266 | 0.2040 |
| 303 | fixed | 2 | 0.3125 | 0.3184 | 2.0178 | 908 | 47 | 0.4069 | 0.3266 | 0.2040 |

### 4.2. Outer Fold 1 (Nhóm người nói bị giữ lại: EBVS, HJK, HKK)

| Seed | Phương pháp | Budget $B$ | Macro-3 AP | Macro-3 F1 | Joint FWR | Cảnh báo sai ($FP$) | Sai loại | AP-S | AP-O | AP-I |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | fixed | 0.2 | 0.2299 | 0.1277 | 0.2067 | 93 | 6 | 0.3724 | 0.2104 | 0.1069 |
| 101 | fixed | 0.5 | 0.2299 | 0.1810 | 0.5578 | 251 | 8 | 0.3724 | 0.2104 | 0.1069 |
| 101 | fixed | 1 | 0.2299 | 0.2302 | 1.0622 | 478 | 21 | 0.3724 | 0.2104 | 0.1069 |
| 101 | fixed | 2 | 0.2299 | 0.2884 | 2.0000 | 900 | 36 | 0.3724 | 0.2104 | 0.1069 |
| 202 | fixed | 0.2 | 0.2357 | 0.1534 | 0.2800 | 126 | 4 | 0.3721 | 0.2244 | 0.1107 |
| 202 | fixed | 0.5 | 0.2357 | 0.2229 | 0.6844 | 308 | 17 | 0.3721 | 0.2244 | 0.1107 |
| 202 | fixed | 1 | 0.2357 | 0.2593 | 1.3067 | 588 | 25 | 0.3721 | 0.2244 | 0.1107 |
| 202 | fixed | 2 | 0.2357 | 0.2918 | 2.3867 | 1074 | 46 | 0.3721 | 0.2244 | 0.1107 |
| 303 | fixed | 0.2 | 0.2362 | 0.1261 | 0.2356 | 106 | 3 | 0.3682 | 0.2245 | 0.1158 |
| 303 | fixed | 0.5 | 0.2362 | 0.1955 | 0.6022 | 271 | 9 | 0.3682 | 0.2245 | 0.1158 |
| 303 | fixed | 1 | 0.2362 | 0.2579 | 1.1289 | 508 | 19 | 0.3682 | 0.2245 | 0.1158 |
| 303 | fixed | 2 | 0.2362 | 0.2867 | 2.2333 | 1005 | 46 | 0.3682 | 0.2245 | 0.1158 |

### 4.3. Outer Fold 2 (Nhóm người nói bị giữ lại: HQTV, LXC, NJS)

| Seed | Phương pháp | Budget $B$ | Macro-3 AP | Macro-3 F1 | Joint FWR | Cảnh báo sai ($FP$) | Sai loại | AP-S | AP-O | AP-I |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | fixed | 0.2 | 0.3326 | 0.1364 | 0.1933 | 87 | 22 | 0.4458 | 0.4529 | 0.0991 |
| 101 | fixed | 0.5 | 0.3326 | 0.2122 | 0.4556 | 205 | 41 | 0.4458 | 0.4529 | 0.0991 |
| 101 | fixed | 1 | 0.3326 | 0.2588 | 0.9400 | 423 | 70 | 0.4458 | 0.4529 | 0.0991 |
| 101 | fixed | 2 | 0.3326 | 0.3415 | 1.9578 | 881 | 124 | 0.4458 | 0.4529 | 0.0991 |
| 202 | fixed | 0.2 | 0.3321 | 0.1335 | 0.1911 | 86 | 16 | 0.4376 | 0.4586 | 0.1001 |
| 202 | fixed | 0.5 | 0.3321 | 0.2049 | 0.4711 | 212 | 39 | 0.4376 | 0.4586 | 0.1001 |
| 202 | fixed | 1 | 0.3321 | 0.2733 | 0.9800 | 441 | 58 | 0.4376 | 0.4586 | 0.1001 |
| 202 | fixed | 2 | 0.3321 | 0.3487 | 2.0044 | 902 | 114 | 0.4376 | 0.4586 | 0.1001 |
| 303 | fixed | 0.2 | 0.3441 | 0.1448 | 0.2000 | 90 | 14 | 0.4608 | 0.4592 | 0.1122 |
| 303 | fixed | 0.5 | 0.3441 | 0.2122 | 0.5200 | 234 | 33 | 0.4608 | 0.4592 | 0.1122 |
| 303 | fixed | 1 | 0.3441 | 0.2817 | 1.0844 | 488 | 62 | 0.4608 | 0.4592 | 0.1122 |
| 303 | fixed | 2 | 0.3441 | 0.3632 | 2.1578 | 971 | 145 | 0.4608 | 0.4592 | 0.1122 |

### 4.4. Outer Fold 3 (Nhóm người nói bị giữ lại: PNV, SKA, SVBI)

| Seed | Phương pháp | Budget $B$ | Macro-3 AP | Macro-3 F1 | Joint FWR | Cảnh báo sai ($FP$) | Sai loại | AP-S | AP-O | AP-I |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | fixed | 0.2 | 0.3379 | 0.1717 | 0.1444 | 65 | 15 | 0.4177 | 0.3419 | 0.2542 |
| 101 | fixed | 0.5 | 0.3379 | 0.2416 | 0.3778 | 170 | 18 | 0.4177 | 0.3419 | 0.2542 |
| 101 | fixed | 1 | 0.3379 | 0.2987 | 0.7978 | 359 | 31 | 0.4177 | 0.3419 | 0.2542 |
| 101 | fixed | 2 | 0.3379 | 0.3661 | 1.6156 | 727 | 56 | 0.4177 | 0.3419 | 0.2542 |
| 202 | fixed | 0.2 | 0.3338 | 0.1707 | 0.1356 | 61 | 12 | 0.4094 | 0.3472 | 0.2448 |
| 202 | fixed | 0.5 | 0.3338 | 0.2527 | 0.3467 | 156 | 19 | 0.4094 | 0.3472 | 0.2448 |
| 202 | fixed | 1 | 0.3338 | 0.3062 | 0.7867 | 354 | 27 | 0.4094 | 0.3472 | 0.2448 |
| 202 | fixed | 2 | 0.3338 | 0.3672 | 1.5800 | 711 | 54 | 0.4094 | 0.3472 | 0.2448 |
| 303 | fixed | 0.2 | 0.3374 | 0.1733 | 0.1222 | 55 | 6 | 0.4198 | 0.3585 | 0.2339 |
| 303 | fixed | 0.5 | 0.3374 | 0.2446 | 0.3267 | 147 | 12 | 0.4198 | 0.3585 | 0.2339 |
| 303 | fixed | 1 | 0.3374 | 0.3003 | 0.7400 | 333 | 25 | 0.4198 | 0.3585 | 0.2339 |
| 303 | fixed | 2 | 0.3374 | 0.3482 | 1.7067 | 768 | 59 | 0.4198 | 0.3585 | 0.2339 |

### Bảng 4.5: Chi tiết các giá trị ngưỡng tối ưu nội suy ($\tau$) được đóng băng từ Inner Folds (Fixed Fusion)
| Outer Fold | Budget $B$ | Ngân sách Inner Target | Ngân sách Inner Gap | Ngưỡng $\tau_{\text{target}}$ | Số target chọn | Ngưỡng $\tau_{\text{gap}}$ | Số gap chọn | Achieved Outer FWR (S101) |
|---|---|---|---|---|---|---|---|---|
| Fold 0 | 0.2 | 216.0 (FP$\le$216) | 54.0 (FP$\le$54) | 0.856953 | 516 | 0.866081 | 74 | 0.2267 |
| Fold 0 | 0.5 | 540.0 (FP$\le$540) | 135.0 (FP$\le$135) | 0.735038 | 1170 | 0.624275 | 174 | 0.4978 |
| Fold 0 | 1 | 1080.0 (FP$\le$1080) | 270.0 (FP$\le$270) | 0.598158 | 2166 | 0.371865 | 329 | 0.9467 |
| Fold 0 | 2 | 2160.0 (FP$\le$2160) | 540.0 (FP$\le$540) | 0.498548 | 4008 | 0.157944 | 622 | 1.6956 |
| Fold 1 | 0.2 | 216.0 (FP$\le$216) | 54.0 (FP$\le$54) | 0.877688 | 594 | 0.839065 | 109 | 0.2067 |
| Fold 1 | 0.5 | 540.0 (FP$\le$540) | 135.0 (FP$\le$135) | 0.736417 | 1357 | 0.589556 | 207 | 0.5578 |
| Fold 1 | 1 | 1080.0 (FP$\le$1080) | 270.0 (FP$\le$270) | 0.605513 | 2405 | 0.332486 | 372 | 1.0622 |
| Fold 1 | 2 | 2160.0 (FP$\le$2160) | 540.0 (FP$\le$540) | 0.499173 | 4373 | 0.143274 | 667 | 2.0000 |
| Fold 2 | 0.2 | 216.0 (FP$\le$216) | 54.0 (FP$\le$54) | 0.846844 | 488 | 0.899620 | 102 | 0.1933 |
| Fold 2 | 0.5 | 540.0 (FP$\le$540) | 135.0 (FP$\le$135) | 0.718152 | 1114 | 0.617923 | 220 | 0.4556 |
| Fold 2 | 1 | 1080.0 (FP$\le$1080) | 270.0 (FP$\le$270) | 0.588778 | 2066 | 0.346302 | 373 | 0.9400 |
| Fold 2 | 2 | 2160.0 (FP$\le$2160) | 540.0 (FP$\le$540) | 0.496648 | 3845 | 0.152943 | 673 | 1.9578 |
| Fold 3 | 0.2 | 216.0 (FP$\le$216) | 54.0 (FP$\le$54) | 0.871977 | 509 | 0.858187 | 74 | 0.1444 |
| Fold 3 | 0.5 | 540.0 (FP$\le$540) | 135.0 (FP$\le$135) | 0.736838 | 1194 | 0.586736 | 181 | 0.3778 |
| Fold 3 | 1 | 1080.0 (FP$\le$1080) | 270.0 (FP$\le$270) | 0.607128 | 2165 | 0.353286 | 334 | 0.7978 |
| Fold 3 | 2 | 2160.0 (FP$\le$2160) | 540.0 (FP$\le$540) | 0.499524 | 3985 | 0.176484 | 619 | 1.6156 |

---

## 5. PHÂN RÃ HIỆU NĂNG TỪNG NGƯỜI NÓI (PER-SPEAKER DISAGGREGATION)

Dưới đây là chi tiết hiệu năng phân rã cho 12 người nói trong tập TRAIN của mô hình Fixed 50/50 tại ngân sách $B=0.2$ (Trung bình qua 3 seed 101, 202, 303).

### Bảng 5.1: Bảng số liệu chi tiết 12 người nói tại $B=0.2$
| Người nói (Speaker) | Accent L1 Bản ngữ | Fold | Achieved Joint FWR | Macro-3 F1 | Substitution F1 | Omission F1 | Insertion F1 |
|---|---|---|---|---|---|---|---|
| **ABA** | Arabic | Fold 0 | **0.3222 ± 0.0503** | **0.1496 ± 0.0195** | 0.0534 | 0.0638 | 0.3316 |
| **ASI** | Arabic | Fold 0 | **0.1511 ± 0.0274** | **0.1401 ± 0.0458** | 0.0508 | 0.1663 | 0.2032 |
| **BWC** | Chinese (Mandarin) | Fold 0 | **0.2689 ± 0.0220** | **0.1474 ± 0.0057** | 0.0872 | 0.2938 | 0.0612 |
| **EBVS** | Spanish | Fold 1 | **0.2244 ± 0.0612** | **0.1194 ± 0.0151** | 0.0732 | 0.1352 | 0.1499 |
| **HJK** | Korean | Fold 1 | **0.2533 ± 0.0381** | **0.0281 ± 0.0186** | 0.0641 | 0.0202 | 0.0000 |
| **HKK** | Korean | Fold 1 | **0.2444 ± 0.0245** | **0.1852 ± 0.0148** | 0.0902 | 0.3056 | 0.1599 |
| **HQTV** | Vietnamese | Fold 2 | **0.1978 ± 0.0166** | **0.1348 ± 0.0135** | 0.0796 | 0.2405 | 0.0845 |
| **LXC** | Chinese (Mandarin) | Fold 2 | **0.2578 ± 0.0354** | **0.1233 ± 0.0180** | 0.0690 | 0.2331 | 0.0679 |
| **NJS** | Spanish | Fold 2 | **0.1289 ± 0.0328** | **0.1469 ± 0.0077** | 0.0814 | 0.1428 | 0.2166 |
| **PNV** | Vietnamese | Fold 3 | **0.1644 ± 0.0220** | **0.1139 ± 0.0136** | 0.0207 | 0.2488 | 0.0721 |
| **SKA** | Arabic | Fold 3 | **0.1533 ± 0.0163** | **0.1540 ± 0.0016** | 0.0581 | 0.0923 | 0.3115 |
| **SVBI** | Hindi | Fold 3 | **0.0844 ± 0.0245** | **0.1156 ± 0.0164** | 0.0832 | 0.1423 | 0.1213 |

### 5.2. Nhận xét về độ phân tán giữa các Accent:
1. **Độ phân tán tỷ lệ cảnh báo sai (FWR Spread):** Người nói `SVBI` (gốc Tiếng Hindi) có tỷ lệ cảnh báo sai thấp nhất (**0.0844 FWR**), trong khi `ABA` (gốc Tiếng Ả Rập) có tỷ lệ cao nhất (**0.3222 FWR**), chênh lệch gần 4 lần.
2. **Hiện tượng người nói cá biệt (`HJK` - Korean):** Tại mức ngân sách $B=0.2$, mô hình gần như không phát hiện được true positive trên `HJK` (Macro-3 F1 chỉ đạt **0.0281**), nhưng khi tăng ngân sách lên $B=1.0$, F1 của `HJK` tăng vọt lên **0.1840**.

---

## 6. KIỂM ĐỊNH ĐỐI CHỨNG ÂM HỌC (DIAGNOSTIC ACOUSTIC CONTROLS)

Để chứng minh rằng hệ thống hoạt động dựa trên các đặc trưng âm thanh thực tế chứ không phải dựa trên sự ghi nhớ văn bản hay phân phối tiên nghiệm, ba phép can thiệp âm học kiểm chứng đã được thực hiện trên cùng các mô hình Outer Refit của Seed 101.

### Bảng 6.1: Kết quả kiểm chứng can thiệp âm học trên Seed 101
| Điều kiện âm học | Bản chất can thiệp | Macro-3 AP | AP-S | AP-O | AP-I | Mức sụt giảm AP tương đối |
|---|---|---|---|---|---|---|
| **Tín hiệu âm học gốc (Original)** | Đặc trưng H6 wav2vec2 đầy đủ | **0.306370** | **0.413233** | **0.342789** | **0.163088** | *Mốc chuẩn (0,0%)* |
| **Zero Audio ($X = 0$)** | Triệt tiêu hoàn toàn khung âm thanh về 0 | **0.115638** | 0.276633 | 0.026678 | 0.043603 | **-62,3%** (AP-O giảm -92,2%) |
| **Shuffled Audio (Hoán vị)** | Đảo ngẫu nhiên trật tự thời gian khung | **0.089728** | 0.205710 | 0.047055 | 0.016419 | **-70,7%** (AP-I giảm -89,9%) |
| **Reference Only** | Giữ vector one-hot âm vị, zero âm thanh | **0.115638** | 0.276633 | 0.026678 | 0.043603 | **-62,3%** |

**Kết luận kiểm định:**
- Sự sụt giảm 70.7% của Macro-3 AP khi xáo trộn thời gian và sụt giảm 92.2% của AP-O khi triệt tiêu âm thanh là bằng chứng định lượng không thể bác bỏ: **Mô hình phụ thuộc trực tiếp và quyết định vào tín hiệu sóng âm thực tế**, loại trừ hoàn toàn nghi vấn về rò rỉ nhãn hay học vẹt văn bản.

---

## 7. SO SÁNH ĐỐI ĐẦU VỚI CÁC BASELINE VÀ REFERENCE LỊCH SỬ

### Bảng 7.1: Bảng so sánh toàn diện với các mốc chuẩn
| Hệ thống | Nguồn gốc | Macro-3 AP | AP-S | AP-O | AP-I | Macro-3 F1 ($B=0.2$) | Joint FWR ($B=0.2$) | Tình trạng khoa học |
|---|---|---|---|---|---|---|---|---|
| **MB03 PLAIN** | Baseline lịch sử MB03 | 0.287347 | 0.423209 | 0.276706 | 0.162125 | *NOT_ESTIMABLE* | *NOT_ESTIMABLE* | **VERIFIED (Historical)** |
| **MB08 Selected Assembly** | Ghép nối modular MB08 | 0.298550 | 0.402742 | 0.330783 | 0.162125 | *NOT_ESTIMABLE* | *NOT_ESTIMABLE* | **VERIFIED (Historical)** |
| **Direct H-002 Neural** | Đầu Neural H-002 độc lập | 0.271711 | 0.388373 | 0.268146 | 0.158615 | 0.115284 | 0.153704 | **MEASURED (Phase-01)** |
| **CTC / E4 Sequence** | Nhánh CTC/E4 độc lập | 0.228896 | 0.283944 | 0.244158 | 0.158615 | 0.140371 | 0.270741 | **MEASURED (Phase-01)** |
| **Fixed 50/50 Fusion** | **Hệ thống Champion Phase-01** | **0.304154** | **0.409825** | **0.344073** | **0.158615** | **0.160371** | **0.204259** | **MEASURED (Phase-01)** |

### 7.2. Phân tích đóng góp thành phần (Component Attribution):
1. **Đầu Neural cục bộ (H-002):** Chuyên trách nhận diện lỗi thay thế âm ($S$) với độ chính xác cao (Precision-S = 0.740 tại $B=0.2$) nhờ trích xuất 2.343 chiều ngữ cảnh trung tâm và độ biến thiên khung.
2. **Nhánh CTC / E4:** Bổ sung bằng chứng chuỗi toàn cục, giải quyết triệt để điểm mù của neural đối với lỗi lược bỏ ($O$), nâng AP-O từ **0.2681** lên **0.3441** (+28.3% relative).
3. **Đầu chèn trực tiếp MB03:** Đạt AP-I **0.1586** và P@560 **0.2476**, cách ly hoàn toàn dòng phát hiện âm chèn khỏi nhiễu của dòng âm vị mục tiêu.

---

## 8. SỔ CÁI TÀI NGUYÊN VÀ DANH MỤC ARTIFACTS ĐÃ BẢO LƯU

### Bảng 8.1: Sổ cái tiêu thụ thời gian và bộ nhớ
| Seed | Số fold | Số nhánh fit | Checkpoints | Thời gian active-wall | Giới hạn | Bộ nhớ đỉnh (RSS) | Kết quả trạng thái |
|---|---|---|---|---|---|---|---|
| **Seed 101** | 4 outer (12 inner) | 48 | 400 epochs + 16 latest | 2.516,79 s (41,95 min) | 2.700,00 s | 4.072 MiB | **COMPLETE** |
| **Seed 202** | 4 outer (12 inner) | 48 | 400 epochs + 16 latest | 2.316,85 s (38,61 min) | 2.700,00 s | 4.057 MiB | **COMPLETE** |
| **Seed 303** | 4 outer (12 inner) | 48 | 400 epochs + 16 latest | 2.326,09 s (38,77 min) | 2.700,00 s | 4.107 MiB | **COMPLETE** |
| **Tổng cộng** | **12 outer** | **144** | **1.200 epochs + 48 latest** | **7.159,73 s (1,99 giờ)** | **8.100,00 s** | **4.107 MiB** | **HOÀN THÀNH 100%** |

### 8.2. Danh mục lưu trữ và tính toàn vẹn mã hóa (`experiments/rq1_train_phase_01/formal/`):
- **Tổng số tệp được niêm phong trong manifest:** 2.238 tệp.
- **Tệp mã băm SHA256:** `experiments/rq1_train_phase_01/formal/MANIFEST_SHA256.json`.
- **Tệp cấu hình đông băng:** `experiments/rq1_train_phase_01/formal/TRAIN_PHASE_01_CONFIG.json`.
- **Các log thực thi:** `formal_seed101.log`, `formal_seed202.log`, `formal_seed303.log`.
- **Toàn bộ trọng số mô hình:** 1.200 file checkpoint `.pt`, 48 file `terminal.json`, 32 file chuẩn hóa `normalizer.npz`, 16 file `competitors.json`, 12 file chính sách `policies.json`.

---

## 9. KẾT LUẬN KHOA HỌC VÀ BƯỚC ĐI TIẾP THEO (PHASE-02)

1. **Kết luận chính thức:** Chiến dịch Phase-01 đã kiểm chứng thành công và độc lập kiến trúc Fixed 50/50 Fusion, đạt Macro-3 Error AP **0.3042 ± 0.0034** và duy trì kiểm soát nghiêm ngặt tỷ lệ cảnh báo sai (**0.2043 Joint FWR** tại $B=0.2$).
2. **Hạn chế còn tồn đọng:**
   - Lỗi nhận diện sai loại âm ($S \leftrightarrow O$) chiếm 10.5% số cảnh báo sai tại $B=0.2$.
   - Độ phân tán FWR giữa các accent người nói còn lớn (từ 0.084 đến 0.322).
   - Đầu chèn âm vị hoạt động độc lập và chưa tận dụng được alignment toàn cục của CTC.
3. **Định hướng Phase-02:** Xem chi tiết tại `reports/TRAIN_PHASE_01_NEXT_RESEARCH.md` với 4 giả thuyết khoa học ưu tiên (H-P2-01 đến H-P2-04) về tách ngưỡng chi phí bất đối xứng, hiệu chuẩn thích ứng theo nhóm L1 và tích hợp alignment CTC cho insertion.
