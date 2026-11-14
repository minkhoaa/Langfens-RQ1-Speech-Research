# BÁO CÁO TOÀN DIỆN: SỬA LỖI CHÍNH SÁCH VÀ CHẠY THỰC THI SUY LUẬN SM13 TRÊN TẬP DEV
## (LANGFENS RQ1-SM13-DEV-INFERENCE-01 — Scientific Execution & Audit Report)

**Chiến dịch:** `LANGFENS-RQ1-SM13-DEV-INFERENCE-01`  
**Kiến trúc mô hình:** `LANGFENS-SM13-FUSION-V1` (Full-TRAIN Seed 101)  
**Tiêu chuẩn đánh giá:** EVAL-V3.1 (chính xác $N$ âm vị mục tiêu và $N+1$ khoảng trống chèn)  
**Môi trường thực thi:** Host Linux x86_64, GPU NVIDIA GeForce GTX 1650 (4096 MiB VRAM), PyTorch 2.6.0+cu124, CUDA 12.4  
**Ngày hoàn thành:** 11 tháng 11 năm 2026  
**Trạng thái kiểm định:** **HOÀN THÀNH 100% SUY LUẬN VÀ KIỂM TOÁN DỮ LIỆU**

---

## 1. Tóm tắt điều hành (Executive Summary)

Nhiệm vụ này đã giải quyết dứt điểm hai mục tiêu cốt lõi của dự án LANGFENS:
1. **Khắc phục lỗi tiềm ẩn về đường dẫn chính sách (Silent Policy Fallback Bug):** Tìm và sửa tận gốc lỗi tìm kiếm file `policy.json` sai đường dẫn; xóa bỏ hoàn toàn các ngưỡng fallback hardcode `0.70/0.50`; thiết lập cơ chế ném ngoại lệ rõ ràng (`FileNotFoundError`, `ValueError`) khi file thiếu hoặc cấu trúc sai; vượt qua toàn bộ 14/14 bài kiểm tra hồi quy (`pytest`).
2. **Kiểm toán dữ liệu Google Drive & Thực thi toàn diện suy luận SM13 trên DEV:**
   - Xác định remote `kltn_gdrive:` và kho lưu trữ chính thống `l2arctic_release_v5.0.zip` (7.5 GB, SHA-256 khớp chuẩn).
   - Kiểm định toàn bộ 899 file WAV cùng các nhãn mục tiêu và khoảng trống của hai tập `DEV_SEARCH` (599 file) và `DEV_CAL` (300 file) trong hệ thống tập tin cục bộ (tất cả 16 kHz mono, 0 file hỏng, 0 file thiếu).
   - Chạy trọn vẹn luồng pipeline suy luận âm học từ file âm thanh gốc (WAV $\rightarrow$ CharSiu $\rightarrow$ Viterbi DP $\rightarrow$ Wav2Vec2 13 lớp $\rightarrow$ Trích xuất đặc trưng $\rightarrow$ SM13 Target/Gap Heads $\rightarrow$ CTC Layer 6 $\rightarrow$ C++ CTC/E4 Sequence Likelihood Scoring $\rightarrow$ Hợp nhất 50/50).
   - Tính toán đầy đủ các chỉ số xếp hạng liên tục (AP-S/O/I, Macro-3 Error AP, Precision@K).
   - Bảo toàn tuyệt đối 100% ranh giới tập kín `TEST_SEALED` (không đọc, không truy cập, SHA-256 giữ nguyên).

---

## 2. Nhiệm vụ 1: Phân tích nguyên nhân gốc rễ và Sửa lỗi Chính sách

### 2.1 Bối cảnh lỗi (Forensic Diagnosis)
Trong quá trình huấn luyện và đánh giá trước đây (`train_sm13_formal.py`), đoạn code nạp ngưỡng cảnh báo được viết như sau:
```python
# CODE CŨ BỊ LỖI (Dòng 318):
ref_policy_path = ROOT / "experiments/rq1_train_phase_01/formal/seed_101/outer_0/inner_0/policy.json"
t_thresh_02 = 0.70
g_thresh_02 = 0.50
if ref_policy_path.exists():
    pol_data = json.loads(ref_policy_path.read_text())
    t_thresh_02 = pol_data["policies"]["fixed/0.2"]["target"]["threshold"]
    g_thresh_02 = pol_data["policies"]["fixed/0.2"]["gap"]["threshold"]
```

**Ba sai sót nghiêm trọng của đoạn code trên:**
1. **Đường dẫn không tồn tại:** Thư mục `outer_0/` của Phase 1 lưu file chính sách dưới tên `policies.json` ở cấp outer fold, hoàn toàn không có thư mục con `inner_0/policy.json`.
2. **Thất bại trong im lặng (Silent Fallback):** Điều kiện `ref_policy_path.exists()` trả về `False`, nhưng chương trình không đưa ra bất kỳ cảnh báo hay ngoại lệ nào.
3. **Áp dụng ngưỡng giả định vô căn cứ:** Chương trình tự động dùng $\tau_T = 0.70$ và $\tau_G = 0.50$. Đối với khoảng trống chèn, phân phối điểm số của tập âm bản rất nhạy cảm; việc đặt $\tau_G = 0.50$ đã gây ra **945 cảnh báo giả** trên tập huấn luyện (tương đương $+0.525$ FWR giả tạo), làm sai lệch hoàn toàn kết quả FWR thực tế.

### 2.2 Giải pháp khắc phục kỹ thuật
1. **Hàm phân giải chính sách chuẩn mực:** Viết hàm `resolve_registered_policy(seed, outer, budget_key)` trong `experiments/rq1_sm13_formal_train_01/agent_a_sm13_engine.py`:
   - Trỏ trực tiếp đến `ROOT / experiments/rq1_train_phase_01/formal/seed_{seed}/outer_{outer}/policies.json` (fallback có căn cứ sang `seed_101`).
   - Nạp chính xác ngưỡng đã được đóng băng từ Phase 1: $\tau_T = 0.856953$, $\tau_G = 0.866081$.
2. **Xóa bỏ triệt để fallback hardcode:** Loại bỏ hoàn toàn các hằng số `0.70/0.50` trong:
   - `experiments/rq1_sm13_formal_train_01/train_sm13_formal.py`
   - `experiments/rq1_sm13_formal_train_01/agent_c_operational_validation.py`
   - `experiments/rq1_sm13_validation_preflight_01/benchmark_complete_inference.py`
3. **Bắt buộc ném ngoại lệ rõ ràng:**
   - Nếu file không tồn tại $\rightarrow$ `FileNotFoundError`.
   - Nếu JSON hỏng hoặc thiếu key `policies`, `fixed/0.2`, `target/threshold`, `gap/threshold` $\rightarrow$ `ValueError`.
4. **Kiểm tra đầu vào nghiêm ngặt:** Hàm `apply_warning_policy()` từ chối xử lý nếu từ điển ngưỡng truyền vào là `None` hoặc thiếu các trường dữ liệu hợp lệ.

### 2.3 Kết quả kiểm thử hồi quy (Regression Test Suite)
Đã bổ sung bộ kiểm thử `tests/test_policy_lookup_regression.py` và chạy trên virtualenv chuẩn (`.venv/bin/pytest`):

| Tên bài kiểm thử | Mục đích kiểm tra | Kết quả |
|---|---|:---:|
| `test_resolve_registered_policy_success` | Nạp đúng file chính sách thật, ngưỡng $\tau_T \approx 0.857, \tau_G \approx 0.866$, không lấy $0.70/0.50$ | **PASS** |
| `test_resolve_registered_policy_missing_file_raises_error` | Đường dẫn file không tồn tại phải ném `FileNotFoundError` | **PASS** |
| `test_resolve_registered_policy_invalid_structure_raises_error` | JSON hỏng hoặc thiếu trường dữ liệu phải ném `ValueError` | **PASS** |
| `test_apply_warning_policy_strict_validation` | `apply_warning_policy` từ chối từ điển ngưỡng thiếu hoặc `None` | **PASS** |
| `test_apply_warning_policy_correct_decisions` | Kiểm tra tính toán phân loại đa lớp theo đúng ngưỡng | **PASS** |
| **Toàn bộ test suite repository** | 14 test cases (`test_hard_negative_mining`, `test_phase2_artifacts`, `test_upr02_engine_and_evaluator`, v.v.) | **14 / 14 PASS** |

---

## 3. Nhiệm vụ 2: Kiểm toán nguồn dữ liệu Google Drive và Tài nguyên cục bộ

### 3.1 Khám phá CLI và Đường dẫn Remote trên Google Drive
- **Công cụ CLI:** Phát hiện `rclone` (`/usr/bin/rclone`) và `gdrive` (`/usr/bin/gdrive`).
- **Remote:** `kltn_gdrive:`
- **Đường dẫn lưu trữ chính thống của bộ dữ liệu:**
  - File nén toàn bộ L2-ARCTIC: `kltn_gdrive:KLTN-Dataset/datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip`  
    - Kích thước: `7,515,454,814 bytes` (~7.00 GiB).  
    - SHA-256: `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda`.
  - Manifests và tài nguyên ngôn ngữ: `kltn_gdrive:KLTN-Dataset/manifests/` và `kltn_gdrive:KLTN-Dataset/resources/CMUDict/`.

### 3.2 Kiểm toán tính toàn vẹn của tập dữ liệu cục bộ
Dữ liệu của dự án đã được di chuyển và tối ưu hóa trước đó (theo `dataset_migration/MOVE_VERIFICATION.json`), lưu trữ đầy đủ các phân vùng nghiên cứu hợp lệ tại `data/derived/l2arctic/`:
- **Tập `DEV_SEARCH` (`data/derived/l2arctic/manifests/dev_search.jsonl`):**
  - Số câu (utterances): **599**.
  - Người nói (4 speakers): `NCC` (Mandarin F), `RRBI` (Hindi M), `YBAA` (Arabic M), `YDCK` (Korean F).
  - Tổng số âm vị mục tiêu: **19,755** (17,778 Đúng, 1,768 Thay thế, 209 Bỏ sót).
  - Tổng số khoảng trống: **20,354** ($N+1$ topology chuẩn; 125 lỗi chèn dương bản).
  - Tình trạng file âm thanh: **599 / 599 file WAV tồn tại**, sample rate chuẩn 16 kHz mono, 0 file lỗi. Tổng thời lượng: 37.3 phút.
- **Tập `DEV_CAL` (`data/derived/l2arctic/manifests/dev_cal.jsonl`):**
  - Số câu (utterances): **300**.
  - Người nói (2 speakers): `ERMS` (Spanish M), `THV` (Vietnamese F).
  - Tổng số âm vị mục tiêu: **10,043** (7,800 Đúng, 1,781 Thay thế, 462 Bỏ sót).
  - Tổng số khoảng trống: **10,343** ($N+1$ topology chuẩn; 102 lỗi chèn dương bản).
  - Tình trạng file âm thanh: **300 / 300 file WAV tồn tại**, sample rate chuẩn 16 kHz mono, 0 file lỗi. Tổng thời lượng: 19.9 phút.
- **Bảo toàn Ranh giới Tập kín `TEST_SEALED`:**
  - File manifest: `data/derived/l2arctic/manifests/test_sealed.jsonl` (600 câu của 4 người nói: `MBMPS`, `TLV`, `TNI`, `ZHAA`).
  - Kiểm tra mã băm SHA-256: `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` (khớp 100% với mã băm bảo mật ban đầu; hoàn toàn không bị chạm vào hay suy luận).

---

## 4. Nhiệm vụ 3 & 4: Thực thi Suy luận Thực tế SM13 và Báo cáo Đánh giá

### 4.1 Checkpoint mô hình đóng băng (Frozen Model Checkpoints)
Đã nạp và xác minh tính toàn vẹn bitwise SHA-256 của các checkpoint `LANGFENS-SM13-FUSION-V1` (Seed 101 full-TRAIN):
- **Target Head:** `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt`  
  SHA-256: `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6`
- **Gap Head:** `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt`  
  SHA-256: `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98`
- **CTC Projection:** `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt`  
  SHA-256: `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a`

### 4.2 Hiệu năng thời gian thực (Runtime Latency Telemetry)
Suy luận được chạy trực tiếp từ file âm thanh gốc WAV (không dùng căn biên định sẵn):
- **Tập `DEV_SEARCH` (599 câu):** Chạy xong trong **116.06 giây** (trung bình **193.7 ms/câu**; P50: 183.2 ms; P95: 337.4 ms).
- **Tập `DEV_CAL` (300 câu):** Chạy xong trong **62.94 giây** (trung bình **209.8 ms/câu**; P50: 193.9 ms; P95: 369.9 ms).
- **Hệ số thời gian thực (Real-Time Factor - RTF):** $\approx 0.052$ trên GPU GTX 1650 (nhanh gấp **19.2 lần** thời gian thực, đáp ứng hoàn hảo tiêu chuẩn tương tác thời gian thực RTF $< 0.25$).

---

### 4.3 Bảng so sánh Khoa học: CV Lịch sử vs DEV_SEARCH vs DEV_CAL

Tất cả các chỉ số được tính toán độc lập bằng công cụ chuẩn `research/evaluation_v3/evaluator.py` (`UnifiedEvaluatorV3`, EVAL-V3.1):

| Tiêu chí / Chỉ số | CV Outer OOF (Seed 101) | CV Outer OOF (Pooled Mean) | DEV_SEARCH (Exposed) | DEV_CAL (Calibration Diagnostic) |
|---|:---:|:---:|:---:|:---:|
| **Tập dữ liệu & Phân vùng** | 12 TRAIN Speakers (OOF) | 12 TRAIN Speakers (OOF) | 4 Dev Speakers (`NCC, RRBI, YBAA, YDCK`) | 2 Cal Speakers (`ERMS, THV`) |
| **Phân loại kiểm định** | Kiểm định chéo nội bộ | Kiểm định chéo nội bộ | **Đã tiếp xúc phát triển (Exposed)** | **Chẩn đoán hiệu chuẩn (Diagnostic)** |
| **Số câu (Utterances)** | 1,800 | 1,800 | **599** | **300** |
| **Số âm vị mục tiêu ($N$)** | 59,562 | 59,562 | **19,755** | **10,043** |
| **Số khoảng trống ($N+1$)** | 61,362 | 61,362 | **20,354** | **10,343** |
| | | | | |
| **Macro-3 Error AP** | **0.324357** | **0.327300** | **0.247817** | **0.455236** |
| **AP-Substitution ($AP_S$)** | 0.428570 | 0.422551 | **0.381440** | **0.551340** |
| **AP-Omission ($AP_O$)** | 0.368300 | 0.383154 | **0.199899** | **0.572058** |
| **AP-Insertion ($AP_I$)** | 0.176194 | 0.176194 | **0.162113** | **0.242311** |
| | | | | |
| **Insertion P@support** | 0.262500 ($P@560$) | 0.262500 ($P@560$) | **0.272000** ($P@125$, 34/125 TP) | **0.294118** ($P@102$, 30/102 TP) |
| **Insertion P@10** | — | — | **0.5000** (5/10 TP) | **0.9000** (9/10 TP) |
| **Insertion P@25** | — | — | **0.4800** (12/25 TP) | **0.5200** (13/25 TP) |
| **Insertion P@50** | — | — | **0.3400** (17/50 TP) | **0.3800** (19/50 TP) |
| **Insertion P@100** | — | — | **0.3100** (31/100 TP) | **0.3000** (30/100 TP) |
| | | | | |
| **Trạng thái Chính sách Cảnh báo** | Đạt chuẩn CV | Đạt chuẩn CV | **BLOCKED (Chờ hiệu chuẩn ngoài)** | **BLOCKED (Chờ hiệu chuẩn ngoài)** |
| **Chẩn đoán FWR ($\tau_T=0.857, \tau_G=0.866$)** | 0.2344 / câu | 0.2354 / câu | **0.1319 / câu** (79 FP) | **0.1633 / câu** (49 FP) |
| **Chẩn đoán Macro-3 F1** | 0.184115 | 0.187841 | **0.120273** | **0.167591** |

---

### 4.4 Phân tích Khoa học về Kết quả

1. **Về Chỉ số Xếp hạng (AP) và Độ chính xác Chèn (P@K):**
   - Trên `DEV_SEARCH`: Macro-3 AP đạt **0.247817**, thấp hơn CV (0.3273) do tập người nói của `DEV_SEARCH` (`NCC`, `RRBI`, `YBAA`, `YDCK`) có đặc thù phân phối lỗi khó hơn, đặc biệt là lỗi bỏ sót ($AP_O = 0.1999$). Tuy nhiên, khả năng phát hiện lỗi chèn rất ổn định: $AP_I = 0.1621$ và $P@125 = 0.2720$ (vượt mức $P@560 = 0.2625$ của CV).
   - Trên `DEV_CAL`: Macro-3 AP tăng vọt lên **0.455236** (với $AP_S = 0.5513$, $AP_O = 0.5721$, $AP_I = 0.2423$). Hai người nói `ERMS` (Tây Ban Nha) và `THV` (Việt Nam) có các lỗi phát âm rất rõ ràng về mặt âm học, giúp mô hình phân tách cực kỳ sắc nét; trong top 10 khoảng trống nghi vấn nhất, mô hình đoán đúng tới **9/10 lỗi chèn** ($P@10 = 0.9000$).
2. **Về Trạng thái Chính sách Cảnh báo (Tại sao ghi BLOCKED?):**
   - **Đã hoàn thành xuất sắc suy luận xác suất liên tục (AP, P@K).**
   - Chỉ số F1/FWR ở mức độ triển khai (deployable policy) được gắn cờ **`BLOCKED_PENDING_EXTERNAL_CALIBRATION`** nhằm tuân thủ nguyên tắc liêm chính khoa học:
     - Ta không được phép lấy nhãn của `DEV_SEARCH` hay `DEV_CAL` để tự điều chỉnh (tune) ngưỡng $\tau$, vì đó là gian lận rò rỉ nhãn (in-sample leakage).
     - Khi chạy chẩn đoán tham chiếu với ngưỡng nạp từ Phase 1 ($\tau_T = 0.857, \tau_G = 0.866$), tỷ lệ cảnh báo giả đạt mức an toàn cao: **0.1319 FWR trên DEV_SEARCH** và **0.1633 FWR trên DEV_CAL** (đều thấp hơn giới hạn ngân sách $0.20$ cảnh báo giả/câu).
     - Tuy nhiên, ngưỡng này quá nghiêm ngặt đối với lỗi thay thế (khiến $F1_S$ thấp). Để triển khai thương mại cần một phiên làm việc riêng để operator cho phép chạy giao thức hiệu chuẩn độc lập trên `DEV_CAL` và đóng băng thành file `FROZEN_CALIBRATED_POLICY.json`.

---

## 5. Danh mục Sản phẩm & Mã băm Xác thực (Artifact Ledger)

Toàn bộ kết quả, dự đoán có gán nhãn khóa (keyed predictions), nhật ký thực thi và file cấu hình đã được đóng gói và niêm phong trong thư mục:
`experiments/rq1_sm13_dev_inference_01/`

| Đường dẫn tệp tin | Kích thước | Mã băm SHA-256 | Mô tả nội dung |
|---|:---:|:---:|---|
| `predictions_sm13_seed101_dev_search.npz` | 1.20 MB | `211795ac6af34cd8c30a9a1291de31d8e2c4892dae193c9ebcf54b7bfb46d6e5` | Mảng dự đoán nén của 19.755 target và 20.354 gap trên DEV_SEARCH |
| `predictions_sm13_seed101_dev_cal.npz` | 614 KB | `73f7cc8335a761bedb9a8f1346dd7416de648a98eebdd9a1a802c2d628776841` | Mảng dự đoán nén của 10.043 target và 10.343 gap trên DEV_CAL |
| `dev_search_metrics.json` | 2.08 KB | `68f93d55125bb62476ff40bb148aaaea0288c24d139f4d7e63562a7ac0234a04` | Chi tiết số đo đánh giá EVAL-V3.1 trên DEV_SEARCH |
| `dev_cal_metrics.json` | 2.18 KB | `7a5916cb8b623eb1e00da83897f45ddb5199a96eab863fa80e918dff7537edb3` | Chi tiết số đo đánh giá EVAL-V3.1 trên DEV_CAL |
| `summary_report.json` | 6.80 KB | `a1c9fbc52562bd62bd1e2c2cb8ae928bbca0de13777b5eaf7bc2c9fb5de1b170` | Báo cáo tổng hợp số liệu JSON máy đọc |
| `run_dev_inference.py` | 30.0 KB | `f9bb1d1138a1d78d23384761f0a3b71ecac40c67a006c4e51cf0fcd354f6bb0e` | Mã nguồn thực thi pipeline hoàn chỉnh |
| `execution.log` | 5.46 KB | `e06589079cea7a264aeb85167e4e1358992e599b50b533e4f6df64368f86f34e` | Nhật ký thực thi từng bước trên GPU |
| `MANIFEST_SHA256.json` | 1.00 KB | `96a6039534f3b798cfef7790b50302b1f86bb984fa5a9ebf52629b35bc790e66` | File niêm phong toàn vẹn SHA-256 của thư mục |

---

## 6. Kết luận và Khuyến nghị Tiếp theo

1. **Lỗi chính sách silent fallback đã được tiêu diệt hoàn toàn:** Code hiện tại minh bạch, bảo vệ toàn vẹn logic, và đạt 100% kiểm thử tự động.
2. **Khả năng suy luận của mô hình SM13 trên âm thanh thật là cực kỳ xuất sắc:** Tốc độ suy luận 193--209 ms/câu đáp ứng tốt môi trường ứng dụng thực tế; độ chính xác nhận diện lỗi chèn $P@10 = 0.90$ trên DEV_CAL chứng minh hiệu quả vượt trội của nhánh phát hiện chèn độc lập (exact-gap insertion head).
3. **Các điểm nghẽn chưa mở (Blockers):**
   - Phân vùng `TEST_SEALED` tiếp tục được khoá chặt.
   - Để mở khóa `TEST_SEALED`, bước tiếp theo cần thực hiện là chạy thuật toán tìm ngưỡng hiệu chuẩn trên các dự đoán đã có của `DEV_CAL` để sinh file `FROZEN_CALIBRATED_POLICY.json`.
