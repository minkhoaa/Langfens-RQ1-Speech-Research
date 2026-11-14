# BÁO CÁO KẾT QUẢ KIỂM THỬ HỒI QUY: LANGFENS SM13 DEV CALIBRATION
## (LANGFENS-SM13-DEV-CALIBRATION-01 — Regression Verification Report)

**Chiến dịch:** `LANGFENS-SM13-DEV-CALIBRATION-01`  
**Kiến trúc:** `LANGFENS-SM13-FUSION-V1` (Seed 101 Full-TRAIN)  
**Tập hiệu chuẩn:** `DEV_CAL` (300 câu, 2 người nói `ERMS` và `THV`)  
**Tiêu chuẩn đánh giá:** EVAL-V3.1 (`UnifiedEvaluatorV3`)  
**Môi trường thực thi:** Host Linux x86_64, Python 3.11.17, PyTorch 2.6.0+cu124, pytest-9.2.0  
**Ngày thực hiện:** 11 tháng 11 năm 2026  
**Trạng thái kiểm định:** **TẤT CẢ 19/19 BÀI KIỂM THỬ ĐẠT (100% PASS)**

---

## 1. Tóm tắt Kiểm thử Tự động (Test Suite Summary)

Toàn bộ các bài kiểm thử hồi quy được thực thi tự động qua test runner chuẩn (`.venv/bin/pytest tests/`):

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

---

## 2. Chi tiết các Bài Kiểm thử Trọng yếu

### 2.1 Kiểm thử Hiệu chuẩn & Đóng băng Chính sách (`tests/test_sm13_calibration_regression.py`)

| Tên bài kiểm thử | Mục đích kiểm tra | Kết quả | Chi tiết kiểm chứng |
|---|---|:---:|---|
| `test_calibrated_policy_file_integrity` | Xác minh file `FROZEN_CALIBRATED_POLICY.json` tồn tại, đúng cấu trúc schema, đủ metadata, khớp 100% mã băm của checkpoint và manifest | **PASS** | `schema_version`: V1.0<br>Mã băm checkpoints (Target, Gap, CTC) khớp bitwise.<br>Ranh giới `TEST_SEALED` được bảo vệ (`accessed=False`). |
| `test_deterministic_policy_resolution` | Xác minh hàm `resolve_registered_policy` nạp chính xác ngưỡng đã hiệu chuẩn ($\tau_T=0.848521, \tau_G=0.873637$) và từ chối nạp cấu hình không hợp lệ | **PASS** | Nạp đúng `fixed/0.2` và `dev_cal_nominal_unmargined/0.2`.<br>Bắt lỗi `ValueError` khi thiếu key.<br>Bắt lỗi `FileNotFoundError` khi seed không tồn tại (fail-closed). |
| `test_policy_application_and_decision_rules` | Xác minh hàm `apply_warning_policy` phân loại chính xác tại ngưỡng biên (boundary threshold) và trên toàn bộ tập dự đoán `DEV_CAL` | **PASS** | Biên chính xác: $\ge \tau \rightarrow$ cảnh báo, $< \tau \rightarrow$ bỏ qua.<br>Trên `DEV_CAL`: phát đúng 132 cảnh báo âm vị và 23 cảnh báo khoảng trống. |
| `test_warning_counting_and_evaluator_compatibility` | Tích hợp toàn diện với bộ đánh giá chuẩn EVAL-V3.1 (`UnifiedEvaluatorV3`), đếm chính xác số lượng cảnh báo giả, kiểm tra FWR $\le B_{eff}$ | **PASS** | Số cảnh báo giả thực tế: **53** (43 target, 10 gap).<br>FWR thực tế: **$0.1767$** $\le 0.1800 < 0.2000$.<br>`fwr_reference_0_20_exceeded`: `False`.<br>Lỗi loại cảnh báo (S$\leftrightarrow$O wrong-type): **7** cảnh báo. |
| `test_historical_reference_comparison_consistency` | Xác minh đối sánh với chính sách tham chiếu lịch sử ($\tau_T=0.856953, \tau_G=0.866081$) trên cùng tập dữ liệu | **PASS** | Cảnh báo giả tham chiếu: 49 ($0.1633$ FWR).<br>Macro-3 F1 tham chiếu: $0.167591$.<br>Khớp hoàn toàn 100% với nhật ký chẩn đoán trước đó. |

---

### 2.2 Kiểm thử Sửa lỗi Tra cứu Chính sách & Loại bỏ Fallback Ngầm (`tests/test_policy_lookup_regression.py`)

| Tên bài kiểm thử | Mục đích kiểm tra | Kết quả | Chi tiết kiểm chứng |
|---|---|:---:|---|
| `test_resolve_registered_policy_success` | Nạp đúng chính sách đăng ký thực tế từ Phase 1, loại bỏ triệt để hằng số hardcode `0.70/0.50` | **PASS** | $\tau_T \approx 0.857, \tau_G \approx 0.866$.<br>Tuyệt đối không lấy fallback giả định `0.70/0.50`. |
| `test_resolve_registered_policy_missing_file_raises_error` | Bắt buộc ném ngoại lệ `FileNotFoundError` khi đường dẫn file không tồn tại hoặc tra cứu chéo seed không khớp (fail-closed) | **PASS** | Thử nghiệm với file giả lập $\rightarrow$ `FileNotFoundError`.<br>Thử nghiệm với seed giả lập `seed=999` $\rightarrow$ `FileNotFoundError`, không fallback ngầm về `seed_101`. |
| `test_resolve_registered_policy_invalid_structure_raises_error` | Bắt buộc ném ngoại lệ `ValueError` khi cấu trúc file JSON bị hỏng hoặc thiếu trường bắt buộc | **PASS** | Bắt lỗi JSON hỏng, thiếu key ngân sách, thiếu `target/threshold`, hoặc thiếu `gap/threshold`. |
| `test_apply_warning_policy_strict_validation` | Hàm `apply_warning_policy` từ chối từ điển ngưỡng thiếu trường hoặc `None` | **PASS** | Ném `ValueError` rõ ràng khi input không hợp lệ. |
| `test_apply_warning_policy_correct_decisions` | Kiểm tra thuật toán phân loại 3 lớp mục tiêu và nhị phân khoảng trống | **PASS** | Khớp 100% ma trận quyết định. |

---

### 2.3 Kiểm thử Tương thích Hệ thống & Module Kế thừa (Legacy Modules)

- `tests/test_hard_negative_mining.py` (2 tests): **PASS** (Khai phá mẫu âm tính khó trong huấn luyện).
- `tests/test_phase2_artifacts.py` (4 tests): **PASS** (Tính toàn vẹn của artifacts pha 2).
- `tests/test_upr02_engine_and_evaluator.py` (3 tests): **PASS** (Động cơ tính điểm và tích hợp chuẩn EVAL-V3).

---

## 3. Xác minh Cơ chế Fail-Closed (Silent Fallback Elimination)

Trước phiên làm việc này, hàm `resolve_registered_policy` và script huấn luyện `train_sm13_formal.py` có đoạn code kiểm tra:
```python
# ĐOẠN CODE CŨ (NGUY CƠ BỊ FAIL THẦM LẶNG):
if not p.exists():
    p = ROOT / f"experiments/rq1_train_phase_01/formal/seed_101/outer_{outer}/policies.json"
```
Đoạn code trên dẫn đến lỗ hổng: khi gọi hàm với `seed != 101` (ví dụ `seed=202` hay `seed=999`), nếu file không tồn tại, chương trình sẽ tự động trỏ về `seed_101` mà không thông báo, vi phạm nguyên tắc bảo mật và liêm chính dữ liệu.

**Hành động sửa đổi:**
1. Đã xóa bỏ hoàn toàn đoạn fallback ngầm trong `experiments/rq1_sm13_formal_train_01/agent_a_sm13_engine.py` (dòng 310-311).
2. Đã xóa bỏ hoàn toàn đoạn fallback ngầm trong `experiments/rq1_sm13_formal_train_01/train_sm13_formal.py` (dòng 318).
3. Đã bổ sung assertion trong `test_resolve_registered_policy_missing_file_raises_error`:
   ```python
   with pytest.raises(FileNotFoundError, match="does not exist"):
       resolve_registered_policy(seed=999, outer=0)
   ```
4. Xác nhận: Khi tra cứu seed không tồn tại, hệ thống lập tức ném `FileNotFoundError` (fail-closed 100%).

---

## 4. Xác nhận Tính toàn vẹn Dữ liệu và Ranh giới Bảo mật

| Tệp tin kiểm toán | Đường dẫn tệp tin | Mã băm SHA-256 xác thực | Trạng thái bảo mật |
|---|---|---|:---:|
| **Predictions DEV_CAL** | `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz` | `73f7cc8335a761bedb9a8f1346dd7416de648a98eebdd9a1a802c2d628776841` | Khớp 100%, không bị sửa đổi |
| **Manifest DEV_CAL** | `data/derived/l2arctic/manifests/dev_cal.jsonl` | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` | Khớp 100%, chỉ đọc nhãn hiệu chuẩn |
| **Manifest TEST_SEALED** | `data/derived/l2arctic/manifests/test_sealed.jsonl` | `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | **KHÓA CHẶT 100%, KHÔNG TRUY CẬP** |
| **Checkpoint Target Head** | `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt` | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` | Đóng băng bitwise |
| **Checkpoint Gap Head** | `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt` | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` | Đóng băng bitwise |
| **Checkpoint CTC Head** | `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt` | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` | Đóng băng bitwise |
| **Chính sách đóng băng** | `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json` | `dc7da3bcbc28918fdc9df40916ffaf24870da3efad12dd8c6e2bba11bf3cb98e` | **ĐÃ NIÊM PHONG BẢO MẬT** |

---

## 5. Kết luận Kiểm thử

Hệ thống đã vượt qua 100% các tiêu chuẩn kiểm thử khắt khe. Các ngưỡng hiệu chuẩn hoạt động xác thực, định lượng chính xác từng cảnh báo giả, kiểm soát rủi ro sư phạm dưới trần FWR cho phép, và ngăn chặn tuyệt đối hiện tượng rò rỉ hay lỗi ngầm.
