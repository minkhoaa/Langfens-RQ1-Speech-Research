# HƯỚNG DẪN TÁI LẬP CHIẾN DỊCH TMCI (REPRODUCTION MANUAL)
## LANGFENS Post EVAL-V3 Top-Model Capability Investigation

Tài liệu này cung cấp toàn bộ hướng dẫn, câu lệnh CLI, mã hash kiểm tra (SHA-256) và quy trình chuẩn hóa để tái lập 100% các kết quả định lượng, ma trận cắt bỏ và phân tích độ bền vững của chiến dịch TMCI.

---

## 1. MÔI TRƯỜNG THỰC THI VÀ YÊU CẦU PHẦN CỨNG

### 1.1. Yêu cầu hệ thống
- **Hệ điều hành:** Linux x86_64 (Ubuntu 22.04 LTS / Debian 12 khuyến nghị).
- **Phần cứng:**
  - CPU: Tối thiểu 4 nhân vật lý (phục vụ tính toán ma trận và đánh giá số liệu).
  - RAM: Tối thiểu 16 GB.
  - GPU: NVIDIA GPU có hỗ trợ CUDA (GTX 1650 4GB VRAM hoặc cao hơn) để chạy các kiểm chứng suy luận dạng sóng âm thanh thực tế.
- **Môi trường Python:**
  - Python 3.10+ ảo tại `.venv/bin/python`.
  - Các thư viện phụ thuộc chính: `torch`, `torchaudio`, `numpy`, `scipy`, `scikit-learn`, `pytest`.

### 1.2. Biến môi trường bắt buộc
Trước khi thực thi bất kỳ câu lệnh nào, đảm bảo biến môi trường `PYTHONPATH` trỏ về thư mục gốc của dự án:
```bash
export PYTHONPATH=.
```

---

## 2. BẢNG MÃ HASH XÁC THỰC CÁC ARTIFACTS NỀN TẢNG (SOURCE FINGERPRINTS)

Trước khi chạy lại các phép đo, xác minh tính toàn vẹn của các tệp nguồn chuẩn hóa EVAL-V3:

| Tệp nguồn / Artifact | Đường dẫn tương đối | SHA-256 Fingerprint |
|---|---|---|
| Manifest DEV_CAL | `data/derived/l2arctic/manifests/dev_cal.jsonl` | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` |
| Manifest DEV_SEARCH | `data/derived/l2arctic/manifests/dev_search.jsonl` | `a4ab6c7df7bd65b5d213d306c2fdfde6c316cbf7768a3eb63d056775351f99d2` |
| Receipt EXP-24 | `experiments/evaluation_v3/closure/checkpoint_replays/EXP-24-FINAL-ROADMAP-EXPANSION/CHECKPOINT_INFERENCE_RECEIPT.json` | `10a5990584d74d29a88eaf6a87c075aa011d8bc5f51d798f5df6e6b184d485cb` |
| Điểm SEARCH EXP-24 | `experiments/evaluation_v3/closure/checkpoint_replays/EXP-24-FINAL-ROADMAP-EXPANSION/DEV_SEARCH.npz` | `bf157001d92e225c24c9da3e329dc70648d8bf6b944961333444ec946299a53a` |
| Checkpoint EXP-24 | `experiments/EXP-24-FINAL-ROADMAP-EXPANSION/CHECKPOINTS/jsac_monotonic_model.pt` | `78e906d0f5369c23196dab3b8a7816b1f061c380dbfb7ddadd31149e88321b59` |
| Công thức CAL EXP-36 | `experiments/evaluation_v3/closure/composition_replays/EXP-36-FINAL-HEXAD-PNC-SUPER-FUSION/FROZEN_CAL_COMPOSITION.json` | `888cf31be4db59850c950d603a11b66db57df29dfa43fb3f521b44ec4e502120` |
| Điểm SEARCH EXP-36 | `experiments/evaluation_v3/closure/composition_replays/EXP-36-FINAL-HEXAD-PNC-SUPER-FUSION/dev_search_scores.npz` | `bf10ce76652eb6f7df20b411d51a6d71b405fcb9ad7818eb28dfbc8bf47517c2` |
| Điểm SEARCH EXP-43 | `experiments/evaluation_v3/closure/composition_replays/EXP-43-CONTINUOUS-AUTONOMOUS-FRONTIER-EXPANSION/dev_search_scores.npz` | `07a6ec420be0db4a1599cebf6fae98e29a996f5b9d2149e1e0a294b4e398ca63` |
| Điểm SEARCH EXP-29 | `experiments/evaluation_v3/closure/composition_replays/EXP-29-PROSODIC-PENTAD-FUSION/dev_search_scores.npz` | `2d4bf31cf54966779434857ddca833f9ca333276644f808fe376174a737f5d94` |
| Publication Receipt | `experiments/evaluation_v3/closure/release_final/PUBLICATION_RECEIPT.json` | Khớp kiểm toán EVAL-V3 |

---

## 3. CÁC BƯỚC THỰC THI TÁI LẬP (REPRODUCTION COMMANDS)

### Bước 1: Kiểm tra tính hợp lệ của môi trường và bộ kiểm thử an toàn
Chạy bộ kiểm thử an toàn và giao thức đánh giá EVAL-V3:
```bash
PYTHONPATH=. .venv/bin/python -m pytest -q research/evaluation_v3/test_numerical_contract.py research/evaluation_v3/test_evaluator_v3.py
```
*Kết quả kỳ vọng:* 100% tests passed.

### Bước 2: Tái lập phân tích nguồn gốc và cắt bỏ thành phần (Agent A)
Thực thi script trích xuất lineage, đo đạc năng lực 9 thành phần đơn lẻ, tính toán ma trận cắt bỏ leave-one-out và đường tích lũy từng bước:
```bash
PYTHONPATH=. .venv/bin/python tools/top_model_investigation/agent_a_lineage_and_ablations.py
```
*Các artifact sinh ra:*
- `experiments/top_model_investigation/lineage/lineage_dependency_graph.json`
- `experiments/top_model_investigation/lineage/component_inventory.csv`
- `experiments/top_model_investigation/ablations/single_component_evaluations.csv`
- `experiments/top_model_investigation/ablations/leave_one_out_ablations.csv`
- `experiments/top_model_investigation/ablations/cumulative_additions.csv`
- `experiments/top_model_investigation/ablations/ablation_summary.json`
*Thời gian chạy:* ~10 giây.

### Bước 3: Tái lập đánh giá phân biệt, độ tin cậy và chẩn đoán lỗi (Agent B)
Thực thi script đánh giá toàn diện bằng `UnifiedEvaluatorV3`, quét các chính sách vận hành (`CAL_MAX_ERROR_MACRO_F1`, `PREDECLARED_RAW_0.9`), tính toán Brier/ECE và Top-K selective points:
```bash
PYTHONPATH=. .venv/bin/python tools/top_model_investigation/agent_b_metrics_and_confidence.py
```
*Các artifact sinh ra:*
- `experiments/top_model_investigation/metrics/all_candidates_comparative_metrics.csv`
- `experiments/top_model_investigation/metrics/operating_policies_breakdown.csv`
- `experiments/top_model_investigation/metrics/confusion_matrices.json`
- `experiments/top_model_investigation/confidence/confidence_brier_ece.csv`
- `experiments/top_model_investigation/confidence/precision_recall_at_k.csv`
- `experiments/top_model_investigation/confidence/pr_curves_top_models.json`
*Thời gian chạy:* ~15 giây.

### Bước 4: Tái lập kiểm toán hợp đồng suy luận và phân tầng độ bền vững (Agent C)
Thực thi script kiểm toán chuỗi suy luận âm thanh thực, tra cứu từ điển CMUdict, phân tích phân tầng theo 4 người nói, 7 họ âm vị, ngữ cảnh khe hở và kiểm định thống kê cặp:
```bash
PYTHONPATH=. .venv/bin/python tools/top_model_investigation/agent_c_validity_and_robustness.py
```
*Các artifact sinh ra:*
- `experiments/top_model_investigation/robustness/audio_inference_verification_receipt.json`
- `experiments/top_model_investigation/robustness/speaker_breakdown.csv`
- `experiments/top_model_investigation/robustness/phone_family_breakdown.csv`
- `experiments/top_model_investigation/robustness/gap_context_breakdown.csv`
- `experiments/top_model_investigation/robustness/candidate_agreement_matrix.csv`
- `experiments/top_model_investigation/robustness/statistical_significance_paired.csv`
*Thời gian chạy:* ~12 giây.

### Bước 5: Đồng bộ và xuất bản các bảng so sánh tổng hợp (Final CSVs)
Xuất bản hai bảng dữ liệu tổng kết chính thức vào thư mục `reports/`:
```bash
PYTHONPATH=. .venv/bin/python tools/top_model_investigation/generate_final_csvs.py
```
*Các artifact sinh ra:*
- `reports/TMCI_MODEL_CAPABILITY_COMPARISON.csv`
- `reports/TMCI_COMPONENT_CONTRIBUTION.csv`
*Thời gian chạy:* ~2 giây.

---

## 4. BẢNG ĐỐI SOÁT CÁC CON SỐ THEN CHỐT CỦA BẢN TÁI LẬP

Để xác nhận việc tái lập hoàn toàn chính xác, đối chiếu các chỉ số đầu ra với bảng chuẩn dưới đây:

| Chỉ số kiểm tra | Giá trị tái lập chính xác | Nguồn kiểm chứng |
|---|---:|---|
| EXP-36 DEV_SEARCH Macro-3 Error AP | **0.2892330268512784** | `all_candidates_comparative_metrics.csv` |
| EXP-36 DEV_SEARCH AP Substitution | **0.3870880226541211** | `all_candidates_comparative_metrics.csv` |
| EXP-36 DEV_SEARCH AP Omission | **0.20468719779213324** | `all_candidates_comparative_metrics.csv` |
| EXP-36 DEV_SEARCH AP Insertion | **0.27592386010758085** | `all_candidates_comparative_metrics.csv` |
| EXP-24 DEV_SEARCH Macro-3 Error AP | **0.2768187528435872** | `all_candidates_comparative_metrics.csv` |
| EXP-24 DEV_SEARCH AP Omission (Kỷ lục đơn lẻ) | **0.22509157986760872** | `single_component_evaluations.csv` |
| Tổ hợp tối giản [EXP-24 + EXP-34] Macro-3 AP | **0.2920124641089199** | `cumulative_additions.csv` |
| EXP-43 Macro-3 AP khi cắt bỏ EXP-38 (Renorm) | **0.2887534479532669** | `leave_one_out_ablations.csv` |
| Kiểm định cặp EXP-36 vs EXP-24: Paired t-statistic | **5.655289496589346** | `statistical_significance_paired.csv` |
| Kiểm định cặp EXP-36 vs EXP-24: p-value | **$2.412566 \times 10^{-8}$** | `statistical_significance_paired.csv` |
| Số câu OOV trong tra cứu từ điển CMUdict | **39 / 599 câu (6.51%)** | `audio_inference_verification_receipt.json` |
