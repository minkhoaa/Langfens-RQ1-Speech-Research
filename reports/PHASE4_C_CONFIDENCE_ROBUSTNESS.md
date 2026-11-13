# LANGFENS Phase 4 — Agent C: độ tin cậy và độ bền vững

## Phạm vi và nguồn có thẩm quyền

Đây là tổng hợp kiểm toán dựa trên đầu ra evaluator/canonical arrays hiện có; không chạy lại inference, không khớp mô hình/calibrator và không dùng DEV_SEARCH để lựa chọn chính sách. DEV_SEARCH là tập phát triển đã bị exposure, không phải test mù. Thư mục `experiments/phase4/confidence/` chỉ chứa các bảng con được dẫn xuất trực tiếp từ các đầu ra EVAL-V3 chính thức.

Nguồn đầu ra chính:
- `experiments/evaluation_v3/closure/release_final/RESULTS_PRIMARY_COMPARABLE.csv` (các score/ranking và provenance canonical).
- `experiments/evaluation_v3/closure/release_final/OPERATING_POINTS.csv` (các chính sách đã fit trên DEV_CAL và FWR quan sát được).
- `experiments/evaluation_v3/closure/classification_selective_v2/CANONICAL_EVENT_ID_TOP_K.csv` và `CLASSIFICATION_RISK_COVERAGE.csv.gz` (xếp hạng Top-K và selective-risk; không gộp vào classification threshold metrics).
- `experiments/evaluation_v3/closure/stratified_full/FAMILY_ALIGNMENT_GAP_STRATA.csv` và `STRATIFICATION_RECEIPT.json` (độ bền vững theo speaker/stratum; canonical score provenance).
- `experiments/top_model_investigation/confidence/confidence_brier_ece.csv` và `precision_recall_at_k.csv` chỉ được đối chiếu để nhận diện báo cáo cũ, không thay thế kết quả canonical khi không cùng evaluator/provenance.

## Phân biệt ba câu hỏi

1. **Ranking (continuous-score discrimination):** AP/ROC-AUC đo thứ tự điểm trên toàn bộ target S/O hoặc toàn bộ exact-gap I; không xác định threshold. Báo cáo dùng `ap_substitution`, `ap_omission`, `ap_insertion_exact_gap`, `macro3_error_ap`. Không diễn giải AP là PR-AUC hình thang.
2. **Classification (điểm quyết định):** các confusion cells, P/R/F1 và `macro3_error_f1` áp dụng cho đúng một chính sách/threshold đã nêu. DEV_CAL là nguồn threshold. S/O được arbitrated thành một lớp target duy nhất; I thuộc gap riêng.
3. **Policy (vận hành):** FWR đo số false warnings/utterance, với lỗi chẩn đoán sai lớp được tính thành cảnh báo sai; mỗi warning được tính đúng một lần. FWR không phải metric ranking, cũng không phải giới hạn chấp nhận mô hình. CAL budget danh nghĩa và FWR thực tế phải được phân biệt.

Canonical population là 599 utterances, 4 SEARCH speakers, 19,755 target slots và 20,354 gaps; gold S/O/I lần lượt 1,768 / 209 / 125. Mọi phân tích speaker bên dưới giữ riêng 4 cụm, không giả định 599 utterances độc lập.

## Chính sách CAL và điểm FWR

Bảng đầy đủ, gồm các policy/mức ngân sách cho bốn frontier models, được trích nguyên cột từ canonical operating-points array ở `experiments/phase4/confidence/cal_policy_fwr_points.csv`. Ví dụ kiểm tra bằng nguồn `OPERATING_POINTS.csv`, EXP-36:

| Chính sách | Ngân sách danh nghĩa | Macro-3 error F1 | FWR quan sát |
|---|---:|---:|---:|
| `ARGMAX_DIAGNOSTIC` | — | 0.3448 | 2.7679 |
| `CAL_CLASS_SPECIFIC_CONFIDENCE` | — | 0.3311 | 5.5309 |
| `CAL_RISK_BUDGETED` | 0.10 | 0.1127 | 0.0935 |
| `CAL_RISK_BUDGETED` | 0.20 | 0.1478 | 0.1386 |
| `CAL_RISK_BUDGETED` | 0.50 | 0.1960 | 0.3856 |
| `CAL_RISK_BUDGETED` | 1.00 | 0.2776 | 0.9783 |
| `CAL_RISK_BUDGETED` | 2.00 | 0.3147 | 1.7679 |
| `CAL_MAX_ERROR_MACRO_F1` | — | 0.3311 | 5.5309 |

FWR thực tế không nhất thiết bằng ngân sách danh nghĩa. Không có ngưỡng FWR nào ở đây được tuyên bố an toàn về mặt giảng dạy. Chính sách được fit trên DEV_CAL; áp dụng trên DEV_SEARCH chỉ là ước lượng phát triển. `SEARCH_ORACLE` không được dùng để lựa chọn model/policy. Mức `0.20` được giữ như một điểm tham chiếu chứ không phải cổng chất lượng.

## Calibration, PR và selective risk

- Canonical score tables giữ nguyên AP độc lập với policy. Bản canonical PR/threshold curves có trong các thư mục closure confidence (`RECALL_FWR_CURVES.csv.gz`, `FP_COUNT_RECALL_CURVES.csv.gz`); curve không thể thay thế ma trận confusion tại policy cụ thể.
- Precision/recall@K cần nêu pool/domain và K. `CANONICAL_EVENT_ID_TOP_K.csv` giữ event identity; bảng cũ `precision_recall_at_k.csv` không được coi là cùng provenance nếu thiếu canonical receipt/hash.
- Calibration: Brier/ECE chỉ có ý nghĩa như xác suất nếu score thực sự probability-valued; evaluator dùng bins equal-width, 10 bins, [0,1], bin cuối đóng, population toàn bộ eligible events. Không gọi score chưa chứng minh hậu hiệu chuẩn là “calibrated probability”. Báo cáo `confidence_brier_ece.csv` lịch sử có raw/cal values nhưng không được ghép ngang với canonical arrays nếu không khớp hash/evaluator. Calibration trên cùng DEV_CAL vừa fit vừa đánh giá là in-sample, không phải held-out calibration validation.
- Selective-risk canonical source là `classification_selective_v2/CLASSIFICATION_RISK_COVERAGE.csv.gz`. Coverage/abstention được giữ trong full-population accounting; không xóa abstain khỏi FN denominator. Bảng rủi ro phụ thuộc confidence/abstention, không phải thay cho AP.

## Độ bền vững, cụm speaker và độ bất định

`experiments/phase4/confidence/speaker_cluster_scores.csv` chứa AP theo class/domain/speaker, lấy từ canonical stratification cho EXP-24/29/36/43 (48 hàng). Hỗ trợ theo lớp trong bốn cụm không đồng đều: S 536/451/331/450, O 55/65/31/58, I 29/44/30/22; giá trị ở mỗi hàng là số event domain của speaker và positive support. Vì vậy điểm giữa các speaker không đồng nghĩa với macro AP toàn tập.

Ví dụ bốn speaker cho EXP-36 (AP):

| Speaker | AP S | AP O | AP I |
|---|---:|---:|---:|
| NCC | 0.4897 | 0.1953 | 0.3475 |
| RRBI | 0.3391 | 0.2961 | 0.0771 |
| YBAA | 0.2208 | 0.1136 | 0.3796 |
| YDCK | 0.5311 | 0.2436 | 0.4554 |

Không có model nào thắng mọi speaker/classes; insertion tại RRBI (44 positive) yếu hơn nhiều các speaker còn lại. Với chỉ 4 cluster độc lập, 95% speaker-cluster bootstrap/t-interval có độ phân giải yếu và giả định phân phối rất mong manh. Bảng stratification chính thức đánh dấu `NO_FAKE_SINGLE_SPEAKER_CI; SMALL_CLUSTER_LIMITATION`; do đó không trình bày CI giả chính xác. Báo cáo điểm theo từng cụm và dùng chúng như phân tích exploratory, không kết luận tổng quát tới speaker population.

## Kiểm toán t-test

Không tái sử dụng tuyên bố p-value t-test từ báo cáo cũ dựa trên “Macro-3 AP từng utterance”: AP là thống kê xếp hạng yêu cầu pool positives/negatives; AP trên từng câu ngắn/có lớp không hỗ trợ thường bị không xác định hoặc tạo ra estimand khác với AP toàn corpus. Đơn vị lặp phù hợp cho suy luận ngoại suy speaker là speaker-cluster, nhưng chỉ có 4 cụm; cỡ mẫu effective không phải 599. Các điểm AP theo 4 speaker trong bảng canonical cho phép kiểm tra hướng/mức không đồng nhất, nhưng không đủ cho test xác nhận đáng tin. Không có continuous paired speaker-level raw arrays cho mọi so sánh trong artifact này để tái dựng kiểm định suy luận; do đó t-test xác nhận được đánh dấu **NOT ESTIMABLE / NOT RUN**, không suy diễn p-value mới. P-values từ paired tests cũ không được coi là kiểm định xác nhận trong báo cáo Phase 4.

## OOV và mẫu số

Nguồn canonical Phase 4 tại các đường dẫn đã kiểm tra không cung cấp danh sách token-level OOV đã hash, lexicon version/hash, quy tắc biến thể phát âm, hay nối từng token tới utterance. Báo cáo Agent C cũ nhắc 39/599 câu không tra CMUdict; đây là tỷ lệ *utterances có OOV được báo cáo* (6.51%), không phải tỷ lệ token OOV, và chưa đủ chứng cứ canonical để tái xác nhận số đếm. Mẫu số phải báo cáo riêng: utterances scanned (599), total reference tokens scanned (chưa có), OOV tokens (chưa có), utterances có ≥1 OOV (39, chỉ là số báo cáo lịch sử chưa xác minh), và utterances không đọc/không có lexicon entry (chưa có). Kết luận hiện tại: **OOV denominator incomplete; no deployability estimate**. Không loại utterance OOV khỏi phân tích hiệu năng mà không ghi coverage/abstention.

## Lease và giới hạn thực thi

Đã kiểm tra `coordination/GPU_RESOURCE_LEDGER.jsonl` và các lock marker trong `coordination/`: ledger hiển thị INIT `active_lease: null`, các lease cũ đã release; các `.phase3_gpu_slot*.lock` là marker rỗng với timestamp 6/11, không phải bằng chứng lease đang hoạt động. Công việc này chỉ đọc CPU-side CSV/JSON, không chạy neural inference/GPU, nên không cần giữ lease hay thay đổi trạng thái GPU. Không suy ra khả năng GPU hiện hành từ marker cũ.

## Kết luận

Canonical evidence hỗ trợ so sánh ranking, policy operating points, calibration/selective-risk provenance và phân tầng theo 4 speaker khi đọc đúng miền/domain. Khả năng tổng quát speaker còn hạn chế rõ bởi 4 cluster; OOV token denominator và kiểm định t-test xác nhận còn thiếu bằng chứng. SEARCH vẫn chỉ là exploratory/development-exposed; không có claim test mù, không có khẳng định confidence ≥0.9 tương đương 90% correctness, và không có khẳng định policy FWR cao an toàn.
