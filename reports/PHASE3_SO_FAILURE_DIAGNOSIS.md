# Vì sao student gần như không phát hiện S/O tại FWR ≤ 0,20?

## Phạm vi và kết luận

Điều tra trực tiếp trên checkpoint/predictions đã công bố ở commit `45cd3d2`; không thay weights, không train thêm, không truy cập holdout, không dùng remote data. Đã tính **42 counterfactual policies trên 6 arm/seed pairs**. Ngưỡng implementable chỉ chọn trên DEV_CAL; DEV_SEARCH vẫn exploratory. Ba statistical tests kiểm tra policy semantics/ties và tái hiện đúng kết quả cũ đã pass.

**Nguyên nhân gần, đã chứng minh:** student có khả năng phân biệt S/O, nhưng chính sách dùng một ngưỡng chung cho hai score có phân bố khác nhau, cộng với ngân sách false warnings chia ba task, làm phần lớn S/O bị tắt. **Giới hạn sâu hơn:** ranking ở đuôi cực ít false-positive chưa đủ tốt; O gặp shift lớn giữa CAL và SEARCH và feature alignment có ambiguity. Không có bằng chứng current KD code còn lỗi softmax/mask. Chưa thể kết luận rằng tăng model/epochs hoặc bỏ class imbalance sẽ giải quyết.

Script tái lập: `research/student/diagnose_so_failure.py`. Output: `experiments/student_evaluation/so_diagnosis/score_policy_diagnosis.json` và `policy_counterfactuals.csv`. Các thí nghiệm này không thay thế benchmark Phase3 chính và không phải search model mới.

## 1. Không phải collapse về CORRECT

Seed101, DEV_SEARCH, **argmax không dùng ngưỡng FWR**:

| Arm | S TP / support | S recall | O TP / support | O recall | Joint false warnings | FWR |
|---|---:|---:|---:|---:|---:|---:|
| Gold | 863 / 1.768 | 48,81% | 144 / 209 | 68,90% | 2.657 | 4,436 |
| KD-All | 871 / 1.768 | 49,26% | 143 / 209 | 68,42% | 2.626 | 4,384 |

Đây không phải operating point dùng được: budget0,20 chỉ cho119 false warnings, không phải2.657. Nhưng nó bác bỏ diễn giải “network không học được S/O hoặc luôn xuất CORRECT”.

Gold S/O AUPRC = **0,3646 / 0,1594**, AUROC = **0,8230 / 0,9450**. Base prevalence lần lượt 0,0895 / 0,0106; ranking tốt hơn random đáng kể. KD gần tương tự: AP0,3685 / 0,1584. Specialist frozen AP cùng SEARCH =0,3833 / 0,1415; không có bằng chứng student thấp hơn rất xa specialist trên mọi ranking metric.

## 2. Ngưỡng chung chặn S — counterfactual trực tiếp chứng minh

`evaluate_student_predictions.py::_predictions` lấy max(p_SUB,p_OMISSION), rồi áp **cùng một target threshold**. Gold seed101 threshold = **0,90857**. Trong true substitutions trên SEARCH:
- median p_SUB =0,4644;
- percentile99 =0,8436;
- vì threshold vượt percentile99 nên hầu hết S bị loại, dù argmax nhận ra nhiều S.

Giữ nguyên weights và insertion policy, chỉ tách threshold S/O, chọn trên CAL với cùng ngân sách:

| Arm seed101 | Policy | S TP | O TP | I TP | False warnings /599 | Có vượt119? |
|---|---|---:|---:|---:|---:|---|
| Gold | Shared hiện tại | 1 | 3 | 29 | 63 | Không |
| Gold | Separate S/O, giữ I threshold | 27 | 3 | 29 | 75 | Không |
| KD-All | Shared hiện tại | 3 | 2 | 32 | 73 | Không |
| KD-All | Separate S/O, giữ I threshold | 37 | 2 | 32 | 87 | Không |

Gold chọn S threshold0,83708 và O0,90827; KD chọn S0,83015 và O0,88933. S recall tương ứng tăng0,057%→1,527% và0,170%→2,093%, không train thêm và không đổi recall O/I ở seed này. Đó là evidence về bottleneck policy, **không** phải claim model đã đạt chất lượng tốt. Realized FWR tăng nên đây là cùng trần ngân sách, không phải cùng realized risk.

Tách ngưỡng không luôn giúp mọi seed: Gold202 S TP28→26, KD202 S TP32→26 (KD O TP3→5); Gold303 S0→67, KD303 S1→40. Vì vậy không khẳng định đây là cải thiện universal. Tất cả separate-joint policies trong đợt diagnostic đều dưới119 trên SEARCH; không có bảo đảm cho future split.

**Lưới thô không phải toàn bộ vấn đề:** thay target threshold grid bằng toàn bộ unique CAL score, vẫn giữ shared S/O và cùng insertion threshold, Gold seed101 chỉ lên S TP2/O TP4/FW73; KD S4/O4/FW85. Tăng độ mịn ngưỡng chung không khôi phục khả năng của ngưỡng riêng.

## 3. Ngân sách cảnh báo ưu tiên insertion/O và để S gần như bằng0

Gold seed101 CAL policy hiện tại dùng FP S/O/I = **0 /18 /31**, tổng49 trong allowance60. Shared threshold để S ở mức gầnzero; một phần budget chưa dùng không được khai thác bởi lưới/chính sách coupled. Ngưỡng riêng dùng7/19/31=57FP CAL và vẫn giữ nguyênI decision.

Counterfactual seed101 khác, chọn ngưỡng trên CAL:
- Tắt insertion, tách S/O: Gold S/O TP27/13, FW47; KD37/13, FW49. O recall lên6,22% thay vì1,44%/0,96%, nhưng **I recall về0**. Đây là trade-off, không phải joint dominance.
- Cho toàn bộ budget vào S-only: Gold S TP189 (10,69% recall), FW118; O/I bị tắt. KD S TP196, FW130 **vượt budget SEARCH**, dù đã thỏa CAL. Không được dùng điểm KD này như safe improvement.
- Cho toàn bộ budget vào O-only: Gold O TP16 (7,66%), FW54; KD19 (9,09%), FW54; S/I bị tắt.

Các kết quả bác bỏ việc gán S/O recall thấp chỉ cho architecture. Một phần lớn là do chính sách đa task tại trần rủi ro thấp. Đồng thời chúng không cho phép cộng riêng recall S-only/O-only/I để tạo một hệ thống joint giả.

## 4. Omission bị distribution shift và feature ambiguity

Các dữ kiện trực tiếp:
- O prevalence: CAL462/10.043=**4,60%**, SEARCH209/19.755=**1,06%** (giảm khoảng4,35lần).
- Median Gold p_O trên true O: **0,7342 CAL →0,5861 SEARCH**. Không chỉ base rate thay đổi; phân bố positive score cũng đổi.
- Với p_O≥0,90: CAL có40 trueO/65warnings (precision61,54%); SEARCH4/23 (17,39%). High-confidence tail không chuyển tốt giữa speaker groups.
- TrueO có median aligned duration **2frames CAL →4frames SEARCH**; mean intended-phone CTC probability **0,3427→0,5251**. O trên SEARCH trông “có acoustic presence” hơn theo những feature model đang thấy.
- **207/209 gold omissions trên SEARCH vẫn được gán duration>0**; chỉ2có duration0. Ở high p_O≥0,90, median aligned duration1frame cả CAL và SEARCH; trên SEARCH phần lớn warning lại là goldCORRECT (18/23).

Trong `tools/extract_acoustic_features.py::viterbi_align`, mỗi intended phone có emitting state; skip transition chủ yếu bỏ silence state, không có epsilon deletion transition cho một target phone bất kỳ. Vì vậy duration này là **model/reference-derived alignment duration**, không phải bằng chứng gold phone thực sự có mặt. Short legitimate phones và omitted phones có thể có summary tương tự.

**VERIFIED:** representation/shift và nhầm lẫn trên tail như trên. **Chưa causal-verified:** thay alignment hoặc thêm frame context sẽ cải thiện bao nhiêu; cần matched feature/alignment ablation riêng. Không được suy từ việc207O có duration>0 rằng toàn bộ alignment hoặc gold labels sai.

## 5. Ranking có trần thấp tại risk tail, không chỉ threshold xấu

Diagnostic oracle trên SEARCH: sort class score và cho mỗi binary class dùng **riêng toàn bộ119FP**. Đây là upper ceiling không implementable vì dùng SEARCH labels để chọn threshold, không phải kết quả mới:

| Seed101 | S max TP / recall | O max TP / recall |
|---|---|---|
| Gold | 189 /10,69% | 29 /13,88% |
| KD-All | 192 /10,86% | 32 /15,31% |

Ngay cả oracle class-specific này cũng không đạt high recall. Các trần không thể cộng vì mỗi lớp đã tiêu riêng toàn bộ joint budget. Do đó sửa policy là cần nhưng **không đủ**: muốn tăng recall mạnh ở FWR0,20 còn cần nâng khả năng xếp hạng lỗi trước các correct-phone false positives.

## 6. Điều gì chưa được chứng minh?

- Class weights sqrt inverse frequency và unweighted CAL target+gap CE là facts trong trainer. Có khả năng checkpoint objective không hợp lỗi hiếm, nhưng chưa có per-epoch safe-error metric replay để kết luận nó là nguyên nhân chính. Checkpoint seed101 được chọn epoch11, không phải model vừa train1epoch hoặc training thất bại.
- Chưa chứng minh shared encoder bị gradient interference từ gap head; counterfactual tắt I decisions không phải retrain bỏ gap loss.
- Chưa chứng minh cần Conformer. Cached51D chứa referenceonehot+12summary, thiếu competitor identity/frame trajectory; đó là limitation có thật, chưa phải causal ablation.
- Reference-phone TRAIN-frequency baseline S AP0,3063 trên SEARCH, so với student0,3646: có phone-prior signal đáng kể, nhưng network vẫn bổ sung thông tin. O baseline prior0,0360 so với student0,1594; không thể gọi model chỉ học prior.
- T=1 KD weight0,1 chỉ làm score/ranking thay đổi nhẹ trong thiết lập này. Không kết luận mọi KD formulation đều không hiệu quả.

## 7. Hành động phù hợp với evidence

1. **Sửa policy trước, không tăng model ngay:** threshold riêng S/O/I, conflict rule rõ, có risk allocation/coverage theo lớp. Chọn trên CAL; theo dõi SEARCH breaches trung thực. Không mặc định threshold0,9 cho mọi error class.
2. **Ablation kế tiếp nếu tiếp tục training:** cùng seed/budget, lưu checkpoint từng epoch để so unweightedCE selection với CAL risk-constrained error-F1 selection; tách khỏi feature/architecture changes.
3. **Sau đó mới thay representation/alignment:** giữ frozen encoder nhưng cache context không oracle, cho omission/delete state hợp lệ; so matched Gold trước khiKD. Đánh giá precision ở đuôi rủi ro, không chỉ AUROC hoặc argmax recall.

Không có neural training CPU hoặc training mới trong điều tra này. Việc dùng CPU để đọc checkpoint metadata, tính threshold/metrics và TRAIN reference-frequency statistics là I/O/statistics. Không spawn thêm agent sau yêu cầu operator làm trực tiếp.

## Tái lập

```bash
cd /home/khoa/Projects/New-Method-Research
.venv/bin/python research/student/diagnose_so_failure.py
PYTHONPATH=research/student .venv/bin/python -m pytest research/student/test_so_diagnosis.py -q
```

Script giữ nguyên prediction/checkpoint cũ, chỉ ghi thư mục `experiments/student_evaluation/so_diagnosis/`. Hash các prediction inputs và script được lưu trong diagnosis JSON. OOF/frozen teacher AP chỉ dùng comparison ranking, không được gọi in-sample TRAIN là out-of-sample.
