# LANGFENS Phase 3 — Báo cáo khoa học cuối, local CUDA

## 1. Kết luận cho operator

**VERIFIED: đã huấn luyện student hai đầu ra trên GTX 1650 CUDA; chưa chứng minh KD-All tốt hơn Gold-Only một cách ổn định.**

Đã đánh giá 14 cặp arm/seed tại 7 ngân sách FWR (98 dòng kết quả). Gold-Only và KD-All có ba seed ghép cặp 101/202/303, cùng kiến trúc, dữ liệu, khởi tạo theo seed, trọng số lớp, optimizer, 12 epochs / 684 updates và tiêu chí chọn checkpoint trên DEV_CAL. Không train neural trên CPU. Các agent thực thi cuối dùng `openai-codex/gpt-6-luna` theo yêu cầu operator.

Tại ngân sách joint FWR ≤ 0,20, trung bình thay đổi KD−Gold macro-3 **lỗi** F1 là **+0,0003 ± 0,0047** (mean ± sample SD của ba seed, không phải CI). KD thường tăng insertion/substitution recall rất nhỏ nhưng giảm omission recall. Đây không phải bằng chứng về ưu thế tổng thể hoặc tổng quát hóa test.

## 2. Dữ liệu và chống oracle

Các số dưới đây đã đối chiếu trực tiếp manifest và thứ tự/nhãn cache, không chỉ lặp lại báo cáo cũ:

| Split | Utterances | Speakers | Targets | Gaps | Gold S/O/I |
|---|---:|---:|---:|---:|---:|
| TRAIN | 1.800 | 12 | 59.562 | 61.362 | 7.381 / 1.825 / 560 |
| DEV_CAL | 300 | 2 | 10.043 | 10.343 | 1.781 / 462 / 102 |
| DEV_SEARCH | 599 | 4 | 19.755 | 20.354 | 1.768 / 209 / 125 |

Speaker partitions rời nhau; mỗi utterance giữ đúng N target và N+1 gap. Hash manifest/feature/teacher checkpoints: `coordination/PHASE3_PREFLIGHT.json`. HEAD Phase 1 ban đầu là `41f02ad9ad4ea2df4734ebc38bc2052503586fb6`; không reset Git.

Inference dùng audio và **chuỗi ARPAbet dự định**, không dùng gold timing, realized phone hoặc nhãn lỗi. Alignment lấy từ CTC frozen + intended reference. Gold chỉ dùng cho supervision/metrics/calibration đúng vai trò. Không truy cập TEST_SEALED, GPG, replication holdouts; không dùng mạng, cloud, Colab hoặc SSH. DEV_SEARCH đã development-exposed, chỉ là đánh giá exploratory.

Cache tái sử dụng: target 51D = reference one-hot 39D + 12 acoustic/CTC summaries; gap 6D. Đây **không phải frame hidden states** và không chứng minh đủ cho frame cross-attention. Kiến trúc chọn là shared bidirectional GRU-64, target head 3 lớp, gap head 2 lớp, 78.661 parameters đo từ model. STU-00 GRU-32 có 20.901 parameters. Packed sequence, mask và adapter xử lý padding, N=0 và một gap hợp lệ; không loại slot khó khỏi mẫu số. Không train Conformer trong đợt này: dùng phương án lightweight hợp lệ thay vì tải model hoặc fine-tune encoder nặng.

## 3. Teacher OOF và KD

Ba fold theo speaker trên TRAIN; mỗi fold train bốn neural specialist heads trong 8 epochs bằng CUDA. Normalization chỉ fit speaker thuộc fold-training; S loại omission khi fit theo hợp đồng specialist nhưng vẫn score mọi held-out target. Lưu 12 checkpoint fold, danh sách speaker, score và provenance. OOF là replica tiết kiệm, không phải tái hiện đầy đủ ngân sách champion gốc.

- S/O là hai binary specialists riêng; không nối thành teacher simplex 3 lớp.
- I-F1: `TeacherI_Network / TEACHER_I_EXP01`, đặc trưng 23D.
- I-Precision: `TeacherI_ChampionNet / TEACHER_I_CHAMPION`, đặc trưng 25D.
- Nguồn chính thức: `experiments/student_training/kd/teacher_scores_v2_*.npz` và `provenance_v2_correct_i_sources.json`.
- Isotonic chỉ fit trên DEV_CAL; báo cáo raw/calibrated Brier, ECE15, AP trong provenance. Áp dụng mapping frozen-teacher DEV_CAL sang OOF TRAIN là distribution shift đã ghi rõ. ECE gần 0 trên chính tập fit isotonic **không** phải xác nhận calibration out-of-sample.

Gold loss = weighted target CE + weighted gap CE. Trọng số lấy sqrt inverse TRAIN frequency, chuẩn hóa mean 1. KD = KL(Bernoulli teacher || student marginal từ **full softmax**) tại T=1, hệ số 0,1 mỗi task; không có hệ số T² khác 1 hoặc teacher logits giả. Mask KD dùng Y/YY != -100. Dual-I lấy trung bình hai objective để giữ tổng trọng số insertion task. Confidence gate liên tục từ calibrated probability là ablation, không phải quy luật uncertainty đã chứng minh. Không thêm consistency loss mặc định.

Đã chạy: STU-00, STU-01, STU-02 S-only, STU-03 O-only, STU-04 I-F1, STU-05 All, STU-06 confidence-gated, STU-07 I-Precision, STU-08 dual-I, STU-09 calibrated All; Gold/All thêm seed202/303. Kết quả từng arm nằm trong `PHASE3_EXPERIMENT_RESULTS.csv`; không chọn seed đẹp nhất làm kết luận.

## 4. Hợp đồng đánh giá cuối

Protocol `student-eval-v2-upper-tail`: lưới 21 quantile thô + 8 quantile đuôi định trước, endpoints và no-warning option; 32×32 candidate pairs trên DEV_CAL. Mục tiêu chọn threshold là mean F1(S,O,exact-gap I), macro-4 chỉ tie-break. Hai ngưỡng target/gap được khóa rồi áp dụng DEV_SEARCH; không fit lại để ép FWR an toàn. Tất cả 98 điểm student hiện tại không vượt ngân sách realized; điều này không bảo đảm cho split tương lai.

Joint FWR theo evaluator lịch sử = (FP_SUB + FP_OMISSION + FP_INSERTION) / utterances. Sai loại lỗi tính một false warning của lớp được dự đoán; không cộng FP_CORRECT và không đếm hai lần một target. Trên 599 utterances, budget0,20 cho phép tối đa floor(119,8) = **119** false warnings. Macro-3 lỗi gồm S/O/I; macro-3 target gồm C/S/O là chỉ số khác được đặt tên riêng. AP xử lý score ties; không dùng accuracy/CORRECT-F1 làm tín hiệu thành công chính.

Chi tiết TP/FP/FN/TN, P/R/F1/AP, ECE10/Brier, correct/error confusion, selectivity, speaker/family strata, ngưỡng, warning ledgers và always-CORRECT control có trong `experiments/student_evaluation/individual/` và `PHASE3_DETAILED_RESULTS.json`. Các event được nối manifest bằng dataset/speaker/utterance/type/index, giữ full population.

## 5. Kết quả ghép cặp tại budget0,20

Recall dưới đây là %, F1 ở thang 0–1. Đây là **cùng ngân sách**, không phải cùng số false warnings realized.

| Seed | Arm | False warnings | FWR | S recall | O recall | I recall | Macro-3 lỗi F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 101 | Gold | 63 | 0,1052 | 0,057 | 1,435 | 23,2 | 0,10490 |
| 101 | KD-All | 73 | 0,1219 | 0,170 | 0,957 | 25,6 | 0,10625 |
| 202 | Gold | 69 | 0,1152 | 1,584 | 2,392 | 20,8 | 0,11313 |
| 202 | KD-All | 73 | 0,1219 | 1,810 | 1,435 | 20,8 | 0,10826 |
| 303 | Gold | 47 | 0,0785 | 0,000 | 1,914 | 15,2 | 0,08228 |
| 303 | KD-All | 56 | 0,0935 | 0,057 | 0,957 | 18,4 | 0,08661 |

Seed101: Gold S/O/I TP = 1/3/29, FP = 0/15/48; KD-All TP = 3/2/32, FP = 0/15/58. Insertion recall tăng nhưng precision giảm 37,66% → 35,56%; omission F1 giảm 0,02643 → 0,01770. S/O recall vẫn rất thấp — không gọi đây là thành công cân bằng ba loại lỗi. Mean paired recall delta qua ba seed: S +0,13 percentage points, O −0,80 pp, I +1,87 pp.

Tại budget1,00, mean paired macro-3 lỗi F1 thay đổi −0,0001 ± 0,0014. Confidence-gated seed101 đạt macro-3 lỗi F1 0,11319 và I recall27,2% tại realized FWR0,1135, nhưng chỉ một seed nên không khẳng định gating tốt hơn. Dual-I seed101 không cải thiện so với Gold. Xem đủ bảy budget trong CSV; không cherry-pick điểm.

Bootstrap resample theo bốn speaker có trong `PHASE3_SPEAKER_CLUSTER_BOOTSTRAP.json`; chỉ mang tính exploratory. SD của ba seed và khoảng bootstrap speaker không tương đương test-set confidence interval.

## 6. Đối chứng frozen và Phase 2 corrections

Rescore complete ledgers bằng cùng confusion/FWR implementation, không retune quyết định lịch sử:

| Control | False warnings | FWR | S/O/I recall % | Macro-3 lỗi F1 |
|---|---:|---:|---|---:|
| UPR03 | 113 | 0,18865 | 5,939 / 4,306 / 5,600 | 0,09332 |
| CPL02 best frozen F5-RISK-B020 | 53 | 0,08848 | 1,244 / 4,306 / 3,200 | 0,05202 |

Student có insertion recall cao hơn hai control này tại budget0,20 nhưng S/O thấp hơn. Gold-only cũng có xu hướng đó: không thể gán toàn bộ cải thiện insertion cho KD. Một số student macro-3 lỗi F1 cao hơn UPR03 dưới cùng trần rủi ro; **không phải dominance** ở mọi lớp hoặc cùng realized FWR. Không gọi control có sẵn là best toàn bộ frontier lịch sử khi thiếu ledger.

Corrections lưu tại `experiments/student_evaluation/PHASE2_TARGETED_CORRECTIONS.json`: omission union85/209 là correction operator-directed từ table, chưa độc lập tái dựng union bằng complete ledger; không lặp94/209. F2-O-PRIO/F1-UNION không phải student baseline. I-Precision11/51 =21,57% ở threshold báo cáo. UPR03 FWR113/599 và CPL02 frozen53/599 đã tái tính từ ledger. Historical frontier đầy đủ chưa có linked ledgers nên không khẳng định đã kiểm chứng mọi Pareto membership; Pareto student hiện tại được tính lại riêng.

## 7. Tài nguyên, tốc độ, lỗi và giới hạn

- CUDA:0 GTX1650 4GB, PyTorch2.6.0+cu124/CUDA12.4. FP32 và FP16 backward đã finite; BF16 native không hỗ trợ dù API mặc định có emulation.
- Gold seed101: 49,99s training-run wall time, peak allocated44,09MiB; KD-All seed101: 96,51s,44,14MiB. Thời gian phụ thuộc concurrency và gồm phần công việc trong runner, không chỉ kernel time. Checkpoint khoảng320KB.
- Gold cached student forward DEV_SEARCH: mean1,124ms/utterance. Một audio/reference adapter call gồm load model: khoảng0,89s; đây không phải warm streaming benchmark. Sample acoustic+student peak khoảng422MiB ở receipt trước; không suy ra peak này cho mọi audio.
- Wrapper ledger ghi nhận tối đa ba lease intervals chồng nhau; các nhóm job CUDA thực sự đã được launch. **Không có CUDA kernel trace để chứng minh ba-way kernel overlap**. PyTorch peak allocated không bao gồm toàn bộ context/driver; nvidia-smi telemetry được giữ riêng.
- OOF v2: khoảng21,85s,22,23MiB allocated peak; head replicas tiết kiệm không tương đương champion budget.
- Đã lưu lỗi Gemini provider high/low unavailable, lỗi calibration đầu và artifact superseded. Sửa source I-teacher, GRU padding, KD logit/probability/mask, macro lỗi/AP và threshold objective trước kết quả cuối. Số liệu sơ bộ trước sửa là invalid/superseded, không được dùng làm bằng chứng.
- Adapter nhận intended ARPAbet, chưa tích hợp orthographic text-to-phoneme. Alignment kế thừa không có state leading-gap độc lập trong trường hợp N>0; gap0 vẫn giữ slot nhưng acoustic evidence bị hạn chế. Cache không giữ frame context phong phú; đây là giới hạn kiến trúc đáng kể.
- Calibration isotonic fit trên cùng DEV_CAL với checkpoint/threshold selection làm tăng mức reuse development; không diễn giải là unbiased calibration/generalization estimate.

## 8. Nghiệm thu và tái lập

Supervisor tự chạy: **26 tests passed**, compile scripts thành công. Independent Luna reviewer: không phát hiện blocker sau sửa; chấp nhận với giới hạn nêu trên. `coordination/PHASE3_ACCEPTANCE_TESTS.log` và artifact SHA index `coordination/PHASE3_ARTIFACT_MANIFEST.json` giữ evidence. Code/report/provenance Phase3 được commit riêng; binary checkpoints/NPZ lớn được giữ local theo `.gitignore`; logs được giữ trong thư mục thí nghiệm, không tự thêm dữ liệu lịch sử vào Git.

Điều tra trực tiếp S/O tiếp theo (không đổi weights, không train thêm): `reports/PHASE3_SO_FAILURE_DIAGNOSIS.md`. Counterfactual cho thấy model không collapse; shared S/O threshold và phân bổ risk làm giảm mạnh recall, nhưng tail ranking và CAL→SEARCH shift vẫn hạn chế chất lượng. Các policy diagnostic được lưu riêng, không thay thế benchmark chính.

Lệnh tái lập: `reports/PHASE3_REPRODUCTION.md`. Raw predictions/checkpoints/OOF folds ở local paths ghi trong receipt. Không có dịch vụ remote hoặc URL training. Không tuyên bố80%, breakthrough, Conformer cần thiết, hoặc generalization trên sealed test.
