# LANGFENS Phase 3 — Hoàn tất thực nghiệm bounded

**VERIFIED:** đã train local CUDA unified target/gap student; 14 arm/seed pairs, 98 điểm ngân sách; Gold/KD-All ghép cặp seed101/202/303. Mỗi run 12epochs / 684CUDAupdates. Tests cuối: **26 passed**; independent Luna review không có blocker sau sửa.

**Kết luận:** chưa chứng minh KD-All vượt Gold-Only ổn định. Tại budgetFWR0,20, mean paired macro-3 lỗi F1 delta **+0,0003 ± 0,0047** (SD ba seed). Insertion recall tăng ở một số seed nhưng omission giảm. Không tuyên bố test-set generalization.

| Seed | Gold false warnings | KD-All false warnings | Giới hạn |
|---:|---:|---:|---:|
| 101 | 63 | 73 | 119 |
| 202 | 69 | 73 | 119 |
| 303 | 47 | 56 | 119 |

Seed101 Gold S/O/I recall = 0,057% / 1,435% / 23,2%; KD-All = 0,170% / 0,957% / 25,6%. Đây là cùng budget, không phải cùng realized warning count. S/O recall còn yếu; không dùng CORRECT-F1 làm tín hiệu thành công chính.

## Deliverables
- `reports/PHASE3_FINAL_SCIENTIFIC_REPORT.md`
- `reports/PHASE3_REPRODUCTION.md`
- `reports/PHASE3_EXPERIMENT_RESULTS.csv`
- `reports/PHASE3_GOLD_VS_KD_PAIRED.md`
- `reports/PHASE3_FWR_RECALL_PARETO.csv`
- `reports/PHASE3_TEACHER_OOF_PROVENANCE.md`
- Local checkpoints `models/student/`; raw score/prediction arrays `experiments/student_training/`.
- Final evaluator `experiments/student_evaluation/individual/`, provenance/controls/bootstrap cùng thư mục.
- Hash index `coordination/PHASE3_ARTIFACT_MANIFEST.json`, dataset fingerprints `PHASE3_PREFLIGHT.json`, GPU ledger và queue cùng `coordination/`.

## Sửa lỗi trước nghiệm thu
Source I-F1/IPrecision đã sửa đúngEXP01/23D vàCHAMPION/25D; packedGRU/masks/TRAINweights; KD full-softmax probability/mask/dualIweight; macro3 lỗi gồm insertion, APties, error-F1 threshold objective và upper-tailCALgrid. Kết quả cũ invalid được lưu superseded; không dùng làm kết luận. Gold seed metadata đã sửa với model-state hashes xác nhận weights không đổi.

## Giới hạn
DEV_SEARCH development-exposed với4speaker, OOFheadreplicas8epochs không tương đương championbudget. Student dùng 51D/6D slot summaries, khôngframeConformer. Adapter nhận audio+intendedARPAbet, chưa text→phone; localCUDA/nooracle sample đã chạy. Max3leaseinterval overlap được ghi nhận, **kernel-overlap chưa VERIFIED**. Teacher calibration trên chínhDEV_CAL có reuse/shift đã ghi rõ. Historical fullfusionfrontier chưa đủledger để rederive tất cảPareto. Không truy cập sealed/encrypted/holdouts, không remoteGPU/network.

Workflow Luna chính `24a399cd-6ce2-40b1-a63a-7fcbfebbbcde`; correction/review `1899aa34-0299-4da0-ba1d-bc00a30ef8c4`, đều hoàn tất. Gemini trước đó lỗi provider trước output và đã dừng theo log. Chỉ commit files Phase3; binary artifacts bị ignore giữ local với hashes; logs lưu trong thư mục thí nghiệm, không tự add historic/unrelated changes.
