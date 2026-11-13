# EVAL-V3 — Giao thức khoa học có thể thực thi

## Quần thể và nguồn

Entrypoint duy nhất: `research.evaluation_v3.evaluator.UnifiedEvaluatorV3.evaluate(target_predictions, gap_predictions, gold_targets, gold_gaps, expected_utterances=None, calibration_provenance=None)`, version `EVAL-V3.1`. Schema chi tiết: `research/evaluation_v3/specification.md`. CPU chỉ dùng cho metrics/statistics; inference neural duy nhất trong đợt là sampled WAV CUDA, không training.

Event key `dataset_id:speaker_id:utterance_id:target|gap:slot_index`. Target N slots có C/S/O mutually exclusive, gap N+1 binary slots có cả leading/trailing. Duplicate, extra IDs, alias mâu thuẫn, topology đứt đoạn bị từ chối. Missing/ABSTAIN không xóa event; không-warning là FN cho gold error, coverage giảm khi abstain. Target đúng nhưng abstain cũng là FN_CORRECT. Binary MD coi abstain là không cảnh báo, không biến insertion thành target.

## Metrics

- `ap_substitution`, `ap_omission`: stepwise sklearn average precision one-vs-rest trên mọi target; `ap_insertion_exact_gap`: trên mọi exact gap. Ties nguyên vẹn. `macro3_error_ap` là mean ba AP khác domain. ROC-AUC chỉ bổ sung; không gọi trapezoidal PR-AUC là AP.
- Mỗi lớp giữ integer TP/FP/FN/TN và support. P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Mẫu số0 trả null/status; F1 khi có positive support nhưng không warning là0. Không ẩn missing component trong macro.
- `macro3_error_f1`=mean(F1_S,F1_O,F1_I); `macro3_target_f1`=mean(F1_C,F1_S,F1_O); `macro4_mixed_domain_f1`=mean(F1_C,F1_S,F1_O,F1_I), alias `macro_4_f1`, không phải4-way softmax. Correct P/R/F1 secondary do prevalence.
- `binary_md_*`: S hoặc O vs C trên target, không quan tâm predicted diagnosis type. AP dùng documented fused score hoặc p_S+p_O **chỉ khi mọi target có complete valid simplex**. Independent teacher S/O không tự thành simplex.
- `joint_fwr`=(FP_S+FP_O+FP_I)/utterances. Wrong-type error cảnh báo tính một FP của predicted class, không cộng FP_CORRECT. Báo thêm false alarms chỉ trên gold CORRECT, per-class FP, fraction utterances với FP, histogram warning/utterance và false alerts/1000 eligible positions. FWR≤0,20 là reference, không quality veto hoặc chứng nhận an toàn sư phạm.

## Scores và confidence

`score_*`/`raw_score_*` giữ arbitrary finite ranking scores/logits, higher-is-positive; không clip/normalize. `prob_*` probability-valued có thể vẫn **uncalibrated**. AP không suy từ decision/confusion; thiếu score trả NOT_ESTIMABLE. Brier/ECE chỉ dùng probability fields đủ toàn population. ECE10 equal-width [0,1], bin cuối đóng; reliability output có counts/frequency/mean probability và baseline Brier prevalence-constant. Normalized AP lift=AP/prevalence chỉ mô tả.

`calibration.py:fit_isotonic` chỉ fit DEV_CAL, lưu frozen knots và fit count/speakers. Assess lại DEV_CAL phải `IN_SAMPLE_CALIBRATION`; assess SEARCH ghi speaker-disjoint/development-exposed, checkpoint-selection reuse DEV_CAL là flag riêng. Confidence0,9 không tự nghĩa empirical correctness90%. Không dùng SEARCH labels để fit bất kỳ threshold/calibrator/weight nào.

## Chính sách đã khóa

`ARGMAX_DIAGNOSTIC` không giới hạn risk. `select_cal_policies` chọn S/O riêng, arbitration chỉ giữa class eligible, score cao hơn thắng, tie→S; insertion độc lập. Objective là actual post-arbitration `macro3_error_f1`, tối ưu **finite DEV_CAL grid định trước**, không continuous global optimum. Candidate quantiles33 điểm đều cộng đuôi .97/.98/.99/.995/.9975/.999/.9995/.9999, all/no-warning. Tie-break macro cao hơn, FP thấp hơn rồi thresholds cao hơn.

Các ngân sách .10/.20/.50/1/2 và unbounded chỉ chọn trên CAL, sau đó apply SEARCH bất biến. `CAL_CLASS_SPECIFIC_CONFIDENCE` là policy threshold riêng. `LEGACY_REPLAY` phải có recorded thresholds; giữ policy shared lịch sử để audit. SEARCH oracle chỉ diagnostic, không deployable/champion selection. Abstain giữ masks và full-population FN/coverage, không cắt population.

## Đầu ra và kiểm định

Migration dùng cùng evaluator, không tạo scoring implementation mới. Independent Luna audit dùng arithmetic riêng để đối chứng raw seed101, frozen shared/separate thresholds và CPL01 union. Test suite có known-vector/tie/reverse AP, AP-vs-trapezoid, zero-support, exact cross-utterance gaps, raw-vs-probability, wrong-type FWR, legacy parity fixture, row shuffle, abstention, calibration/speaker leakage, TRAIN-only normalizers, authorized manifest counts và1499-source-hash immutability.

`tools/eval_v3_robustness.py` tạo per-speaker/reference-phone AP/support, fixed K25/50/100/200 với tie-break event-ID không gold, reliability SVG/bins, CAL→SEARCH shift, raw-vs-frozen-isotonic confidence quality, matched seeds101/202/303 mean±sample SD và200 speaker-cluster bootstrap exploratory trên4clusters.

**Validity hạn chế:** canonical target identities/order có nguồn TextGrid; không suy zero leakage từ việc inference bỏ gold timing. Genuine sampled orthographic-reference WAV smoke không xác minh toàn bộ cached references. DEV_SEARCH đã development-exposed, không final blind test. Missing artifacts/unsupported domains giữ nguyên trong non-comparable inventory; không retrain/khôi phục holdout/restart Phase3B.
