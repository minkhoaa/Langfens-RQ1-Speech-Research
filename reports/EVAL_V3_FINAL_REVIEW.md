# Review độc lập EVAL-V3

> Snapshot review baseline, không phải acceptance cho toàn bộ closure mới. Kết luận bàn giao hiện hành: `reports/EVAL_V3_CLOSURE_REPORT.md` và `reports/EVAL_V3_REQUIREMENTS_ACCEPTANCE.md`; independent closure artifacts ở `experiments/evaluation_v3/closure/independent_*`. Các số test/scope dưới đây thuộc thời điểm baseline.

Reviewer fresh-context Luna `openai-codex/gpt-6-luna`, run `53e5f804-c399-4e85-99f5-04e24eb27236`. Phạm vi: evaluator/schema/policy/calibration, independent audit, waveform smoke, robustness và safety tests; migration được tích hợp sau, không gán review này cho inventory completeness.

## Kết luận reviewer

**OK with notes**: raw scores tách khỏi probability simplex; không suy score từ decisions; exact-gap matching; wrong-type target warnings tính FWR đúng một lần. WAV smoke dùng WAV thật, orthographic prompt và CMUDict local; full-population reference provenance vẫn UNVERIFIED.

Finding P2: wrapper `select_cal_max_error_macro_f1` dùng `num_utterances=1`, làm receipt `cal_joint_fwr` bằng warning count khi caller chưa cấp denominator. Parent đã sửa: nhận denominator thực tùy chọn; thiếu thì trả null + `NOT_ESTIMABLE_UTTERANCE_DENOMINATOR_MISSING`, không báo tỷ lệ giả. Joint budget selector luôn yêu cầu số utterance thực. Regression test bổ sung để không tái xuất lỗi.

## Arithmetic verification riêng

Luna lane C dùng tie-group AP/confusion arithmetic độc lập, không import evaluator. Đã tái dựng seed101 shared/separate TP1/3/29→27/3/29, FP63→75 và CPL01 omission union85 từ raw events. Receipt: `experiments/evaluation_v3/INDEPENDENT_AUDIT.json`.

## Review migration/historical bổ sung

Fresh-context Luna run `a6b8fbd2-fc7b-48e0-aa9a-cf8e2ab94fd2` review migration/historical/frozen/curve/selective/reconciliation. Hai P1 ban đầu: discovery có thể gán FULL_SCORE_REPLAY trước khi đủ AP S/O/I; migration replay ghi báo cáo vào source/reports. Parent sửa cả hai: discovery chỉ partial, full rows phải có đủ class AP/macro AP và full topology; replay ghi report vào output/reports, publication vào source là tool tường minh riêng. Reviewer kiểm lại tại resumed run `09ffbb24-4f11-4ff4-8075-31e4a1b51864`: **đóng cả hai P1, Merge verdict OK**, không thấy blocker cụ thể còn lại trong scope sửa. Đây là static review, không phải reviewer đã chạy toàn replay/tests.

Review không chứng nhận mọi scientific gate, full-waveform cohort, inventory completeness, zero leakage, hay KD superiority. Exact final test result và validation commands: `coordination/EVAL_V3_ACCEPTANCE_TESTS.log`.
