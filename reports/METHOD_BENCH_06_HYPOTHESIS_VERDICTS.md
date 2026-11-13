# METHOD-BENCH-06 — Phán quyết giả thuyết

| Giả thuyết | Verdict | Evidence / limitation |
|---|---|---|
| E4 có ranking khác neural | **SUPPORTED (descriptive)** | Spearman `.4105 S/.2294 O`; có E4-only top-5% TPs 194 S/56 O. Khác biệt không đồng nghĩa hữu ích tại cùng policy. |
| E4 tăng pooled AP riêng lẻ so với neural | **REJECTED** | Neural `.42321/.27671` cao hơn E4 `.31972/.25936`. |
| E4 có lợi trong neural-low-confidence region | **INCONCLUSIVE / suggestive** | E4 AP `.33927/.28315` so với neural `.33016/.19626` trong lowest max-prob quartile; pooled retrospective subset. |
| E4 mang conditional S evidence vượt phone priors | **NOT SUPPORTED by current diagnostics** | phone-prevalence-only AP `.33608`; within-phone E4 `.22168`; phone control retrospective, chưa fitted nested. |
| E4 mang conditional O evidence vượt phone priors | **INCONCLUSIVE / suggestive** | within-phone E4 `.20351` vs phone-only `.10240`; không phải held-fold conditional estimator. |
| Fixed integration cải thiện S/O | **INCONCLUSIVE** | Pooled rank-mean `.44547/.35708` là exploratory, leakage-sensitive score normalization; không có fixed fold-valid comparison. |
| Learned integration cải thiện S/O | **NOT TESTED** | Không train meta-model. |
| Nested grouped cross-fitting hoàn thành | **NOT COMPLETED** | Không có inner folds/meta predictions; không claim leakage-free fusion. |
| Calibration/F1/warning policy được cải thiện | **NOT TESTED** | Không fit thresholds/calibrator, không báo FWR mới. |
| Direct exact-gap temporal probe improves I | **NOT SUPPORTED for tested local pooling; broader timing INCONCLUSIVE** | MB06 six-statistic predicted-alignment CTC-frame pooling AP-I `.02220`, below static matched `.03557` and MB03 neural `.162125`; held-fold `.02004/.03068/.01982/.02621`, better than static only in Fold 3. One seed; does not reject all temporal methods. |
| MB03 neural exact-gap baseline | **VERIFIED HISTORICAL** | `.162125` pooled; fold AP `.18879/.10271/.10945/.25154`, support 560/61,362; `gap_PLAIN_0..3` and receipt verified and used as matched context. |
| GMM/MixGoP adaptation là cải thiện | **REJECTED for tested adaptation** | K4 `.19307/.12649` < neural `.42321/.27671`; one seed, no I. |
| GMM K4 beats K1 reliably | **NOT SUPPORTED** | pooled uplift `.00136/.01140`, fold behavior mixed, not replicated seeds. |
| Có căn cứ xây unified E4+neural+insertion checkpoint | **NO** | Nested fusion unsupported; new insertion probe is below static and neural; no calibration evidence. |

Định nghĩa trạng thái: VERIFIED = thực thi/receipt/artifact có thể kiểm chứng; REPORTED = literature claim; exploratory = descriptive, không confirmatory; NOT TESTED/BLOCKED = không tạo số liệu.
