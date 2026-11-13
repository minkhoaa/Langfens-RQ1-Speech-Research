# LANGFENS RQ1-METHOD-BENCH-01 — Báo cáo khoa học hiện tại

**Cập nhật:** 2026-11-08  
**Đánh giá:** EVAL-V3.1  
**Đây là báo cáo vòng hiện tại:** nghiên cứu đa họ đã hoàn tất cổng lựa chọn ban đầu; benchmark đang tiếp tục. Chưa có benchmark S/O/I đầy đủ cho phương pháp mới.

## Kết luận điều hành

Nghiên cứu ngoài đã thu thập 22 bản ghi nguồn duy nhất và phân loại 15 phương pháp hoặc đối chứng trong chín họ: GOP/alignment, nhận dạng chuỗi, mục tiêu mềm/uncertainty, đặc trưng ngữ âm học, anomaly/normality, SSL rời rạc và surprisal, đồ thị nhầm lẫn, mô hình speech-text/attention và chuẩn chấp nhận theo người nghe. Cơ sở dữ liệu ghi rõ mức kiểm chứng của bài, code và checkpoint; không dùng số liệu khác corpus như LANGFENS kết quả.

Năm cơ chế được chọn để benchmark lần lượt là nhận dạng âm vị đã phát ra với supervision mềm; anomaly/normality; surprisal trên đơn vị SSL; GOP với đối thủ bị giới hạn; và CTC không phụ thuộc forced alignment. Cách chọn bao phủ nguồn bằng chứng khác nhau, đồng thời cho phép chạy CPU/frozen-score trước khi xin lease GPU.

Hai hướng đã có screen CPU: nhận dạng target-window với phone soft targets so với hard-target control, và surprisal trên đơn vị rời rạc từ cached embeddings. Trên DEV_SEARCH, soft target không cải thiện AP substitution hoặc omission; binary target MD AP tăng nhẹ. Discrete-unit surprisal đạt binary target AP 0.14318. Cả hai là adaptation một seed, không có insertion-gap output. Hiện chưa có hypothesis method mới nào được hỗ trợ; chưa thể kết luận về các paper gốc.

## Trạng thái theo chuẩn báo cáo

| Trạng thái | Đối tượng | Kết quả |
|---|---|---|
| **RESEARCHED** | 22 source records, 15 method candidates/controls | Nguồn 2018–2026; 9 họ. Danh mục ở PAPER_DATABASE.csv. |
| **SELECTED** | H01–H05 | Portfolio và điều kiện triển khai nằm trong EXPERIMENT_PORTFOLIO.md. |
| **IMPLEMENTED** | H01 target-window phone recognizer; H02 correct-only cosine prototype; H03 discrete-unit surprisal | H01 và H03 là adaptation. H02 không phải SylPh reproduction. |
| **TRAINED** | H01 hard-target và similarity-soft MLP; H03 K-means codebook và correct-window token counts | Seed 101, CPU. H01 mất 8.44 và 8.03 giây/arm. H03 mất 4.074 giây, gồm token counts. H02 tạo centroid correct-only, không phải neural training. |
| **BENCHMARKED** | H01 C/S/O target ranking; H02 và H03 binary target anomaly | H01 S/O AP trên DEV_CAL và DEV_SEARCH; H02/H03 binary target AP. Chưa có phương pháp mới nào đủ cả S/O/I. |
| **SUPPORTED** | Chưa có hypothesis mới | Không có kết luận ủng hộ phương pháp mới sau một screen. |
| **NOT SUPPORTED (cấu hình hiện tại)** | H01 similarity-soft targets; H02 centroid; H03 unit surprisal | H01 soft arm giảm AP S/O so với hard arm; H02/H03 binary target AP trên SEARCH dưới H-002 receipt. Không bác bỏ toàn bộ các họ method. |

## Kết quả định lượng

| Method/arm | Population | AP S | AP O | Binary target MD AP | AP I exact-gap | Macro-3 Error AP |
|---|---|---:|---:|---:|---:|---:|
| H01 hard-target MLP | DEV_SEARCH, 599 utterances, 4 speakers | 0.36356 | 0.15360 | 0.38365 | N/A | N/A |
| H01 similarity-soft MLP | DEV_SEARCH, 599 utterances, 4 speakers | 0.36214 | 0.14672 | 0.38832 | N/A | N/A |
| H02 correct-only cosine centroid | DEV_SEARCH, 19,755 targets | N/A | N/A | 0.15644 | N/A | N/A |
| H03 discrete-unit surprisal | DEV_SEARCH, 19,755 targets | N/A | N/A | 0.14318 | N/A | N/A |

H01 soft-minus-hard changes: AP S −0.00142; AP O −0.00688; binary target MD AP +0.00468. Đây là các thay đổi từ một seed, không có bootstrap/seed uncertainty và DEV_SEARCH đã được dùng trong chiến dịch rộng trước đây; xem như exploratory. H01 argmax operating decisions chưa calibrated và có warning rates cao; chúng không phải kết quả tối ưu ngưỡng và không dùng tiêu chí FWR để loại nghiên cứu.

H02 comparator receipt từ H-002 cho binary target MD AP 0.34570 trên cùng SEARCH target population. H02 dùng frozen 17-frame acoustic windows, pooling trung bình và centroid theo phone; H03 dùng codebook 64 đơn vị được fit từ TRAIN cùng token counts correct-only theo phone. Cả hai không có S/O diagnosis hay gap outputs. Do đó chỉ so AP target-binary trên cùng miền, không đặt vào Macro-3 leaderboard.

DEV_SEARCH gold support trong corpus: 1,768 substitution; 209 omission; 125 insertion; 20,354 exact gaps tổng. H01, H02 và H03 không có điểm insertion và không được điền thay bằng nhãn giả.

## Diễn giải khoa học

Screen hiện tại hỗ trợ khả năng huấn luyện nhanh CPU một recognizer mục tiêu từ cache CTC-derived; nó không kiểm tra free-running recognized sequence hay edit alignment như Zhou et al. Bằng chứng cho soft target yếu và không thuận lợi với AP S/O ở cấu hình được thử. Binary AP nhích lên nhưng không đại diện đầy đủ chất lượng chẩn đoán. Bước tiếp theo hợp lý là kiểm tra S/O bằng một objective được định trước khác hoặc tiến sang sequence-level decoder nếu checkpoint/cache phù hợp, giữ hard-target làm matched control.

Prototype screen của SylPh-inspired có ranking yếu hơn H-002 nhưng chỉ test một centroid/simple-pooling mechanism. SylPh gốc dùng positive/pseudo-negative bags và adaptive text embeddings trên speech audiometry; không có đủ bằng chứng để tuyên bố cả anomaly family thất bại. Cần kiểm tra speaker confounding trước khi đầu tư thêm.

Những vấn đề như omission yếu, insertion hiếm, alignment, representation và generalization được giữ là giả thuyết. Không khẳng định bất kỳ nguyên nhân nào chỉ từ paper hoặc screen hiện tại.

## Khả năng so sánh và giới hạn

- H01 hiện đánh giá trên target windows đã được tạo từ model CTC với chuỗi đích; inference của mô hình MLP nhận target-conditioned acoustic summaries và target phone vector. Đây là adaptation, không phải benchmark trực tiếp của paper Zhou.
- H01 hỗ trợ target C/S/O; không tạo sequence hoặc N+1 gap I. H02 và H03 chỉ là binary target specialists.
- Không có full three-class AP/F1 hoặc class PR curves cho toàn bộ S/O/I trong vòng benchmark mới. Các CSV giữ trường không ước lượng là rỗng/N/A.
- Kết quả ngoài dùng SpeechOcean762, L2-ARCTIC, LATIC, child speech, synthetic accent hoặc speech audiometry cùng các metric PER/F1/PCC/FAR/FRR khác nhau; không dùng để xếp hạng với LANGFENS.
- No CUDA was used in the new H01/H03 runs. Local GPU was idle at one inspection, but earlier RQ1-UNIFIED-02 reservation was not verifiably released through an active lease receipt; campaign did not claim or take that GPU.
- Session agent registry showed only the root agent. The prior Agent A/B/C discovery, status and protocol artifacts were recovered and retained, but live follow-up messages could not be delivered. No replacement agents were started.

## Tệp kết quả và tái lập

- Research synthesis, taxonomy, database and gaps: corresponding METHOD_BENCH_01 research deliverables.
- Method selection matrix: METHOD_BENCH_01_METHOD_COMPARISON.csv.
- Results and leaderboard: METHOD_BENCH_01_EXPERIMENT_RESULTS.csv and METHOD_BENCH_01_BENCHMARK_LEADERBOARD.csv.
- Reproduction command, receipts and source limitations: METHOD_BENCH_01_REPRODUCTION.md.
- Code and artifacts: experiments/rq1_method_bench_01/agent_b/soft_phone_recognition/ and experiments/rq1_method_bench_01/agent_b/discrete_unit_surprisal/.

External primary sources include [CTC phoneme-level assessment](https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.pdf), [phonological-level MDD](https://www.isca-archive.org/interspeech_2024/shahin24_interspeech.pdf), [similarity-aware phone recognition](https://doi.org/10.21437/Interspeech.2025-2417), [SylPh](https://doi.org/10.21437/Interspeech.2025-1285), [constrained phonological GOP](https://doi.org/10.21437/Interspeech.2025-829), [discrete SSL token surprisal](https://arxiv.org/abs/2606.19910), and [OpenPA](https://arxiv.org/abs/2610.13826). Full source status is recorded in the paper database.

## Next decision

Continue E1 with the selected diversity set. First test whether local reusable CTC outputs can support an alignment-free score and a TRAIN-only constrained-GOP screen without new audio extraction; then replace the H03 unigram adaptation with the paper's sequence/text conditioning if source artifacts are obtainable. If these sources are unavailable, run labeled CPU feasibility variants and mark them as adaptations. A positive result must have matched controls, exact-gap coverage if claiming I, and a new seed before being called supported.
