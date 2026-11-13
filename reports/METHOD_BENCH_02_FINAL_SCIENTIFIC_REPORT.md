# LANGFENS RQ1-METHOD-BENCH-02 — Báo cáo khoa học cuối kỳ

**Ngày:** 08-11-2026 · **Đánh giá:** UnifiedEvaluatorV3 / EVAL-V3.1 · **DEV_SEARCH:** thăm dò, đã exposure; không phải test mù.

## Khuyến nghị

Chưa đưa các phương pháp mới vào U2. Tiếp tục đầu tư có điều kiện vào nhận dạng chuỗi bằng CTC âm vị thật và ánh xạ edit path sang gap N+1, sau khi có checkpoint với provenance tách split và kiểm soát phone-token phù hợp. Trong các thử nghiệm hiện tại, H04 không vượt prior-only, OTTC dạng window yếu hơn neural specialist, discrete-unit bigram giảm AP, H05 optional-deletion chỉ cải thiện so với forced-path control của chính nó; decoded extra-phone insertion AP giảm mạnh.

Độ tin cậy **vừa** cho các kết luận âm về cấu hình đã chạy; **thấp** cho kết luận về toàn bộ họ phương pháp vì run mới chỉ một seed và DEV_SEARCH có bốn speaker. Dấu hiệu đổi quyết định là một CTC sequence model hợp lệ cải thiện O và exact-gap I so với matched controls, ổn định qua speaker/seed.

## Dữ liệu và kiểm tra tính nhất quán

Manifest chính thức `data/derived/l2arctic/manifests/dev_cal.jsonl` xác nhận **300 utterances, 2 speakers, 10.043 targets và 10.343 gaps**; mọi utterance đều đạt N+1. Số **9.744 targets** trong `reports/RQ1_UNIFIED_02_BAO_CAO_TONG_HOP_CHI_TIET.md` là typo báo cáo, đã sửa thành **10.043**. Manifest SHA256 `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6`.

DEV_SEARCH: 599 utterances/4 speakers, 19.755 targets/20.354 gaps (S=1.768, O=209, I=125). TRAIN: 1.800 utterances/12 speakers, 59.562 targets/61.362 gaps. DEV_SEARCH không fit/select bất kỳ model, threshold hay hyperparameter nào. Không mở TEST_SEALED/protected data. `nvidia-smi` không liên lạc được NVIDIA driver; GPU lease an toàn không có, nên các run mới dùng CPU.

## Phương pháp ngoài và thực nghiệm

- **CTC GOP:** Cao et al. (2024) dùng phone-CTC để hỗ trợ alignment-free S/O/I scoring. Cache cục bộ `facebook/wav2vec2-lv-60-espeak-cv-ft` có CTC config/tokenizer (392 token, eSpeak/IPA) nhưng thiếu model weights. Không tạo CTC từ frame-classifier scores.
- **H04 constrained alternatives:** Parikh et al. (2025) hạn chế competitor phones theo phonological confusions. Đã chạy frozen frame-classifier GOP-style score screen với unrestricted, TRAIN confusion và prior-only controls.
- **H05 edit path:** đã chạy optional-phone deletion và silence-topology ablation, cùng frame-argmax sequence decode + Levenshtein exact-gap mapping. Nguồn là `Wav2Vec2ForFrameClassification`; score path không phải likelihood.
- **Discrete units:** Sara & Chowdhury (2026) đề xuất unit surprisal và text-guided DTW. Đã kiểm tra matched unigram/bigram adaptation; native audio/text unit assets không có.
- **OTTC/CROTTC:** Geng et al. (2026) đề xuất ordered subphonetic states/optimal temporal transport; đã chạy một static-plan window adaptation. Full CROTTC/IF code và corpus bundle có công khai, nhưng bundle không dùng vì split provenance không an toàn cho TRAIN-only result.
- H01 soft targets và H02/H03 prototype/surprisal được khôi phục làm controls, không chạy lại công thức cũ.

Nguồn gốc: [Cao et al. 2024](https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.html), [Parikh et al. 2025](https://www.isca-archive.org/interspeech_2025/parikh25_interspeech.html), [Sara & Chowdhury 2026](https://arxiv.org/abs/2606.19910), [Geng et al. OTTC 2026](https://arxiv.org/abs/2609.13694), [Geng et al. prompt-free MDD 2026](https://arxiv.org/abs/2604.22133), [IF-MDD repository](https://github.com/Secondtonumb/IF-MDD).

## Kết quả định lượng chính

### H04 — constrained alternative scoring

| S score | AP-S | Ghi chú |
|---|---:|---|
| Gold-Only S probability | 0.36460 | neural reference |
| Hard frame mismatch | 0.14508 | acoustic |
| Canonical-vs-all alternatives | 0.18817 | acoustic |
| TRAIN-confusion alternatives | 0.19487 | acoustic, coverage 19.755/19.755 |
| TRAIN phone prior only | 0.29923 | không dùng acoustic |

Constrained alternatives nhỉnh hơn unrestricted acoustic score **+0.00670 AP**, nhưng thấp hơn prior-only 0.10435 và Gold-Only 0.16973. Không ủng hộ H04 như scorer S độc lập hiện tại. Đây là frame-classifier GOP-style score, không phải CTC likelihood.

### H05 — omission topology và exact-gap insertion

| DEV_SEARCH arm | AP-O | AP-S | AP-I exact-gap | Macro-3 AP | Macro-3 F1 | Macro-4 F1 | False warnings | Joint FWR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Forced Viterbi | 0.03677 | 0.36460 | 0.18672 | 0.19603 | 0.21586 | 0.39320 | 1.735 | 2.8965 |
| Optional phone; adjacent SIL retained | **0.08793** | 0.36460 | 0.18672 | **0.21308** | 0.21565 | 0.39313 | 1.720 | 2.8715 |
| Optional phone; adjacent SIL collapsed | 0.08439 | 0.36460 | 0.18672 | 0.21190 | **0.21944** | **0.39611** | 1.702 | 2.8414 |
| Frame-argmax extra-phone count | 0.08439 | 0.36460 | **0.02825** | **0.15908** | 0.16982 | 0.35890 | 1.964 | 3.2788 |

Optional deletion AP-O tăng 0.05117 so với forced-path control, nhưng vẫn dưới Gold-Only AP-O **0.15943**. Gộp SIL giảm AP-O 0.00354 so với giữ SIL.

Exact-gap decode chỉ chấm insertion nếu **mọi** minimum-edit path gán extra span vào cùng một gap; vùng mơ hồ abstain, không dùng gold để phân xử. 568 extra spans không mơ hồ được gán vào 488 gaps; 1.015/20.354 gaps (4.99%) mơ hồ; 359/599 utterances có nhiều edit path tối ưu. AP-I trên toàn gap population **0.02825**, so với frozen gap head **0.18672**. Trên riêng gaps không mơ hồ AP=0.03448; subset mơ hồ AP=0.02660. Với quy tắc cảnh báo predeclared `count > 0`, precision/recall/F1 là 0.0533/0.2080/0.0848 (26 TP, 462 FP, 99 FN). Đây là bằng chứng âm trực tiếp cho frame-argmax/edit-gap mapping này.

H05 substitution/insertion heads trong omission ablation được kế thừa từ Gold-Only; AP-I 0.18672 không phải gain mới. H05 insertion decode dùng raw count extra spans, không probability/calibration.

### Representation mới và các control

| Phương pháp | DEV_CAL | DEV_SEARCH | Hỗ trợ |
|---|---:|---:|---|
| H03 unigram binary target AP | 0.34781 | 0.14318 | anomaly target |
| H03 phone-conditioned bigram binary target AP | 0.25997 | 0.11736 | anomaly target; kém unigram 0.02582 |
| OTTC adaptation binary target AP | 0.43744 | 0.19146 | S AP .15842; O AP .01058; không gap score |
| H01 hard / soft AP-S | — | 0.36356 / 0.36214 | soft giảm 0.00142 |
| H01 hard / soft AP-O | — | 0.15360 / 0.14672 | soft giảm 0.00688 |

OTTC adaptation là static transport từ 17 frame đến 3 state, MLP nhỏ, 6 epoch, 48.73 s CPU; checkpoint SHA256 `6c830bd653fc078b738d370c1b0e7a59a20e2529ea8f699f628d15cbffb41da8`. Không có sequence decoder, blank state hay gap features; I và Macro-3 AP không ước lượng. Bigram cũng chỉ binary target score, không chẩn đoán riêng S/O.

## So sánh, operating points và bổ sung

Full-model Macro-3 AP tham chiếu DEV_SEARCH: Gold-Only 0.236917; H-002 0.243940; U2 mean 0.227894; EXP24 0.276819; EXP29 0.282223; EXP36 0.289233; EXP43 0.284760; Phase 4 fixed fusion ≈0.2955; omission-specialist fusion ≈0.3018. H05 full composites 0.15908–0.21308 thấp hơn Gold-Only. H04 S AP và OTTC/H03 specialist scores không vượt baseline tương ứng. Không phương pháp mới nào tiến gần specialist frontier.

Theo EVAL-V3.1 `CAL_MAX_ERROR_MACRO_F1` fit trên DEV_CAL rồi đóng băng: H05-C đạt S P/R/F1 .2796/.6261/.3866, O .1165/.2967/.1673, I .2193/.3280/.2628; Macro-3 Error F1 .2723, Macro-4 F1 .4243, **3,468** false warnings / 599 utterances (joint FWR 5.7896). Giá trị arm C fixed-policy ghi trước đó (O F1 .0094; FWR 2.841) thuộc một ngưỡng/policy khác, không phải kết quả của CAL_MAX_ERROR_MACRO_F1; xem bảng threshold và cả ba arm tại `reports/METHOD_BENCH_02_C_COMPARATIVE_ANALYSIS.md`. Frame-decoded I không có DEV_CAL raw-count predictions nên không có CAL-selected insertion threshold; với predeclared count > 0, hybrid H05-A target + frame I đạt I P/R/F1 .0501/.2080/.0807, Macro-3 .2084, Macro-4 .3707 và joint FWR 7.5743. Đây là exploratory operating point, không phải policy CAL đầy đủ.

Agent C kiểm tra top-100 overlap của các target predictions cũ: hard phone bắt 57 lỗi, soft phone 58, discrete surprisal 30; union bắt 99 lỗi trong 229 alert độc nhất. Đây chỉ là mô tả overlap, không phải fusion fixed-budget hay chứng cứ bổ sung nhân quả.

## Kết luận theo câu hỏi

1. Đã điều tra CTC GOP, constrained alternatives, edit paths, discrete units, SylPh/normality, OTTC/CROTTC, soft phone recognition và prompt-free MDD.
2. Họ thực sự chạy: H04 frame-GOP, optional deletion DP, frame classifier sequence decode + exact-gap mapping, discrete unit bigram và ordered-state OT adaptation.
3. H04 chỉ nhỉnh hơn unrestricted +0.0067 AP-S nhưng prior-only tốt hơn; không hỗ trợ cấu hình hiện tại.
4. H05 cải thiện omission so với control yếu, nhưng chưa vượt neural O baseline; exact-gap insertion decode AP .02825, kém frozen gap head .18672.
5. Representation mới không cải thiện target ranking so với neural specialist; bigram thua unigram; OTTC adaptation âm.
6. Không cải thiện exact-gap insertion. Inherited 0.18672 không được quy cho sequence method.
7. Không method mới vượt Gold-Only/H-002 hay các full-model frontier. Không run nào lặp seed.
8. H04 cho thấy prior-only nổi trội hơn frame acoustic alternatives. Edit-path difference gợi ý acoustic score có thông tin O trong matched comparison, nhưng không chứng minh causal tổng quát.
9. CPU đủ cho DP, score-screen, discrete LM, và small MLP; CUDA không chạy do driver/lease. Full CROTTC/IF trên GTX1650 chưa xác định.
10. Không hỗ trợ: H01 soft-target recipe, H03 within-window bigram, H04 frame acoustic constrained GOP làm scorer độc lập, và OTTC window adaptation. H05 chỉ đáng giữ làm nghiên cứu về CTC/null-path thật, không phải cách mapping hiện tại.

**Hướng tiếp theo:** có genuine phoneme-CTC weights với split provenance và phone mapping; tạo sequence edit lattice cùng uncertainty; đánh giá S/O và N+1 exact-gap I độc lập; chỉ sau khi có tín hiệu CAL/SEARCH hợp lý mới lặp nhiều seed hoặc tích hợp unified model.

## Mức độ xác minh và giới hạn

- **Đề xuất nghiên cứu:** full CTC GOP, CROTTC/IF-MDD, transcript-guided unit DTW.
- **Thuật toán đã triển khai:** H04 score functions; H05 optional deletion/SIL-collapse DP và frame decode mapping; H03 bigram; OTTC static-plan MLP.
- **Đã chạy:** receipts, scripts, CPU time, checkpoint/input hashes lưu dưới `experiments/rq1_method_bench_02/agent_{a,b,c}/`.
- **Số đã xác minh:** EVAL-V3.1 coverage đúng event populations; manifest audit xác nhận N+1.
- **Kết luận khoa học:** exploratory vì DEV_SEARCH exposed, bốn speakers và một seed. Intended reference vẫn annotation-derived, independence chưa xác minh. Frame-classifier outputs không phải CTC; không có blind generalization claim.

Reproduction: [METHOD_BENCH_02_REPRODUCTION.md](METHOD_BENCH_02_REPRODUCTION.md). Chi tiết: [Agent A](METHOD_BENCH_02_A_SEQUENCE_METHODS.md), [Agent B](METHOD_BENCH_02_B_ALTERNATIVE_REPRESENTATIONS.md), [Agent C](METHOD_BENCH_02_C_COMPARATIVE_ANALYSIS.md).
