# LANGFENS RQ1-METHOD-BENCH-04 — Báo cáo khoa học cuối

**Ngày:** 08-11-2026 · **Evaluator:** EVAL-V3.1 · **Phạm vi:** TRAIN speaker-grouped validation, DEV_CAL và DEV_SEARCH exposed (chỉ exploratory). Không huấn luyện lại recognizer CTC MB03, không truy cập protected holdout.

## Kết luận chính

E4 được triển khai trên posterior thật của ba Wav2Vec2-CTC recognizer bằng dynamic programming CTC forward trong log-space. So với E3 dùng cùng recognizer, E4 tăng AP substitution (S), omission (O), exact-gap insertion (I) và Macro-3 trên cả ba nhóm speaker TRAIN holdout. Macro-3 tăng từ `.0802–.0829` lên `.1465–.1526`; classwise gains ổn định qua seed 101/202/303. Trên DEV_CAL, E4 tiếp tục tăng S/O và Macro-3 nhưng giảm I; DEV_SEARCH seed 101 cũng tăng S/O và Macro-3, giảm nhẹ I. Do đó E4 là cải thiện có cơ sở cho target S/O ranking và đáng tiếp tục nghiên cứu, nhưng chưa phải scorer cân bằng S/O/I hoặc model vượt neural frontier.

Candidate mass thấp đã được kiểm tra bằng hai cách độc lập: CTC forward kernel so với Python implementation và exhaustive enumeration trên các fixture nhỏ. Tính toán hợp lệ; mass nhỏ đến từ beam/candidate truncation rất mạnh. Các likelihood ratios là evidence scores, không phải calibrated event posterior.

Hai can thiệp speaker-generalization của Agent B không cho thấy gain vững chắc: utterance-centered residualization làm giảm grouped Macro-3; Gaussian jitter cho omission gain nhỏ ở một seed nhưng fold behavior không đồng nhất. Phương pháp non-CTC lấy cảm hứng từ SylPh cải thiện so với positive-only anomaly control nhưng thấp hơn neural baseline và không có insertion score.

## Kết quả đã thực thi

### E4 so với matched E1/E3 trên TRAIN speaker holdout

Mỗi seed giữ nguyên recognizer Wav2Vec2, posterior, tokenizer, blank ID 42, frame clock 20 ms và event population (300 utterances từ hai held-out TRAIN speakers; 9,839 targets/10,139 gaps). E4 dùng tối đa bốn substitution competitors và tám insertion phones được chọn từ 1,500 TRAIN-fit utterances; mọi `N+1` exact gaps được giữ riêng.

| Seed | Arm | AP-S | AP-O | AP-I exact-gap | Macro-3 Error AP |
|---:|---|---:|---:|---:|---:|
| 101 | E1 | .11970 | .01944 | .00581 | .04832 |
| 101 | E3 | .17852 | .05217 | .00997 | .08022 |
| 101 | **E4** | **.28233** | **.11844** | **.03872** | **.14650** |
| 202 | E1 | .11239 | .01880 | .00587 | .04568 |
| 202 | E3 | .17565 | .05671 | .01640 | .08292 |
| 202 | **E4** | **.28952** | **.12144** | **.03404** | **.14833** |
| 303 | E1 | .11214 | .01742 | .00485 | .04480 |
| 303 | E3 | .17409 | .04936 | .01743 | .08029 |
| 303 | **E4** | **.28737** | **.12549** | **.04480** | **.15255** |

E4-minus-E3 Macro-3 gains là `+.0663`, `+.0654`, `+.0723`. S/O/I đều tăng trong từng seed. Tuy vậy holdout chỉ gồm hai speakers; bốn-fold event OOF neural baseline có population lớn hơn và không nên coi đây là ước lượng speaker-transfer đầy đủ.

### DEV_CAL và DEV_SEARCH

| Split / seed | Arm | AP-S | AP-O | AP-I exact-gap | Macro-3 Error AP |
|---|---|---:|---:|---:|---:|
| CAL / 101 | E1 | .25776 | .09182 | .02029 | .12329 |
| CAL / 101 | E3 | .37671 | .21035 | .06120 | .21608 |
| CAL / 101 | **E4** | **.45752** | **.40937** | .03299 | **.29996** |
| CAL / 202 | E3 | .38499 | .20276 | .05733 | .21503 |
| CAL / 202 | **E4** | **.45841** | **.41100** | .03352 | **.30098** |
| CAL / 303 | E3 | .37061 | .19435 | .06568 | .21021 |
| CAL / 303 | **E4** | **.46037** | **.39847** | .04520 | **.30135** |
| SEARCH / 101 | E1 | .14660 | .02257 | .01050 | .05989 |
| SEARCH / 101 | E3 | .23547 | .04186 | **.02871** | .10202 |
| SEARCH / 101 | **E4** | **.31685** | **.15774** | .02357 | **.16606** |

CAL per-seed corroborates S/O and Macro-3 improvements but also confirms E3 has higher I AP for all three seeds. SEARCH is exposed and exploratory. Against the historical DEV_SEARCH neural/composition references, E4 S/O/I `.3169/.1577/.0236` and Macro-3 `.1661` remain below Gold-Only `.3646/.1594/.1867` (Macro-3 `.2369`), H-002 `.3778/.1673/.1867` (`.2439`), U2 `.3714/.1853/.1180` (`.2249`), EXP24 `.2768`, EXP36 `.2892`, EXP43 `.2848`, and Phase 4 omission-specialist composition ≈`.3018`. E4 is not a new champion.

### Speaker-generalizable learning

Historical grouped TRAIN OOF plain neural-head AP was S/O/I `.423209/.276706/.162125`, Macro-3 `.287347`, compared with utterance-random `.478970/.359747/.220765`, Macro-3 `.353161`. The difference is split-sensitive, but it does not isolate speaker identity as the sole cause.

| Intervention | AP-S | AP-O | AP-I | Macro-3 | Kết luận |
|---|---:|---:|---:|---:|---|
| Plain grouped OOF | .423209 | .276706 | .162125 | .287347 | Historical matched reference |
| Gaussian feature jitter, S101 | .424092 | .277921 | .166973 | .289662 | O +.0065, mixed folds; unconfirmed single seed |
| Utterance-centered residualization, S101 | .406526 | .257928 | .163893 | .276116 | Macro-3 −.0112; rejected |

No intervention improved speaker-held-out ranking reproducibly. These are direct error heads, not phone recognizers, so PER is not applicable. Prior GRL showed no consistent fold gain and was not rerun.

### Non-CTC acoustic method

External discovery selected SylPh, an Interspeech 2025 method that compares normal positive acoustic patterns with pseudo-negative patterns and uses adaptive text/audio contrastive learning. The implementation here is a clearly labeled adaptation: correct-only, phone-conditioned acoustic centroids and a cosine margin to the nearest competing phone; it does not reproduce SylPh architecture or train its contrastive objective. The primary paper describes syllable-level Korean audiometry and phoneme-level abnormality localization ([Jin et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/jin25c_interspeech.html)). A foundational one-class pronunciation-verification approach also uses phoneme-specific normal models ([Shahin & Ahmed, 2019](https://doi.org/10.1016/j.specom.2019.06.003)).

Cosine-margin grouped OOF AP-S/O was `.211316/.165356`, lower than plain neural grouped `.423209/.276706`. Exposed SEARCH was `.164551/.060511`, below historical neural target scores. Positive-only diagonal distance was weaker (`.142699/.038721` grouped). The method supports target S/O only; no I score, Macro-3, or invented gap evidence is reported. No external alternative beat the relevant neural baseline.

## Trả lời các câu hỏi nghiên cứu

1. **E4 với posterior CTC thật?** Có. Dùng đúng MB03 Wav2Vec2-CTC arrays; không retrain.
2. **Likelihood scoring tăng AP S/O/I so với E1/E3?** Trên TRAIN holdout, tăng cả ba lớp qua cả ba seeds. Trên CAL, tăng S/O nhưng giảm I; trên SEARCH S101, tăng S/O và Macro-3, giảm I nhẹ so với E3.
3. **Candidate mass được xác minh?** Có. Forward kernel khớp Python forward; exhaustive path fixtures khớp xác suất sequence và tổng xác suất bằng 1. Posterior FP16 được row-normalize lại.
4. **Mass thấp do đâu?** Candidate/beam coverage, không phải accounting bug. Top-10 + canonical exact mass trên W2V2 S101 có mean khoảng `1.3e-5` và median `3.0e-13` trên SEARCH. Một diagnostic MB03 riêng trên log-mel CTC cho thấy width 10→100 chỉ tăng top-10 mean từ `2.32e-6` lên `3.02e-6`; mở beam đơn thuần chưa giải quyết coverage.
5. **Aggregation rộng hơn có giúp?** E4 aggregate likelihood của nhiều event-conditioned alternatives và vượt E3 cho S/O, nhưng không cải thiện insertion trên CAL/SEARCH. Bên trong E4, logmeanexp và max gần nhau, không có aggregator thắng ổn định; mean hơi tốt hơn insertion/grouped Macro-3, max nhỉnh hơn S trên CAL.
6. **Ổn định qua CTC seeds?** Có trên TRAIN speaker holdout cho cả S/O/I và Macro-3; CAL S/O/Macro-3 cũng ổn định. I không ổn định theo hướng tăng trên CAL.
7. **Quan hệ PER–MDD AP?** PER W2V2 beam-10 trung bình `.3303` holdout, `.3458` CAL, `.3216` SEARCH. E4 không thay recognizer; chỉ ba seed không đủ kiểm tra tương quan. PER thấp hơn không mặc nhiên cho event AP tốt hơn.
8. **Thử nghiệm speaker-grouped nào?** E4 matched E1/E3 trên TRAIN-speaker holdout (hai speakers, ba seeds); Agent B dùng bốn fold x 3 held-out speakers trên TRAIN OOF; non-CTC prototypes cũng dùng cùng grouped folds.
9. **Speaker-generalizable training cải thiện lớp nào?** Chưa có gain lặp lại. Jitter có O +.0065 một seed; residualization hại S/O. OOF fold biến thiên lớn.
10. **Phương pháp non-CTC được tìm và thử?** SylPh-inspired correct-only phone-normal prototypes/cosine margin; đã chạy grouped OOF, CAL và SEARCH exploratory. One-class phoneme verification được dùng làm foundational context, không tái lập trực tiếp.
11. **Có method nào vượt neural baseline liên quan?** Không. E4 vượt matched CTC E1/E3, không vượt các neural/champion references DEV_SEARCH. Prototype margin cũng thấp hơn neural target models.
12. **Omission và insertion?** E4 là cải thiện mạnh nhất cho O; I tăng trên grouped TRAIN holdout nhưng thấp hơn E3 trên CAL và SEARCH. Non-CTC anomaly có target-only scores; Agent B jitter không cho I gain đáng kể.
13. **Giả thuyết nào bị bác bỏ?** Utterance mean removal giúp speaker transfer; prototype anomaly vượt neural baseline; beam widening đơn thuần sửa candidate coverage; CTC E4 đã giải quyết exact-gap insertion. Không bác bỏ mọi dạng speaker invariance, toàn bộ lattice marginalization, hoặc non-CTC methods.
14. **Cải thiện lặp lại lớn nhất?** E4 so E3 trên grouped holdout: AP-O tăng `.0647–.0761` qua seeds; Macro-3 tăng `.0654–.0723`; AP-I tăng `.0176–.0288`. Đây là matched within-recognizer comparison.
15. **Hướng tiếp theo có cơ sở nhất?** Giữ E4 cho S/O research, tập trung xây gap-local acoustic likelihood/alignment để sửa I; sau đó chạy 4-fold grouped TRAIN OOF với nhiều held-out speakers, chọn chính sách trên CAL, không dùng SEARCH để tune. Chưa tích hợp E4 vào unified model.

## Thực thi, đo lường và diễn giải

**Đã triển khai và chạy:** CTC forward kernel và kiểm chứng độc lập; E4 S101/202/303 trên TRAIN speaker holdout/CAL; E4 S101 SEARCH; matched E1/E3; max aggregation, per-frame normalization, reference-prior và prevalence controls; grouped event-head jitter/residualization; non-CTC phone-normal anomaly.

**Số đo chính:** E4 CPU khoảng 30–32 giây mỗi 300 utterance TRAIN holdout, 36–39 giây mỗi 300 CAL, 65 giây cho SEARCH (262,206 candidate sequence scores, 20,354/20,354 gap events). Peak memory không được instrument. Agent B residualization CPU 20.61 giây; jitter CUDA 20.44 giây trên GTX 1650; C prototype CPU 3.31 giây. Không có checkpoint E4 mới; B đã huấn luyện head fold-specific trong bộ nhớ nhưng chỉ giữ predictions/receipt, không lưu checkpoint.

**Published literature:** SylPh báo cáo syllable MDD và khả năng phoneme-localization trên dữ liệu speech audiometry tiếng Hàn; đó là tuyên bố của paper, không phải kết quả LANGFENS.

**Diễn giải exploratory:** E4 cung cấp bằng chứng target acoustic event likelihood hữu ích hơn CTC edit paths hiện có trên cùng recognizer. Split variation và chỉ hai TRAIN holdout speakers giới hạn kết luận. SEARCH numbers chỉ là kiểm tra thăm dò.

**Chưa giải quyết:** E4 probability calibration; full candidate/lattice coverage; insertion rank; xác nhận grouped transfer qua nhiều folds cho chính E4; non-CTC exact-gap evidence; protected holdout.

## Artefact và reproduction

- [E4 event CTC](METHOD_BENCH_04_A_EVENT_CTC.md)
- [Speaker learning](METHOD_BENCH_04_B_SPEAKER_LEARNING.md)
- [External alternatives](METHOD_BENCH_04_C_EXTERNAL_ALTERNATIVES.md)
- [Experiment leaderboard](METHOD_BENCH_04_EXPERIMENT_LEADERBOARD.csv)
- [Ablation matrix](METHOD_BENCH_04_ABLATION_MATRIX.csv)
- [Reproduction](METHOD_BENCH_04_REPRODUCTION.md)
- [Next research](METHOD_BENCH_04_NEXT_RESEARCH.md)

Các lệnh, checkpoint/posterior source, hash và giới hạn runtime nằm trong reproduction receipt. Score ledgers được giữ ở `experiments/rq1_method_bench_04/`; mọi artifact lịch sử MB03 được bảo toàn.
