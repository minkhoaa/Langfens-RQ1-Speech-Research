# Kiểm chứng sâu protocol và ba phương pháp — 2026-11-13

Trạng thái: **VERIFIED_WITH_LIMITATIONS — REVIEW_ONLY, NOT_ADOPTED**.

Đã tính lại từ manifest TRAIN, replay evaluator trên dự đoán đã lưu, đối chiếu paper gốc và nguồn tác giả. Không chạy training hoặc inference mới; không mở official test, audio, checkpoint hay archive. Không sửa protocol, GOAL, nhãn hoặc kết quả cũ. Nhánh: `feat/protocol-deep-verification-20261113`.

## Kết luận có thể dùng để quyết định

Hai kết luận số học trong RECOMMENDATION_VI.md đúng trong phạm vi đã nêu. Các điểm F1 của ba paper cũng đúng. **Chúng không chứng minh một combo nào đạt yêu cầu đúng loại S/O/I, đúng gap, recall từng lớp ≥80% và FWR ≤0,20.**

Đề xuất trước đây thiếu bước nối giữa metric của paper và acceptance của sản phẩm. Tuy nhiên, bằng chứng hiện tại cũng không đủ để kết luận ba phương pháp đã được tái lập đầy đủ rồi thất bại. Cần phân biệt: đã đọc reference, đã chạy adaptation, đã chạy nguyên cơ chế, và đã tái lập kết quả paper.

Hai phát hiện bổ sung đáng chú ý:

- Ngay nhóm đủ supervision chuỗi vẫn có macro F1 rất thấp. Việc loại 557 câu là vấn đề thật, nhưng chưa phải lời giải thích đầy đủ cho chất lượng thấp.
- Ngoài PER-MDD, bảng CROTTC-IF cũng cho thấy cần kiểm tra insertion kỹ hơn. Không dùng detection F1 làm bằng chứng thay thế exact-gap insertion.

## 1. Bảng kiểm các phát biểu

Các verdict đánh giá phát biểu ở cột hai, không đánh giá toàn bộ công trình. “Unverifiable” nghĩa là bằng chứng kiểm tra chưa đủ, không có nghĩa phát biểu sai.

| ID | Phát biểu cần kiểm | Verdict | Bằng chứng / hiệu chỉnh |
|---|---|---|---|
| C01 | Pipeline hiện tại loại 557/1.800 câu khỏi supervision chuỗi đầy đủ | Accurate | Tính lại JSONL và đọc điều kiện lọc trong code; 1.243 eligible, 557 ineligible. |
| C02 | Vẫn đánh giá MDD trên các câu đó | Accurate | Replay đủ 1.800 câu, coverage target/gap 100%, không thiếu prediction. |
| C03 | Con số 557 áp dụng pipeline hiện tại, không suy rộng mọi experiment | Accurate | Eligibility là policy của realized pipeline; lịch sử có recipe dùng supervision khác. |
| C04 | Recall S/O/I ≥80%, FWR ≤0,20 cần pooled typed precision khoảng 95,6% | Accurate | Tính lại support và công thức ở mục 2. |
| C05 | Điều kiện này khả thi về toán học | Accurate | TP tối thiểu từng lớp không vượt support; FP=0 là nghiệm lý thuyết. Không suy ra khả thi về học máy. |
| C06 | CROTTC dùng OT alignment, consistency và teacher transfer | Accurate | [Paper, §§3–4](https://arxiv.org/html/2604.22133v1); IF là Indirect Fusion. |
| C07 | CROTTC-IF F1 71,77% | Accurate | [Table 3, trang PDF 6](https://arxiv.org/pdf/2604.22133); detection F1. |
| C08 | CROTTC-IF hiện chỉ là preprint 04/2026 | Outdated | Có [Interspeech 2026 Long Track, DOI 10.21437/Interspeech.2026-711](https://www.isca-archive.org/interspeech_2026/geng26_interspeech.html). Bản arXiv được kiểm vẫn là v1. |
| C09 | Các lượt vừa chạy chưa reproduce đầy đủ CROTTC-IF | Mostly accurate | Đúng với M1/MB02 đã truy dấu; chưa chứng minh phủ mọi experiment trong toàn bộ lịch sử. |
| C10 | PER-MDD dùng HuBERT/pool phone, không train thêm model MDD | Accurate | [§§2–3](https://arxiv.org/html/2511.20107v1); vẫn cần nhãn và alignment TRAIN để dựng pool. |
| C11 | PER-MDD là ICASSP 2026 | Accurate | [Trang tác giả](https://vnpt-ai-official.github.io/ICASSP2026-MDD/). arXiv ban đầu là 25-11-2025. |
| C12 | PER-MDD F1 69,60%, PER 104,08%; chưa chứng minh insertion tốt | Accurate | [Table 1 và §3.4.2](https://arxiv.org/html/2511.20107v1). |
| C13 | PER-MDD phù hợp làm đối chứng “nhanh” | Unverifiable | Chưa đo thời gian/memory trên máy này; repo được trang tác giả liên kết chưa có implementation chạy được trong tree kiểm tra. |
| C14 | X-MDD dùng Qwen3-ASR-1.7B, canonical speech/phones, SFT và GRPO với F1/PER | Accurate | [§§3–4](https://arxiv.org/html/2610.23079v1). |
| C15 | X-MDD English F1 58,78%, recipe hai L40S | Accurate | [Table 1 và §4.2](https://arxiv.org/html/2610.23079v1). |
| C16 | X-MDD học “gần mục tiêu đánh giá hơn CTC thuần” | Mostly accurate | Có reward detection/PER; chưa tối ưu trực tiếp typed SOI, exact-gap và FWR của LANGFENS. |
| C17 | Không nên reproduce nguyên recipe X-MDD trên GTX 1650 4 GB | Mostly accurate | Là đánh giá kỹ thuật hợp lý từ recipe, chưa phải kết quả profile/OOM trên máy này. Không kết luận mọi bản thu gọn đều bất khả thi. |
| C18 | Ba hướng này đều chưa từng được thử trong repo | Unverifiable | CROTTC có adaptation; không tìm thấy run PER-MDD/X-MDD xác thực trong phạm vi tìm kiếm. Không đồng nhất “không tìm thấy” với “chưa từng chạy”. |
| C19 | IF trong CROTTC-IF là “Iterative Fusion” | False | `literature/SOTA_SURVEY.md:163` dùng tên sai; paper và repo tác giả dùng **Indirect Fusion**. Ghi hiệu chỉnh tại đây; chưa sửa survey cũ. |

Tổng 19 phát biểu: **12 Accurate, 3 Mostly accurate, 1 Outdated, 1 False, 2 Unverifiable**.

Các nhận định “đáng xem xét”, “khắt khe”, “không đề xuất reproduce nguyên bản” là đánh giá nghiên cứu, không phải cam kết hiệu năng. Không có bằng chứng về nguyên nhân nhân quả hay bảo đảm đạt 80%.

## 2. Kiểm toán lại supervision và điều kiện 95,6%

Nguồn trực tiếp: `m0/train_supervision.jsonl` và `data/derived/l2arctic/manifests/train.jsonl`. Đã kiểm tra tập ID bằng nhau; không đọc manifest test/dev lịch sử.

| Nhóm câu | Số câu | S | O | Gap có I |
|---|---:|---:|---:|---:|
| Eligible chuỗi | 1.243 | 4.480 | 1.352 | 361 |
| Ineligible chuỗi | 557 | 2.901 | 473 | 199 |
| Toàn bộ | 1.800 | 7.381 | 1.825 | 560 |

“560 insertion” ở đây là **560 gap dương tính**, không mặc định 560 phone chèn. Evaluator chấm một quyết định insertion trên mỗi gap; nhiều phone chèn tại một gap vẫn là một gap dương tính.

557 câu chiếm 30,94% số câu nhưng mang 39,30% S, 25,92% O và 35,54% gap I. Có 462 câu mang deviation marker và 126 câu có `ERR`; các issue giao nhau nên không cộng chúng thành tổng câu. Eligibility của toàn câu bị tắt khi có issue; không chỉ mask token chưa biết. Đây là mất nguồn supervision khả dụng, không chứng minh các nhãn event sai.

Đọc `contract.py`, `train.py`, `m2_local.py`, `remaining_train.py` và `remaining_decoder.py` xác nhận việc lọc theo eligibility. M1 reference có declared deviation về strict local sequence population; không lấy bộ đếm này áp vào mọi recipe cũ, hoặc nói 557 câu hoàn toàn không xuất hiện ở bất kỳ supervision nào khác.

| Fold | FIT tổng / eligible | CAL tổng / eligible | HELD tổng / eligible |
|---|---:|---:|---:|
| 0 | 900 / 628 | 450 / 386 | 450 / 229 |
| 1 | 900 / 473 | 450 / 384 | 450 / 386 |
| 2 | 900 / 615 | 450 / 244 | 450 / 384 |
| 3 | 900 / 770 | 450 / 229 | 450 / 244 |

Mỗi fit thực tế chỉ lấy phần eligible trong 900 câu FIT của fold; không train trên cả 1.243 câu rồi đánh giá lại chính chúng. Đã kiểm tra speaker FIT/CAL/HELD rời nhau trong cả frozen folds và receipt M6. Các held này vẫn là development TRAIN-speaker cross-validation, không phải xác nhận độc lập bằng speaker mới.

Cần tối thiểu:

`TP_S = ceil(7381×0.8)=5905; TP_O=1460; TP_I=448; ΣTP=7813`.

EVAL-V3.1 định nghĩa `FWR = (FP_S + FP_O + FP_I)/N_câu`, bao gồm cảnh báo sai loại S↔O. Với 1.800 câu, `ΣFP ≤360`, nên:

`P_micro_typed = ΣTP/(ΣTP+ΣFP) ≥7813/(7813+360)=95.59525266%`.

Đây là ràng buộc pooled của đúng quần thể và cách đếm này. Không yêu cầu precision riêng từng lớp đều 95,6%; không tương đương binary detection precision, FRR trên phone đúng, hay tỷ lệ câu có ít nhất một FP. Nếu bỏ FWR khỏi acceptance hoặc đổi quần thể/support, phải tính lại. `protocol.json` hiện ghi `product_acceptance: NOT_AGREED`; chưa được diễn giải điều kiện giả định thành acceptance đã được thông qua.

## 3. Replay M6 và kiểm tra giả thuyết “do 557 câu”

Đã kiểm SHA-256 của 6 nguồn trong evidence cũ: tất cả còn khớp. Kiểm thêm 48 artifact B0.2 trong 12 receipt seed/fold: prediction target/gap, policy và metric đều khớp hash. Chạy lại **UnifiedEvaluatorV3** trên prediction đã lưu, ghép gold TRAIN theo ID và vị trí. Summary và toàn bộ classes khớp chính xác pooled JSON và SUMMARY ở cả ba seed, bao gồm AP. Đây là replay chấm điểm, không tái tạo forward model hay train lại.

| Seed | Macro F1 SOI (%) | HELD FWR | FP | Macro F1 eligible (%) | Macro F1 ineligible (%) |
|---|---:|---:|---:|---:|---:|
| 101 | 10,05 | 0,22333 | 402 | 10,90 | 8,66 |
| 202 | 8,84 | 0,20278 | 365 | 9,30 | 8,42 |
| 303 | 10,44 | 0,22056 | 397 | 11,29 | 9,13 |

Các subgroup dùng nguyên dự đoán và policy đã freeze; không hiệu chỉnh threshold lại trên subgroup. FP budget 360 bị vượt ở cả ba seed. Với seed 101, recall S/O/I là 4,66% / 9,86% / 2,68%, cho thấy khác biệt rất lớn với yêu cầu mỗi lớp 80%.

Nhóm ineligible có FWR cao hơn ở cả ba seed; nhưng nhóm eligible cũng kém. **Kết luận suy luận:** sửa supervision là giả thuyết hợp lý cần thử, chưa phải nguyên nhân đã được xác lập, và không đủ cơ sở hứa sửa xong sẽ đạt 80%. Phân tầng này không kiểm soát khác biệt speaker, độ khó hay kiểu lỗi; không phải thí nghiệm nhân quả.

## 4. Ba paper: bằng chứng, giới hạn và độ phù hợp

### CROTTC-IF

**Trust: Promising** đối với áp dụng SOI ở đây. CR + OTTC và Indirect Fusion; “prompt-free” nói về recognizer, diagnosis vẫn cần reference. [Paper §§3–4](https://arxiv.org/html/2604.22133v1).

Table 3: detection P/R/F1 = 76,94/67,24/71,77; PER/COR = 46,52/92,42. F1 tính TR gồm cả chẩn đoán sai. Từ công thức §6.3, **suy ra** `I/N≈46,52−(100−92,42)=38,94%`, với N là perceived phones. Không chuyển tỷ lệ này thành exact-gap FP hoặc FWR. Đã đối chiếu trực quan bảng PDF trang 6. Đây là lý do cần audit insertion, không phải chứng minh phương pháp không dùng được. [Table 3 và công thức](https://arxiv.org/pdf/2604.22133).

Recipe: WavLM Large, Conformer, AM 300 epoch, LM/teacher 200 epoch, GH200. Có [source và bundle công khai](https://github.com/Secondtonumb/IF-MDD), nhưng source license không tự bao phủ weights/data. Dùng bundle L2 chưa rõ speaker exclusion có nguy cơ trùng population của local CV; cần provenance trước khi dùng làm bằng chứng generalization.

### PER-MDD

**Trust: Promising** cho canonical-phone detection; bằng chứng yếu cho exact-gap insertion của mình. Mặc định: HuBERT Large ASR, pool 500 câu TRAIN, mid-frame, k=10, threshold 0,7. “Training-free” vẫn cần phone labels/timing để xây pool; không cần gold timing cho query inference. [§§2–3](https://arxiv.org/html/2511.20107v1).

Table 1 báo F1 69,60%, P/R 71,78/67,56, PER 104,08%, COR 90,42. Tác giả tự nêu insertion hạn chế và metric chủ yếu trên canonical phones. Không dùng các số này để nhận định đạt full SOI. [§3.4.2](https://arxiv.org/html/2511.20107v1).

[Repo do trang tác giả liên kết](https://github.com/VNPT-AI-Official/ICASSP2026-MDD) tại tree `8860bb4fe7b147ec79a818c212b7e18c908fc9ed` chứa website, PDF, ảnh; không có mã pipeline runnable trong tree đã kiểm. Vì thế “đối chứng nhanh” là ý tưởng, chưa có runtime hay recipe local đã xác minh. Không kết luận tác giả không có code ở nơi khác.

### X-MDD

**Trust: Promising**; chưa xác minh reproduction độc lập/code release. Qwen3-ASR-1.7B nhận canonical speech do TTS tạo với learner audio làm voice reference; xuất actual-phone sequence rồi alignment. English P/R/F1 = 60,04/57,57/58,78, DER 19,71: F1 detection, không phải typed SOI. [§§3–4, Table 1, Appendix C.3](https://arxiv.org/html/2610.23079v1).

Recipe: hai L40S, SFT 7.000 step, GRPO 300 step, reward detection F1/PER; English dùng demonstration SpeechOcean762. Chưa tối ưu FWR/typed errors hay sửa nhãn realized chưa biết. Đánh giá tài nguyên 4 GB chưa được profile; recipe thu gọn phải ghi là adaptation. [§§3.3–4.2](https://arxiv.org/html/2610.23079v1).

Không thấy báo cáo phân phối nhiều seed/CI cho các headline score được dùng ở đây. Không xem chênh lệch headline giữa ba paper là head-to-head đã kiểm soát data, split, metric và compute. Truy vấn reproduction chưa tìm ra nguồn độc lập đáng tin để nâng trust lên Strong.

## 5. Repo đã thử đến đâu?

Tìm theo tên/ID paper trong source, config, Markdown, registry CSV/JSON; bỏ data, prediction khổng lồ và input snapshot khi tìm text. Không mở test để truy tìm phương pháp. Danh sách registry hiện vẫn là 113 thư mục EXP, 110/14/12 dòng ở ba CSV; các tập chồng lấn, không cộng thành số lần train.

| Dấu vết | Điều xác minh được | Điều không được suy ra |
|---|---|---|
| `m1_completion/REFERENCE_FIDELITY.json` | PhnMonoSSL baseline từ IF-MDD, commit `cca150ec4ff00e01d61a613092d1d178e661dfb2`; local encoder/inventory/split/epoch khác paper; `paper_result_replication=false` | Không phải CROTTC-IF; đổi linear thành MLP không thêm OT/CR/teacher. |
| `m1_reference/upstream/` | Có source/config CROTTC-IF được lưu | Có file upstream không chứng minh đã chạy. |
| `rq1_method_bench_02/agent_b/ottc_results.json` và `.py` | Có receipt/static-plan 17-frame window adaptation; khai báo thiếu learned marginals/full sequence decoder và exact-gap I | Không đủ phủ định cơ chế paper; lượt này không load lại checkpoint hay replay benchmark cũ. |
| `literature/SOTA_SURVEY.md` | Có reference PER-MDD và CROTTC-IF; có lỗi tên IF | Reference table không phải run registry. |
| PER-MDD, X-MDD | Không tìm thấy run/config/receipt mang cơ chế đầy đủ trong phạm vi rà soát | Không bảo đảm không có experiment dùng tên khác hoặc artifact bên ngoài repo. |

Điều chỉnh hai cách nói cực đoan: **“đã thử cả ba đầy đủ” chưa có bằng chứng; “cả ba hoàn toàn mới và chưa thử” cũng chưa có bằng chứng.**

## 6. Protocol nào đáng giữ, điều gì cần sửa trong đề xuất?

Giữ official test niêm phong; giữ strict exact-gap evaluator làm chuẩn hồi quy. Không loại 557 câu khỏi evaluation để làm đẹp điểm. Chốt acceptance và ý nghĩa FWR trước bất kỳ so sánh model mới nào.

Audit người độc lập vẫn hữu ích: phân biệt S có phone thay thế rõ với deviation không có identity rời rạc. Tận dụng nhãn event đã biết, mask phần phone chưa rõ; không tự điền gold bằng canonical hay pseudo-label rồi gọi là nhãn thật. Lượt kiểm này chưa nghe/adjudicate nên chưa định lượng được annotation noise hoặc ceiling.

**Hiệu chỉnh khuyến nghị cũ:** tạm dừng sweep kiến trúc là hợp lý về ưu tiên, nhưng chưa có chứng cứ để tuyên bố protocol là nguyên nhân chính hoặc mọi method đã thất bại. Có thể tiếp tục một pilot được đăng ký để phân biệt lỗi implementation, coverage supervision và generalization.

Pilot nên có control cũ và event supervision trên cùng data split, frontend/backbone, budget và evaluator; giữ CAL riêng, freeze threshold, không chọn checkpoint bằng held. Báo riêng subgroup eligibility/deviation và từng lớp S/O/I; mọi kết quả đều tính toàn population held. Oracle alignment chỉ làm chẩn đoán, không đầu vào inference.

Chỉ mở reproduction paper khi đã làm rõ provenance và có budget phù hợp. CROTTC cần đúng OT/CR/IF; PER-MDD cần pool FIT-only và kiểm insertion; X-MDD cần recipe tài nguyên khác nếu chạy máy này, khi đó ghi rõ adaptation. Chưa có cơ sở ưu tiên một combo như lời giải đạt 80%.

## 7. Tái kiểm và giới hạn

Chạy từ repo root:

```sh
.venv/bin/python reports/PROTOCOL_REVIEW_20261113/verify_deep.py
```

- `deep_evidence.json`: hash input, coverage, support, fold counts, metric replay cả ba seed và subgroup.
- `external_sources.json`: URL/version/ngày truy cập, hash snapshot paper và tree PER-MDD; full paper snapshot chỉ ở `/tmp/langfens_verify_20261113`, không đưa vào Git.
- Paper: đối chiếu số được tác giả công bố, không tái lập training. Không có bằng chứng độc lập xác nhận score trên protocol LANGFENS.
- Local: kiểm receipt và replay evaluator, chưa forward lại weights, xác minh lại mọi audio hash hay replay mọi run lịch sử. Hash nhất quán chứng minh artifact không đổi so với receipt, không chứng minh mọi quy trình tạo artifact đều đúng.
- Audit nhãn độc lập và fresh-speaker confirmation vẫn chưa hoàn thành. Không báo PASS sản phẩm từ lượt verify này.
