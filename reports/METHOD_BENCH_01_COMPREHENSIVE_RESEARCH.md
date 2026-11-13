# LANGFENS RQ1-METHOD-BENCH-01 — Comprehensive research synthesis

**Synthesis date:** 2026-11-08  
**Language:** Vietnamese  
**Stage:** R1/R2 research synthesis and selection gate  
**Evidence boundary:** This document synthesizes primary research leads and existing campaign artifacts. It does not equate published task metrics with LANGFENS EVAL-V3.1 performance.

## Tóm tắt quyết định

Danh mục hiện có bao phủ chín họ phương pháp và 22 công trình/nguồn nghiên cứu duy nhất từ năm 2018 đến 2026. Bốn cơ chế có đủ giá trị thông tin và khả năng chuyển giao để ưu tiên là: (1) nhận dạng chuỗi âm vị thực tế với mục tiêu mềm theo độ tương tự ngữ âm; (2) mô hình hóa phát âm chuẩn bằng prototype hoặc anomaly detection; (3) đơn vị SSL rời rạc cùng surprisal và căn chỉnh theo văn bản; (4) GOP giới hạn các âm vị thay thế bằng đồ thị nhầm lẫn có nguồn TRAIN. CTC alignment-free được giữ làm nhánh cấu trúc/đối chứng quan trọng. Các nhánh này không phải biến thể siêu tham số của cùng một mô hình: chúng dùng lần lượt chuỗi âm vị, mật độ của phân bố phát âm chuẩn, surprisal trên đơn vị SSL và prior cấu trúc về thay thế.

Đề xuất nghiên cứu mạnh nhất không phải là một kết quả thắng cuộc đã được chứng minh. Đây là giả thuyết có thể bác bỏ: tách riêng ước lượng âm thanh “đã phát ra âm vị nào” khỏi âm vị kỳ vọng, sau đó căn chỉnh chuỗi bằng phép edit để phân loại S/O và gán các âm vị dư vào chính xác một trong N+1 khe cho I. Giả thuyết này có cơ sở vì các mô hình GOP theo đoạn thường dựa vào alignment; các nghiên cứu CTC cho thấy có thể biểu diễn thêm deletion/insertion không cần alignment chuẩn; và nhận dạng chuỗi cho phép so sánh kỳ vọng với thực tế. Tuy nhiên, map âm vị dư vào khe chính xác của LANGFENS chưa được xác nhận trong các điểm số được công bố và phải được kiểm tra như một thành phần nghiên cứu riêng.

**Cổng nghiên cứu được đóng cho vòng triển khai ban đầu:** ba bộ đầu ra cũ của A/B/C còn trong repo, có nghiên cứu ngoài, thiết kế tái lập và quy tắc đánh giá độc lập; bản tổng hợp này mở rộng chúng thành danh mục có thể hành động. Chọn năm nhánh đa dạng ở trên để đi tiếp sang E1. Phân loại rõ: SylPh screen hiện có là `INSPIRED_IMPLEMENTATION`, không phải tái lập trung thực; kết quả CPU của nó được giữ làm kết quả thăm dò âm/giới hạn, không dùng làm bằng chứng rằng toàn bộ họ anomaly thất bại. Mọi kết quả khác chỉ được báo là đã huấn luyện/đánh giá khi có biên nhận chạy và đánh giá thực tế.

## Phạm vi, cách tìm và giới hạn

Câu hỏi nghiên cứu: những cơ chế từ pronunciation assessment, MDD, ASR, SSL, phát hiện bất thường và structured prediction nào có thể tạo điểm số sự kiện S/O/I ở mức âm vị cho audio người học cùng chuỗi âm vị đích? Phạm vi bao gồm các bài báo gốc, kỷ yếu và preprint có cơ chế hoặc kết quả liên quan trong giai đoạn 2018–2026; bao gồm cả ngôn ngữ/quần thể khác nếu chuyển giao được và ghi rõ giới hạn.

Nguồn tìm gồm ISCA Archive/Interspeech, arXiv, IEEE Xplore records, trang nhà xuất bản và kho mã được liên kết. Các truy vấn theo nhóm: `phoneme-level mispronunciation detection diagnosis CTC alignment-free`; `pronunciation assessment discrete speech tokens surprisal`; `mispronunciation detection phonological features articulatory`; `pronunciation anomaly detection one-class`; `uncertainty multiple pronunciations learner speech`; cùng với truy vấn theo các paper 2025–2026 đã xác định. Log tìm kiếm ban đầu ở `research/method_benchmark_01/external_discovery/DISCOVERY_LOG.md`; lượt bổ sung gồm tìm kiếm ISCA/arXiv về CTC, hybrid attention, OCSVM, uncertainty, discrete units và mô hình đa phương thức ngày 2026-11-08.

22 bản ghi trong cơ sở dữ liệu là bản ghi đã chuẩn hóa theo DOI/arXiv/tên bài; một số bản ghi là đường dẫn proceedings/challenge chứ không phải bằng chứng thực nghiệm độc lập. Có 5 nguồn mới được mở trực tiếp tới toàn văn/abstract chính trong lượt bổ sung: CTC phone-level assessment 2024, hybrid CTC-attention MDD 2020, RNN-T có pitch 2024, ATP-LLM 2024 và anomaly pronunciation verification 2019; cùng các nguồn cho uncertainty modeling 2021, edit-automaton decoder 2020, articulatory MDD 2025 và perceptual-norm assessment 2026. Không thực hiện hai người sàng lọc độc lập trong lượt tổng hợp hiện tại; hồ sơ agent C là kiểm tra benchmark độc lập, không thay thế double-screen paper. Những trường code/checkpoint/license được ghi `unknown` hoặc `not verified` khi chưa kiểm chứng.

Hạn chế chính: đây là scoping review chứ không phải systematic review/meta-analysis; không suy luận effect size chung từ các dataset, ngôn ngữ, định nghĩa lỗi, split hoặc operating point khác nhau. Các số liệu từ abstract chỉ được nhắc để mô tả báo cáo của tác giả, không phải xếp hạng hiệu năng. Trạng thái xuất bản và phiên bản 2026 có thể còn thay đổi.

## Synthesis theo cơ chế

### 1. GOP và alignment-free CTC trả lời hai câu hỏi khác nhau

GOP và các posterior-margin methods đặt câu hỏi liệu khung hoặc đoạn đã được căn chỉnh có âm học phù hợp với âm vị kỳ vọng hay không. Điều này tạo tín hiệu trực tiếp cho substitution nhưng phụ thuộc vào chất lượng alignment và ít tự nhiên với âm vị bị xóa hoặc âm vị phát sinh ngoài chuỗi. Cao và cộng sự xây dựng một framework GOP dựa trên CTC không phụ thuộc căn chỉnh âm vị, đồng thời đưa deletion và insertion vào đánh giá của họ; đây là bằng chứng công bố phù hợp với việc thử một hệ thống chuỗi cho S/O/I, nhưng các định nghĩa và đầu ra của bài không tự động tạo ra điểm số insertion tại từng khe N+1 của LANGFENS. Parikh và cộng sự dùng tập âm vị cạnh tranh bị giới hạn bởi tri thức ngữ âm để điều chỉnh GOP alignment-free; lợi ích tiềm năng là bớt các đối thủ không khả dĩ, nhưng nguy cơ là prior tăng điểm mà không có thêm chứng cứ âm học. Nên benchmark cả GOP không giới hạn và giới hạn trên cùng logits, đồng thời giữ nguyên chuẩn tham chiếu.

### 2. Nhận dạng chuỗi đã phát âm có đường đi rõ cho cả S/O/I nhưng cần kiểm soát lỗi căn chỉnh edit

Các hệ thống hybrid CTC-attention, RNN-T và phone decoder giới hạn edit distance tạo ra giả thuyết chuỗi âm vị thực tế thay vì chỉ một nhãn lỗi cho âm vị kỳ vọng. Một edit alignment có thể cung cấp substitution và omission ở phone target; một phone dư cung cấp ứng viên insertion. Nghiên cứu Cao et al. 2024 và Lo et al. 2020 cho thấy các cấu hình end-to-end có thể giảm nhu cầu forced alignment; Chu et al. 2020 dùng phone decoder chỉ cho phép các chuỗi cách chuỗi canonical một edit để phát hiện và nhận dạng lỗi; Wang et al. 2024 đưa pitch vào RNN-T cho tiếng Quan thoại. Những công trình này đại diện cho các giả định khác nhau, không phải một benchmark đồng nhất: phương pháp của Chu dùng dữ liệu spontaneous lớn và tối ưu mô hình âm học; RNN-T có tone/pitch phụ thuộc ngôn ngữ; và recognition errors có thể bị hiểu nhầm thành lỗi người học. Thử nghiệm LANGFENS cần kiểm tra độ ổn định của edit mapping, đặc biệt với repeated phones, ties và insertion span giữa hai target.

### 3. Soft phone similarity và uncertainty có thể giảm nhầm lẫn “sai khác” với “không thể phân biệt”

Zhou et al. 2025 dùng nhận dạng phoneme verbatim cùng mục tiêu huấn luyện mềm theo độ tương tự âm vị, cho phép lỗi gần nhau về phát âm ít bị phạt như lỗi xa. Cách này có thể cải thiện thứ hạng substitution khi âm học thể hiện một phone gần với target, và duy trì chuỗi output để phát hiện O/I. Điểm yếu là kết quả gốc dùng các accent/synthetic-error conditions và WPER/AER, chưa chứng minh event AP; mô hình mềm cũng có thể làm mờ các lỗi quan trọng nếu độ tương tự không khớp với nhãn LANGFENS. Korzekwa et al. 2021 xử lý bất định phone recognition và nhiều cách phát âm hợp lệ, báo cáo tăng precision tương đối trên các nhóm L1 khác nhau. Đây là lý do để kiểm tra posterior uncertainty và alternate-reference sensitivity, nhưng không phải lý do để chấp nhận mọi biến thể: chuẩn âm vị LANGFENS và nhãn gold phải được giữ nguyên, còn alternate pronunciations chỉ nên là ablation có quy tắc định trước.

### 4. Phonological features tạo tín hiệu chẩn đoán khác với phone identity

Shahin và Ahmed 2024 cho thấy nhận dạng feature ngữ âm học có thể cung cấp thông tin articulatory chi tiết và cải thiện các chỉ số FAR/FRR/DER trong thiết lập của họ. Shahin et al. 2019 dùng các bộ phát hiện manner/place kết hợp one-class SVM theo phone, chỉ huấn luyện từ phát âm đúng, và so sánh với GOP trên các tập native, foreign-accented và trẻ em có speech disorder. Wei et al. 2025 phân tích các mô hình PHN và ART với Conformer/Wav2Vec2; abstract báo cáo đánh đổi giữa phát hiện tổng thể và chẩn đoán articulatory, cùng vấn đề biến thiên theo người nói. Nhóm này có thể hỗ trợ S, giải thích loại khác biệt, hoặc tạo feature cho anomaly detector; O/I vẫn cần occupancy/duration hoặc chuỗi. LANGFENS đã có H-006, nên đây là mở rộng/đối chứng hiện hữu chứ không được tính như ý tưởng mới.

### 5. Normality/anomaly và weak supervision xử lý thiếu mẫu lỗi nhưng đối mặt speaker confounds

OCSVM per-phone trên speech attributes và SylPh contrastive syllable bags đều học ranh giới giữa phát âm trong phân bố với phần ngoài phân bố, thay vì cần đầy đủ nhãn của mọi lỗi. SylPh dùng positive bag, pseudo-negative bag và adaptive text embedding để hỗ trợ open-set syllable assessment; phù hợp nhất như score chuyên biệt S/O tại vùng phone. Một prototype cosine dùng frozen feature đã được chạy cục bộ: AP lỗi nhị phân target-only trên DEV_SEARCH là 0.1564, thấp hơn score tham chiếu H-002 0.3457 trên target population. Đây là một screen của cách pooling/centroid đơn giản, không phải tái lập SylPh và không kiểm định exact-gap I. Kết quả làm giảm ưu tiên cho đúng adaptation này; không loại bỏ toàn bộ anomaly learning vì representation, bag objective và speaker-normalization chưa được thử. Các thí nghiệm kế tiếp phải phân tầng theo speaker và đối chiếu lỗi thật với độ lệch speaker/phone.

### 6. Discrete SSL surprisal là hướng ít nhãn có thể khai thác occupancy và phonotactics

Sara và Chowdhury 2026 chuyển audio sang codebook SSL, học token language model trên speech native và tính surprisal; Text2DUnit-DTW cung cấp mẫu token theo transcript để so khớp. Bài báo báo cáo PCC utterance-level 0.60→0.66 khi thêm transcript guidance và cross-dataset gains trên L2-ARCTIC theo abstract. Điều này gợi ý tín hiệu khác với classifier có nhãn lỗi: độ bất thường cục bộ của chuỗi đơn vị âm học. Nhưng PCC không phải AP sự kiện; token có thể gộp nhiều phone, và khó xác định chính xác gap cho insertion. Do đó chọn làm biến thể mới nếu encoder/codebook/corpus có thể chạy hoặc cache được; đánh giá phone/gap mapping trước khi đầu tư fine-tuning.

### 7. Cross-modal attention, speech-text encoders và LLMs tạo tương tác tham chiếu nhưng còn xa event contract

OpenPA dùng cross-attention của Whisper, mixed transcripts và multi-granular assessment nhưng native output chính ở mức từ/utterance; phone alignment hoặc khe chèn sẽ là adaptation. ATP-LLM prompt một LLM đa phương thức bằng audio và canonical phones để sinh chuỗi phone thực tế rồi alignment; ý tưởng chuỗi tương tự recognition-based nhưng footprint và khả năng tái lập phụ thuộc checkpoint. Gonzales et al. 2026 báo cáo frozen/fine-tuned joint speech-text embeddings trên L2-ARCTIC trong record IEEE, song full method, code/checkpoint và exact label/gap mapping chưa được kiểm tra đủ. Các hướng này có giá trị nghiên cứu vì dùng conditioning và biểu diễn đa phương thức mới, nhưng hiện chưa nên xếp vào lượt GPU đầu tiên cho GTX 1650 4GB. Attention weights cũng không được xem là explanation fidelity nếu chưa có localization audit.

### 8. Confusion graphs và language-specific priors có thể tăng hiệu suất hoặc chỉ mã hóa thiên kiến

Tu et al. 2026 đề xuất đồ thị nhầm lẫn phone có hướng và chiến lược theo L1 cho L2-ARCTIC. Parikh et al. 2025 giới hạn đối thủ GOP theo tri thức ngữ âm. Cả hai đều có khả năng giúp S vì phương án thay thế không đồng xác suất; cả hai cũng có nguy cơ nâng thứ hạng các phone thường gặp dựa trên prior. Bài test quan trọng là đối chiếu score acoustic-only với prior-only, tạo graph chỉ từ TRAIN, kiểm tra theo speaker/L1 nếu metadata hợp lệ, và xác nhận lợi ích S không đánh đổi O/I hoặc tạo false positive ở phone không thuộc confusion set.

### 9. Chuẩn perceptual và norm của người nghe có thể làm thay đổi định nghĩa “lỗi”

Perception Norm 2026 nêu lập luận rằng đánh giá phát âm phụ thuộc vào người nghe, chuẩn xã hội và phân bố chấp nhận, thay vì chỉ độ lệch sản xuất so với một chuẩn native duy nhất. Đây là vấn đề đo lường quan trọng cho tương lai nếu nhãn LANGFENS phản ánh annotator judgment: disagreement có thể là uncertainty của nhãn chứ không chỉ lỗi mô hình. Tuy nhiên không có listener-label collection hoặc multi-rater gold trong benchmark hiện tại để triển khai ngay; giữ hướng này như gap về construct validity, không tự điều chỉnh nhãn.

## Candidate portfolio và kế hoạch kiểm chứng

Chi tiết 12+ method cards, các trường S/O/I, input/supervision, compute, giả thuyết, control và rủi ro nằm ở `METHOD_BENCH_01_EXPERIMENT_PORTFOLIO.md`. Năm candidate được chọn theo khác biệt cơ chế, giá trị thông tin và khả năng có test event-level: similarity-aware sequence recognition; normal-only/prototype; discrete SSL surprisal; constrained confusion GOP; alignment-free CTC. Các candidate không có hỗ trợ tự nhiên cho toàn bộ ba lớp vẫn được đánh giá trên miền hợp lệ hoặc như specialist.

## Research gate và biên nhận đóng cổng

| Điều kiện | Trạng thái | Bằng chứng/giới hạn |
|---|---|---|
| Tìm kiếm nhiều họ và nguồn gần đây | Đạt | 22 source records; 2018–2026; taxonomy 9 họ; external log + lượt bổ sung 2026-11-08 |
| Phân tích sâu candidate đáng kể | Đạt có giới hạn | 9 nhóm cơ chế được tổng hợp theo applicability; paper/code chưa kiểm tra được giữ trạng thái chưa xác minh |
| Kiểm tra nguồn chính và claim quan trọng | Đạt có giới hạn | Mở nguồn ISCA/arXiv/nhà xuất bản cho các phương pháp cốt lõi; database chỉ rõ record cần mở toàn văn/repo thêm |
| Danh mục khác biệt khoa học | Đạt | 5 nhánh chọn thuộc 5 nguồn evidence/mechanism khác nhau; các specialist được giữ riêng |
| Kế hoạch thực thi khả thi | Đạt | CPU screens/frozen scores trước; GPU ưu tiên sau lease an toàn; phương án CPU được phép |
| Xếp hạng so sánh | Đạt | Ma trận transparent triage và selection rationale ở recommendations, portfolio, comparison CSV |
| Kiểm tra độc lập ba vai trò | Đạt theo artifact lịch sử, không có live agent registry | Agent A discovery log, Agent B status/repro artifacts, Agent C protocol review còn trong repo; session hiện tại chỉ liệt kê root, nên không thể gửi follow-up hoặc xác nhận agent còn chạy. Không tạo agent thay thế theo chỉ thị giữ nguyên agent cũ. |

**Quyết định:** đóng R1/R2 cho mục đích khởi động benchmark đa phương pháp. Đây là cổng lựa chọn ban đầu, không phải kết luận rằng danh mục đã vét cạn văn liệu. Tiếp tục tìm kiếm bên ngoài song song với E1 như yêu cầu. Nguồn và code chưa rõ sẽ không bị mô tả như tái lập đã hoàn tất.

## Nguồn cốt lõi

Danh sách đầy đủ và status theo từng record ở `METHOD_BENCH_01_PAPER_DATABASE.csv`. Các primary source tiêu biểu: [Cao et al. 2024 CTC framework](https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.pdf); [Shahin & Ahmed 2024 phonological MDD](https://www.isca-archive.org/interspeech_2024/shahin24_interspeech.pdf); [Zhou et al. 2025](https://doi.org/10.21437/Interspeech.2025-2417); [Jin et al. 2025 SylPh](https://doi.org/10.21437/Interspeech.2025-1285); [Parikh et al. 2025](https://doi.org/10.21437/Interspeech.2025-829); [Korzekwa et al. 2021](https://arxiv.org/abs/2101.06396); [Sara & Chowdhury 2026](https://arxiv.org/abs/2606.19910); [OpenPA 2026](https://arxiv.org/abs/2610.13826); [Tu et al. 2026](https://arxiv.org/abs/2606.05569).

## Feedback awal từ E1 và cập nhật agenda

Hai thí nghiệm CPU cho thấy lý do cần giữ nhiều họ trong danh mục thay vì suy rộng từ một adaptation. H01 similarity-soft target-window scorer không vượt hard-target control về AP S hoặc O trên SEARCH; H03 phone-conditioned unigram surprisal đạt binary target AP 0.14318, dưới H-002 0.34570 và dưới centroid H02 0.15644. Cả hai không có insertion output. Đây là bằng chứng against cho đúng cấu hình đã chạy, không phải bằng chứng phủ định nhận dạng chuỗi soft-target hoặc discrete token surprisal theo paper: H01 không có free-running decoding; H03 không có native speech token LM/Text2DUnit-DTW.

Vì vậy thứ tự tiếp theo đổi theo giá trị thông tin: kiểm tra cached CTC/GOP artifact để chạy H04/H05 bằng score reuse; chỉ sau đó mới mở thêm training lớn. Nếu artifact không có competitor posteriors hoặc timing span, ghi rõ giới hạn và không tạo pseudo-gap score. Báo cáo đầy đủ số liệu và biên nhận ở METHOD_BENCH_01_FINAL_SCIENTIFIC_REPORT.md, METHOD_BENCH_01_EXPERIMENT_RESULTS.csv và METHOD_BENCH_01_REPRODUCTION.md.
