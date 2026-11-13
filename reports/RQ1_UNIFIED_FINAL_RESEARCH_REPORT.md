# LANGFENS RQ1-UNIFIED-01 — Báo cáo nghiên cứu tích hợp

**Câu hỏi:** một mô hình neural audio-conditioned có thể học S/O/I chuyên biệt trong cùng một pipeline, dùng frame representation giàu hơn và không cần các historical expert checkpoints khi dự đoán hay không?

## Kết luận chính

Đã huấn luyện thành công mô hình thật hai miền đầu ra trên CUDA: U2 flat dự đoán C/S/O cho từng intended phone và học một insertion head mới trên đúng N+1 gaps. Inference mới nhận WAV thật và danh sách phoneme dự định; smoke run cho 39 target và 40 gap. Checkpoint neural U2 là một file duy nhất và không nạp EXP24/36/43 hoặc Gold-Only expert checkpoints. Frontend vẫn cần hai snapshot frozen local: CharSiU frame classifier để căn tâm dự đoán và Wav2Vec2 Base 960h để trích hidden state.

U2 **không vượt Gold-Only hoặc H-002 về Macro-3 AP**. Omission AP tăng so với Gold-Only, nhưng insertion head mới làm giảm AP-I đáng kể. Hierarchical U3 không khắc phục được lỗi này và giảm AP S/O. Thay gap positive weight sang 3:1 tạo cải thiện seed101 nhưng không lặp lại trong seed202/303. Không có căn cứ gọi GAPW3 là cải thiện ổn định.

## Kiến trúc và dữ liệu

- **Representation:** WAV được chuẩn hoá theo `AutoFeatureExtractor` của `facebook/wav2vec2-base-960h`; encoder đóng băng; dùng `hidden_states[6]`, crop 17 frame quanh centers do frame classifier + intended phones tạo ra. Đây là checkpoint ASR-finetuned, không phải encoder chỉ pretraining SSL.
- **Target:** H-002 reference-conditioned local attention, intended-phone one-hot 39 chiều và 3-way C/S/O softmax. U2 dùng gold weighted cross-entropy.
- **Gap:** MLP mới dùng hai pooled 768D acoustic edge vectors và hai 39D intended-phone vectors (1614 chiều), xuất softmax No-Insertion/Insertion. Internal gap lấy frame phía sau center của phone trái và frame phía trước center phone phải. Leading/trailing gap giữ đúng slot và dùng zero sentinel cho neighbor vắng. Nhãn gap chỉ vào loss.
- **Tập:** TRAIN 59,562 targets, 61,362 gaps, gồm 560 insertion positives; DEV_CAL 10,043/10,343; DEV_SEARCH 19,755/20,354 trên 599 utterances, bốn speakers. DEV_SEARCH là exploratory/exposed.
- **Tính hợp lệ:** mọi model có 100% target/gap coverage; zero missing IDs, target probability simplex violation bằng 0. PR curve points theo lớp S/O/I cho U2, U3 và GAPW3 seed101 nằm ở `reports/RQ1_UNIFIED_CLASSWISE_PR_CURVES.csv.gz`; AP trong file khớp EVAL-V3.1. `Macro3_Error_AP=(AP_S+AP_O+AP_I)/3`.

## Kết quả AP và F1

AP dưới đây là DEV_SEARCH và không phụ thuộc threshold. F1 mới của U2/U3/GAPW3 dùng policy riêng fit trên DEV_CAL; hàng unrestricted chỉ để chẩn đoán warning trade-off. Gold-Only/H-002/EXP historical policy có khác biệt và được gắn nguồn riêng.

| Mô hình | Seed | AP-S | AP-O | AP-I | Macro-3 AP | Macro-3 F1, CAL 20% | Macro-3 F1, unrestricted | Macro-4 F1, unrestricted | Binary MD AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gold-Only GRU | 101 | .36460 | .15943 | .18672 | .23692 | .10490 | .31021 | .46194 | .34570 |
| H-002 target + Gold-Only gap | 101 | .37781 | .16729 | .18672 | .24394 | .12529 | .30955 | .45923 | .38633 |
| **U2 flat, learned two heads** | 101 | .37136 | .18532 | .11799 | .22489 | .12542 | .28035 | .43674 | .36332 |
| U3 hierarchical, learned two heads | 101 | .36241 | .17612 | .12382 | .22078 | .07234 | .26545 | .42697 | .33899 |
| U2 gap-weight-3 | 101 | .37136 | .18532 | .12735 | .22801 | .11834 | .29556 | .44815 | .36332 |
| U2 flat arithmetic mean | 101/202/303 | .36833 | .17636 | .13900 | .22789 | .11398 | .27782 | .43489 | .36932 |
| U2 gap-weight-3 mean | 101/202/303 | .36833 | .17636 | .13902 | .22790 | .12008 | .29432 | .44726 | .36932 |

Gold-Only and H-002 seed101 scores reuse Cycle 2 receipts. Their Macro-3 AP values are historical controls. GAPW3 mean F1 does not mean stable threshold benefit: CAL20 per-seed deltas vary, and AP paired deltas reverse sign.

At U2 CAL 20% policy, class P/R/F1 are S `0.4783/0.0124/0.0243`, O `0.3299/0.1531/0.2092`, I `0.3793/0.0880/0.1429`; Macro-3 error F1 `0.1254`, Macro-4 mixed-domain F1 `0.3307`, binary MD F1 `0.0575`, joint FWR `0.1786`. This policy misses most substitution positives. Unrestricted thresholds lift F1 but raise FWR to 4.654 warnings per utterance, so that row is not a recommended operating point.

### So sánh với Phase 4 và specialist references

| Reference | AP-S | AP-O | AP-I | Macro-3 AP | Macro-3 F1 |
|---|---:|---:|---:|---:|---:|
| EXP24 | .357737 | **.225092** | .247628 | .276819 | .305855 |
| EXP29 | .378062 | .196539 | .272068 | .282223 | **.342914** |
| EXP36 | **.387088** | .204687 | .275924 | **.289233** | .331090 |
| EXP43 | .363453 | .205771 | **.285056** | .284760 | .332231 |
| Phase 4 omission-specialist score fusion, 5-seed mean | — | — | — | **.3018** | .3178 |
| Phase 4 fixed head-wise score blend, 5-seed mean | — | — | — | .2955 | .3296 |

Đây là score compositions (ngoại trừ replay EXP24 standalone), không phải một neural checkpoint thống nhất. U2 seed101 thấp hơn EXP24 Macro-3 AP 0.05193 và EXP36 0.06434; thấp hơn EXP43 AP-I 0.16706. Macro-3 F1 unrestricted U2 `.28035` cũng dưới EXP29 `.342914`, nhưng policy không đồng nhất tuyệt đối. Phase 4 có kết quả specialist mà một model đơn hiện chưa học lại được.

## Ablation và độ ổn định

**Hierarchical U3:** chỉ AP-I tăng nhẹ so với U2 `+.00583`; AP-S `−.00895`, AP-O `−.00920`, Macro-3 AP `−.00411`. Ở CAL20, Macro-3 F1 giảm `.12542→.07234`. Omission BCE dùng mọi target; conditional S/C loss chỉ dùng gold non-omission rows. Omission examples có zero conditional loss, nhưng vẫn góp omission loss. Cấu trúc xác suất `pO=sigmoid(zO)`, `pS=(1-pO)sigmoid(zS|nonO)`, `pC=(1-pO)(1-sigmoid(zS|nonO))` tạo simplex. Kết quả không ủng hộ hierarchical specialization trong cấu hình này.

**GAPW3:** giữ target branch y hệt, đổi duy nhất tỷ lệ insertion/non-insertion loss sang 3. Paired AP-I/Macro-3 AP thay đổi theo seed:

| Seed | Δ AP-I GAPW3−U2 | Δ Macro-3 AP |
|---:|---:|---:|
| 101 | +.00936 | +.00312 |
| 202 | +.03019 | +.01006 |
| 303 | −.03948 | −.01316 |
| Mean 3 seeds | +.00002 | +.00001 |

Vì dấu đảo ở seed303, không có hiệu ứng ranking lặp lại. U2 3-seed mean AP-I `.13900`, GAPW3 `.13902`; Macro-3 AP `.22789` và `.22790`. Chênh lệch gần zero. Không chạy thêm weight sweeps.

### Phân tích lỗi

U2 omission AP theo speaker SEARCH không đồng đều: RRBI `.3117` (65 positives), YDCK `.2236` (58), YBAA `.1108` (31), NCC `.1096` (55); bốn speaker chỉ là lát cắt mô tả. Ở unrestricted threshold, 96/209 omission bị bỏ sót. Phone slices yếu gồm DH `.0083` (2 positive), ER `.0106` (2), HH `.0189` (3), TH `.0227` (3), IH `.0243` (8); support nhỏ nên không ổn định.

Insertion là thất bại chính: U2 I AP `.11799`, GAPW3 mean `.13902`, so với Gold-Only/H-002 `.18672` và EXP43 `.285056`. Tín hiệu target/representation U2 có ích nhẹ cho O, nhưng gap representation/rare-positive learning chưa chuyển thành ranking đủ mạnh. U2 AP-S `.37136` cao hơn Gold-Only `.36460` một chút, nhưng thấp hơn H-002 `.37781`; U2 AP-O `.18532` vượt Gold-Only `.15943` và H-002 `.16729`, song còn dưới EXP24 `.22509`.

## Chuyển giao teacher knowledge

Không chạy K1–K4. Không tìm thấy speaker-held-out TRAIN OOF outputs cho EXP24/36/43. EXP36/43 replay là score composition trên DEV_CAL/DEV_SEARCH, không có neural TRAIN forward; EXP24 checkpoint đã fit TRAIN nên tự dự đoán TRAIN sẽ in-sample. Agent B thêm `distillation.py` chỉ chấp nhận hashed TRAIN_OOF receipts và tạo loss phụ trợ, nhưng chưa có dữ liệu hợp lệ để dùng. Đây là thiếu provenance OOF, không phải kết quả phủ định distillation. H-004 feature-cosine KD không lặp lại.

## WAV inference và checkpoint

U2 checkpoint seed101: `experiments/rq1_unified_01/agent_a/RQ1-U2-FLAT-TWOHEAD-S101.pt`, 653,893 learned parameters, SHA-256 `1978633b08034904d566273cb29f3f8305c604ed86e3be1145e1bc70d6375c92`; train 8 epochs trên GTX 1650 trong 24.88 s, peak allocated VRAM 41.33 MiB. Neural inference head file là một checkpoint thống nhất; frozen pretrained frontends (CharSiU alignment + Wav2Vec2 Base) vẫn cần để chuyển WAV thành acoustic features. Không cần các historical error specialist checkpoints.

Đã chạy WAV `YBAA_arctic_a0003.wav` với phone list phát sinh từ prompt text + CMUdict: 39 target/40 gaps, 1.61 s, peak 438.76 MiB; xác suất target và gap có simplex sai số tối đa `1.2e-7`. Hash WAV `45a0246c2aeee9f9348f5b8690f17dfc7f627ce860a772b74331ef23781f9813`; đầu ra `experiments/rq1_unified_01/agent_a/YBAA_arctic_a0003_RQ1-U2_predictions.json`. N=0 code trả đúng 1 gap, nhưng chưa runtime-verified và whole-utterance pooled gap là OOD so với TRAIN. Reference independence của canonical training references chưa được chứng minh hoàn toàn.

## Trả lời câu hỏi nghiên cứu

1. **True two-head đã train?** Có: U2 học S/O/C target và neural insertion trên exact N+1 gap; không tái sử dụng gap score Gold-Only.
2. **Acoustic representation?** Frozen, normalized Wav2Vec2 Base 960h layer 6, 17 frames, reference-conditioned; checkpoint ASR-finetuned.
3. **Hierarchical specialization?** Không cải thiện tổng thể; AP O giảm so với U2 và CAL20 Macro-3 F1 giảm.
4. **Gap head insertion?** Chưa cải thiện so với Gold-Only/H-002. Loss ratio 3 có một-seed gain nhưng không lặp lại.
5. **Teacher distillation?** Chưa kiểm định do thiếu OOF targets hợp lệ.
6. **Vượt Gold-Only/H-002?** Không về Macro-3 AP; U2 chỉ tăng AP S/O riêng lẻ so với Gold-Only, nhưng AP-I giảm mạnh.
7. **Khoảng cách Phase 4?** U2 Macro-3 AP `.22489` so với EXP24 `.27682`, EXP29 `.28222`, EXP36 `.28923`, EXP43 `.28476`, omission fusion `.3018`; gap AP `.11799` so với EXP43 `.28506`.
8. **AP/F1?** Nêu trong bảng, cùng CAL20 class P/R/F1 và receipt EVAL-V3.1.
9. **Ổn định seed?** U2 ba seed Macro-3 AP mean `.22789` (SD `.00652`); GAPW3 `.22790` (SD `.00564`) nhưng paired gain gần 0 và từng seed đảo dấu. Không có multi-seed U3.
10. **Trở ngại chính?** Exact-gap insertion ranking yếu/không ổn định; omission vẫn khó ở một số speakers/phones; specialist OOF transfer chưa thể chạy.

## Kết luận

Chiến dịch đã đạt yêu cầu mô hình thống nhất có thể train và suy luận từ WAV + intended phone sequence, nhưng **chưa đạt khả năng phân biệt S/O/I bằng Phase 4 score compositions**. U2 là mô hình học đầy đủ hai output domains; U3 không chứng minh ích lợi của hierarchical specialization; GAPW3 không tạo lợi ích lặp lại. Kết luận khoa học chính là phần target acoustic học được có tiến bộ O nhỏ, trong khi gap head/representation và số insertion dương ít là nút thắt. Mọi số DEV_SEARCH đều là khám phá trên split đã exposed, không phải khẳng định generalization độc lập.

## Tệp đính kèm chiến dịch

- Agent A: `reports/RQ1_UNIFIED_A_ACOUSTIC_TRAINING.md`
- Agent B: `reports/RQ1_UNIFIED_B_SPECIALIST_TRANSFER.md`
- Agent C: `reports/RQ1_UNIFIED_C_MODEL_ANALYSIS.md`
- Registry: `experiments/rq1_unified_01/EXPERIMENT_REGISTRY.csv`
- Leaderboard, ablation, reproduction và next work: các tệp RQ1 tương ứng trong `reports/`.
