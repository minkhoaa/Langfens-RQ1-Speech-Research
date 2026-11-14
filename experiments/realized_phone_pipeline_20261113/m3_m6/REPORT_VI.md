# M3–M6 — Triển khai, kết quả và giới hạn nghiệm thu

Đã chạy 52 run training mới ×16 epoch, ngoài các smoke và decoder replay. Tất cả các nhánh còn lại đã có implementation/artifact và được thực thi trong phạm vi dữ liệu hiện có. Goal tổng thể chưa hoàn thành; không promote hoặc coi prototype là sản phẩm.

## Trạng thái thực tế

| Mốc | Đã làm | Điều kiện chưa đạt |
|---|---|---|
| M3 | Hard/soft-edit/FIT-informed CTC-LR, full S/O/I/exact-gap, coverage/convergence, policy replay | Finite graph, không full event posterior hoặc mọi multi-edit path; chưa xác nhận mới |
| M4 | Phone-only / attribute-only / joint, 4fold×2seed, realized attribute CTC | Chưa chứng minh gain SOI hữu ích, nhất quán |
| M5 | 600 audio nguồn thật, silver FIT-only, matched compute và sửa control speaker exposure; human-review batch100 | 0 speaker mới, 0 nhãn human mới; chưa mở rộng gold đa speaker |
| M6 | Freeze trước seed303, 3seed replication, offline inference bundle và confirmation runner | Không có fresh-speaker human-gold confirmation; product acceptance chưa chốt |

## M3 — Gain thực, trên cùng acoustic posterior

Các arm dùng cùng normalized fp16-exported posterior, cùng 6fit/3cal/3held TRAIN CV và đủ1.800 held câu/seed. Threshold CAL B=.20; FWR dưới đây là held thực tế. Chỉ profile ba alias reference có thực trong tập được chuẩn hóa tường minh; một SPN được giữ bằng neutral-scored research row và ghi unsupported riêng. Không remap gold hoặc bỏ denominator. Inference package từ chối SPN, không đoán phone.

| Decoder | Seed | PER | Macro-3 AP | Macro-3 F1 | FWR |
|---|---:|---:|---:|---:|---:|
| hard | 101 | 26.03% | 0.13407 | 0.01929 | 0.18944 |
| hard | 202 | 26.31% | 0.12906 | 0.02232 | 0.19611 |
| soft | 101 | 26.03% | 0.17147 | 0.02623 | 0.19667 |
| soft | 202 | 26.31% | 0.16546 | 0.02594 | 0.19056 |
| lr | 101 | 26.03% | 0.27238 | 0.07841 | 0.19333 |
| lr | 202 | 26.31% | 0.25858 | 0.07201 | 0.21333 |

Nhánh AP screen so hard: True; chi tiết delta từng lớp trong completion.json. Đây là gain của decoder/evidence scoring, không phải recognizer mới đột nhiên tăng chất lượng. FWR cal0,20 không đảm bảo held0,20.

SM13 full-fit historical (cùng speaker protocol/full denominator): AP0,24687/0,25071. M3 LR cao hơn về điểm số, nhưng khác capacity, supervised coverage và training budget; không có matched CharSiU-vs-new-aligner causal comparison hoặc fresh-speaker superiority claim. Chưa kết luận vượt toàn bộ các baseline mạnh nhất hoặc đạt sản phẩm.

### Precision/recall tại operating point

| Seed | Lớp | Support | TP | FP | FN | Precision | Recall | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | SUBSTITUTION | 7381 | 424 | 223 | 6957 | 65.53% | 5.74% | 0.38744 |
| 101 | OMISSION | 1825 | 94 | 63 | 1731 | 59.87% | 5.15% | 0.36196 |
| 101 | INSERTION | 560 | 11 | 62 | 549 | 15.07% | 1.96% | 0.06775 |
| 202 | SUBSTITUTION | 7381 | 368 | 243 | 7013 | 60.23% | 4.99% | 0.36937 |
| 202 | OMISSION | 1825 | 73 | 77 | 1752 | 48.67% | 4.00% | 0.34149 |
| 202 | INSERTION | 560 | 16 | 64 | 544 | 20.00% | 2.86% | 0.06488 |

ORACLE_OPERATING_POINTS.json là phép đo dùng HELD gold để chọn threshold chẩn đoán tại matched FWR=.20 và precision=.75/.80; tuyệt đối không đưa những threshold này vào handoff. Không thay oracle bằng policy triển khai hoặc claim precision/recall cùng tăng chỉ vì AP tăng.

### Độ phủ và uncertainty

- Seed101: candidate chứa S thật 3903/4480, phone I thật 258/405 trên strict subset. Median mass của các sequence DISTINCT đã xét (CAL+HELD) 3.4587e-22.
- Seed202: candidate chứa S thật 3903/4480, phone I thật 258/405 trên strict subset. Median mass của các sequence DISTINCT đã xét (CAL+HELD) 3.21831e-21.

Soft-edit tích phân tất cả alignment Levenshtein tối ưu cho đúng một greedy hypothesis. LR chỉ xét single-edit competitors FIT-derived (4S/8I); nhiều lỗi đồng thời, S-vs-O+I và multi-phone insertion chưa có full posterior. Candidate mass không phải xác suất sự kiện. Beam1/4/8 trên4CAL probe, top12 token expansions/frame, được ghi BEAM_CONVERGENCE.json; không gọi truncated coverage là hội tụ lattice đầy đủ. DIAGNOSTICS giữ phone families, edges/multiple-gap, lệch gap và nhóm recall0; AP trên nhóm chỉ có positives được ghi hạn chế.

## M4 — Cấu âm: đã thử, chưa có lợi ích được xác nhận

Bảng ARPABET-ATTR-CAT-v1 gồm40phones/9 categorical groups, không collision trong inventory. Nominal English engineering mapping dựa trên [CMU phone classes](https://github.com/cmusphinx/cmudict/blob/master/cmudict.phones) và [khái niệm IPA chart](https://www.internationalphoneticassociation.org/content/chart); các lựa chọn onset/glide/AX được ghi rõ. Đây không phải phép đo cấu âm của người học. Nhãn lấy từ realized gold, không reference; unknown/deviation không được tự biến thành gold.

Cùng architecture/features/16epochs/batch16/AdamW lr.001 wd.001/cosine/clip5. Joint loss=phoneCTC+.2×mean9attributeCTCs; attribute-only decode bằng factorized compatibility, không có học reference shortcut. Shared heads instantiated giống nhau nhưng inactive gradients/compute khác ở từng arm. Khác thử H006/MB07 cũ ở sequence-realized supervision và categorical objectives. Ngân sách này không đại diện mọi cách dùng attribute.

| Arm | Seed | PER | Macro-3 AP (hard decoder) | FWR |
|---|---:|---:|---:|---:|
| phone | 101 | 25.87% | 0.13185 | 0.20222 |
| phone | 202 | 26.23% | 0.12977 | 0.19500 |
| attribute | 101 | 52.07% | 0.08500 | 0.21000 |
| attribute | 202 | 137.84% | 0.07268 | 0.20167 |
| joint | 101 | 25.95% | 0.13000 | 0.17944 |
| joint | 202 | 26.21% | 0.13068 | 0.18722 |

Joint vs phone AP screen: False. Attribute-only decoder kém/không ổn định; các head CTC có alignment/blank riêng và factorization có giới hạn. Không suy ra mọi attribute model đều vô ích. Attribute sequence error từng nhóm được lưu, không dùng auxiliary accuracy làm bằng chứng MDD. Không chạy full LR mọi M4 arm trong budget; không gán LR gain ở M6 cho attribute causality.

## M5 — Dữ liệu và đối chứng

600 WAV thật từ archive TRAIN12speaker hiện có, nguyên archive/Drive không đổi; liên kết source revision/license/hash, continuous H6 frozen mới, resampling+PCM16 khớp2TRAIN probes. Không lấy CharSiU windows cũ làm continuous frames. Mỗi fold chỉ300extras của6FIT speaker; teacher cùng FIT split, không cal/held speaker. Có2.400 phiên bản silver qua fold/seed, chỉ600audio unique. Confidence=.6 là mean emission score chưa calibrate; cả300/fold đều qua nên không xem filter như chứng nhận gold. CAL teacher PER là checkpoint-selected diagnostic, không human validation trên extras.

Control repair executed: True. Control cũ lặp cyclicfirst300FIT IDs làm lệch speaker exposure (ví dụfold0không đủ6×50); giữ nguyên artifacts. Repair M5_CONTROL_REPAIR.md đăng ký trước khi xem M5 held metrics, đổi đúng chọn source real cùng speaker mỗi extra row, giữ training order/update/epochs/model. Chỉ bỏ thêm lượt diagnostic per-attribute sau train để giữ budget; toàn bộ phone/MDD/policy/posterior vẫn đủ.

| Arm | Seed | PER | Macro-3 AP (hard decoder) | FWR |
|---|---:|---:|---:|---:|
| repeated_real | 101 | 25.87% | 0.13192 | 0.18389 |
| repeated_real | 202 | 26.26% | 0.12841 | 0.20833 |
| pseudo | 101 | 25.76% | 0.13580 | 0.19944 |
| pseudo | 202 | 26.08% | 0.13113 | 0.19833 |
| balanced_real | 101 | 26.28% | 0.13140 | 0.19222 |
| balanced_real | 202 | 26.10% | 0.13165 | 0.20333 |

Pseudo vs balanced_real AP screen: False. Không dùng control cũ riêng để claim causal data gain. Không thêm human gold hoặc speaker mới; SpeechOcean score không tự thành realized/S/O/exact-gap labels. Các dev manifest lịch sử tồn tại nhưng không được tự tái phân loại thành fresh confirmation. M5 chất lượng/đa dạng supervision vẫn mở.

Đã chuẩn bị human_review/100WAV gốc có SHA:50disagreement/lowconfidence+50randomcontrol, forms rỗng và guideline hai annotator/adjudication, không hiển thị teacher hypotheses để giảm anchoring. Đây là batch chờ human review, 0nhãn được tạo, không phải IAA study hoặc speaker mới. PROVENANCE_AUDIT.json có prompt-relative C/S/O/I distributions của silver (không human error rates), teacher exclusion và600feature hash.

## M6 — Freeze và handoff

Primary joint16epoch+FIT-LR+CAL B=.20 đã đăng ký trước seed303; cố định fold0/seed303 cho prototype, không chọn seed tốt nhất hoặc đổi candidate theo M5 held results. Ba seed dùng cùng development speakers, không phải fresh-speaker confirmation.

| Candidate decoder | Seed | PER | Macro-3 AP | Macro-3 F1 | FWR |
|---|---:|---:|---:|---:|---:|
| joint + FIT-LR | 101 | 25.95% | 0.27589 | 0.10053 | 0.22333 |
| joint + FIT-LR | 202 | 26.21% | 0.26907 | 0.08843 | 0.20278 |
| joint + FIT-LR | 303 | 25.85% | 0.26789 | 0.10439 | 0.22056 |

handoff/ là bundle độc lập với backbone/head offline, mapping, model/decoder source, CAL policy, method freeze, requirements, checksums, predict.py và confirm.py + EVAL-V3.1. Không kèm dataset, credentials hoặc oracle thresholds. Head/head-backbone SHA được kiểm; waveform→38target/39gap; perturb reference giữ acoustic decode; missing/unknown/gold fields bị từ chối. Multiple inserted candidates giữ nguyên qua greedy edit path, không đồng nhất chúng với LR posterior. Resampling và fp16 profile cố định; CPU/GPU backend có thể có drift ở câu khác.
Fresh confirmation runner từ chối reused corpus speaker/missing human gold, yêu cầu provenance/attestation; một JSON flag không chứng minh human identity/provenance. Không có tập fresh để chạy. Product acceptance chưa được người dùng chốt; không promote, không claim acoustic diagnosis readiness, arbitrary text parity hoặc M6 acceptance hoàn thành.

## Kiểm chứng, chi phí và tái lập

- 8 tests mới pass; independent float64 C++ vs CUDA fp32 trên actual probe (CTC_NUMERIC_PARITY.json); 528 frozen held-policy application checks + CAL provenance/false-budget audit; EVAL-V3 counts/classes, exactly-once1.800held/seed và independent PER. Đây không phải recompute CAL selection của mọi LR run; hạn chế đó được ghi rõ.
- 2721 historical hash khớp, old GOAL resolve bằng snapshot. Failed M3 reference-profile/SPN attempts và queued-control restart đều giữ log; không đổi checkpoint/manifest lịch sử. Original sealed test/dev_rep audio/labels không được stage/extract; split-manifest speaker metadata chỉ dùng denylist/audit.
- 52 full train run mới ×16epoch, không nhầm decoder replay thành training. Atomic last/best lưu model/optimizer/scheduler/scaler-disabled/RNG/source/fit metadata. State đủ để xây resume; CLI16epoch hiện từ chối overwrite, chưa thử kill/restart campaign này. Không gọi điều đó là operational resume đã được xác nhận.
- Recorded/conservative GPU-related wall: 3488.25s trong budget3600s nếu balanced control ghi within_cap. Stage wall gồm cal/export và chuẩn bị; phần CPUanalysis trước không profile đầy đủ, không gọi là0. final_analysis_timing.json đo riêng M5/M6 replay. Cache/extraction lịch sử không bị gọi là miễn phí.
- Feature branch giữ nguyên; weights/arrays/WAV/archives được .gitignore, chưa commit/push/merge. Không Colab/Kaggle, chỉ local GPU. Artifact manifests seal nguồn mới; live GOAL có snapshot trước/sau.

Lệnh: python -m research.realized_phone_pipeline.remaining_pipeline (chỉ thư mục mới; không overwrite); remaining_m3; remaining_report --stage m3|m4|m5|m6; handoff/predict.py --bundle handoff --request request.json --output prediction.json. Đọc PREREGISTRATION.md/M5_CONTROL_REPAIR.md và handoff/README.md trước khi tái chạy. Tất cả code dùng .venv hiện có.

Phần công việc còn lại để đạt goal: supervision thật đa dạng speaker và các lỗi khó/biến thể; xác nhận trên speaker human-gold mới sau freeze; đạt precision/recall/FWR theo tiêu chí đã thống nhất. Những việc đó chưa được tạo ra bằng việc hoàn tất code hoặc chạy thêm epoch.
