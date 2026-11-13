# LANGFENS Phase 4 — Báo cáo khoa học tổng hợp (kết quả có giới hạn)

**Trạng thái:** Hoàn thành bốn cuộc kiểm toán song song; chưa hoàn thành toàn bộ chiến dịch thực nghiệm UPED. Đây là báo cáo tích hợp các kết quả đã kiểm chứng và công khai các thí nghiệm chưa chạy. Không truy cập TEST_SEALED, protected cohorts hay dữ liệu ngoài phạm vi.

## Tóm tắt quyết định

- **EXP-24+34:** Macro-3 Error AP **0.292012** là kết quả SEARCH lịch sử mang tính **EXPLORATORY**, không phải cải thiện độc lập đã xác nhận. Chưa đủ artifact để tái dựng chính xác selection/weight recipe/SEARCH feedback của kết quả cũ. Không được thăng hạng thành winner.
- **EXP-36:** cấu hình composition canonical có thể tái tạo chính xác từ các đầu ra thành phần và trọng số đã fit trên DEV_CAL; score target/gap khớp official replay với sai số cực đại 0. Đây xác nhận provenance số học của replay hiện tại, không chứng minh superiority trên test mù hoặc parity với lịch sử. Nó có Macro-3 AP cao nhất trong các cấu hình được nêu (0.289233), nhưng chỉ trên DEV_SEARCH đã exposure.
- **Bằng chứng acoustic causality:** Chưa xác lập. Agent B kiểm toán nguồn và đăng ký ma trận can thiệp nhưng không chạy inference/ablation; không kết luận lexical/prosodic/acoustic contribution từ AP lịch sử.
- **UPED:** không có mô hình neural Phase 4 được huấn luyện. Agent D xây dựng interface và xác minh 4 unit tests, nhưng thiếu EXP24 training/preprocessing contract, reference/OOV gate và CUDA lease. Không có cơ sở nói UPED vượt EXP24/EXP36/student.
- **Tổng quát hóa/confidence:** SEARCH có 599 utterances nhưng chỉ 4 speaker clusters; đây không phải test mù, uncertainty speaker-level rất thô. Một số điểm operating policy có FWR cao; FWR<=0.20 không phải điều kiện loại trừ. Calibration không được khẳng định nếu chưa fit/đánh giá tách biệt.

## Bằng chứng VERIFIED

### Source, score provenance và tái tạo composition

Revision gốc `aa4e6ea2147180a4a820d77a71911f333fcb6721`; working tree đã có nhiều untracked user/research artifacts, được bảo toàn. Agent A chạy CPU-only verifier `experiments/phase4/composition/recompute_official_compositions.py`: các composition EXP-29/36/43 được tái dựng bằng các vector component score đóng băng và frozen DEV_CAL weights. Đủ 19,755 target và 20,354 exact-gap slots; event metadata trùng nhau, max absolute score delta=0. EXP-36 recipe SHA256 `1b1a5a644e51e399def0e41b7d87e885fa5420ee9e9da434bf3c38524a282f01`; frozen score SHA256 `ad06fde5813406089fccb3a7ff92aa07c95a8e7bd07371940888c54204b4b1ae`.

Weights của composition canonical EXP-36 được chọn trên DEV_CAL theo Macro-3 F1 objective trong grid; không có FWR constraint. EXP-36 dùng 7 source heads (22,17,21,10,24,28,34); EXP-29 dùng 6; EXP-43 dùng 9. Tổng parameters theo closure: EXP-24 533,272; EXP-29 2,999,732; EXP-36 3,535,372; EXP-43 4,626,882. Đây là tổng số constituent parameters, không phải latency measurement. Không có cold end-to-end runtime benchmark tương đương.

### Khả năng ranking và policy — DEV_SEARCH, mô tả phát triển

Official EVAL-V3.1 closure (599 utterances, 4 speakers; S=1,768, O=209, I=125; 100% target/gap coverage):

| Cấu hình | AP-S | AP-O | AP-I exact-gap | Macro-3 AP | CAL_MAX_ERROR Macro-3 F1 | Macro-4 F1 | FWR tại policy này |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXP-24 | 0.357737 | **0.225092** | 0.247628 | 0.276819 | 0.305855 | 0.450294 | 5.713 |
| EXP-29 | 0.378062 | 0.196539 | 0.272068 | 0.282223 | **0.342914** | **0.482871** | 4.598 |
| EXP-36 | **0.387088** | 0.204687 | 0.275924 | **0.289233** | 0.331090 | 0.470515 | 5.531 |
| EXP-43 | 0.363453 | 0.205771 | **0.285056** | 0.284760 | 0.332231 | 0.472479 | 5.172 |

CAL_MAX_ERROR thresholds were fitted on DEV_CAL and evaluated on SEARCH. Per-class P/R/F1 at those thresholds: EXP-36 S .306/.675/.421, O .184/.512/.271, I .254/.368/.301; EXP-29 S .333/.623/.434, O .190/.545/.282, I .370/.272/.313; EXP-24 S .289/.644/.399, O .172/.483/.253, I .231/.312/.265; EXP-43 S .316/.677/.431, O .184/.493/.268, I .395/.240/.299. These are policy-specific estimates, not ranking scores. EXP-43 provides the strongest binary MD AP reported in the canonical record (~0.3872); chosen threshold binary F1 varies by policy and does not identify a universally best model.

EXP-36's argmax diagnostic FWR is 2.768; CAL_MAX_ERROR FWR is 5.531; selected risk budgets 0.1/0.2/0.5/1/2 yielded observed FWR about 0.093/0.139/0.386/0.978/1.768 and macro-3 F1 0.113/0.148/0.196/0.278/0.315. These illustrate a policy trade-off only; no FWR threshold is a mandatory success gate.

### Confidence and robustness

Canonical raw score values are explicitly `UNKNOWN_NOT_POSTHOC_CALIBRATED`. For EXP-36, ECE equal-width 10 bins / Brier from EVAL-V3 arrays: S=.1867/.10685, O=.0160/.01180, I=.00633/.00663. These are raw reliability diagnostics, not evidence of calibrated confidence; the frequently quoted isotonic improvements were not independently re-fit and assessed in this campaign. Reliability data and CAL operating curves remain in official closure artifacts.

EXP-36 speaker AP (NCC/RRBI/YBAA/YDCK): S .490/.339/.221/.531; O .195/.296/.114/.244; I .348/.077/.380/.455. Effects are heterogeneous, especially insertion at RRBI. Four clusters do not support broad population claims or precise cluster CI. Agent C did not rerun paired tests; historic event-wise t-tests are not confirmatory because phone events are dependent and AP is not a per-utterance metric suitable for treating thousands of events as independent. OOV figure 39/599 is historical reported *utterances containing OOV*, not token-level denominator and not independently verified here.

## EXPLORATORY findings and unresolved hypotheses

TMCI reported EXP-24+34 Macro-3 AP .292012, O AP .242612. It exceeds official EXP-36 on exposed SEARCH by .002779 Macro-3 AP, but its historical candidate selection provenance is insufficiently auditable and may reflect post-hoc search. The Phase 4 score verifier does not reconstruct this two-model composition. Treat as development-set discovery only.

Legacy cumulative/leave-one-out reports attribute gains to EXP-34/28/24, but source variant arrays/weights and full scripts were not present for independent reconstruction. Correlations between EXP-36 component target scores are high (roughly .72–.91), including EXP-24/34 .873 and EXP-28/34 .906; this is descriptive association, not evidence of independent causal acoustic information.

Agent B's reference-only, prior/prosody neutralization, mismatch sensitivity and CTC omission experiments were not executed. No gold TextGrid/realized-phone artifact was admitted to deployed inference. Exact effects on omissions and nonzero Viterbi duration remain unresolved. Intended reference is annotation-derived; its independence from realized gold is explicitly unverified.

## UPED and student status

Agent D added `data_contract.py` and unit tests verifying genuine local 16-kHz WAV input, reference phone validation, N/N+1 mask cardinality and empty-reference gap. Four tests passed; this is interface validation, not neural inference. No Phase 4 D0/D1/D2 training ran. Existing Phase 3 Gold-Only GRU (78,661 params; 12 epochs/684 updates; CUDA receipts) and KD-All student headline 0.236510 are historical controls, not matched new experiments. EXP-24's full primary preprocessing/training contract could not be reconstructed; UPED handoff is a hypothesis, not a validated architecture. No CUDA lease was acquired; no GPU training/inference evidence in Phase 4.

## Hạn chế và kết luận

EVAL-V3 has full event IDs and continuous score arrays for the reported official configurations, but DEV_SEARCH is repeatedly exposed. It is legitimate for exploratory selection/debugging, not independent generalization. Raw audio inference receipts exist for certain closure replay; reference construction is not certified independent. No Phase 4 acoustic intervention or newly trained unified network exists. Therefore strongest validated statement is: canonical frozen EXP-36 replay ranks highest among listed canonical controls on DEV_SEARCH; EXP-24 has best O AP; EXP-43 best exact-gap I AP; EXP-29 has best CAL-selected Macro-3/Macro-4 F1 in this comparison. EXP24+34 remains an exploratory configuration, and UPED improvement is **not established**.

## Reports and reproducibility

See `PHASE4_A_COMPOSITION_VALIDATION.md`, `PHASE4_B_ACOUSTIC_DEPENDENCE.md`, `PHASE4_C_CONFIDENCE_ROBUSTNESS.md`, and `PHASE4_D_UNIFIED_STUDENT.md`. Dataset/revision provenance is retained in closure receipts and per-report references. Parent preflight observed HEAD `aa4e6ea...`, dirty/untracked tree and GTX 1650 4GB; initial `nvidia-smi` showed idle, but this is not a lease. No Phase 4 GPU job ran. Reproduction commands and limitations are in `reports/PHASE4_REPRODUCTION.md`.
