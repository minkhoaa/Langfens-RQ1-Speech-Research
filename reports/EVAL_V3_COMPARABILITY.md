# EVAL-V3 — Ma trận đối sánh và missingness

Matched subset chỉ gồm14 current student arm/seed pairs với N=19.755, gaps20.354, utterances599, speakers4; manifest hash và integer support được kiểm tra trước metrics. Không trộn historical policies hoặc teacher score semantics vào một universal ranking.

## Event-cohort audit của các control cũ

| Control | Targets | Gaps | Extra targets | Missing targets | Trạng thái |
|---|---:|---:|---:|---:|---|
| UPR02_A0_CTC_BASELINE | 19664 | 20263 | 49 | 140 | PROTOCOL_NONCOMPARABLE |
| UPR02_A1_L6_RAW_UNCONSTRAINED | 19664 | 20263 | 49 | 140 | PROTOCOL_NONCOMPARABLE |
| UPR02_B1_L6_ACOUSTIC_GATED | 19664 | 20263 | 49 | 140 | PROTOCOL_NONCOMPARABLE |
| UPR02_B2_WHITENED_BOUNDED | 19664 | 20263 | 49 | 140 | PROTOCOL_NONCOMPARABLE |
| UPR02_B3_AGBV_INSERTION | 19664 | 20263 | 49 | 140 | PROTOCOL_NONCOMPARABLE |
| UPR02_CANDIDATE_CHAMPION_CALIBRATED | 19664 | 20263 | 49 | 140 | PROTOCOL_NONCOMPARABLE |
| UPR02_CHAMPION_CERTIFIED | 19664 | 20263 | 49 | 140 | PROTOCOL_NONCOMPARABLE |
| UPR02_SYNTHESIS_CHAMPION | 19755 | 20354 | 0 | 0 | MATCHED_FULL_POPULATION |
| UPR03_SYNTHESIS_CHAMPION | 19755 | 20354 | 0 | 0 | MATCHED_FULL_POPULATION |

Bảy control cũ không được ép join/drop slots; hai synthesis controls có complete matching cohort. Older EXP prediction arrays thiếu event IDs không được gán IDs theo row order. Individual specialists chỉ có observed AP của domain/class thật; missing AP khác giữ null. CPL02 deterministic F0/F1/F2/F3/F5 được dựng lại từ frozen teacher scores và recorded parameters; F4 thiếu frozen coefficients chỉ partial, không retrain. F3 temperature constants1,15/1,25 được hardcode trong source cũ, **không có evidence được fit**, không xác nhận calibrated confidence. V3 raw AP của reconstructed CPL02 dùng teacher-native scores, khác legacy synthetic simplex; frozen-best ledger giữ semantic cũ và được gắn nhãn riêng.

Mọi registered ID được giữ cùng full legacy JSON. Các file superseded inventory mang INVALID_SUPERSEDED, không xếp hạng. OOF TRAIN không là final test; isotonic mapping CAL→OOF TRAIN có distribution shift chưa xác minh. No-waveform cached rows không thừa hưởng sampled WAV validity gate.
