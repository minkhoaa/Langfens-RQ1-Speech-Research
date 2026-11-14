# Kết quả đối soát EVAL-V3

Nguồn chỉ đọc; NPZ scores được đối chiếu ID/nhãn với manifest canonical. DEV_SEARCH là development-exposed, không phải holdout độc lập.
Số rows registry và experiments: 274; FULL_SCORE_REPLAY: 100 rows (14 arm/seed NPZ, 84 DEV_CAL policy rows, 2 controls).

Lệnh:

`python research/evaluation_v3/migration.py --root /home/khoa/Projects/New-Method-Research --output /home/khoa/Projects/New-Method-Research/experiments/evaluation_v3`

Trạng thái:
- FULL_SCORE_REPLAY: 100
- CHECKPOINT_INFERENCE_REPLAY: 0
- COUNTS_ONLY_PARTIAL: 45
- MISSING_ARTIFACT: 83
- PROTOCOL_NONCOMPARABLE: 46
- INVALID_SUPERSEDED: 0
