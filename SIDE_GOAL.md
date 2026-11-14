# SIDE_GOAL — Compatibility pointer, không phải GOAL thứ ba

Theo yêu cầu user2026-11-14, decision **`RESEARCH-GOALS-SPLIT-20261114-v1`**, repository có hai GOAL điều hành rõ ràng:

1. **[GOAL_CROTTC_IF.md](GOAL_CROTTC_IF.md)** — tiếp nhận/bảo toàn phiên CROTTC hiện tại, đóng fidelity gates và tái lập đầy đủ paper. Current local-protocol AM không được gọi là full reproduction.
2. **[GOAL_SM13.md](GOAL_SM13.md)** — phương pháp SM13 với backbone lớn, common detectionF1, matched controls và retainedS/O/IAP. OriginalS0local50/50 đã COMPLETE/VERIFIED; S1/S2productionfit và commonevaluator cònpending.

[GOAL.md](GOAL.md) là index/quy tắc chung. **Cập nhật công việc SM13 ở GOAL_SM13.md**, không duy trì hai workqueue mâu thuẫn.

## Giữ nguyên thảo luận và authorizations trước đây

- Toàn bộ nội dung SIDE_GOAL trước tách, bao gồm discussionepoch/dataset/frozen-vs-finetuned/backbone, dimensions, cache/leakage và các amendments: **[PRE_SPLIT_SIDE_GOAL.md](reports/GOAL_SPLIT_20261114/PRE_SPLIT_SIDE_GOAL.md)**.
- [ArchiveSHAreceipt](reports/GOAL_SPLIT_20261114/ARCHIVE_RECEIPT.json) bảo toàn snapshot, không rewrite lịch sử.
- `SM13-BACKBONE-SIDE-20261114-v1` và `SM13-EPOCH50-20261114-v1` vẫn là provenance; latest user instruction **SM13local, không break current CROTTC** tiếp tục có hiệu lực.
- Test/DEV_CAL vẫnsealed. TáchGOAL không mởsweep, tựlaunchtraining, đổi runtime chính, canceljob hay xóaartifacts.
