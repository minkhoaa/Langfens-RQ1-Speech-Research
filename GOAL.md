# LANGFENS — Hai GOAL nghiên cứu hiện hành

**Quyết định tách:** `RESEARCH-GOALS-SPLIT-20261114-v1`, theo yêu cầu user ngày2026-11-14.
**Trạng thái:** `RESEARCH_NOT_COMPLETE`.
**Snapshot trạng thái:** 2026-11-14 **16:46 +07**; số liệu bên dưới không phải monitor realtime.

## 1. Hai GOAL tách biệt, không dùng kết quả của nhánh này chứng nhận nhánh kia

| GOAL | Tài liệu điều hành chi tiết | Mục tiêu | Trạng thái tại snapshot |
|---|---|---|---|
| **1 — CROTTC-IF full-paper reproduction** | **[GOAL_CROTTC_IF.md](GOAL_CROTTC_IF.md)** | Đóng data/recipe/metric fidelity gates, train đủ stages và phủ các thí nghiệm paper | Current AM local-protocol continuation trên Kaggle RUNNING, humanepoch52/update2950; **chưa tái lập full paper** |
| **2 — SM13 phương pháp lớn hơn** | **[GOAL_SM13.md](GOAL_SM13.md)** | Frozen WavLM Large + SM13 heads, đối chứng S0 và common detectionF1; giữ S/O/I AP/falsewarnings/cost | Original-backbone S0 local50/50 **đã COMPLETE/VERIFIED**; Large-backbone fit và common detection evaluation **chưa thực hiện** |

GOAL.md là **index và quy tắc chung**, không phải GOAL thứ ba. Hai file trên là nơi cập nhật kế hoạch, trạng thái, gates và file references cho từng nhánh. `SIDE_GOAL.md` chỉ là compatibility pointer, không còn là work queue độc lập.

## 2. Trạng thái thực tế cần giữ đúng tên

### GOAL1
- Kernel [optimized-window001](https://www.kaggle.com/code/minkhoaa/langfens-crottc-if-optimized-window001), PIDcontroller1291861; 2T4/FP16+FP32loss-master-Adam/batch32; không dừng/restart để tách tài liệu hoặc phục vụ SM13.
- Snapshot live: update2950/internalepoch51cursor1376, đanghumanepoch52/300AM. Completedepoch51 F1=51,4233%/PER=27,4774%; bestF1epoch46=54,3590%.
- Latest **downloaded/full-state verified** checkpoint vẫn482 tại snapshot; không gọi live2950 là đã exportverify.
- Population local1800TRAIN/599DEV_SEARCH, preprocessing/augmentation/IFrecipe còn fidelity gaps. Đây là **implementation/training development**, không benchmark reproduction và không bằng chứng đã đạt paper71,77%.
- SDKquota snapshot báoused31899.255732.0s >allowed21600s, reserved0, reset21NovUTC. CurrentadmittedkernelvẫnRUNNING; khôngstopnó. Cửa sổ sau có thểquota-blocked; phải preserve/export/verify và báo limits, không hứa uninterrupted300/fullpaper.

### GOAL2
- S0local thử target50/gap50 hoàn tất trong551,57s, 102checkpointstates/predictions/metrics verified; train/verifierPID1448557/1451451đãterminal.
- Same450TRAINheld (ABA/ASI/BWC), seed101: neural-onlymacro3AP **30,8974%→33,4066% (+2,5092pp)**; omissiongiảm, 1seed/1reusedfold, khôngfullfusion/backbonegainclaim.
- S1nativeWavLMLarge và S2provisionaladapted mới có8×2frozenfeaturesmoke; chưaproductionheadfit. Không gọi smoke là phương pháp đã cải thiện.
- RemoteS0SM13 vẫnQUEUED/notcancelled; khôngguesssessionID/deleteartifacts hoặc khởiduplicate. Latestuserchọn **SM13local**, khôngbreakcurrentCROTTC.

Nguồn snapshot: [remote/quota](reports/GOAL_SPLIT_20261114/REMOTE_SNAPSHOT.json), [CROTTCcurve](reports/GOAL_SPLIT_20261114/CROTTC_CURVE/SUMMARY.json), [SM13verification](reports/SM13_EPOCH50_20261114/LOCAL_VERIFICATION.json).

## 3. Quy tắc chung và ranh giới ủy quyền

1. **Giữ nguyên tiến trình CROTTC hiện tại.** GOAL1fulltraining dùngKaggle; không tự resume/train CROTTC local hoặc chuyểnColab. GOAL2SM13local là quyền riêng đã được userchốt.
2. Không tự start model/sweep/corpus mới chỉ vì xuất hiện trong GOAL. Productionrun mới cần protocol/profile/resourcebudget/parent gates tương ứng. Tác vụ táchGOAL này **chỉ sửa tài liệu**, không launch/stop/restarttraining.
3. **Officialtest vàDEV_CAL vẫnSEALED.** Splitpaper mới hoặc benchmarktest cần quyền protocol cụ thể; khôngextract/stage/đọc audio/labels trước. Hiện chỉ allowlistedTRAIN/DEV_SEARCH.
4. Không sửa/xóa originalarchives/features/checkpoints/logs/receipts để làm đẹp lineage. Fullstateatomic, preserveoptimizer/scheduler/scaler/RNG/cursors/pendinggradients, verifyexport trước continuation. Stopnonfinite/mismatch/resourcefailure, khôngskipbatches.
5. Không xếp hạng F1detection vớimacro3AP; không gọi reuseddevelopment là unbiasedtest. Commonmetric/commondata/commoninformationbudget phải được kiểm chứng trước head-to-head.
6. Không dùng currentallTRAIN-adaptedCROTTC backbone để claim independentTRAINOOF. NJSđã nằmTRAINcurrent; exactpaperbenchmarkcóNJStestphải dùngfreshcleanlineage, không reusecurrentcheckpoint.
7. Khôngcron/timer/subagent/commit/push/merge mặcđịnh. Tests cần sharedGPUlock phải chờ productionlease, không nhả/bypasslock để testxanh. Dữ liệu/weights/credentials không vàoGit.

## 4. Bối cảnh thesis giữ nguyên, không mở thêm GOAL

Kế thừa mục tiêu `THESIS-METHOD-DECISION-20261113-v1`: phát triển và kiểm chứng phương pháp chẩn đoán/hướng dẫn sửa phát âm có căn cứ. CROTTC-IF là baseline nghiên cứu; SM13 là nhánh phương pháp đang phát triển, chưa có novelty/gain claim trước bằng chứng.

[L2-Arctic-plus/ALMs4Learning paper](https://aclanthology.org/2026.findings-eacl.190/) và [code](https://github.com/zoeyada/ALMs4Learning) tiếp tục là nguồn thiết kế đánh giá phản hồi, không phải quyền tự train thêm ALM. Không đồng nhất văn bản giống nhau với lời khuyên đúng ngữ âm.

Thesis cần evidence về căn cứ âm học, đúng vị trí/chẩn đoán/hướng dẫn, coverage và cảnh báo sai. Báo:
- **Độ đúng phản hồi:** cảnh báo đồng thời đúng vị trí/chẩn đoán/hướng dẫn trên tổngcảnhbáo.
- **Độ phủ hữu ích:** lỗi thật nhận được phản hồi đúng trọnvẹn trên tổnglỗitrongphạmvi.
- Humanrubric/blinding/disagreement khi phù hợp; không tự đặt80%P/R hayFWR≤0,20 làm điều kiện tốt nghiệp. Không suy người học tiến bộ chỉ từ modelscore.

Hoàn thành GOAL1 hoặcGOAL2 không tự hoàn thành thesis. Các milestoneM0–M6, compactCROTTC/EXP/ATP-LLM/PP-ATP/PER-MDD/X-MDD cũ là tài liệu lịch sử/phụ, không tự phục hồi thành campaignsong song.

## 5. Bàn giao và bảo tồn quyết định

- Đọc [AGENTS.md](AGENTS.md) → GOAL.md → fileGOALtươngứng → [HANDOFF.md](HANDOFF.md) trước hành động.
- [Decision/archives split](reports/GOAL_SPLIT_20261114/DECISION.json), [SHAreceipt](reports/GOAL_SPLIT_20261114/ARCHIVE_RECEIPT.json).
- Bản đầy đủ trước tách: [GOAL cũ](reports/GOAL_SPLIT_20261114/PRE_SPLIT_GOAL.md), [toàn bộ SIDE_GOAL/discussion](reports/GOAL_SPLIT_20261114/PRE_SPLIT_SIDE_GOAL.md), [HANDOFF cũ](reports/GOAL_SPLIT_20261114/PRE_SPLIT_HANDOFF.md), [AGENTS cũ](reports/GOAL_SPLIT_20261114/PRE_SPLIT_AGENTS.md).
- [Thesis supersession20261113](reports/GOAL_DECISION_20261113/DECISION.json) và [Kaggleauthorization](reports/CROTTC_IF_KAGGLE_20261114/DECISION.json) vẫn là provenance hợp lệ. Quyết định mới tách điều hành thành haiGOAL, không xóa lịch sử hoặc ngầm cấp quyền mởtest.
- Khi cập nhật: timestamp/source/verified-vs-live rõ ràng. Snapshot cũ không phải live process status hay completion marker.
