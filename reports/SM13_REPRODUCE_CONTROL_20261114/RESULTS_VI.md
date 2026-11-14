# Tái hiện baseline trước khi kết luận — paired training BLOCKED

## Yêu cầu hiện hành

User yêu cầu không đề xuất quyết định chưa có đối chứng trên SM13, phải tái hiện rồi đối chứng trước. Tiếp nhận theo phạm vi bounded paired50/head/arm đã có cho dropout; **không tự chuyển sang label smoothing, chạy nhiều phương án hoặc fine-tune backbone**. Research recommendation trước chỉ là giả thuyết, chưa được quality evidence ủng hộ.

## Đã thực hiện thật trong lượt này

`tools/replay_sm13_baseline_and_admission.py`, CPU-only/2threads/37.54s:

- Rehash native featurecache/audio/registered inputs, strict-load normalizedtarget/gap50states với SHA từ prior602-state verification.
- Fresh evalmode inference toànfit/held, CE từlogits; so savedGPU probabilities/argmax/TP/FP/FN/TN và heldCE. Khôngthresholdretune.
- Independently replay **đủ50epoch đầu** từ100saved target/gap predictionfiles bằngsklearn, đối chiếu population/eventorder/P/R/F1/S/O/IAP với originalcurve, tolerance1e-12 choAP/F1. Đây là metricreplay của predictions đãcó, khôngtrainlại50epochs.
- GiữnguyênSHA checkpoint/predictionartifacts dùngtrongaudit. NoGPU/backbone/optimizer/backward/remote/testDEV_CAL.

Fresh baseline50 kết quả:

| Population | TP | FP | FN | F1@.5 |
|---|---:|---:|---:|---:|
| Fit |7246|326|170|96.6907%|
| Held |822|856|1528|40.8143%|

**Tái hiện gap thành công ở mức inference/metric. Chưa có freshtraining reproduction, candidatefitting hoặc gainclaim.**

## Vì sao chưa có kết quả đối chứng mới?

Admission tại lượt này:
- `/var/tmp` free14,597,033,984bytes (~13.59GiB).
- CROTTCreserve bắtbuộc15,032,385,536bytes (14GiB).
- **Thiếu435,351,552bytes (~415.18MiB)** đểchỉđạtreserve.
- Home4GiBfloor+1.5GiBoutputbudget/RAMinitial16GiBđạt; GPUtrainingprofile/lock chưaadmit vìgatechặntrước.

Receiptstatus: **BASELINE_REPLAY_VERIFIED_PAIRED_TRAINING_RESOURCE_BLOCKED**.

Khôngdelete/move/cache/checkpoints/hạfloor/stopCROTTC/đổisangCPUtrainingđểláchgate. Khôngscheduler/waiter/autolaunch. Khôngtạoqualityoutcomesgiảchođốichứngchưachạy.

## Phạm vi đối chứng khi được admission

Protocol gốc `../SM13_DROPOUT_VERIFY_20261114/PROTOCOL.json` được giữ nguyên: freshnormalizedbaseline vs2xdropoutcandidate,50/head/arm,sameinit/RNG/batchorder/data/fit-onlyRMS/LR/loss/weightdecay/fixed.5/fullcurves/sixqualitysafeguards. Reproductioncontrol cầnfreshbaselineparity trướcchấpnhậncandidatecomparison; independentstates/keyedpredictions/metricsverification trướcqualityclaim. Chưaimplement/launch pairedproductiondriver trong lượt này. Khôngđổigate sauoutcomes, khôngchọnbestepoch.

Chỉkhibằngchứngthậtđạt mới được báo cógiúp/khônggiúp/regression trong phạmvi onepairedseed/reusedTRAINdevelopment. Khônghứacausalrootcause, backbonegain hoặcindependenttestconfirmation.
