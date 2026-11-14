# Kết quả evaluator replay — COMPLETE / independently VERIFIED

Đã tải/xác minh `retrieval_v1/VERIFICATION.json`: **599 DEV_SEARCH ×5trials =2995predictions**,
fixedcheckpoint3150 SHA `47fbf1d91ab0838970b05d9d0af67d224c9c8294e3f3950c5c9069414ab5ceaf`.
ActualremoteRoPEMHAsourceSHA khớpdiagnosis. Modelweights khôngđổi,optimizerupdates0,
CPU-only,Kagglemaintraining khôngsửa/dừng/restart. Driver1027.68s;retrieval/verifier đãterminal.

| Trial | F1 (%) | P (%) | R (%) | PER online (%) | PER no-sil (%) |
|---|---:|---:|---:|---:|---:|
| Legacy3047 |54.917319|48.613251|63.10|27.314768|15.941661|
| Legacy3048 |54.939759|48.888889|62.70|27.284277|15.885761|
| Legacy3049 |54.720701|48.693957|62.45|27.446895|16.043297|
| Repair3047 |54.581238|48.503692|62.40|27.177559|15.773961|
| Repair3048 |54.581238|48.503692|62.40|27.177559|15.773961|

LegacymeanF1=**54.859260%**;range[54.720701,54.939759]%;repair−meanlegacy=
**−0.278022percentagepoints**. Repairpredictions exactequal acrosstwoseeds.
No bestrepeatselection. Allper-speaker/alignment/phoneoutputs saved and replayed locally
by independent text-only scorer;no localbackbone/audioinference required.

## Interpretation

1. Evaldropoutdefect **cóthật**, deterministicrepair loại inference noise đúng như thiết kế.
2. Trong **fixed3150/599DEV/CPUsharedmemory** này, repair không làm tăngF1,
   khôngủnghộ đây là nguyênnhânchính của khoảngcáchđến60%. Khôngsuy“evaluator hoàn toàn đúng”
   hoặc mọi checkpoint/dropoutseed/backend đều có cùngảnhhưởng.
3. CPU/unpadded/sharedfeaturecache khác historical2T4/padded pipeline. Khôngrewrite
   historicalF1 hoặc lấy54.58%làmfullpaperresult. Đâylàpairedintervention,khôngmodelseedCI.
4. Silencepolicy đổi PER khoảng11.4pp trongreplay, nhưng **không đổi hierarchicalF1**;
   không dùng PERno-sil thay điểmonline rồi gọimodelgain.
5. Cần táchcomponentOT/CR,augmentation vàcleanTRAIN/DEVobjective trước khiđổi recipe/
   backbone. Productionrepair haytrainlại lànewlineage/registrationriêng,khôngtựhotpatch.

## Preservation / rerun boundary

User sau đó yêu cầu “lưu lại toàn bộ rồi chạy lại”. Lưu code/config/predictions/metrics/
logs/receipts và referencehashcheckpoints;khôngxóa hoặc ghiđèbằngchứngcũ. “Chạy lại” chưa
chỉ rõ evaluation-only hayfreshtraining/recipe nào; **chưalaunchnewtraining/stopcurrent**
trong bước preservation. Latestcurrentwindowexport phải đượctải/verify qua controllercósẵn,
khôngnhậnliveweights là downloadedproof hoặc tảitrùng. Test/DEV_CALvẫnsealed.
