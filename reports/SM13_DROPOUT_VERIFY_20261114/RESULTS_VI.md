# Verify hướng chống overfit — PREFLIGHT ONLY / RESOURCE BLOCKED

User làm rõ muốn verify hướng có khả năng cải thiện điểm, không chỉ load backbone đã fine-tune. Chọn một đối chứng hẹp: tăng gấp2 dropout của fresh RMS-normalized SM13 heads. Đây là giả thuyết chốngoverfit, không phải fix đã chứng minh hoặc quyền mở fullbackbonefinetuning/S2.

## Đã thực hiện

CPU-only construction smoke cho cảtarget/gap, khôngoptimizer/backbone/GPU:
- Same seeded initial state và parameter count giữa baseline/candidate.
- Chỉ dropout đổi: target.25/.20→.50/.40; gap.15→.30.
- Eval outputs bằng nhau chính xác trướctraining; train-mode outputs stochastic vàfinite.

`PREFLIGHT.json` ghi các kiểm tra trên đềuđạt. **Chưa đoquality/F1; không gọi phươngphápđãcảithiện.**

## Pilot dự kiến, chưa chạy

`PROTOCOL.json`: fresh normalizedbaseline vs dropoutcandidate,50epoch/head/arm,seed101/gap1101,same1350fit450held/initialweights/batchorder/LR/loss/weightdecay/batches/fixed.5. Khôngthresholdsearch/bestepoch. Sáusafeguards kếthừa priorpilot nhưngdelta lấycandidate−normalizedbaseline, khôngrawbaseline. Localonly/2threads/≤1h/sharedGPUlock/atomicfullstates; khôngautoextension/retry/queue.

Chưa viết/launch productiontrainingdriver củapilot này. Trướcfit phảiquaGPUprofile/admissionvàstate/predictionverifier; protocol khôngtựchứngminhreadinesshayquality.

## Gate đang chặn

Tại preflight, `/var/tmp` free **13,448,781,824bytes (~12.53GiB)** < **14GiB CROTTCexportfloor**, thiếu~1.47GiB đểchỉđạtreserve (chưaphảiquyềndùngphầnreserve). RAMvàhomefloor4GiB+1.5GiBoutputbudget đạt. Khôngnhảgatechỉvìweightsdựkiếnởhome. CROTTCartifactretrievalđanghoạtđộng; khôngstop/delete/moveđểlấychỗ.

Trạngthái **PREFLIGHT_ONLY_RESOURCE_BLOCKED**. Khôngtraining/freshF1/newcache/GPU/remote/sealeddata, khônghẹnjobtựchạy. Cầnkhôiphụcfree/var/tmp≥14GiB bằngphươngánstorageđượccấpphépkhôngđụngartifactsbảovệ, rồirecheckadmission; khôngtựxóaweights/cachehoặchạfloor.

Tool `tools/verify_sm13_dropout_preflight.py` từchốioverwriteoutputdirectory; lầnverifytiếpcầnsnapshotmới. Khôngsửacodeproductionheads/recipehaycheckpointcũ.
