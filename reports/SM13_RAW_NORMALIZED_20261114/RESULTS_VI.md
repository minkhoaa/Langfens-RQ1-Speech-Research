# Raw vs normalized pilot và normalized300 — COMPLETE/VERIFIED

Training PID1763886 đã terminal, `STATUS.json`=COMPLETE_NORMALIZED300_VERIFIED.
Cả target/gap normalized300 xong;26400/13800 optimizer updates. Tổng window3744,75s
(~62,41phút), gồm haiarm pilot, normalized continuation và CPUverification.
Local only, không new remote/CROTTC commands, không test/DEV_CAL access.

## Đối chứng đúng cùng epoch50

| Arm, total epoch/head | Detection micro F1@0.5 | Macro S/O/I AP |
|---|---:|---:|
| Fresh raw50 |38,6507%|25,7442%|
| Fresh fit-RMS normalized50 |40,8143%|30,7764%|

Delta normalized−raw: **+2,1636ppF1/+5,0322ppmacroAP**. S/O/IAP tăng lần lượt
+3,7658/+6,0132/+5,3174pp. Cả3heldspeakers F1 tăng; meanlast10F1delta+2,3965pp;
falsewarnings giảm0,5289/câu. Tất cả6qualitygate đã đăng ký trước outcomes đềuPASS.
`PILOT_VERIFICATION.json`:204states, event/prediction/metric/initialization/fit-only
normalization-buffer/rawhistoricalparity và freshCPUheldinference đãverify trong
subprocessCPU riêng trước conditional300admission. Đây là onepairedseed/reusedTRAIN
exploratory, không independent confirmation.

## Normalized conditional total300

| Epoch/head | F1@0.5 | Macro S/O/I AP |
|---|---:|---:|
|50 |40,8143%|30,7764%|
|100|41,6992%|31,4947%|
|150|40,5812%|32,0566%|
|200|43,2289%|32,3324%|
|250|40,7588%|33,1578%|
|300|40,8843%|33,8588%|

Fixed300precision52,5753%/recall33,4468%; typedS/O/IAP42,2003/29,4380/29,9382%.
Không chọn200đẹp nhất thay fixed300. F1@300 gần ngang50(+0,06997pp), dù macroAP tăng
+3,0825pp. MeanF1epochs41–50=42,1783%,291–300=39,6455%: cuối curve không cho thấy F1
cải thiện bền vững. TargetfitCE0,1096→0,0321 nhưng heldCE1,6889→3,0304: overfitting
vẫn còn. Không tự nối thêm epoch, retune threshold hay thay checkpointselected.

`FINAL_VERIFICATION.json`: **602normalizedstates VERIFIED**, full1..300curves,
Adam/RNG/cursors/buffers, keyedpredictions/metrics, freshCPUheldoutputreplay, recomputed
fit-onlyscales, retainedcachehash và unchangedpilotepochartifacts. `FINAL_VERIFIER_STATUS.json`
xác nhậnNORMALIZED300_STATES_AND_CURVES_VERIFIED. No postprocessing repair needed.

Curve `NORMALIZED300_CURVES.png`; pilot `PILOT_CURVES.png`; rawdata
`PILOT_COMPARISON.json` và `NORMALIZED300_REPORT.json`. Checkpoints nằm ngoàiGit:
`/home/khoa/.local/share/langfens/sm13_raw_normalized_20261114_v1/`.

**Giới hạn:** raw50vsnormalized50 là matchednormalizationcomparison; normalized300vsraw50
KHÔNG matchedepochgain. Fixed0.5 là diagnosticdevelopmentpolicy đãkhóa, không paper
hierarchicalF1. Same450TRAINheldABA/ASI/BWC đãreuse,1350fit, neuralonly, onepairedseed.
FullTRAIN1800→DEV_SEARCH599/commonCROTTCadapter/finalthesis/testevaluation vẫn pending.
