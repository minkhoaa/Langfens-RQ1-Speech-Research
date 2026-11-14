# Phase A — Error analysis từ predictions đã VERIFIED

**COMPLETE text-only**;checkpoint3150,repairedCPUeval,599DEV_SEARCH. No acoustic
forward/backward/training ở laptop; không đổi scoring policy hoặc test/DEV_CAL.
`ERROR_ANALYSIS_V2/ANALYSIS.json` exact matches overall/per-speaker counts trong
independentlyverified5trialreplay. Đây không phải current5672modelresult.

## Counts và nhóm lỗi

- TA16562 / **FR1325** / **FA752** / TR1248; CD988 / ED260.
- DetectionF1 **54.5812%**,precision48.5037%,recall62.40%.
- Với các lỗi thật được nguồn annotate: S1058/1659detected(**63.77%**),
  D110/209(**52.63%**),I80/132(**60.61%**).
- Trong TR, chẩn đoán đúngrealizedsequence là988/1248≈79.17%;260 là phát hiện
  có lỗi nhưng diagnosis sai theohierarchicalmetric. Không đồng nhất vớifeedbackcorrectness.
- **Không báo per-typeprecision/F1 từ các nhóm goldpositive-only**:falsewarnings
  trêncorrectphones không có trueerrorclass để chiađộcnhất. Không giảprecision=100%.

## Speaker breakdown — cùngcheckpoint nhưng khácsupport/prevalence

| Speaker | Utterances | F1% | P% | R% | FR / correct-support |
|---|---:|---:|---:|---:|---:|
| NCC |150|64.16|60.14|68.76|283/4357≈6.50%|
| RRBI |150|46.87|41.10|54.53|440/4429≈9.93%|
| YBAA |149|37.21|29.39|50.70|346/4680≈7.39%|
| YDCK |150|63.90|59.11|69.55|256/4421≈5.79%|

YBAA cópositive284/4964≈5.72%,NCC621/4978≈12.48%:F1/precisiondifference bịảnhhưởng
bởierrorprevalence,khôngtựchứngminhmodelacousticallybias. TuyvậyRRBI cófalsewarningrate
cao nhất vàYBAA cómissederrorratecao, cần matchedcleanTRAIN/DEV vàphone-conditionedanalysis.
**Không có globalhardceiling60%**:hai speaker hiệnđãvượt60 trong cùngprotocol.
Không cherry-pickhai speaker này thành overallclaim.

## False alarms nổi bật

Trong1325false-rejections,top3confusions là:
- `z→s`:137;
- `r→err`:130;
- `dh→d`:75.

Tổng342/1325≈**25.81%**falsewarnings. Đây làmodelprediction trái annotationcanonical/
perceived đúng ở các vị trí này, không phải thực chứng phoneticlistener rằngmọiannotationđúng.
`err` làunknownrealizationtoken,không phải phoneme cụ thể để dùng làm hướngdẫn articulatory.
Predictioncó361errtokens,nhưngDEVreferencechỉ12. TRAIN có1110errtargets,355canonicalr→err
positions (metadataaudit):phùhợp giảthuyết supervision policy/tokencollapse cầnkiểmtra,
**chưa chứng minh causal** vàkhôngquyềnrelabel/droperrrows.

Recognition alignment sau bỏsil:1794substitutions,406deletions,904insertions,
17478matches. Đây làperceived→hypeditcounts,không phải strictS/O/I/exactgapcounts
vàkhôngđược cộng thànhhierarchicalFR/FA. Hypinsertionscóthểbịhierarchicalmetricbỏqua.

## Ví dụ và giới hạn

JSON lưu first3lexicographicIDs peroutcome (FR/FA/TR_CD/TR_ED),không chọnworstcase
sau khi nghetest. Chỉtext/phonearrays;chưa cóaudio/listenerreview hoặc goldtimestampsinference.
V1đượcgiữnhưartifacttrước khiẩnconditionalpositive-onlyprecision/F1 ởV2;aggregatecounts
khôngđổi. Native5672matchedcleanpopulations/componentgradientdiagnostic làphần tiếp theo,
khôngdùng3150erroranalysisđểkhẳngđịnhlatestmodel đãcùngerrorprofile.
