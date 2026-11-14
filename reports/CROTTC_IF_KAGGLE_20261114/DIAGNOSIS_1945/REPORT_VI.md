# Vì sao CROTTC AM chưa vượt60%? — chẩn đoán có chứng cứ

Audit **2026-11-14 19:44–20:00 +07**. User yêu cầu phân biệt lỗi thuật toán,
mô hình hay cách train. **Không dừng/restart/sửa production recipe/dependency**;
không optimizer update/native WavLM forward local, không test/DEV_CAL access.

## Kết luận ngắn

**Có lỗi xác nhận ở implementation đường validation, không chỉ “dao động bình thường”.**
SpeechBrain1.0.3 `RoPEMHA.forward` vẫn truyền `dropout_p=.1` vào PyTorchSDPA
khi module đã `.eval()`. Same weights/input cho outputs khác nhau. Điều này làm
validation/checkpoint selection có stochastic noise không được tắt như dự định.
**Chưa đo actual DEV F1 delta**, nên không quy toàn bộ plateau cho lỗi này và không
hứa sửa xong vượt60%.

**Chưa tìm thấy bằng chứng OTTC toán học sai, optimizer không cập nhật, hoặc backbone
bị freeze nhầm.** Loss/recipe và data/evaluation còn nhiều confounders. Ưu tiên
đo lại checkpoint cố định bằng deterministic eval trước khi đổi backbone hay train thêm dài.

## 1. Evidence matrix — đã biết vs chưa biết

| Giả thuyết | Kiểm tra | Kết luận / confidence |
|---|---|---|
| Optimizer đứng yên/NaN |4173update logs liên tục483–4655;loss+gradnorm finite, probeweightdelta>0;3150fullstateverified|Không thấy failure loại này trong phạm vi đã kiểm. High cho observations, không bảo đảm mọi gradient/module tối ưu tốt.|
| FullSSL bị freeze/không load native |Verified3150:21CNN tensors giống hệt nativepretrained; q_proj cả24Transformer layers đã đổi;strictload heads3822765params|Full315Mbackbone đang được fine-tune, CNNfrozen đúng; không phải mô hình nhỏ/frozenbackbone. High, representative24probes không phải fullquality claim.|
| OTTC cost/gradient sai |8CPUrandomcases (4seeds×FP32/64), exactpinnedoperator chỉ đổi literaldeviceCUDA→CPU;đối chiếu denseCDFoverlaptransport độc lập cảloss,logitgrad,alphalogitgrad|Các fixtures khớp trongtol. Không phát hiện coreOT bug ở caunpadded; chưa phải nativeGPUallshapeproof.|
| CR sign/reduction hỏng |Independent probabilityKL/stopgradient test; đúng registered2view/44class/frame-average interpretation|Core implementation khớp interpretation đã đăng ký. Nhưng không chứng minh paper/source recipe tương đương.|
| Validation deterministic |ActualRoPEMHA1.0.3source+PyTorch2.6docs+sameinputrepeats;trained3150heads repeats8×|**CONFIRMED EVAL-DROPOUT DEFECT**. High ở dependency/head reproduction; remote exactsourcebytes chưa fetch trực tiếp, nhưng environmentpin1.0.3 khớp publictag/localbytehash.|
| BatchNorm nhỏ làm train/eval lệch |ConstructactualsourceAMheads,strictload3150: **0BN,14LayerNorm**,norunningstats|**Loại bỏ giải thích BN trước đây cho AM này.** Historicaldocstrings nói preservingBN không phải bằng chứng module thật.|
| Train không khái quát hóa/overfit |AugmentedTRAINloss giảm, DEVF1 plateau/regress;LRconstant,weightdecay0;allobservednormclip|Consistent generalization/objective gap, không đủ để kết luận overfit đơn thuần vì thiếu cleanTRAIN/DEVmatchedloss vàeval hiệnstochastic. Moderate hypothesis.|
| Metric có trần60% |Sanity dùng perceivedreference làm giảhypothesis trên599DEV,sauinference-free textscoring|F1=99.7745%,PER0%;**goldhypothesissanity**,khôngmodelresult. Không có hardcoded60%ceiling;alignment/metriclimitations vẫn có.|
| Backbone yếu nên không đạt60% |Model315MWavLM+Conformer2layer384;chỉAM,khôngIF;dữliệu/recipekhácpaper|**Chưa có bằng chứng** capacity là bottleneck. Đổi backbone trước khi sửa/đo evaluator sẽ khó diễn giải.|

## 2. Lỗi xác nhận: attention dropout vẫn bật khi validation

### Causal chain tại code

1. `research/crottc_if_official/kaggle_optimized_train.py:evaluate` gọi `heads.eval()`.
2. Current head config dùng2layerConformer, `attention_type=RoPEMHA`,dropout=.1.
3. Installed `speechbrain/nnet/attention.py:1367–1373` gọi:
   `F.scaled_dot_product_attention(..., dropout_p=self.dropout, scale=self.scale)`.
   Không có `if self.training else 0.0`.
4. [PyTorch2.6SDPA documentation](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html)
   xác nhận functionalSDPA luôn áp dropout theoargument, không tự đọcmodule.eval().
5. [SpeechBrainv1.0.3source](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/nnet/attention.py#L1367)
   khớp **byte-for-byte** localsourceSHA
   `2a0a6d1b3839287e3d0820858c73a770c1f6e373d10afca4c47b621d8e5dc40f`.
   Remote `window001/ENVIRONMENT.json` và livecontractpinSpeechBrain1.0.3/Torch2.6.0.
   `SB_SOURCE_IDENTITY.json` ghi rõ chưa đọc trực tiếp bytes attention.py trên remote.

### Controlled probe — không có optimizer/nativebackboneforward

`TRAINED_HEAD_EVAL_PROBE.json`:verified3150checkpoint,headsstrictload,syntheticmemory
shape2×64×1024,seed3047,mỗi nhánh8repeats:

| Disposable head probe | Same-input maxlogitdifference | Outputs bitwiseequal |
|---|---:|---|
| Normal `.eval()`,RoPEMHA.dropout=.1|1.14343|NO|
| Control chỉ set2RoPEattention dropout=0 trong RAM|0|YES|

Normalprobe frameargmaxdisagreement8.59–16.41%vsfirstrepeat. **Syntheticmemorykhôngphải
acousticDEVdistribution**, không được đọc đây là mất8–16%accuracy/F1. Control
khôngsửa weights/source/package hay run production. Weights trước/sau exactequal.

**Hệ quả:** historicalF1 là stochastic-eval observation; có thểnoise/chọnbestlệch,
không được rewrite bằng điểm rerun mới. Eval còn tiêu thụRNG, nên một repair cần
pinpolicy/RNG/provenance riêng, không hotpatch giữa campaign rồi gọi trajectories
unchanged. Sửa đúng eval **không đồng nghĩa** tắt dropout khi TRAIN.

## 3. Loss/cách train: có học nhưng chưa khái quát hóa tốt

![Diagnostic curves](LOG_ANALYSIS/DIAGNOSTIC_CURVES.png)

`LOG_ANALYSIS/LOG_DIAGNOSIS.json` tổng hợp logs483–4655,validation9–81.
Bảng chỉ dùng **full57updates/epoch**; không lấy partialtail để so loss:

| Epochblock |Mean augmentedTRAINloss|Mean DEVF1%|Mean precision%|Mean recall%|
|---|---:|---:|---:|---:|
|41–55|2.1454|52.3263|49.0366|56.1233|
|56–65|1.9298|51.8651|49.2690|54.7750|
|66–75|1.8045|50.4813|48.4308|52.7300|
|76–81|1.7065|51.1936|48.6800|54.0250|

- Loss tiếp tụcgiảm khôngchứngminh F1 tăng. Recall bị giảm trong khi precision không
  bùđủ; không đơn thuần“chưa đủepoch”. Evalnoise có thể đónggóp; chưa đo tách.
- **100%4173observedupdates** normtrướcclip>5,median khoảng16–18;clip là intentional
  safeguard,không phải tự độnglỗi. Không dùngpreclipnorm lớn để kết luận explodinggradients
  khi finite/updateđều vàlossgiảm. Tuy nhiên nên đoOT/CRgradient contributions.
- Checkpoint3150AdamLR1e-5SSL/3e-4heads,weightdecay0,LambdaLRconstant,scalerscale256;
  khôngscheduledecay. Nó làmgiảthuyếtlateoptimization/overfit đángkiểmtra, không đủ
  choautođổiLRhaytrainpolicy giữa chiến dịch.

### CR balance khác publicsource rất nhiều — đã đo đại số, chưa đo tác động model

Public `upstream/utils/LossManager.py` dùng `.5*OT_sum + .1*mean_symmetric_KL`.
Current `kaggle_native_batch.py` dùng `OT_two_view_sum + KL_sum/(2T)` cho từngutterance,
rồi meanacrossutterances. Với **mộtunpaddedutterancepair,C=44**:

- CRtermcurrent/source=**220×**;
- OTcoefficientcurrent/source=**2×**;
- relativeCR:OTcoefficient=**110×**publicsource trong ca này.

Đây là **intentionalpaper-equationinterpretation** đãđăngký, không phải arithmeticbug
hay patchlén. PublicbatchOTsum vs localutterancemean cũngkhác; **110×không phải số đo
CRdominance trên cảbatch32/nativecampaign**. Không thể suygradientratio từcoefficient
màthiếucomponentlogs. L1chỉlogtotalloss, chưa cóOT/CRseparate/gradients, nên
**over-regularization là giả thuyết ưu tiên**, chưa phải nguyên nhân được chứng minh.
Không tự thay vềpublicrecipe để cố đạt60%.

## 4. Data/architecture/evaluator confounders

- Local1800TRAIN/599DEV khácpaper2429/268/900;NJS150đãtrain nhưngpapertestcóNJS.
  Điều này chặn exactpaperclaim, **không chứng minh NJSleakage gây giảm localDEVF1**.
- TRAIN1110/65933targettokensunknownerr≈**1.6835%**;DEV12/19678≈**0.0610%**.
  557câu affectedkhông có nghĩa557câu hoàn toànvôdụng. Không thể đổ mọi gap chounknowns.
- TRAIN7577silencetokens(~11.49%),DEV0:rawTRAINvsderivedDEVpolicyasymmetry đãxácnhận.
  `MpdStats` loạihypothesissilence; **PERonline/offlinesil mismatch không tự giải thích
  việc F1 dưới60%**, vì detection pipeline không dùng PER làmselection.
- Metadata có1963TRAIN/145DEVadjacentrepeats. OTblankless+greedycollapse/repeattarget
  policies đángaudit, nhưng chưa đoactualoutputmistakes; không tự deduplicate targets.
- AugmentationwaveformDropFreq/DropChunk/speed cộng nativeSSLmasktime. PublicSBwrapper
  mặcđịnhnativeSpecAugmentoff;currentexplicitlyon. Recipe không khớppaperwarp80/masks;
  cần đo riêng, không dùng công thức paper để mặc định augmentation đúng.
- Model cófullWavLM315453120+heads3822765,notcompact. **KhôngBN trong AM**;
  CNNfrozen21tensorsđúng,24TransformerqprojectionsđềuđổiRMS7.64e-4..1.21e-3ở3150.
  StridewordinggapvẫnOPEN, chưa có bằng chứng nó gâyplateau.
- Currentstage chỉ**CROTTCAM**. [Paper v1Table3/4a](https://arxiv.org/html/2604.22133v1)
  báoAMF162.39%;71.77%là**CROTTC-IF**,chưa phải currentpipeline.
  Khácprotocolnên không lấy62.39−54.40làmfairreplicationgap.
- Goldhypothesissanity99.7745% không100% còn9FAdoalignment/evaluatorbehavior;
  nó chỉ loại hardceiling60, không đóng F3/F7 hay chứng minh audio-modelquality.

## 5. Recommendation và next gates — không áp dụng lén

**Ưu tiên sửa/đo đường evaluation trước, không đổi backbone hoặc train thêm dài để dò60%.**

1. **Registration mới cho eval-only repair**: chỉ conditionalRoPEattentiondropout0 ởeval,
   giữ nguyêntrain-mode. CPUregressiontest repeatedsamecheckpoint/inputdeterminism,
   no weights/optimizerupdates,save/restoreRNG để diagnostic không perturb parenttraining.
   Không sửa site-packages toàn máy/remote hotpatch; không upgradepackage hàng loạt.
2. **BoundednativeDEVreplay trên Kaggle**, dùng existing verified checkpoint3150 và
   selectedbest46snapshot (sau nàybest57 nếuexportverified), **audio-only**,exact599IDs,
   saveperutterancehypotheses. So originalevalmultiplefixedrepeats vsdeterministicrepair;
   reportF1/P/R/FR/FA/typederrors/per-speaker,varianterange;khôngbestrepeatselection.
   Không chiếm GPU window đangtrain/launch duplicate production. Cần native resource/
   budget/registration trước admission; không tự chuyển local/Colab.
3. Đo**cleanTRAIN/DEVmatchedobjective** vàcomponentOT/CRloss+gradnorms trênauthorized
   fixedTRAINsample ởnativeprofilegate, không lấyaugmentedTRAINlosssoDEVF1làmoverfitproof.
4. Chỉsauđó mớiđăngkýtừngmatchedcontrol:lossbalance/augmentation/LRschedule,
   giữsamemodel/data/seed/budget;khôngbroad sweeps,tựresetcampaign hay retuneon test.
   Correctpaperprotocol cầncleanpretrainedlineage/manifestauthorizationriêng.

**Confidence:** High choevaldropoutbug/mathprobes/non-BN/integrity;Moderate cho
optimization/generalizationhypotheses;Low choattributiontừngnguồnđếnđiểmF1vàkhả năngvượt60%.

## 6. Verification/replay và giới hạn

- `MATH_PROBES.log` giữlượtđầuthấttại evaldeterminism:10passed/1failed;failure này
  chính là phát hiện bug,không bịxóađể tạo“allpass”.
- `DIAGNOSIS_TESTS.log`:**58passed,1xfailed**;xfailstrict làinvariant
  evaldeterminism **vẫnKNOWNBROKEN**,khôngđượcgọiđãfix. Independentmathchecks pass.
- `tools/diagnose_crottc_plateau.py`:log/metadataonly,fullblockmeans,8production
  implementationhasheskhớplivecontract. Không sửa8drivers/sourceupstream.
- `tools/probe_crottc_eval_dropout.py`:CPUdisposableheadsforward+weightsinspection,
  khôngnativebackbone/audio/optimizer. Inputsverified3150/nativehashes.
- Test/DEV_CALvẫnsealed;khôngcontactauthors/push/commit/productionrunmới.

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-crottc-official/bin/python -m pytest -q \
  tests/test_crottc_diagnosis.py tests/test_crottc_fidelity_audit.py \
  tests/test_crottc_resource_profile.py tests/test_crottc_validation_report.py
```

Không có native repeatedDEVscore trong report này; những gì cần futureproof đãghi rõ.
