# SM13 plateau/overfit: có thực sự do dropout không?

## Verdict

**Chưa có bằng chứng dropout là nguyên nhân gốc.** Lựa chọn tăngdropout ở bước trước là một giả thuyết regularization dễ tách biến, không phải kết luận chẩn đoán. Không nên gọi nó là fix đã xác định hoặc hứa tăngF1.

Đã xác nhận overfitting/overconfidence trên normalizedSM13. **Không thấy lỗi runtime dropout** trong đường head hiện tại. Độ mạnh dropout có đủ hay không vẫn chưa được kiểm chứng bằng paired refit. Pilot ấy chưa chạy và không được coi preflight/smoke là quality evidence.

## Phép kiểm mới

`tools/debug_sm13_dropout_causality.py`, localCPU-only/2threads/22.74s. Nooptimizer/backward/GPU/backboneforward/thresholdsearch/testDEV_CAL/remote. Không sửa recipe/head/checkpoint cũ.

- Hash production source khớp registration gốc; cache/audio/input hashes được đọc lại.
- Strict-load trained normalizedtarget/gap checkpoints50/300, SHA khớp prior602-state verification.
- Với mỗi head/epoch: lấy256fit và256held events cách đều theoeventorder, không chọn bằngnhãn/predictions; chạy repeated-forward controls.
- Forward lại toàn bộ held; savedGPUprobabilitymaxdelta≤1e-5/argmaxsame, pooledweightedCE khớp recordedheldloss.
- Tensors/checkpoint/predictionSHA khôngđổi; test xác nhận mode/p/RNG được khôi phục saucontrols.
- `AUDIT.json`: **DROPOUT_RUNTIME_CONTROLS_VERIFIED_NO_CAUSAL_REFIT**. `CPU_TESTS.log`: **62passed**, gồm3tests mới; khôngfullrepoPASS.

## 1. Dropout đang hoạt động đúng, không bị bỏ quên

Code `epoch50.fit_head` gọi `model.train()` mỗi epoch trước batch/backward; `evaluate` gọi `model.eval()`. Heads dùng PyTorch `nn.Dropout`, không phải attentionSDPA dropout của nhánhCROTTC. Wrapper RMS không override `.train()` hoặc `.eval()`.

| Trained head | p đã đăng ký | Tỷ lệ activation bị dropout đo ởtrainmode* |
|---|---:|---:|
| Target first hidden |25%|24.54–25.12%|
| Target second hidden |20%|20.05–20.16%|
| Gap hidden |15%|14.84–14.94%|

*Dải trên4sampleprobes/head gồmfit/held tại50/300. Chỉđếm activation input≠0 để không nhầm ReLU/GELU zero với dropoutmask. Không phải replay lịch sử từng mask của trainingGPU; source/recipe đãpin vàtrainedhead kiểmhiệntại phùhợpcontract.

Các controls đềuđạt trên cả4trainedheads/2populations:

1. Eval forward lặp lại **bằng nhau chính xác**: không có dropout leak vào head evaluation.
2. Trainmode forward stochastic, maskrate khớp p: dropout không bị vôhiệu hóa trong đường head được kiểm.
3. Trainmode với p=0 **bằng evaloutput chính xác**: không có BatchNorm/runningstats hoặc mode-dependent path khác giải thích khác biệt tronghead này.
4. Tăngp gấp2 **sau khi đãtrain**, vẫn đểeval, output **không đổi**: đổi configdropout lúcinference không thể sửa model đã học. Muốn đo regularization effect phải freshrefit.

Do đó **không có bằng chứng bug thiếu/bật sai dropout giải thíchplateau**. Điều này không đồng nghĩa p hiện tại là tối ưu hoặc đủregularization.

## 2. Những gì thực sự được evidence ủng hộ

### 2.1 Model fit rất tốt nhưng sai tự tin trên held speakers

Fresh toànfit/heldCPUaudit trước, cùngevalmode/pooledweightedCEestimator:

| Metric | Normalized50 | Normalized300 |
|---|---:|---:|
| PooledfitdetectionF1@.5 |96.6907%|99.6100%|
| PooledhelddetectionF1@.5 |40.8143%|40.8843%|
| TargetfitweightedCE |0.04650|0.00516|
| TargetheldweightedCE |1.68894|3.03045|
| Targetheld sai, confidence≥99% |867|1290|

Overfitting khôngbiếnmất khi đo fit/held ởcùngmode/estimator. Khôngphải chỉ trainloss dùngdropout cònheldloss khôngdùngdropout nên nhìnnhầm.

### 2.2 PhầnheldCE xấuđi tập trung ở trueerror S/O

Phép mới tính CE từlogits, không probabilityfloor:

| Goldclass, held | Count | MeanCE50 | MeanCE300 |
|---|---:|---:|---:|
| Correct |12600|0.1721|0.2067|
| Substitution |1859|4.5968|7.9076|
| Omission |357|5.6086|12.0728|
| Insertion (gaphead) |134|7.5234|8.5959|

S/O đónggóp khoảng98.3% phần targetpooledweightedCE tăng50→300. Model không chỉ bị lỗi tínhF1 hoặc score sátthreshold: nó gán probabilitygold rất thấp cho nhiều lỗiheld. CE lớn không phải sốlượnglỗi; nó cũng phạt độ tự tin sai.

### 2.3 Training dài thêm làmfit tốt hơn, không tạo heldtrend tăng

Descriptivewindows đãcốđịnh trướcphân tích trongtool, khôngcheckpointselection:

| Window | MeanheldF1@.5 | Meanmacro3AP |
|---|---:|---:|
|1–10|40.4601%|26.1714%|
|11–20|44.4078%|29.9441%|
|21–30|43.7830%|30.4074%|
|41–50|42.1783%|29.6480%|
|291–300|39.6455%|33.4927%|

Dấuhiệu F1 đãxấuđi có trước300; khôngchỉdo mộtđiểmcuối xấu. Khônglấywindow11–20 làmquyền thayendpoint đãđăngký hoặcchọnđượcearlystoprule hậu nghiệm.

LR tại50và300 vẫn.001target/.0008gap; schedulerNone; weightdecay.01/.0001. Targetfirstlinearweightnorm41.42→75.82/last3.66→6.31; gapfirst22.44→44.58. Scalar scale đồngthời giảm, nên **không được dùng weightnorm riêng để suy effectiveacousticamplification hoặc causalproof**. Đây là diễnbiến đãđo, khôngchứngminh chỉcầnLRdecay/weightdecay/dropout sẽsửađược.

### 2.4 AP/F1 mismatch không phải riêngdropout

Đãverify: macroAPgain50→300 phầnlớnởinsertion(+8.0268pp), trongkhi S/O94.30%positives; substitutionAP+1.3427/omissionAP−.1220pp. F1@.5 phụthuộc cảrecall/precision tại mộtoperatingpoint, khácmacroAP cả vềranking vàclassweighting. TổngTP822→786 vàFP856→709: ítfalsewarnings hơn nhưngíttrueerrors đượcbắt hơn.

## 3. Giả thuyết cạnh tranh, không chốt bừa một nguyên nhân

| Giải thích | Evidence hiện tại | Verdict |
|---|---|---|
| Dropout bịtắt trongheadtrain hoặc bậttrongeval | Source/mode/mask controls đúng; evaldeterministic | Khôngđượcủnghộ trongđườngđãkiểm |
| p hiện tại quáthấp | Overfit dùdropout đanghoạtđộng; chưađốichứngrefit | **Chưa biết** |
| Traindài/selection khôngngănmemorization | Fit tănggần100%, heldconfidence/loss xấu, F1windowgiảm | Overfittrajectory xácnhận; chưaisolateoptimalepoch/cause |
| Capacity vs effective independentdata | ~1.11Mheadparams,9fitspeakers; events tươngquan,426fitinsertions/1468omissions | Plausible, khôngcausalproof chỉtừparam/samplecounts |
| Speaker/taskrepresentation mismatch | Fit/heldspeakerdisjoint, verylargegeneralizationgap; nativebackbonefrozen | Plausible; chưaisolate speaker/accent/phone/context |
| Fixedpooling/predictedalignment bỏmấtcues | Backbone/cache/alignment khônghọc theoheadepochs;18nonmonotonicIDs lịch sử | Plausible; chưađogiảithíchđượctoànplateau |
| Label/order/resume/evaluatorbug | Priorlabel/order/fullstatereplay + freshheadmetricreplay khôngmismatch | Chưathấybuggiảithíchregression; annotationtruthvẫnchưaadjudicate |
| Classimbalance/base-rate alone | Fit/heldpositiveprevalence gầnnhau; fitpositiveF1rấtcao | Imbalancecóthật, nhưngkhôngđủđểquymọilỗi |

**Mô tả cơ chế phù hợp nhất:** recipe tiếp tục tối ưu CE trên biểu diễn cố định của một populationfit hẹp; head có thể khớp/gánscore cựcđoan cho fit nhưng ranhgiới ấy không chuyển tốt sang speakersheld. Đó là overfitting dưới protocol này. Vìsao population khôngchuyển tốt (data/representation/label/alignment/capacity/regularization) chưa được phânrã đủ để chọn một rootcause duy nhất.

## 4. Có nên tăngdropout không?

Có thể thử nhưmột **intervention cóđốichứng**, khôngphảikếtluậnchẩnđoán. Dropout mạnhhơn cũng cóthể làm mất cuehiếm/underfit; khôngcóbằngchứng.5/.4/.3 là tốiưu.

Muốn phátbiểu dropout cógiúp trongrun này: freshnormalizedbaseline vs2xdropoutcandidate, sameinitialweights/seed/batchorder/epochbudget/loss/LR/threshold, so fixed50/fullcurve/fit-vs-held/cácsafeguards đãđăngký. Khôngchỉ sửa p củacheckpoint300 hoặc bậtMCdropout rồi gọilà refitđãthànhcông. NếuPASS cũng chỉ là mộtpairedseed/reusedTRAINdevelopment, khôngchứngminhdropout lànguyênnhângốcduynhất hoặctựđộnggiúpDEV_SEARCH/test.

Pilot ở`../SM13_DROPOUT_VERIFY_20261114/` vẫn **PREFLIGHT ONLY**, không training. Currentrootdisk probe<14GiBCROTTCreserve; không bypassfloor/chuyểnCPUtrainingđểláchgate/delete/stopCROTTC. Investigation hiện tại chỉ nhỏgọnCPUinference vàaudit, không phải productionrun.

## Evidence

- `AUDIT.json`:8mode/maskcontrolprobes,4trainedheads,fullheldCEreplay,SHA/sourcechecks.
- `CPU_TESTS.log`:62passed.
- `../SM13_NORMALIZED_DIAGNOSIS_20261114/AUDIT.json`:freshallfit/heldF1/CE/confidence.
- `../SM13_RAW_NORMALIZED_20261114/FINAL_VERIFICATION.json`:prior602states/fixedendpoints.
- `../SM13_RAW_NORMALIZED_20261114/NORMALIZED300_REPORT.json`:fullcurve.
- `../SM13_DROPOUT_VERIFY_20261114/PREFLIGHT.json`:construction/resourcecheck ONLY, noqualityoutcomes.
