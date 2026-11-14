# Vì sao SM13 hiện tại chưa tăng khả năng bắt lỗi?

## Kết luận

**Chưa có bằng chứng model đã chạm giới hạn khả năng tuyệt đối. Có bằng chứng rất rõ recipe hiện tại đã overfit và thêm epoch không cải thiện detection F1 bền vững.** RMS normalization xử lý được một điểm yếu của raw mixing, nhưng không giải quyết được khoảng cách fit–held. Đây không phải dấu hiệu cần tiếp tục tăng epoch.

Đối tượng: S1 native pretrained WavLM Large frozen + fresh fit-only-RMS heads, seed101/gap1101, fit1350 câu/9 speakers, held450 câu/ABA+ASI+BWC; neural-only. Held là reused TRAIN development, không phải test độc lập. F1 ở đây là event-level diagnostic @0.5, không phải paper hierarchical F1. Không so trực tiếp với F1 CROTTC.

## 1. Phép kiểm mới trong session này

- Đọc GOAL_SM13/HANDOFF, các report raw350 và paired raw/normalized50→normalized300. Lịch sử déjà vu session `01a1d97f-a6fc-737d-836d-3dfa0fb86b91` đã chẩn đoán raw350; **không dùng kết luận raw350 thay cho đo lại normalized300**.
- Process probe không thấy SM13 trainer; work STATUS là `COMPLETE_NORMALIZED300_VERIFIED`. GPU idle. Không có dấu hiệu đây là run treo/chưa đủ300. dmesg không được hệ điều hành cho đọc, nên không khẳng định đã kiểm hết kernel events.
- Chạy mới `tools/diagnose_sm13_normalized_plateau.py`: CPU-only,2threads,49.03s; không backbone forward, optimizer step, GPU, remote, threshold search hay sealed-data access.
- Rehash cache/audio/registered inputs; kiểm reference/label/event order với TRAIN manifest; strict-load **4 checkpoint** normalized target/gap tại50/300, kiểm SHA với receipt602-state đã có.
- Forward lại toàn bộ fit và held bằng eval mode; CE tính từ logits, **cùng pooled weighted estimator** ở hai population. Đối chiếu saved GPU probabilities, argmax, TP/FP/FN và held CE. Artifact được dùng có SHA không đổi sau audit.
- Kết quả `AUDIT.json`: `CPU_NORMALIZED_ENDPOINT_DIAGNOSIS_VERIFIED`. Không tuyên bố session này đã verify lại toàn bộ602states.
- Regression tests CPU: **59 passed** (`CPU_TESTS.log`), không full-repository PASS claim.

Lệnh tái chạy cần outputdirectory mới vì tool cố ý từ chối overwrite audit cũ:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
timeout 330 .venv-crottc-official/bin/python -m tools.diagnose_sm13_normalized_plateau
```

## 2. Nút thắt đã xác nhận: generalization, không phải model chưa học được tập fit

Fresh CPU inference, cùng estimator/cùng population ở hai checkpoint:

| Chỉ số | Normalized50 | Normalized300 |
|---|---:|---:|
| Pooled target+gap **fit F1@.5** |96.6907%|99.6100%|
| Pooled target+gap **held F1@.5** |40.8143%|40.8843%|
| Target fit pooled weighted CE |0.04650|0.00516|
| Target held pooled weighted CE |1.68894|3.03045|
| Target held sai với confidence≥99% |867|1290|
| Gap fit F1@.5 |98.1481%|100.0000%|
| Gap held F1@.5 |28.6792%|35.6436%|

Các head học rất tốt trên dữ liệu đã thấy, nhưng không chuyển được phần lớn năng lực ấy sang speakers held. Confidence sai tăng; target held CE tăng khoảng79.4% dù fit CE giảm~88.9%. Đây là **overfitting/overconfidence thực**, không chỉ chênh train/eval dropout hoặc cách lấy trung bình loss trong log. Các số fit CE ở bảng là fresh eval-mode pooled CE, không được thay ngược vào train-loss history vốn dùng batch/dropout estimator khác.

F1 mean theo cửa sổ10epoch:41–50 **42.1783%**,91–100 **41.8154%**,141–150 **41.7935%**,191–200 **40.7797%**,241–250 **39.7701%**,291–300 **39.6455%**. Điểm300 gần ngang50 không phải một xu hướng tăng bền vững; epoch200 tốt riêng lẻ không được lấy hậu nghiệm thay fixed300.

## 3. Vì sao AP vẫn tăng mà F1 không tăng?

Hai yếu tố khác nhau, không nên nhập thành một:

1. AP đo thứ hạng score qua các operating points; F1@.5 đo quyết định tại threshold đã khóa. Model có thể xếp hạng tốt hơn nhưng vẫn sai/overconfident tại.5. Audit **không** retune threshold để làm kết quả đẹp hơn, và chưa chứng minh calibration đơn thuần sẽ giải quyết toàn bộ giới hạn.
2. **Macro3 AP và micro detection F1 có trọng số population rất khác.** S/O/I AP50→300 tăng lần lượt **+1.3427/−0.1220/+8.0268pp**. Phần tăng AP lớn nhất là insertion, nhưng held chỉ có134 insertion positives so2216 S/O positives. S/O chiếm **94.30%** toàn bộ lỗi; target binary AP chỉ43.8872→44.8967% (+1.0095pp).

Vì thế macroAP30.7764→33.8588% không chứng minh detection tăng tương ứng. Model có một cải thiện insertion ranking thật trên population này, nhưng main target detector gần plateau.

## 4. Model cụ thể đang bỏ sót gì?

| Held ở fixed.5 | Epoch50 | Epoch300 |
|---|---:|---:|
| Substitution bắt được /1859 |643|647|
| Omission bắt được /357 |141|103|
| Insertion bắt được /134 |38|36|
| Tổng TP |822|786|
| Tổng FP |856|709|
| Tổng FN |1528|1564|

Đến300, precision tăng48.99→52.58%, nhưng recall giảm34.98→33.45%; cảnh báo ít hơn đồng thời bỏ sót nhiều lỗi hơn. Omission recall giảm39.50→28.85%, còn substitution recall gần như đứng yên34.59→34.80%.

Target có1240 lỗi bị bỏ sót **ở cả50 lẫn300**.226 lỗi từng bắt đúng ở50 thành bỏ sót ở300, chỉ192 lỗi đi chiều ngược lại. Target positives có score<.1 tăng1115→1282/2216: rất nhiều lỗi được model gán điểm rất thấp, không chỉ vài trường hợp sát ngưỡng. Insertion cũng bỏ sót92 ca ở cả hai checkpoint. Đây không phải chỉ thiếu vài epoch để phân loại xong các ca khó.

## 5. Vì sao tăng backbone/epoch chưa giải quyết?

### Sự thật implementation

- Native WavLM315M **frozen**, acoustic cache cố định; optimizer chỉ cập nhật scalar mixing và hai MLP (~1.11M params). Train300epoch không phải fine-tune WavLM300epoch và không tạo thêm speaker/phonetic diversity.
- Features đã bị nén thành target mean/std/delta và gap left/right means quanh predicted alignment; head không có quyền thay đổi backbone hoặc alignment. Đây là giới hạn của đường truyền thông tin hiện tại, **chưa phải bằng chứng causal rằng pooling/alignment quyết định trần40%**.
-9fit speakers, chỉ426 insertion/1468 omission positives ởfit; hàng chục nghìn events tương quan trong cùng câu/speaker không tương đương hàng chục nghìn speakers độc lập.
- LR giữ.001(target)/.0008(gap), không scheduler; target weightedCE[1,2.5,4], gap sqrt-inverse-frequency weights. Những lựa chọn này đã xác minh đúng recipe, **không tự là bug**, và chưa có controlled trial kết luận LR/loss là nguyên nhân gốc.

### Yếu tố có bằng chứng can thiệp

Matched fresh raw50→normalized50 cải thiện F1 **38.6507→40.8143%**, macroAP25.7442→30.7764%; all6preregisteredgates passed. Vậy layer-RMS conditioning là yếu tố có thể cải thiện trong một paired seed, không còn chỉ giả thuyết từ đo amplitude. Nhưng normalized300 vẫn overfit, nên conditioning **không phải lời giải đầy đủ**, cũng không chứng minh WavLM Large tốt hơn S0.

### Giả thuyết còn mở — không gọi root cause đã chứng minh

- Thiếu diversity/khác biệt speaker và cách biểu diễn lỗi so với corpus pretraining.
- Backbone/task mismatch: S0 dùng ASR-fine-tuned W2V2-960h, S1 native pretrained WavLM; thay này không chỉ là tăng số params.
- Pooling cố định, shared scalar mix và predicted alignment làm mất cue duration/context cần cho omission/insertion. Có18 nonmonotonic alignments lịch sử, nhưng chưa có evidence lỗi này giải thích plateau toàn bộ.
- Regularization/checkpoint policy chưa đủ để ngăn head học các quy luật chỉ đúng trênfit.
- Annotation ambiguity/ground-truth noise chưa được adjudicate độc lập. Zero mapping mismatch không chứng minh mọi annotation đúng về ngữ âm.

Class prevalence fit/held khá gần: target positive~15.62/14.96%, insertion~0.924/0.878%. Class imbalance có thật, nhưng **chỉ khác base rate giữa hai population** chưa giải thích được khoảng cách fit~100%/held~41%.

## 6. Những lời giải thích không được evidence hiện tại ủng hộ

- Chưa train đủ / optimizer không học: fit gần100%, updates và state liên tục.
- Labelshuffle/eventorder/resumereset/evaluator-countbug: prior raw audit và fresh normalized replay không thấy mismatch giải thích regression.
- Cache corruption lớn: full cache SHA hợp lệ, CPU head replay tái tạo output. Sparse strict nativeCPU/GPU feature parity failures của audit cũ vẫn là giới hạn được giữ, không tuyên bố universal parity.
- Chỉ do double waveform normalization: counterfactual trước rất nhỏ và chỉ1sample, không đủ quy trách nhiệm.
- Chỉ do insertion: target chứa94.3% positives và vẫn là main bottleneck.
- Chỉ cần chuyển.5 để chắc chắn tăng khả năng: threshold thay operating point, không tự cải thiện representation/ranking hay chứng minh generalization.

## 7. Hướng tiếp theo có thứ tự, CHƯA launch

1. **Không nối thêm epoch theo recipe cũ.** Giữ fixed300 và toàn bộ history; không lấybestepoch200 hậu nghiệm làm kết quả chính.
2. Hoàn tất TRAIN1800→commonDEV_SEARCH599/evaluator gates theoGOAL để biết phương pháp đứng ở đâu soS0. Chưa được gọi450TRAINheld là final/generalizable score hoặc soCROTTC trên đó.
3. Nếu user cấp run mới: đăng ký một đối chứng hẹp chốngoverfit (baseline normalized so với một thay đổi regularization hoặc smallerhead), sameinitialization/data/compute/threshold; checkpoint/earlystop rule khóa trước trên TRAIN-derived calibration. Một yếu tố mỗi lần, không combine LR/loss/layer/backbone trong một trial không thể giải thích.
4. Trước adaptation/backbone fine-tuning, cần discriminating audit về lỗi omission theo speaker/phone/alignment và ground-truth quality trên authorizedTRAIN. Không tự mởtest/DEV_CAL, gold timing làm inference input, corpus mới, backbone-backward hay broad sweep.

**Mức chắc chắn:** cao về overfitting/recall bottleneck và AP–F1 mismatch trong run này; vừa về normalization gain nội bộ một paired seed; chưa đủ để chọn một nguyên nhân gốc duy nhất hoặc tuyên bố model không thể cải thiện nữa. Không có training mới hay tác động CROTTC trong investigation này.

## Evidence map

- `AUDIT.json`: fresh CPU endpoint replays, matchedfit/heldCE, classcounts/FNs/transitions, checkpoint/prediction SHA.
- `CPU_TESTS.log`:59passed.
- `../SM13_RAW_NORMALIZED_20261114/FINAL_VERIFICATION.json`: prior602stateverification, fixedendpoints/per-speaker/AP.
- `../SM13_RAW_NORMALIZED_20261114/PILOT_VERIFICATION.json`: prior204pilotstates/gates/rawhistoricalparity.
- `../SM13_RAW_NORMALIZED_20261114/NORMALIZED300_REPORT.json`: fullcurve.
- `../SM13_S1_DEBUG_20261114/REPORT_VI.md`: raw350audit, nativefeatureparitylimits, label/order/resumechecks.
