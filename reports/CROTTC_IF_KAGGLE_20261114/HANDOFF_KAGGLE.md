# HANDOFF — chuyển full CROTTC-IF từ local sang Kaggle

Cập nhật: **2026-11-14, Asia/Ho_Chi_Minh**. Workspace:
`/home/khoa/Projects/New-Method-Research`.
Nhánh: `feat/crottc-if-official-20261113`, Git HEAD tại bàn giao:
`879edc2ea5f85ce4ba8f56a97ceb49b4f83cb8a0`.

## 1. Quyết định mới nhất và thứ tự ưu tiên

**Quyết định có hiệu lực: `CROTTC-IF-FULL-KAGGLE-20261114-v1`.**
Người dùng yêu cầu: “thôi, tôi nghĩ nên chuyển lên kaggle để chạy full, tôi có
kaggle cli. Đây là quyết định mới, cần ghi lại lịch sử phiên train, quyết điện
thay đổi lên kaggle vào HANDOFF và GOAL để tôi switch qua session mới, yêu cầu
HANDOFF phải chi tiết nhé”.

- Chuyển full training/reproduction sang **Kaggle**. Local giữ vai trò chuẩn bị,
  phát triển, test, audit, phân tích, lưu checkpoint và log lịch sử.
- Quyết định này thay hướng “hoàn toàn local” ngày 2026-11-13 và “ưu tiên GPU,
  thiếu VRAM dùng RAM” ngày 2026-11-14 về nơi chạy tiếp. Những quyết định cũ vẫn
  là lịch sử hợp lệ của các run đã thực hiện, không phải lệnh tự resume local.
- Job train local đã dừng sạch và checkpoint được xác minh; **không tự khởi chạy
  lại local**. Không có Kaggle training/kernel/dataset mới được tạo trong bước
  bàn giao này. Session mới tiếp nhận việc chuẩn bị rồi thực thi trên Kaggle.
- Giữ nguyên phạm vi full model/full schedule/toàn bộ coverage paper. Không
  đổi thành compact pilot hoặc gọi một AM run là “full paper reproduced”.
- Official test vẫn SEALED. Chuyển nền tảng không cấp quyền mở test, đổi split
  hoặc upload toàn bộ archive có test. Không provisioning Colab như fallback.
- `GOAL.md` vẫn là mục tiêu nghiên cứu duy nhất: xây baseline CROTTC-IF rồi phát
  triển một phương pháp cải tiến cho chẩn đoán và phản hồi. Chưa hoàn tất nghiên cứu.

Quyết định, snapshot GOAL/HANDOFF/AGENTS trước thay đổi và bằng chứng bàn giao ở
`reports/CROTTC_IF_KAGGLE_20261114/`. Đọc `DECISION.json`,
`LOCAL_STOP_VERIFICATION.json`, `LOCAL_SESSION_HISTORY.json`,
`KAGGLE_CLI_READINESS.json`. HANDOFF trước chuyển Kaggle được giữ nguyên tại
`reports/CROTTC_IF_KAGGLE_20261114/HANDOFF_BEFORE_KAGGLE.md`; các dòng “ACTIVE local”
hoặc “IF chưa triển khai” trong snapshot đó chỉ phản ánh các mốc lịch sử.

## 2. Trạng thái thật tại điểm dừng local

Đã gửi **SIGTERM**, trainer nhận signal, dừng ở ranh giới microbatch và ghi
atomic checkpoint đủ partial gradient. Không SIGKILL, không bỏ batch lỗi.

- PID train cuối 324148, exec session 69438: đã kết thúc, kiểm `/proc` không còn.
- Shared GPU lock đã kiểm có thể acquire độc quyền:
  `/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock`.
- Status cuối: `WINDOW_EXHAUSTED_RESUMABLE`, không phải training failure hoặc
  hoàn tất lịch300 epoch. Lý do dừng thực tế: yêu cầu chuyển Kaggle của người dùng;
  chuỗi status này cũng được trainer dùng cho signal stop.
- `RUNNING_WINDOW.json` và `EXECUTION_CONTROL.json` dưới experiment root đã đổi
  thành `LOCAL_STOPPED_FOR_KAGGLE_SESSION_HANDOFF`, `local_auto_resume_authorized=false`.
  PID/session trong manifest được giữ làm lịch sử, không còn là job live.
- Full native WavLM Large315,453,120 tham số; SSL trainable311,246,528;
  acoustic heads3,822,765. Native CNN frozen; không phải toàn SSL frozen.
- Đã xong **2/300 epoch**. Internal `epoch=2` nghĩa là đang epoch 3.
  `updates=124`, `cursor=320`, pending7/32 microbatch: đã xử lý327 câu của epoch 3,
  nhưng mới320 câu thuộc các optimizer update hoàn tất trong epoch đó.

### Checkpoint chính phải giữ và chuyển

`experiments/crottc_if_full_local_20261113/am_crottc/last.pt`

- SHA-256: `d060833722f1898800d00778c05e39df636c3aecf3584723a3ae0dfe0a71ee5c`.
- Bytes: **5,058,796,894** (~4.71GiB). Lớn hơn checkpoint3.8GB thông thường vì
  chứa thêm khoảng1.26GB gradient RAM của batch chưa xong.
- `epoch=2`, `cursor=320`, `updates=124`.
- Accumulation: `completed=7`, `group_size=32`, `loss_sum=32.19321870803833`.
  Nếu tiếp tục cùng recipe, bắt đầu **microbatch 8**, không lặp lại 7 câu đầu hoặc
  xóa gradient. Không lấy `best.weights.pt` thay full resumable state.
- 537 Adam parameter states đều `step=124`; scheduler `last_epoch=124`.
- Có SSL/head weights, optimizer, scheduler, disabled scaler FP32, RNG
  Torch/CUDA/Python/NumPy, named gradients, cursor, objective, provenance,
  execution contract. Gradient inventory: SSL 467 tensors/1,244,986,112bytes;
  heads 70 tensors/15,291,060bytes.
- Đã hash khớp `CHECKPOINT_RECEIPT.json`, load CPU bằng
  `torch.load(..., weights_only=True, mmap=True)`; metadata/status/receipt khớp.
  Năm hash implementation đang train giữ nguyên.
- Data/source/native và toàn bộ2399 authorized WAV đã verify lại bằng
  `full_train.verify_inputs()` sau dừng, khớp provenance trong checkpoint.

Selected weights chỉ phục vụ model selection:
`experiments/crottc_if_full_local_20261113/am_crottc/best.weights.pt`.
Được chọn ở human epoch 2 (`epoch=1`, `cursor=1800`, `updates=114`),
source F1 `0.46682054241200227`. SHA-256: `ddddc065bdc13c022b5f86f21c7ced2d6afcb32d721b034114ac4c8c266caff0`;
kích thước 1,277,328,598 bytes. Metadata đầy đủ ở `LOCAL_STOP_VERIFICATION.json`.
File này **không có Adam/RNG/partial gradients để resume đầy đủ**.

Checkpoint profile ban đầu vẫn nguyên tại `am_crottc_profile/last.pt`, SHA
`f4c97a353ce392e166c60965b68aa49868cab598d169ff218bd5626ddde87bd7`,
update 1/cursor 32/internal epoch 0. Đã verify lại SHA lúc bàn giao. Không resume
nhầm profile này thay checkpoint production update 124.

## 3. Lịch sử train của phiên này

Các mốc theo Asia/Ho_Chi_Minh; JSON `LOCAL_SESSION_HISTORY.json` chứa giờ bắt đầu,
giờ kết thúc, PID, cursor, pending gradients và SHA của các log.

1. Ban đầu có native profile thật trên 32 câu ngắn nhất, một Adam update,
   WavLM/Adam trên CPU, head/loss CUDA. Profile này chứng minh backward/update,
   chưa đại diện câu dài; không dùng nó làm ETA toàn campaign.
2. Sửa strict resume: `--resume`/`--resume-from` không âm thầm reset khi thiếu
   checkpoint; chỉ nhận legacy migration cho đúng SHA profile đã đăng ký.
   Bổ sung RNG/partial-gradient checkpoint, checkpoint mỗi5update và clean stop.
3. Numerical continuation tests4passed trên model nhỏ CPU/CUDA. Profile đại
   diện 32 câu:16 median neighbors+8 p95 neighbors+8 longest, max9.054s, một scratch
   update; input checkpoint không thay đổi. Peak RSS~9.89GiB ở CPU backend.
4. CPU production tiếp tục qua nhiều cửa sổ clean stop: update 1→17→19→20→22,
   giữ pending gradients khi ngắt để profile GPU. Các log WINDOW001–005 giữ nguyên.
5. Bốn profile full-GPU trước chưa offload gradient sau từng microbatch đều CUDA
   OOM, **zero scratch optimizer updates**: CPU Adam đơn thuần, thêm activation
   offload, sửa frozen-CNN waveform autograd, rồi AMP. Đây là thất bại của các
   cấu hình đó, không phải bằng chứng full model không thể chạy trên local.
6. Profile thứ5 thành công: toàn SSL/head FP32 trên CUDA, Adam/master weights,
   accumulated gradients và saved activations trong CPU RAM. Sau từng microbatch
   copy/add gradient vào RAM và set GPU `p.grad=None`. Peak CUDA allocated
   2,696,370,688bytes (~2.51GiB), RSS8,588,914,688bytes (~8GiB), đủ 32 câu/một update.
7. CPU→GPU migration có receipt riêng: update 22/cursor 704/pending 3 →
   update 23/cursor 736. Sau đó native GPU clean-stop giữ 12/32 gradient, resume từ
   microbatch 13; đã chứng minh optimizer/scheduler/RNG tiếp tục, không train lại.
8. GPU/RAM sustained window chạy khoảng3902 s (~65 phút), từ update 23 đến 124,
   xong epoch 1 và 2, vào epoch 3. Dừng sạch theo quyết định chuyển Kaggle.

### Các attempt production, không tính scratch profile

| PID | Bắt đầu (local) | Updates đầu→cuối | Cursor cuối | Pending cuối | Kết thúc |
|---|---|---:|---:|---:|---|
|246890|2026-11-13T23:12:45.951669+07:00|1→1|32|23/32|WINDOW_EXHAUSTED_RESUMABLE|
|248779|2026-11-13T23:14:21.836931+07:00|1→17|544|28/32|WINDOW_EXHAUSTED_RESUMABLE|
|289349|2026-11-13T23:47:26.786790+07:00|17→19|608|19/32|WINDOW_EXHAUSTED_RESUMABLE|
|295868|2026-11-13T23:53:01.732195+07:00|19→20|640|3/32|WINDOW_EXHAUSTED_RESUMABLE|
|300886|2026-11-13T23:57:18.838845+07:00|20→20|640|17/32|WINDOW_EXHAUSTED_RESUMABLE|
|306911|2026-11-14T00:02:36.639216+07:00|20→22|704|3/32|WINDOW_EXHAUSTED_RESUMABLE|
|320568|2026-11-14T00:14:45.398879+07:00|22→23|736|0/0|UPDATE_WINDOW_COMPLETE_RESUMABLE|
|322697|2026-11-14T00:16:25.338587+07:00|23→23|736|12/32|WINDOW_EXHAUSTED_RESUMABLE|
|324148|2026-11-14T00:17:40.938555+07:00|23→124|320|7/32|WINDOW_EXHAUSTED_RESUMABLE|

### Validation đã đo

Mỗi epoch decode audio-only đủ599 DEV_SEARCH; gold chỉ được dùng tính metric
sau decode. Log AM cũ chỉ lưu aggregate, không lưu hypothesis từng câu để rescore.

| Epoch hoàn tất | Updates | Source new hierarchical F1 | Source old F1 | PER (%) |
|---|---:|---:|---:|---:|
|1|57|40.0283%|23.0536%|65.9569|
|2|114|46.6821%|26.1382%|52.0276|

Đây là **local development** với annotation policy/split khác paper; không phải
strict typed S/O/I, không chứng minh acoustic diagnosis/feedback readiness.
Old/new source metric có định nghĩa khác nhau, giữ cả hai; không chọn định nghĩa
đẹp hơn để claim gain. Paper F1=71.77% vẫn là kết quả tác giả, chưa được tái lập.
Bằng chứng: `AM_EPOCH_001_VERIFICATION.json`, `AM_EPOCH_002_VERIFICATION.json`
trong `reports/CROTTC_IF_FULL_LOCAL_20261113/`.

### Vì sao đổi nơi chạy

GTX 1650 4GB + RAM31 GiB/swap35 GiB thực sự chạy full model được, nhưng phải offload
nhiều. Snapshot khoảng 1 giờ cho~38.7 s/update kể cả validation/checkpoint;
AM CROTTC 300 epoch riêng ước~7.7 ngày nếu giữ tốc độ. Đây là ước lượng từ cửa sổ
ngắn, không bảo đảm hiệu năng nhiều ngày. Chưa đo tốc độ native AM CTC/IF nên
không gán cùng ETA; toàn campaign có thể nhiều tuần. Chi tiết
`reports/CROTTC_IF_FULL_LOCAL_20261113/GPU_RAM_COMPUTE_PROJECTION.json`.
Người dùng quyết định chuyển Kaggle, không phải trainer chứng minh “local bất khả thi”.

## 4. Dữ liệu, source và môi trường có thể tái sử dụng

Experiment root cũ: `experiments/crottc_if_full_local_20261113/`.
Không đổi tên/xóa để giả vờ toàn bộ run đã chạy trên Kaggle.

- Source `Secondtonumb/IF-MDD`, commit
  `cca150ec4ff00e01d61a613092d1d178e661dfb2`, bản pin ở `upstream/`.
  Không có `.git` trong bản sao; `git -C upstream rev-parse HEAD` có thể trả
  HEAD repo cha. Dùng `SOURCE_RECEIPT.json` và hash từng file.
- Native `microsoft/wavlm-large`, revision
  `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, đã tải tại `native_wavlm_large/`.
  `pytorch_model.bin` SHA
  `fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d`.
  Không dùng released L2-finetuned checkpoint làm native initialization.
- TRAIN 1800/12 speakers: ABA, ASI, BWC, EBVS, HJK, HKK, HQTV, LXC, NJS, PNV,
  SKA, SVBI. DEV_SEARCH 599/4: NCC, RRBI, YBAA, YDCK. Không fitting DEV_CAL.
- Prepared `train.jsonl` SHA
  `8db52b5dcc86ca0c0f6854a2ffc920ec44e5bf506f8414fd8006dcb9fa4cd3cc`;
  `dev_search.jsonl` SHA
  `41c26dc0afa58a515d543a1043bc67e5525fe576c9a9d52e041c71c6a25773bc`.
- Source manifests và archive hashes/counts/label policy đầy đủ trong
  `DATA_RECEIPT.json`; source manifest paths
  `data/derived/l2arctic/manifests/train.jsonl` và `dev_search.jsonl`.
- Authorized WAV ở `data/derived/l2arctic/wavs_16k/`; package Kaggle phải lấy
  **allowlist IDs từ đúng hai prepared manifests**, không copy cả thư mục vì có
  thể chứa population khác. Mỗi row có `audio_sha256`; train annotation provenance
  có `source_member`, `archive_sha256`, `annotation_sha256`.
- Raw archive local phục vụ lịch sử chuẩn bị:
  `experiments/sm13_three_directions_20261113/data_expansion/raw_archives/`.
  Không upload full archive hoặc original Drive zip. Chỉ stage train/validation
  được phép; không mở/extract/stage official test audio hoặc labels.
- Local sealed speakers: ZHAA, TNI, MBMPS, TLV. Paper test còn có NJS/TXHC/YKWK;
  NJS đang ở local TRAIN, nên không thể gọi current checkpoint là paper-split clean.
- `.venv-crottc-official/bin/python`: Python3.11, torch2.6.0+cu124,
  torchaudio2.6, SpeechBrain1.0.3, transformers4.57.0, numpy2.4.6, numba0.68.1;
  tokenizers0.22.2, huggingface-hub0.36.0, wandb0.21.4.
  Base `.venv` dùng transformers5 không phù hợp source này. Freeze tham khảo
  `experiments/crottc_if_official_20261113/RUNTIME_FREEZE.txt`; effective versions
  trong checkpoint execution contract mới là bằng chứng của run local.
  Không copy nguyên venv local sang Kaggle; dựng môi trường và ghi versions thực tế.

## 5. Code đã có và phần chưa được chứng minh

| Thành phần | Trạng thái tại bàn giao |
|---|---|
|`full_train.py`, `checkpointing.py`|CPU backend, strict resume, full state/partial gradient, atomic save; đã thực thi native training.|
|`cpu_adam_offload.py`, `gpu_train.py`|GPU/RAM native AM backend; đã chạy production đến update 124.|
|`placement_migration.py`|Migration CPU→GPU đã dùng đúng một lần; receipt cũ không dùng cho Kaggle.|
|`if_components.py`|Full source IF decoder/teacher15,965,361 params, 384 dim/2 layers, C/S/D/I và audio-only beam decode ; 7 component tests.|
|`stage_contract.py`|Strict completed-parent gates,300/300/200/200, hash/validation/selected weights checks ; 5 tests.|
|`if_train.py`|Đã triển khai IF CTC initialization và CROTTC transfer, GPU/RAM, partial resume, atomic checkpoint, audio-only validation ; 13 trainer+gate tests. **Chưa native IF resource profile/training.**|
|`validation_report.py`|Offline DEV scoring/report helper và tests; không được import vào AM driver đang chạy để thay evaluator giữa campaign.|
|`local_storage.py`|11 storage tests, root private đã provision và verify; **chưa production relocation**.|
|Kaggle adapter/package/notebook|**Chưa tạo, chưa upload, chưa launch**; session mới phải triển khai và kiểm chứng.|
|Automatic next-local-window controller|Mới có ý định; chưa tạo, không có background auto-resume controller.|

Năm code hash đã pin trong local checkpoint:

| File | SHA-256 |
|---|---|
|`full_train.py`|`dacbd50baeec9910700653a1fdc726cd66b1af5338da89a85abf859760f88450`|
|`checkpointing.py`|`fa886cb24816fb03520788249f20accae967874b832363e50ec55447ca7076e8`|
|`gpu_train.py`|`bf18c682e3811e1c5c4a63799dcc3d448cbc3fe5a8d97af9b393c3252f52bd50`|
|`cpu_adam_offload.py`|`4f43514c8ab026f27514c9bf912db19a53fe818281367be2f7ab6876f130ead1`|
|`placement_migration.py`|`176c8f339af7507457ba5cf76c31ef0699a0facdb95169b88fe29dff53ea6d57`|

Không sửa hash trong checkpoint hoặc receipt cũ cho khớp code mới. Kaggle adapter,
đổi package/device placement/path resolution cần migration receipt mới, hash input
checkpoint/config/runtime cũ và mới, cùng kiểm tra giữ nguyên weights/Adam/cursor/RNG/
partial gradients. GPU khác hoặc runtime khác không được claim bitwise-equivalent
chỉ vì load thành công. Không âm thầm bỏ 7 pending gradients để chạy batch 32 mới.

## 6. Lịch full stages và fidelity còn thiếu

| Stage | Target đăng ký | Dependency | Đã đạt |
|---|---:|---|---|
|AM CROTTC|300 epoch|Native WavLM Large|2epoch +320 committed/7pending TRAIN rows của epoch 3;124updates|
|AM CTC|300 epoch|Native WavLM Large|Chưa train|
|IF CTC initialization|200 epoch provisional|AM CTC 300 hoàn tất|Trainer đã triển khai; chưa profile/train|
|IF CROTTC transfer|200 epoch provisional|AM CROTTC 300 + IF CTC 200 hoàn tất|Trainer đã triển khai; chưa profile/train|

- `stage_contract.py` không nhận live/profile checkpoint làm completed parent.
  IF CTC: train SSL+AM heads+IF (native CNN frozen); IF CROTTC: thay toàn AM bằng
  selected CROTTC AM, chỉ chuyển decoder/teacher từ IF CTC, freeze/eval toàn AM,
  fresh IF optimizer. Không bypass gates bằng AM2epoch.
- TRAIN giữ raw silence; starred→`err`; AX→AH; malformed er)/d_/z_/AH)→`err`;
  bỏ deletion khỏi target.557 TRAIN rows có1110 `err` tokens. Không suy diễn lại
  exact realized phone từ canonical. DEV derived thiếu raw silence/star,12 `err`.
- Exact paper split2429/268/900 khác local1800/599. Author preprocessing script/
  frozen manifests chưa có trong pin; preprocessing parity UNVERIFIED.
- AM dùng full WavLM + Conformer 384/2 layer, kernel override31→3 theo paper;
  paper Eq5/6 detached symmetric KL. Waveform DropFreq/DropChunk+speed và native
  SSL SpecAugment chưa khớp paper timewarp 80/ratio masks. `objective=paper` không
  đồng nghĩa toàn recipe fully faithful. Không đổi augmentation/objective giữa
  resumed campaign mà không tách/đăng ký thay đổi.
- Microbatch1 accumulation32 không có cùng BatchNorm statistics như real batch 32.
- Source IF có alias fusion branches, thiếu acoustic Conformer registration,
  binary teacher labels dù head4classes, ConPCO ngoài paper. Local components
  đã tách branches, nối full AM memory, C/S/D/I, bỏ ConPCO theo registered choice.
- Source LM decoder thực tế regular MHA + positional embedding, teacher RelPosMHA;
  không gọi fully RoPE chỉ vì tên YAML. Paper omega1 .3 ở Eq18 vs .5 implementation;
  local registered .5, omega2=1, omega3=10.200 epochs mỗi IF stage là provisional vì
  paper không chia rõ allocation; chưa gọi là đã giải quyết ý định tác giả.
- `PAPER_COVERAGE.csv` trong report local theo dõi main/baselines/ablations,
  LLM variants, SO762, ERJ, Iqra, figures. Các phần chưa có bằng chứng vẫn
  `NOT_REPRODUCED`; chuyển Kaggle không tự hoàn thành chúng.

## 7. Kaggle CLI đã kiểm tra và việc session mới cần làm

CLI có sẵn: `/home/khoa/.local/bin/kaggle`, **2.2.5**, user **minkhoaa**, OAuth.
`kaggle kernels list --mine --page-size 1 --csv` chạy thành công (exit0), đã xác
thực read-only, không chỉ kiểm binary tồn tại. Không in/copy/upload credential.
Chưa xác minh accelerator được cấp, quota còn lại, notebook limits, internet và
storage thực tế của tài khoản. Không hứa GPU cụ thể hoặc300 epoch trong một session.

**Thứ tự triển khai tiếp theo:**

1. Đọc AGENTS/GOAL/HANDOFF và decision mới, xác minh local checkpoint SHA124,
   đủ 7 gradients, local job đã terminal. Không tự resume local.
2. Khảo sát Kaggle runtime/accelerator/quota thực tế bằng CLI/UI phù hợp; chọn
   budget từng window trước optimizer update, chừa thời gian checkpoint và
   xuất output trước giới hạn nền tảng. Giữ target300/300/200/200.
3. Tạo package private tối thiểu từ allowlist 2399 WAV + authorized supervision,
   manifest/hash/speaker allowlist/denylist. Stage riêng source pin, code/config,
   native weights và full checkpoint/selected weights/receipts/logs cần thiết.
   Không upload nguyên repo, full data tree, raw zip hoặc credentials. Kiểm SHA
   cả trước upload và sau mount. Tạo một lần/version, tái sử dụng qua các window.
4. Triển khai adapter cho `/kaggle/input` và writable work/output. Current driver
   **chưa portable nguyên trạng**: prepared rows có WAV path tuyệt đối `/home/khoa/...`,
   source receipt chứa path tương đối repo, native/source roots được suy ra từ
   `ROOT/OUT`, GPU flock dùng absolute laptop path, threads/preflight/placement
   theo local. Nếu rewrite manifest, ghi mapping old/new hashes và chứng minh
   IDs/audio/labels/order không đổi; không sửa receipt cũ rồi gọi same provenance.
5. Chọn resume-compatible path/placement trước. Nếu tận dụng VRAM lớn hơn để đổi
   Adam placement, precision, microbatch/BatchNorm hoặc augmentation, đăng ký rõ
   migration/recipe mới và giữ historical local run riêng; không tự reset native,
   giảm epoch hoặc che mất124updates. Preserve partial group trước mọi thay đổi
   không tương thích. Hạn chế claim numerical equivalence giữa hardware/runtime.
6. Profile scratch native full model đủ 32 median/p95/max, real forward/backward/
   Adam, đo VRAM/RAM/disk/time. Profile không được tự promote thành production.
   Kiểm load/resume ở Kaggle với full optimizer/scheduler/RNG/partial gradients,
   tiếp tục microbatch 8, save/download rồi verify hash/load local. Không gọi chỉ
   load weight hay inference smoke là full training validation.
7. Khởi chạy AM CROTTC continuation bằng Kaggle CLI khi adapter/profile/provenance
   đã xác minh; lưu kernel slug/version, dataset versions, accelerator/runtime,
   log, checkpoints và status. CLI auth đã có; không hỏi lại user quyền chuyển
   Kaggle vốn đã được quyết định rõ ở phiên này.
8. Mỗi window phải có checkpoint atomic, nonfinite/data/resource stop, log thất
   bại, full resume state; lấy output về và verify trước restart. Không giả định
   filesystem của một runtime còn nguyên khi runtime kết thúc. Không có output
   verified thì không claim checkpoint đã persist.
9. Tiếp tục AM CTC/IF theo dependency gates, validate DEV-only/audio-only, rồi
   thực hiện coverage paper còn lại trong quyền dữ liệu. Tách paper MDD metric,
   strict S/O/I và feedback thesis. Chỉ chuyển cải tiến chính sau baseline đủ chứng cứ.

Tài liệu CLI chính thức đã đối chiếu ngày 2026-11-14:
[Kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
[kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
[dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).
CLI `push` sẽ chạy kernel trừ khi dùng `--no-run`; dataset create mặc định private.
Luôn xác minh metadata private và nguồn input trước khi push. Chưa có slug để đưa
lệnh launch thật ở đây; không copy slug của experiment Fold1 cũ vào population mới.

Các lệnh read-only có thể dùng đầu session mới:

```sh
kaggle --version
kaggle kernels list --mine --page-size 1 --csv
kaggle kernels push --help
kaggle kernels output --help
```

## 8. Storage local, log, tests và các lưu ý bàn giao

- Checkpoint production vẫn ở HOME, không symlink/copy sang vùng mới.
  Root riêng `/var/tmp/langfens-crottc-full-20261113-t_car8m4` còn khoảng62 GiB
  lúc provision; registry `LOCAL_STORAGE_REGISTRY.json`. Chỉ có owner marker,
  retention lock và16 MiB I/O probe; **không chứa production checkpoint**.
- Guard PID389676/session34894 đã dừng, status `GUARD_STOPPED_FILES_PINNED`;
  sticky regular files được giữ. Không cần resume guard để chuyển Kaggle. Code
  relocation/local-next-window trong report cũ là hướng cũ, không phải queue mới.
- Không xóa checkpoint/profile cũ để lấy chỗ. Home chỉ~10 GiB free sau partial
  checkpoint; đo lại trước tạo package lớn. Drive archives giữ nguyên.
- Tests đã có log: resume4passed; IF components7passed; IF trainer+stage13passed;
  optimizer/migration/stage16passed ở nhóm kiểm trước; storage11passed.
  Các nhóm có overlap, **không cộng thành một tổng độc lập**. Scope IF tests là
  components/synthetic AM memory/transfer/freeze/gates, không native IF training.
- Tests/reports thường ở `reports/CROTTC_IF_FULL_LOCAL_20261113/` và
  `experiments/crottc_if_full_local_20261113/*TESTS.log`. Không rerun toàn bộ
  training chỉ để xác minh docs. Native Kaggle profile/continuation vẫn chưa có.
- Official released bundle smoke cũ:3 FIT audio-only,
  `reports/CROTTC_IF_OFFICIAL_20261113/REPORT_VI.md`, release revision
  `9149e5107b3332b9aa3f5573f6328a4e312006ad`; không thay native training evidence.
- Repo có nhiều untracked artifacts từ trước. Không `git add -A`, không commit
  dataset/weights/checkpoints/credentials, không force-push/merge. Chưa commit/push
  bàn giao này. Git HEAD không chứa đủ untracked implementation; package source
  cho Kaggle phải dựa trên file/hash inventory, không chỉ git clone HEAD.
- Không spawn agents nếu user/AGENTS/skill không yêu cầu. Không resume M0–M6/
  compact pilot/parallel method sweeps vì GOAL thesis hiện hành đã supersede chúng.

## 9. Prompt mở session mới

> Đọc AGENTS.md, GOAL.md và HANDOFF.md. Resume mục tiêu full CROTTC-IF theo quyết
> định CROTTC-IF-FULL-KAGGLE-20261114-v1: chuyển training sang Kaggle bằng CLI đã
> xác thực. Local đã dừng sạch ở update 124, epoch 2/cursor320 với 7/32 pending
> gradients; giữ checkpoint và lịch sử, không khởi tạo lại hoặc tự train local.
> Chuẩn bị private package chỉ TRAIN 1800/DEV_SEARCH 599, adapter/migration/profile
> Kaggle, xác minh resume rồi chạy các stage đầy đủ theo dependency. Official test
> vẫn SEALED; fidelity và coverage paper còn thiếu phải theo dõi, không claim full
> reproduction chỉ vì chạy được một AM stage.
