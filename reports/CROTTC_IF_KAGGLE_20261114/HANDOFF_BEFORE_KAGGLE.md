# HANDOFF — CROTTC-IF full local reproduction

Ngày lưu: 2026-11-13, Asia/Ho_Chi_Minh. Nhánh: `feat/crottc-if-official-20261113`.
Workspace: `/home/khoa/Projects/New-Method-Research`.

## Trạng thái và yêu cầu người dùng

**RESUMED ngày 2026-11-13 theo yêu cầu “Đọc HANDOFF.md và resume GOAL”.**
Phần dưới giữ nguyên trạng thái tại lần pause làm lịch sử. Trạng thái tiếp tục mới xem
`reports/CROTTC_IF_FULL_LOCAL_20261113/RESUME_REGISTRATION.md` và `GOAL.md`.
## Trạng thái hiện hành — GPU + RAM (2026-11-14)

- Người dùng yêu cầu ưu tiên GPU và dùng RAM khi thiếu VRAM. Đã thực thi.
- Native AM CROTTC đang chạy bằng `research.crottc_if_official.gpu_train`;
  PID tại lần xác minh 324148, đã hoàn tất epoch2/300, checkpoint update115/cursor32/epoch2 (đang epoch3).
- Toàn bộ WavLM Large 315M và head trên CUDA FP32; 311M SSL trainable giữ nguyên.
  Adam/master weights, gradient tích lũy sau từng microbatch và activation trên RAM.
  GPU + RAM profile đủ 32 median/p95/max, một Adam update: peak CUDA ~2.51 GiB,
  peak RSS ~8.0 GiB. Bốn profile GPU trước đó OOM được giữ nguyên log/status.
- Migration từ CPU update22/cursor704/pending3 -> GPU update23/cursor736 đã xác minh.
  Sau đó save 12/32 gradient RAM và resume native GPU từ microbatch13; checkpoint
  định kỳ update25/cursor800 đã hash và kiểm counters scheduler/Adam.
- Nguồn trạng thái live: `experiments/crottc_if_full_local_20261113/RUNNING_WINDOW.json`,
  `EXECUTION_CONTROL.json`, `am_crottc/STATUS.json`,
  `AM_CROTTC_GPU_RAM_WINDOW_001.log`. Luôn kiểm tra PID sống và GPU lock; không
  khởi chạy job trùng. Goal API vẫn active, lịch AM vẫn300, mỗi cửa sổ≤14400s.
- Entry resume hiện tại: `.venv-crottc-official/bin/python -m
  research.crottc_if_official.gpu_train --stage am_crottc --resume --wall-seconds 14400`
  với env offline/thread như `GPU_RAM_EXECUTION.md`. Không dùng CPU `full_train`
  để resume checkpoint GPU mới, không lặp lại receipt migration đã dùng một lần.
- Giữ hash `gpu_train.py`, `cpu_adam_offload.py`, `placement_migration.py`,
  `full_train.py`, `checkpointing.py` trong lúc chạy. Đổi code cần migration rõ ràng.
- IF components 7 tests qua; stage parent gates 5 tests qua; optimizer/migration/
  stage tests tổng16 qua. Full IF trainer đã triển khai (13 trainer/gate tests); native IF profile/training và các stage300/200 chưa hoàn tất.
- Chi tiết `reports/CROTTC_IF_FULL_LOCAL_20261113/GPU_RAM_EXECUTION.md`,
  `GPU_RAM_EXECUTION_VERIFICATION.json`, `GPU_RAM_PLACEMENT_MIGRATION.json`,
  `IF_STAGE_GATE_VERIFICATION.json`. Test sealed; chưa có paper F1 mới.

## Tiến độ IF và epoch đầu — 2026-11-14

- AM GPU/RAM PID324148 vẫn live; đã hoàn tất epoch1 và vào epoch2.
  Bằng chứng `AM_EPOCH_001_VERIFICATION.json`: 57 updates/1800 TRAIN và
  validation599 DEV_SEARCH; source new hierarchical F1=0.4002833865. Đây là
  local development, không paper benchmark hoặc strict S/O/I; source old/new
  definitions khác nhau được giữ nguyên trong biên nhận.
- `if_train.py` đã triển khai full IF CTC initialization và CROTTC transfer,
  GPU/RAM, checkpoint/RNG/gradient resume, validation audio-only rồi mới gold
  metrics, scratch resource profile không thể thành checkpoint production.
- 13 trainer/stage tests passed; full-size source IF components với synthetic
  AM memory là unit evidence, **chưa native IF profile/training**. Parents vẫn
  thiếu AM CTC300 và AM CROTTC300/IF CTC200. Không bypass bằng checkpoint pilot.
- Tài liệu `IF_TRAINER_PROTOCOL.md`, `IF_TRAINER_IMPLEMENTATION_VERIFICATION.json`
  trong `reports/CROTTC_IF_FULL_LOCAL_20261113/`. Có pin code/runtime/parent
  provenance trước mọi IF optimizer update. Native IF profile cần chạy khi
  đủ điều kiện và GPU trống; không tranh GPU với AM đang chạy.

## Epoch2 và checkpoint tiếp tục — 2026-11-14

- Hoàn tất epoch2 với1800 TRAIN/599 DEV_SEARCH,114 updates tổng. Source new
  hierarchical F1=0.4668205424 (46,68%), old F1=0.2613821138; PER52.027645.
  Đây là local development, chưa paper benchmark/strict S/O/I hay cải tiến cuối.
- `AM_EPOCH_002_VERIFICATION.json`: last.pt update115/cursor32/internal epoch2
  load bằng CPU mmap và SHA khớp receipt;537 Adam parameter states đều step115,
  scheduler115, đủ torch/CUDA/Python/NumPy RNG. GPU PID324148 vẫn live, vào epoch3.
- Năm hash code trainer đang chạy giữ nguyên. Profile/train IF vẫn đợi full
  parent stages, không dùng AM2epoch để vượt gate. GOAL API vẫn active.

## Dung lượng checkpoint bổ sung — 2026-11-14

- Khi người dùng hỏi GPU local có đủ không: native model chạy được, nhưng
  thời gian là hạn chế lớn. Snapshot >1giờ,98 optimizer updates mới cho thấy
  khoảng38.7s/update (gồm validation/checkpoint); AM CROTTC300 riêng khoảng7.7ngày
  nếu giữ nhịp này. AM CTC/IF chưa đo throughput, không gán cùng tốc độ hoặc
  hứa full paper vài ngày. `GPU_RAM_COMPUTE_PROJECTION.json` ghi dữ liệu/giới hạn.
  User chưa yêu cầu dừng hoặc đổi platform; GPU job vẫn tiếp tục lịch đã đăng ký.

- Đã tạo private local root `/var/tmp/langfens-crottc-full-20261113-t_car8m4`
  (phân vùng khác `/home`, còn62.4GiB), kiểm real atomic I/O16MiB/SHA và mode0700.
  Registry `experiments/crottc_if_full_local_20261113/LOCAL_STORAGE_REGISTRY.json`.
- Guard PID389676/session34894; kiểm process thực tế và
  `LOCAL_STORAGE_GUARD_STATE.json`. Sticky regular files + directory flock đã
  kiểm bằng installed systemd-tmpfiles với config/prefix chỉ trong test fixture.
  11 storage tests passed; không chạy cleanup global.
- **Chưa chuyển production checkpoint** vì AM GPU PID324148 vẫn live, update113
  tại lần quan sát. Helper từ chối shared GPU lock đang bận và không đổi run.
  Chờ cửa sổ training terminal rồi copy/hash/fsync/atomic exchange, giữ nguyên
  original directory, resume logical path cũ bằng `gpu_train --resume`.
- Hướng dẫn và bằng chứng `LOCAL_STORAGE_PROTOCOL.md`,
  `LOCAL_STORAGE_VERIFICATION.json` trong report. Không đổi hash code trainer
  đang chạy; không xóa checkpoints cũ hoặc stage dataset vào storage này.

## Lịch sử: checkpoint resume ban đầu

- Goal API hiện `active`; user đã yêu cầu resume. Các dòng pause bên dưới là lịch sử.
- Trainer đã sửa strict `--resume-from` / `--resume`, không fallback khởi tạo mới.
- Kiểm thử số học CPU/CUDA: 4 passed; profile 32 câu median/p95/max hoàn tất.
- Giữ nguyên checkpoint lịch sử `am_crottc_profile/last.pt` và SHA cũ.
- `am_crottc/last.pt` đã lưu partial accumulation 23/32, epoch0/cursor32/update1,
  SHA `d070e922d66fd220dd535fd1c30c9a371f0dbfde8eb52247b8d54a5f46ba3f5d`.
- Cửa sổ 4 giờ đang chạy đã nạp partial checkpoint và đạt update2/cursor64.
  Số này là trạng thái trong RAM tại lúc xác minh; last.pt tiếp tục cập nhật mỗi5
  update, mỗi epoch và clean stop. Xem live `am_crottc/STATUS.json`,
  `RUNNING_WINDOW.json`, `AM_CROTTC_WINDOW_001.log` trước khi khởi chạy gì khác.
- Không sửa `full_train.py` hoặc `checkpointing.py` trong lúc chạy; checkpoint
  kiểm hash implementation khi resume. Nếu thay code cần migration đăng ký rõ.
- Bằng chứng: `reports/CROTTC_IF_FULL_LOCAL_20261113/RESUME_VERIFICATION.json`.
- Full paper vẫn chưa được tái hiện; fidelity local/split và IF stages còn lại
  được ghi trong `RESUME_REGISTRATION.md`, `PREREGISTRATION.md`, `PAPER_COVERAGE.csv`.

## Lịch sử: tiến độ trước khi chuyển GPU + RAM

- AM process PID 248779 đã xác minh live; checkpoint định kỳ update10/cursor320,
  epoch0, SHA `1b4f04a3f799fb17314effd125f342ab177218d1b316c39726a88478b7c377af`.
  Biên nhận `AM_PERIODIC_CHECKPOINT_10_VERIFIED.json` trong report. Xem trạng thái
  live trước mọi action; không chạy lại process còn sống.
- Đã thêm `research/crottc_if_official/if_components.py`: LM/teacher thật 384 dim,
  2 layer, hai fusion branch độc lập, full AM memory gồm Conformer, C/S/D/I,
  beam10 không nhận canonical/gold. Bảy CPU component tests passed; không IF
  optimizer update, chưa full IF trainer hoặc stage transitions. Xem
  `IF_COMPONENT_REGISTRATION.md` và `IF_COMPONENT_VERIFICATION.json`.
- IF source LM decoder thực tế regular MHA + positional embedding; teacher
  RelPosMHAXL; không gọi cấu hình này fully RoPE chỉ dựa vào tên YAML. Giữ sai
  khác paper/source rõ ràng. Không sửa trainer AM đang chạy.

## Trạng thái pause lịch sử

Goal API tại lần bàn giao trước ở `paused`; không đánh dấu complete. Người dùng muốn mở phiên mới
để tiếp tục công việc. Tại bàn giao không còn process `full_train`/`download_native`
và không có CUDA compute process. Phiên sau chỉ resume khi được yêu cầu.

Yêu cầu hoạt động gần nhất trước pause: ghi quyết định vào GOAL rồi **tái hiện full
paper CROTTC-IF hoàn toàn trên local**. Training local đã được cho phép rõ ràng;
không hỏi lại quyền training khi người dùng yêu cầu tiếp tục. Đọc `AGENTS.md`,
`GOAL.md`, rồi preregistration trước khi thực thi.

Mục tiêu khóa luận vẫn là phát triển **một phương pháp cải tiến** cho chẩn đoán có
căn cứ và hướng dẫn sửa chính xác/rõ ràng/phù hợp. CROTTC-IF là nền đối chứng;
tái lập paper chưa phải đóng góp cuối cùng. GOAL M0–M6 cũ đã bị thay vai trò điều hành,
giữ các kết quả cũ làm lịch sử. Không mở lại các nhánh PER-MDD/X-MDD như nghĩa vụ.

## Checkpoint thật vừa tạo

`experiments/crottc_if_full_local_20261113/am_crottc_profile/last.pt`

- SHA-256: `f4c97a353ce392e166c60965b68aa49868cab598d169ff218bd5626ddde87bd7`.
- Kích thước: 3,798,326,543 bytes (khoảng 3.54 GiB).
- `epoch=0`, `cursor=32`, `updates=1`; chưa xong epoch đầu.
- Có weights SSL/head, Adam, constant-LR scheduler, scaler disabled, RNG
  Python/NumPy/Torch/CUDA, data cursor, objective và provenance hashes.
- Checkpoint được ghi atomic, tải lại `weights_only=True`, so mọi weight bằng
  weight trong RAM trước khi thay `last.pt`; có `CHECKPOINT_RECEIPT.json`.
- Trạng thái: `am_crottc_profile/STATUS.json`: `PROFILE_UPDATE_COMPLETE`.
- Log: `experiments/crottc_if_full_local_20261113/PROFILE.log`.

Profile chạy đủ forward/backward/Adam với **WavLM Large thật**, không compact/LoRA:
SSL 315,453,120 tham số, SSL trainable 311,246,528; heads 3,822,765 tham số.
SSL + FP32 Adam đặt CPU; head + official OTTC loss đặt CUDA; activation checkpointing.
4 CPU threads; microbatch 1, accumulation 32. Mean loss 8.0182338357; gradient norm
11.82722673; max thay đổi một tensor SSL được kiểm tra là 1.001358e-5.
Một update mất 64.34s; toàn profile gồm khởi tạo/lưu ~73.10s. Peak RSS 9,331,494,912
bytes (~8.69 GiB); peak CUDA allocated 82,714,624 bytes (~78.9 MiB).

**Giới hạn:** đây là 32 câu ngắn nhất do sort duration. Chưa profile câu dài/trung
bình, chưa kiểm tra tiếp tục optimizer cho kết quả tương đương chạy liền, chưa chạy
validation, chưa có F1 mới. Không ngoại suy ETA/full-memory từ nhóm câu ngắn này.

## Các file cần đọc trước

1. `reports/CROTTC_IF_FULL_LOCAL_20261113/PREREGISTRATION.md`: bảng stage,
   khác biệt code/paper, data, resource/checkpoint contract và coverage toàn paper.
2. `reports/CROTTC_IF_FULL_LOCAL_20261113/DECISION.json`: quyết định
   `CROTTC-IF-FULL-LOCAL-20261113-v1`, user authorization và hash root GOAL/AGENTS
   tại lúc chốt. **Hash GOAL lịch sử không còn bằng root sau bản ghi pause này;
   không sửa receipt quyết định cũ để ép bằng root.**
3. `experiments/crottc_if_full_local_20261113/COMPUTE_BUDGET.json`.
4. `research/crottc_if_official/full_train.py`, `prepare_full.py`, `download_native.py`.
5. `experiments/crottc_if_full_local_20261113/DATA_RECEIPT.json`,
   `SOURCE_RECEIPT.json`, `NATIVE_DOWNLOAD_RECEIPT.json`.

Môi trường: `.venv-crottc-official/bin/python` (Python 3.11), dùng `.pth` trỏ base
venv nhưng override transformers 4.57.0, tokenizers .22.2, HF hub .36.0, wandb .21.4.
Torch 2.6.0+cu124, torchaudio 2.6, SpeechBrain 1.0.3. Base `.venv/bin/python` dùng
Transformers 5.x không phù hợp code official. Freeze effective packages nằm ở
`experiments/crottc_if_official_20261113/RUNTIME_FREEZE.txt`.

## Source và native initialization

- Author source `Secondtonumb/IF-MDD`, commit
  `cca150ec4ff00e01d61a613092d1d178e661dfb2`; bản sao riêng ở
  `experiments/crottc_if_full_local_20261113/upstream/`, mọi file khớp hash lịch sử.
  Không có `.git` trong bản sao; `git -C ... rev-parse HEAD` sẽ trả commit repo cha,
  **không phải commit upstream**. Dùng SOURCE_RECEIPT.
- Không sửa bản source lịch sử ở
  `experiments/realized_phone_pipeline_20261113/m1_reference/upstream/` hoặc bundle
  inference đã xác minh. Override/config sửa phải tách riêng và ghi hash.
- Native `microsoft/wavlm-large`, revision
  `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, đã tải đầy đủ/check SHA vào
  `experiments/crottc_if_full_local_20261113/native_wavlm_large/`.
  `pytorch_model.bin` SHA:
  `fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d`.
  Không cần tải lại. **AM training khởi tạo từ native này, không từ released L2
  checkpoint đã fine-tune.**

## Dữ liệu và các khác biệt chưa giải quyết

Đã freeze/verify audio SHA, manifest SHA, raw archive SHA trước update:

- TRAIN 1,800 câu / 12 speakers theo `data/split_specs/L2_ARCTIC_SPLIT.yaml`.
- DEV_SEARCH 599 câu / NCC, RRBI, YBAA, YDCK. Không dùng DEV_CAL cho fitting.
- Đủ 1,800 TRAIN được đưa vào supervision; 557 câu có `err` target, tổng 1,110
  token `err`. Đây là ký hiệu unknown/deviation, **không khôi phục exact realization**.
- TRAIN đọc chỉ TextGrid member thuộc allowlist từ archive local, giữ silence;
  starred -> err; AX -> AH; er)/d_/z_/AH) -> err; deletion bỏ khỏi transcript perceived.
- DEV_SEARCH dùng derived manifest sẵn có, thiếu raw silence/star markers; không
  được gọi score của nó là paper-equivalent. 12 câu có `err` target.
- Source không cung cấp preparation script/frozen manifests, preprocessing parity
  còn UNVERIFIED. Một vài label parsing mang tính policy local đã đăng ký.
- Exact paper split khác hẳn: 2,429 train / 268 val / 900 test; test paper gồm
  NJS, TLV, TNI, TXHC, YKWK, ZHAA. NJS đang thuộc local TRAIN; local sealed gồm
  ZHAA, TNI, MBMPS, TLV. **Không mở/stage/extract test audio/labels.** Generic full
  reproduction authorization không hủy test seal. Cần protocol/quyền cụ thể cho
  exact benchmark về sau; vẫn làm được phần training/implementation được phép.

Preregistration ghi rõ các lệch đáng chú ý:

- README AM YAML không register Conformer; `_confEnc` YAML mới có đủ. Local dùng
  `_confEnc`, override kernel 31 -> 3 theo paper, hidden 384, 2 layers.
- Local AM objective mặc định dùng paper Eq.5/6; `--objective source` là recipe khác:
  source dùng .5*OT + .1*mean KL theo cả class. Không đổi objective khi resume.
- Augmentation hiện **vẫn là source waveform DropFreq/DropChunk + speed**, chưa
  khớp time warp 80 / spectral masks ratio của paper. Chưa thể gọi recipe fully faithful.
- IF YAML bỏ register acoustic Conformer, alias 2 fusion branches, có ConPCO loss
  ngoài paper; DataIO teacher labels chỉ binary mismatch dù head 4 classes. Local
  staged labels giữ C/S/D/I nhưng **IF trainer chưa được triển khai**.
- Paper có omega1 .3 ở Eq.18, .5 ở implementation; 200 epochs LM/teacher không
  phân bổ rõ hai stage. Registered 200 mỗi stage là provisional, không biết chính
  xác tác giả. Cần giải quyết/ghi rõ trước IF training claim.
- Accumulation microbatch1 giữ effective count32 nhưng không cùng BatchNorm
  statistics với batch32 thật. Cần công bố đây là adaptation.

## Việc phiên tiếp theo cần làm khi user resume

1. Xác minh checkpoint hash, source/data receipt và trạng thái process/GPU lock.
   Không chạy lại download hoặc reset từ native làm mất bước đã có.
2. Kiểm tra/sửa resume trước chạy dài. Code hiện chỉ có `--resume`, tìm `last.pt`
   ở thư mục stage đang chọn. **`--stage am_crottc --resume` chưa tự tìm checkpoint
   trong `am_crottc_profile/`; nếu không có file nó khởi tạo lại.** Thêm
   `--resume-from` có strict provenance/objective/stage check hoặc promote checkpoint
   với receipt rõ ràng. `--profile --resume` hiện chạy thêm một update profile, không
   phải continuation 300 epochs. Chưa thực hiện promotion/move/copy nào.
3. Profile đại diện median/p95/max duration và steady-state Adam RAM trước chạy
   dài; giữ failed/interrupted attempt provenance. Không dự báo full training từ
   batch ngắn. Kiểm tra RAM/disk/jobs, acquire flock.
4. Sửa window stopping: code check wall budget ở đầu group32, không giữa group;
   SIGTERM chỉ set flag, đợi xong group. Đăng ký grace hoặc checkpoint partial
   accumulation/RNG nếu cần hard budget. Hiện chỉ save epoch/clean window end;
   nên có checkpoint định kỳ để tránh mất nhiều update khi chưa hết epoch.
5. Kiểm tra đúng semantics resume weights/optimizer/RNG bằng test có ý nghĩa;
   hiện mới compile + thật một update + atomic reload weight equality, chưa test
   numerical continuation. Record hash implementation/effective config trong
   checkpoint (hiện provenance chỉ data/source/native receipts).
6. Giải quyết/đóng băng fidelity augmentation/objective trước mở campaign dài.
   Không biến pilot adaptation thành “full paper reproduced”.
7. Tiếp tục AM CROTTC target300 và AM CTC đối chứng, triển khai IF CTC initialization
   và CROTTC transfer với teacher branches thật; checkpoint stage transitions.
   Scope toàn bộ tables/ablations/generalization/LLM/Iqra phải track riêng;
   thiếu dữ liệu/quyền đánh dấu NOT_REPRODUCED, không lược bỏ rồi claim full paper.
8. Validation audio-only; gold chỉ metric sau decode. Paper F1 hierarchical tách
   strict typed S/O/I và đánh giá feedback khóa luận. Chưa có F1 mới để nói cải thiện.

Compute đã đăng ký: profile <=900s/1 update/32 microbatches, initial sustained
window <=14,400s; không tự hạ schedule. GPU flock:
`/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock`.
GTX1650 4GB, RAM32GB. Sau profile còn khoảng **17GB disk free**. Full checkpoint
~3.54GiB; tính cả temp atomic + last/best + native weights; không xóa checkpoint
lịch sử để lấy chỗ. Không đưa dataset/checkpoint/model weights/credentials vào Git.

## Lệnh và kiểm tra đã chạy, không cần tự chạy lại khi đang pause

```sh
# Tạo data/source receipt (đã hoàn tất; intentionally refuses overwrite receipt)
.venv-crottc-official/bin/python -m research.crottc_if_official.prepare_full
# Native download đã VERIFIED_COMPLETE
.venv-crottc-official/bin/python -m research.crottc_if_official.download_native
# Profile đã hoàn tất, không phải lệnh resume full campaign
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv-crottc-official/bin/python -m research.crottc_if_official.full_train \
  --profile --wall-seconds 900
```

Entrypoint tự acquire flock. Không cần bọc flock lần hai cùng path.
Các attempt chuẩn bị data ban đầu fail loud với unknown `ax`, `er)` và derived
validation `AH)` trước khi policy được ghi rõ và manifest freeze hoàn tất. Không có
optimizer update trong các attempt này. Không dùng nhãn canonical để lấp phát âm thiếu.

## Bằng chứng trước đó có thể tái sử dụng

Official release inference bundle đã tải/check 27 files, strict load 2 checkpoints,
chạy 3 FIT audio-only: `reports/CROTTC_IF_OFFICIAL_20261113/REPORT_VI.md`,
`experiments/crottc_if_official_20261113/SMOKE_RECEIPT.json`. Revision release
`9149e5107b3332b9aa3f5573f6328a4e312006ad`. Đây chỉ là inference, không training.
Compact CROTTC pilot/scaling và các EXP cũ không phải reproduction official.

Repo có rất nhiều untracked/user artifacts từ trước. Không `git add -A`, không
overwrite tài liệu không liên quan, không force-push/merge; chưa commit/push bàn giao.
Không spawn agents mới trừ khi user hoặc AGENTS/skill yêu cầu rõ. Các Luna trước đó
đã chạy pilot; không suy ra đang tiếp tục full reproduction.

Có thể bắt đầu phiên mới bằng:

> Đọc AGENTS.md, GOAL.md và HANDOFF.md. Tôi yêu cầu resume GOAL tái hiện full
> CROTTC-IF trên local; tiếp tục từ checkpoint profile đã lưu, xử lý các mục resume,
> profile đại diện và fidelity trước khi chạy dài. Không khởi tạo lại hoặc mở test niêm phong.
