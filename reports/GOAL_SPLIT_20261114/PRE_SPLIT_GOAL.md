# LANGFENS — Phát triển phương pháp chẩn đoán và hướng dẫn sửa lỗi phát âm

Decision checkpoint: `THESIS-METHOD-DECISION-20261113-v1`.

Ngày chốt: 2026-11-13. Trạng thái: **DECISION_FROZEN — RESEARCH_NOT_COMPLETE**.

Checkpoint thực thi mới nhất: **CROTTC-IF-FULL-KAGGLE-20261114-v1 — KAGGLE_TRAINING_AUTHORIZED, HANDOFF_READY, REPRODUCTION_IN_PROGRESS**. Ngày2026-11-14, người dùng quyết định chuyển full training sang **Kaggle**, xác nhận có Kaggle CLI và yêu cầu bàn giao chi tiết để đổi session. Quyết định này thay chỉ định “full hoàn toàn local” về nền tảng thực thi; giữ nguyên mục tiêu/full architecture/full schedule và toàn bộ lịch sử. [Quyết định mới](reports/CROTTC_IF_KAGGLE_20261114/DECISION.json), [snapshot trước thay đổi](reports/CROTTC_IF_KAGGLE_20261114/GOAL_BEFORE_KAGGLE.md).

**Trạng thái thực thi: LOCAL_STOPPED_CLEANLY — OPTIMIZED_LONG_TRAINING_RUNNING.** **Cập nhật14:18:** remoteoptimizedwindow001 vẫnRUNNING; localcontrollerstatusAPI403đã làmPID1141950dừng13:22, đãpreserveerror/re-attach existingprivatekernel/script/parent **khôngrestart/push** bằngPID1291861; xem `OPTIMIZED_CAMPAIGN_02/window001/ATTACH_RECEIPT.json`. Sidegoal userauthorized đã có CPUaudit/209regressiontests và KaggleCPUfrozenfeaturesmoke riêngCOMPLETE/downloadverified lúc14:22 (8TRAIN×2full315Mfrozen), không dùngGPUmain; headfit/gainclaim vẫnpending; [sideevidence](reports/SM13_BACKBONE_SIDE_20261114/STATUS.md). **Cập nhật13:05:** Longoptimized đã trainthật, livelog updates483–485/internalepoch8cursor928,loss3.6988,step4.18s,SSLdelta9.82e-6. Exportverifiedparent vẫn482 chođến nextdownload; xem `OPTIMIZED_CAMPAIGN_02/window001/LIVE_START.log`. **Snapshot13:02:** optimizedsmoke480→482 đã download/SHA/load/fullstate/order/537Adamledgerverified. Longkernel private `minkhoaa/langfens-crottc-if-optimized-window001` v1 **RUNNING/initializing**, source482pinned,2T4FP16mb32/effective32,4hdriver/16800splatform,**khôngcapupdates**, target300totalAMepoch. [Monitor](https://www.kaggle.com/code/minkhoaa/langfens-crottc-if-optimized-window001), [launchreceipt](reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_CAMPAIGN_02/LAUNCH_RECEIPT.json). ControllerPID1141950nối windows sauverifiedexport; nhiệm vụfullpaper còn đủ stages/coverage, khôngdừngởbenchmark/AMstage. Mốc12:50/bên dưới là lịch sử. **Cập nhật12:50:** user yêu cầu full training optimized, không dừng benchmark/smoke; [mandate và link training dài dự kiến](reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_FULL_TRAINING_MANDATE.md). Latestlegacy480/epoch8/cursor768 đã download/SHA/load/fullstate/orderverified; khôngrollback125. Allocatorconfirmation đã verify chọn2T4/FP16/mb32/effective32, median5.03s/update (~7.74x benchmark), giữ safetyheadroom. Native optimized smoke đã COMPLETE với realupdates481–482; đangdownloadverifyexport, chưa mở longwindow. ControllerPID1141950 sẽ nối optimizedwindow001 target300totalAM sau gate, khôngcapupdates. Các mốc12:24/bên dưới là lịch sử. **Cập nhật12:24:** window002 COMPLETE, remote receipt480/internalepoch8/cursor768/no pending; đangdownloadfulloutput để verify. AllocatorconfirmationRUNNING, optimizedcontroller1141950chờ cả hai gates; chưalaunchproductiontốiưu.163CPUtests đạt. **Snapshot12:14:** theo user yêu cầu tận dụng2T4/CPU/RAM, benchmark native đã đo và download/verify: safe microbatch4 FP16/GPUAdam7.36s/update so38.65s legacy (~5.25x). Đây chưa phải speedup production. Window002 vẫn train (snapshot11:56update450/epoch8); STOP chặn legacywindow kế tiếp, giữ latestexport. Allocator confirmation kernel đang QUEUED để kiểm tra batch lớn với1.5GiBreserve không đổi. Optimized controller PID1141950 chờ profile mới và latestwindow002verified trước capped2update smoke rồi4hcontinuation. Không rollback125/không mất pendinggradient. Các trạng thái08:26/bên dưới là lịch sử. [Handover live và toàn bộ evidence](reports/CROTTC_IF_KAGGLE_20261114/RESOURCE_HANDOVER_STATUS.md), [controller](reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_CAMPAIGN_02/CONTROLLER_STATUS.json). **Cập nhật08:26:** window002/code-v5 đã vượt resume và có optimizer updates thật; log live đến141, internal epoch2/cursor864 (epoch3/300), loss4.5744 hữu hạn, SSL weight delta5.72e-6. Log báo checkpoint remote140; checkpoint download/verify gần nhất vẫn125. Controller campaign02 PID863482 ACTIVE, max63bounded windows từ002; không reset state. Bản sửa115tests đạt; không tạo cron/timer, chỉ chờ300s và kiểm tra một lần theo yêu cầu. [Log kiểm tra](reports/CROTTC_IF_KAGGLE_20261114/AM_CAMPAIGN_02/window002/LIVE_AFTER_300S.log), [controller](reports/CROTTC_IF_KAGGLE_20261114/AM_CAMPAIGN_02/CONTROLLER_STATUS.json). **Lịch sử08:10:** window001 đã ERROR lúc02:51, trước batch mới; controller cũ dừng. Lỗi audit coi hai biểu diễn gradient rỗng là khác nhau, không phải lỗi training/data/quota. Checkpoint đã chứng nhận vẫn125. Adapter repair code-v5 đã qua115tests và migration125được đăng ký minh bạch; chưa được gọi resumed cho đến khi window mới chạy. [Lỗi và bản sửa](reports/CROTTC_IF_KAGGLE_20261114/WINDOW001_FAILURE_REPAIR_V5.md). Người dùng tái xác nhận nhiệm vụ **tái lập toàn bộ paper**, không dừng sau training300epoch; xem [phạm vi và điều khiển thực thi](reports/CROTTC_IF_KAGGLE_20261114/FULL_PAPER_EXECUTION.md). Native profile kernel v3 đã download/verify thành công32mẫu TRAIN và1scratch update; YAML đã sửa, private packages READY, code-v4 khớp workspace. Resume smoke kernel v1 đã COMPLETE và checkpoint125 đã download/SHA/load/full-state/order audit đạt, tiếp tục đúng microbatch8–32. Controller CLI local PID511439/campaign01 đã dừng; window001 thất bại trước update mới. Campaign02 sẽ dùng parent125 pinned, code-v5 và target300total AM epoch, không cap updates; không sửa lịch sử hoặc reset state. Mỗi export phải download/verify trước cửa sổ sau; dừng quota/error/diskfloor14GiB hoặc hết64windows, không xóa artifact.98CPU regression tests đạt. Local chỉ điều khiển CLI, không training; cần giữ máy awake/networked để nối windows. [Profile verification](reports/CROTTC_IF_KAGGLE_20261114/NATIVE_PROFILE_V3_VERIFICATION.json), [nhật ký khôi phục và thất bại được giữ nguyên](reports/CROTTC_IF_KAGGLE_20261114/SESSION_RECOVERY.md). Probe thực tế cấp2×T4/RAM31GiB/output20GB; dùng1GPUvisible để giữ RNG, Python3.11/Torch2.6 cô lập. [Kế hoạch cửa sổ](reports/CROTTC_IF_KAGGLE_20261114/WINDOW_PLAN.json), [runtime GPU](reports/CROTTC_IF_KAGGLE_20261114/KAGGLE_GPU_RUNTIME_V1.json). Job GPU/RAM local đã nhận SIGTERM và lưu checkpoint atomic tại update 124, internal epoch 2/cursor320, còn 7/32 pending microbatches; đã xong2/300 epoch AM CROTTC. Đã verify SHA/loadable, 537 Adam states step124, scheduler124, đủ RNG/gradients; PID324148 đã terminal và GPU lock free. Không tự resume local. [Bằng chứng dừng/checkpoint](reports/CROTTC_IF_KAGGLE_20261114/LOCAL_STOP_VERIFICATION.json), [lịch sử phiên train](reports/CROTTC_IF_KAGGLE_20261114/LOCAL_SESSION_HISTORY.json).

Kaggle CLI2.2.5 user `minkhoaa` đã upload/verify ba dataset private (authorized assets-v1, full update124 state-v1, runtime code-v4). Migration124→125 đã chứng minh bằng export download/verify (`RESUME124_SMOKE_V1_VERIFICATION.json`); sustained AM300 đang chạy, số update/epoch của window hiện hành chỉ chứng nhận sau export; accelerator đã đo thực tế, số giờ quota tài khoản còn lại chưa được public CLI cung cấp — không suy diễn từ quota mặc định. Session mới chuẩn bị private authorized-data package, path/runtime/placement migration và profile full native model rồi kiểm resume trước khi chạy dài. [CLI readiness](reports/CROTTC_IF_KAGGLE_20261114/KAGGLE_CLI_READINESS.json), [HANDOFF chi tiết](HANDOFF.md). Source new F1 epoch 2=46,68% trên 599 DEV_SEARCH chỉ là local development; không phải paper benchmark. Toàn bộ stages300/300/200/200, fidelity và coverage paper vẫn chưa hoàn tất. Test vẫn SEALED. Checkpoint/log/profile/native weights giữ nguyên local; vùng storage bổ sung chưa nhận production checkpoint và guard đã dừng.

**Nhánh phụ được user cho phép2026-11-14:** [SIDE_GOAL.md](SIDE_GOAL.md), decision `SM13-BACKBONE-SIDE-20261114-v1`, lưu toàn bộ thảo luận epoch/dataset/frozen-vs-fine-tuned và thử nghiệm SM13-style với backboneWavLM native/adapted có matchedcontrol. Triển khai bounded song song, không thay/gián đoạn currentfullpapertraining; không mở sweep hoặc localproductiontraining. CPUaudit1800WAV/26featurefiles và209CPUregressiontests đã đạt; sideCPUnativefrozenfeaturesmoke đãCOMPLETE/downloadverified, chưaheadfit/gainclaim. Gain/leakage/data/metric gates ghi riêng. GOAL vẫn là mục tiêu khóa luận chính; sideexperiment không chứng nhận fullpaper/thesiscomplete.

**Latest user15:55: SM13local, không break fullpapertraining.** SM13S0frozenheads50epoch đangtrainlocalPID1448557/GTX1650, targetepoch16đãobserved15:58; originalfeaturecacheSHAverified/sharedGPUlock/atomicfullstate/4hwindow. MainCROTTCvẫnKagglePID1291861, liveupdate2400/humanepoch43, khôngstop/restart/resubmit. SDKquotaAPIđãverifyGPUused>21600sallowed/reset21Nov; currentadmittedmainvẫnRUNNING, nextwindowcóthểquota-blocked. RemoteSM13QUEUEDchưacancel, localguardtránhduplicateruntrong4hwindowtrướcreset. [Localreceipt](reports/SM13_EPOCH50_20261114/LOCAL_LAUNCH_RECEIPT.json), [quota/localproof](reports/SM13_EPOCH50_20261114/LOCAL_ADMISSION.json).

**Amendment15:17 user:** `SM13-EPOCH50-20261114-v1` trong SIDE_GOAL.md cho phép thử SM13frozenheads tới50epoch/head, Kagglefirst; **chỉ SM13 được fallbacklocal khi có bằng chứng Kagglethiếutài nguyên**, không restartlocalCROTTC. S0epoch-extension trước, seed101, historicalTRAINouter0heldABA/ASI/BWC, exploratoryneural-only12/5vs50/50; giữtest/DEV_CALsealed vàmainruntimeunchanged. Privatejob/receipts: `reports/SM13_EPOCH50_20261114/`.

Đây là GOAL hiện hành duy nhất của repository. Theo yêu cầu người dùng, kế hoạch/milestone M0–M6 cũ được thay thế về vai trò điều hành; không tiếp tục chúng như danh sách công việc bắt buộc. Giữ kết quả, checkpoint và báo cáo cũ làm bằng chứng nghiên cứu, không xóa hoặc đổi kết luận.

## 1. Mục tiêu khóa luận

Phát triển và kiểm chứng **một phương pháp cải tiến** giúp hệ thống xác định một số nhóm lỗi phát âm có căn cứ và đưa ra hướng dẫn sửa lỗi **chính xác, rõ ràng, phù hợp**.

Đóng góp phải nằm ở phương pháp và bằng chứng thực nghiệm về cải tiến. Chỉ tích hợp model có sẵn với giao diện hoặc sinh lời khuyên không đủ để coi mục tiêu đã hoàn thành. Tái lập công trình trước là bước xây nền và đối chứng, không phải đóng góp cuối cùng.

Không đồng nhất thành công với một điểm Macro-F1, recall hoặc đứng đầu baseline. Không tự đặt precision/recall 80% cho mọi S/O/I hay FWR ≤0,20/câu thành điều kiện tốt nghiệp khi chưa được thống nhất. Vẫn báo đầy đủ chất lượng, độ phủ và cảnh báo sai; không bỏ ca khó để làm đẹp kết quả.

## 2. Nền nghiên cứu đã chọn

**CROTTC-IF là nền chính cho nhận diện/chẩn đoán.** Ưu tiên implementation, recipe, checkpoint và evaluator chính thức để xác lập đối chứng trước khi sửa phương pháp.

- [Paper — Geng và cộng sự, Interspeech 2026](https://www.isca-archive.org/interspeech_2026/geng26_interspeech.html).
- [Code — Secondtonumb/IF-MDD](https://github.com/Secondtonumb/IF-MDD).
- [Model bundle — Haopeng/CROTTC-IF-l2-arctic](https://huggingface.co/Haopeng/CROTTC-IF-l2-arctic).
- F1 71,77% là detection theo protocol paper, không phải Macro-F1 S/O/I của EVAL-V3.1 hoặc kết quả local đã tái lập.

**L2-Arctic-plus / ALMs4Learning là nguồn thiết kế benchmark và đánh giá phản hồi.** Tận dụng dữ liệu, cấu trúc giải thích/gợi ý sửa và công cụ đánh giá phù hợp; không mặc định ghép nguyên hai kiến trúc hoặc train lại toàn bộ ALM.

- [Paper — Liu và cộng sự, Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.190/).
- [Code — zoeyada/ALMs4Learning](https://github.com/zoeyada/ALMs4Learning).
- Phân biệt nhãn do người xác minh, dữ liệu sinh tự động và kết quả người đánh giá. Tương đồng văn bản không tự chứng minh lời hướng dẫn đúng ngữ âm.

ATP-LLM, PP-ATP, PER-MDD, X-MDD và các EXP cũ là tài liệu/đối chứng phụ khi có câu hỏi cụ thể. Chúng không phải các nhánh bắt buộc chạy song song. Không chọn công trình chỉ vì điểm F1 cao.

### Ưu tiên hiện tại — Chuyển sang Kaggle để tái hiện CROTTC-IF đầy đủ

- **Session mới tiếp nhận thực thi trên Kaggle theo quyết định 2026-11-14**: tận dụng source/data/native/checkpoint đã pin; chuẩn bị private package, adapter/migration và profile thực tế trước full training có checkpoint/đánh giá. Không tự khởi chạy lại local hoặc dừng ở việc load model/inference smoke. CLI đã có và đã xác thực; không hỏi lại quyền chuyển Kaggle đã được chốt.
- Giữ kiến trúc WavLM Large, acoustic objective CROTTC/consistency, IF teacher và decoder của tác giả. Tái hiện các giai đoạn AM, khởi tạo CTC/Transformer và thay AM CROTTC để fine-tune IF theo paper; không thay thành frozen-H6/GRU compact hoặc âm thầm rút lịch học để gọi là full.
- Paper ghi AM 300 epoch, LM/teacher 200 epoch, batch 32. Đối chiếu paper với source trước đăng ký lịch từng stage. CPU/GPU placement, microbatch/accumulation, activation checkpointing/offload và multi-GPU trên Kaggle phải profile và ghi sai khác. Run local đã chứng minh full model chạy được nhưng chậm (~7.7 ngày cho riêng AM CROTTC 300 theo cửa sổ ngắn); đây là căn cứ chuyển nền tảng, không phải chứng minh local bất khả thi.
- Trước optimizer update phải có manifest/hash, speaker split, supervision audit, budget theo từng cửa sổ hữu hạn, resource profile và cơ chế ngăn job trùng khi dùng GPU (shared flock trên local; khóa/định danh run tương ứng trên Kaggle). Checkpoint atomic phải lưu đủ weights/optimizer/scheduler/scaler/RNG để tiếp tục; điều kiện dừng rõ khi nonfinite, mismatch hoặc hết tài nguyên.
- Theo dõi riêng **training recipe**, **checkpoint inference**, **metric paper**, **strict S/O/I/feedback**; inference smoke 3 câu đã có không hoàn thành các phần còn lại. Điểm 71,77% vẫn là kết quả tác giả đến khi được đo lại đúng protocol.
- Lập bảng phủ các thí nghiệm trong paper: kết quả chính, baselines/ablations, generalization và challenge liên quan. Phần thiếu dữ liệu/artefact/quyền truy cập phải ghi `NOT_REPRODUCED`, không bỏ khỏi điều kiện tuyên bố full paper.
- Giữ nguyên test niêm phong. Nếu paper split hoặc corpus chưa được phép truy cập, tiếp tục các phần train/development được phép trên nền tảng đã chọn, ghi rõ sai khác; không tự mở test hoặc trình bày protocol local như benchmark gốc.
- Sau khi baseline được tái hiện có bằng chứng mới chuyển sang phương pháp cải tiến chính của khóa luận. Ngoại lệ userđãcho phép2026-11-14: nhánh SM13backbone bounded ở SIDE_GOAL.md được phát triển/audit/profile riêng song song, không thay goalchính hoặc tự mở rộng thành methodsweep.

## 3. Phạm vi đóng góp và giới hạn độ phức tạp

- Chọn **một vấn đề trọng tâm, một phương pháp cải tiến chính, một baseline gốc và ablation tối thiểu cần thiết**.
- Ghi trước phạm vi lỗi, giả thuyết/cơ chế, protocol, compute budget và quy tắc chọn model, trước khi đánh giá phương pháp mới. Quyết định hiện tại chưa chốt các chi tiết này.
- Đối chiếu implementation, experiment và literature trước khi gọi cơ chế là mới. Không đổi tên OT, alignment, distillation, attribute hoặc calibration đã thử thành đóng góp mới.
- Chỉ mở nhánh bổ sung để giải quyết thiếu hụt cụ thể của giả thuyết chính; không sweep kiến trúc hoặc mở rộng đủ mọi S/O/I như nghĩa vụ mặc định.
- Không cam kết tái lập điểm paper hoặc gain trước thực nghiệm. Có code/checkpoint không đồng nghĩa recipe training vừa GPU 4 GB.

## 4. Bằng chứng thành công

| Thành phần | Bằng chứng cần tạo |
|---|---|
| Căn cứ âm học | Cảnh báo đối chiếu được với audio và nhãn độc lập; inference không nhận gold transcript, nhãn lỗi hoặc gold timestamp của câu đánh giá. |
| Chẩn đoán | Đúng vị trí và loại/đặc điểm lỗi trong phạm vi; đánh giá âm thực tế khi có gold phù hợp; ghi bất định thay vì đoán phần thiếu. |
| Hướng dẫn sửa | Tương ứng lỗi thực, đúng ngữ âm, cụ thể, dễ hiểu và phù hợp người học; có người đánh giá theo rubric định nghĩa trước. |
| Cải tiến phương pháp | So baseline gốc với phương pháp mới trên dữ liệu/protocol phù hợp và ablation; công bố khác biệt compute, dữ liệu, seed và giới hạn kết luận. |
| Toàn tuyến | Đánh giá từ audio đến phản hồi cuối cùng, tính cả cảnh báo sai và bỏ sót; không chỉ chấm lời khuyên cho lỗi đã phát hiện đúng. |

Báo precision/recall/F1 theo phạm vi, cảnh báo sai/câu và độ phủ. Bổ sung hai phép đo toàn tuyến với đơn vị/rubric đăng ký rõ:

1. **Độ đúng của phản hồi:** số cảnh báo đồng thời đúng vị trí, đúng chẩn đoán và đúng hướng dẫn / tổng cảnh báo đã đưa ra.
2. **Độ phủ hữu ích:** số lỗi thật trong phạm vi nhận được phản hồi đúng trọn vẹn / tổng lỗi thật trong phạm vi.

Báo số đếm và trường hợp không ước lượng được. Ngưỡng định lượng, cỡ mẫu và deadline chưa chốt; phải đăng ký, không tự gán trạng thái đạt. Người đánh giá cần bằng chứng phù hợp, được che danh tính phương pháp khi phù hợp, và ghi bất đồng. Phản hồi tốt không tự chứng minh người học tiến bộ sau luyện tập.

## 5. Dữ liệu và tài nguyên

- Giữ ranh giới dữ liệu và official test niêm phong theo AGENTS.md. Benchmark chứa test không đồng nghĩa được phép mở test.
- Kiểm tra training-speaker provenance của checkpoint L2 trước phép so sánh speaker-disjoint. Public checkpoint không mặc nhiên sạch với các fold local.
- CROTTC compact trong `research/three_method_pilot/` và `research/three_method_scaling/` không phải bản gốc đạt 71,77%; điểm của nó không thay kết quả tái lập chính thức.
- Giữ EVAL-V3.1 strict S/O/I và exact-gap làm mốc kỹ thuật khi phù hợp. Metric paper/word-level feedback phải ghi đơn vị, mẫu số và split; không xếp hạng trực tiếp khác protocol.
- Chọn nhóm lỗi trước đánh giá; không chỉ chọn ví dụ đẹp. Tập đã dùng nhiều lần vẫn là development, không đổi tên thành xác nhận mới.
- Full training tiếp theo dùng **Kaggle** theo quyết định mới, thay chỉ định cũ không provisioning Kaggle. Local dùng cho chuẩn bị/test/audit/phân tích và lưu bằng chứng. Không tự provisioning Colab hoặc resume local. Profile accelerator/RAM/disk/quota thực tế, budget hữu hạn, checkpoint đầy đủ và provenance; chỉ tính checkpoint đã persist sau khi output được tải/verify.
- Migration từ checkpoint update 124 phải giữ weights/optimizer/scheduler/RNG/cursor và7 pending gradients. Current drivers có path/hash/runtime contract local, chưa thể chạy nguyên trạng trên Kaggle. Ghi old/new hashes, path mapping và runtime/placement changes; không sửa receipt cũ hoặc âm thầm reset native.
- Chỉ stage private TRAIN 1800/DEV_SEARCH 599 từ manifest allowlist, verify SHA sau mount; không upload full archives, official test, credentials hoặc toàn bộ data tree. Giữ nguyên original Drive archives và không đưa weights/data vào Git.

## 6. Công việc tiếp theo

- [ ] **Kaggle migration:** kiểm runtime/accelerator/quota thực tế, private package/source/checkpoint inventory, adapter cho input/output/path/lock và migration receipt giữ đủ state update 124.
- [ ] **Kaggle execution:** native resource profile đại diện, resume/save/download/reload chứng minh được, rồi AM CROTTC 300, AM CTC 300, IF CTC 200 và IF CROTTC 200 theo parent gates.200 epoch mỗi IF stage vẫn provisional theo preregistration; không hạ lịch hoặc gọi native IF đã kiểm khi mới có unit tests.

- [ ] **Nền thực nghiệm:** xác minh artifact chính thức, provenance, khả năng chạy và baseline ở phạm vi ghi rõ. Tận dụng audit/smoke có sẵn, không làm lại phần đã xác minh.
- [ ] **Đăng ký cải tiến:** chọn một thiếu hụt có bằng chứng, phạm vi lỗi, giả thuyết/cơ chế, đối chứng, rubric phản hồi và budget theo thời gian còn lại; kiểm tra trùng nghiên cứu cũ.
- [ ] **Thực nghiệm:** triển khai phương pháp, so sánh có kiểm soát với baseline và ablation cần thiết; giữ kết quả âm và ca thất bại.
- [ ] **Đánh giá toàn tuyến:** tạo phản hồi từ dự đoán thật, thẩm định chẩn đoán/hướng dẫn, báo độ đúng/độ phủ và giới hạn; hoàn thiện báo cáo khóa luận.

Các ô trên chưa được hoàn thành bởi milestone M0–M6 cũ. Cập nhật này chỉ đóng băng quyết định, không chứng nhận kết quả phương pháp mới.

## 7. Quyết định thay thế và lưu vết

- [GOAL trước quyết định, nguyên văn](reports/GOAL_DECISION_20261113/GOAL_SUPERSEDED.md).
- [Quyết định hiện hành chuyển full training sang Kaggle](reports/CROTTC_IF_KAGGLE_20261114/DECISION.json).
- [Lịch sử local và checkpoint bàn giao](reports/CROTTC_IF_KAGGLE_20261114/LOCAL_SESSION_HISTORY.json).
- [Checkpoint cho phép full local trước đây — lịch sử, không còn là nền tảng mặc định](reports/CROTTC_IF_FULL_LOCAL_20261113/DECISION.json).
- [Bản ghi quyết định](reports/GOAL_DECISION_20261113/DECISION.json).
- [Biên nhận hash snapshot](reports/GOAL_DECISION_20261113/ARCHIVE_RECEIPT.json).
- Mọi `GOAL_before_*`, `GOAL_after_*`, báo cáo/marker hoàn tất campaign cũ chỉ là lịch sử, không phải GOAL hoạt động song song hoặc bằng chứng hoàn thành khóa luận.
