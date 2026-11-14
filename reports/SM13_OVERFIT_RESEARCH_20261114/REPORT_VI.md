# Xử lý overfit SM13: nghiên cứu và thứ tự đáng thử

**Ngày nghiên cứu:**2026-11-14. **Phạm vi:**focusedreview11nguồnprimary, đối chiếu auditSM13 hiện tại; không systematicreview/khôngtuyênbốbaoquátmọipaper2026. **Không training hoặc thay recipe.**

## Kết luận ngắn

**Không nên coi tăng dropout là hướng mặc định.** Với fitF1~99.61%/held~40.88%, heldS/OCE tăng mạnh và dropout runtime đúng, nên ưu tiên:

1. **Quy tắc checkpoint/early stopping khóa trước**, dùng TRAIN-derived speaker-disjoint calibration; không đổi kết quả fixed300 lịch sử.
2. **Output regularization nhẹ**, trước hết một pairedtrial label smoothing; confidencepenalty là alternative, không gộp cảhai từđầu. Các phương pháp này khớp triệu chứng tựtinquámức hơn việc giảđịnhdropoutbịlỗi. [E3,E4,L1,L2]
3. **Giảm capacity head** như một đối chứng riêng, hoặc **R-Drop consistency** nếu outputregularization không đủ. Không thay cảarchitecture/loss/dropout/LR cùng lúc. [E1,E5,L1]
4. **Về dài hạn:** thêm errorlabels thực và speakerdiversity hoặc augmentation có kiểm chứng semanticlabels; không chỉ thêm pseudo-labelCORRECT cùngspeakers. MDDaugmentation có bằng chứng phù hợp hơn augmentationASRđemápmùquáng. [E7,E8,L3]

**Độ tin cậy:** cao về hiện tượngoverfit/runtime; vừa về thứtựưu tiên theomechanism; **chưa biết** gainF1 của bấtkỳ canthiệp mới nào trênSM13. Đây là recommendation nghiên cứu, không tựthay authorization hoặc protocolpilotdropout cũ.

## 1. Chúng ta cần chữa cái gì?

Audit local xác nhận:
- PooledfitF1@.5:96.6907→99.6100%; held:40.8143→40.8843% từnormalized50→300.
- TargetheldCE1.6889→3.0304; sai nhưngconfidence≥99%867→1290.
- Omissionbắtđúng141→103/357; substitution643→647/1859.
- ~94.3%positives làS/O; macroAPgain lớnnhất ởinsertion nên APtăng khôngđủchứngminh main detectiontăng.
- Dropout bật/tắtđúng trong head; p quáthấpchưachứngminh. [L1,L2]

Có ba mục tiêu khác nhau:
1. **Ngănmemorization/độ tựtinquámức trongtraining**: earlystop,capacity,lossregularization,dropout/RDrop.
2. **Tăngthôngtin/generalization sangspeakerskhác**: diversity/labels,augmentation/taskadaptation.
3. **Chọnoperatingpoint/calibrateconfidence**: calibration/threshold. Mục3khôngtựchữamục1/2. [E11]

## 2. Các nhóm phương pháp và mức phù hợp

| Hướng | Tác động kỳ vọng | Dùngcache hiện tại? | Bằng chứng / hạn chế | Ưu tiên |
|---|---|---|---|---|
| Earlystopping/checkpointpolicy | Khôngtiếptụcfitkhivalmetricxấu; khôngtựnângtrầnrepresentation | Có | Classicalvalidationevidence[E1]+localcurves; cầnselectionsplitriêngđểkhôngtựchọnrồichấmcùngheld | Nền tảng bắt buộc cho lượt mới |
| Label smoothing nhẹ | Giảmáp lựcđẩygoldprob→1 trênfit | Có | Calibration/generalization[E3]; phonemerecognitionwithdropout[E4], khôngMDDexactF1 | Ứng viên rẻ đáng thử đầu |
| Confidencepenalty | Phạtoutputentropythấp bằng−βH(p) | Có | E4cóTIMIT/WSJ; hiệuứngdiminishingtrongmộtsốsetting | Alternative riêng choLS |
| Smallerhead/linearprobe | Hạnchếkhảnăngghi nhớ, kiểmtracapacityhypothesis | Có | Classicalcapacity[E1], localfitnearperfect; chưacómatchedsmallheadSM13result | Cao,vai trò diagnostic |
| R-Drop | Hai dropoutviews phải nhấtquán ởoutput | Có | NeurIPS2021[E5],18NLP/CVdatasets, chưaphảiMDD; tăngcompute | Cao saucheapcandidate |
| Dropoutmạnhhơn | Giảmcoadaptationhiddenunits | Có | JMLR[E2]; p2xkhôngđượctài liệuđảmbảo vàruntimehiệntạiđúng | Đốichứng,khôngmặcđịnhfix |
| WD/LRschedule | Regularizationweights/optimizationtrajectory | Có | AdamW[E6], nhưngSM13đãdùngAdamW; LRgiảmkhôngđồngnghĩageneralizationtăng | Secondarysinglefactor |
| Speaker/errorlabeldiversity | Họcđượccácquyluậtítphụthuộcspeaker/corpus | Khôngnếuaudiomới | E7/E8treatscarcity; localselftrainingtrialkhônggiảiquyếtnewdiversity | Ưu tiên dài hạn; dataauthorization |
| MDDspecificaugmentation | Thêmpositiveerrorsrealistic/referencecounterfactual | Thườngkhông | E7referencebias; E8waveformblending, metric/modelkhácSM13 | Promisingnhưngsemanticgate |
| GenericSpecAugment/noise/speed | Robustacousticviews | Khôngđốivớiaudio/frameaugmentation | E9làASR; nhãnlỗiphátâmphảiđượcgiữđúng | Chỉsaulabelpreservationaudit |
| PartialFT/LoRA/S2adapted | Đổirepresentationnếutaskmismatch | Không | E10cảnhbáoOOD; localLoRAcũkhônggainổnđịnh, khôngbácbỏmọiLoRA | Saucheapablation/lineage/profile |
| Temperaturescaling/threshold | Scorecalibration/P-Roperatingpoint | Thườngcó, nhưngcầnlogits/calibration | E11calibrationpaper, khôngchữaheadmemorization | Báoriêng,khônggọioverfitfix |

## 3. Vì sao output regularization đáng cân nhắc trước khi tăngdropout?

### 3.1 Label smoothing

Thaytargetonehot bằngphânphối:

`q_k = (1−epsilon)·1[k=y] + epsilon/K`.

Khôngsửa goldannotations; đây là losstarget trongtrain. Nó giảmáp lực phải cho probability1 cho từngfitexample. Đề xuất **epsilon=.05** cho mộtpilot là lựa chọnpragmaticnhỏđểthử, **khôngphảigiátrịpaperchứngminhtốiưuchoSM13**. Giữdropout/LR/WD nhưcũ. [E3,E4]

E4§4.4.1/Table5 báoTIMITphonePERdropoutbaseline23.2→21.6 vớiLS,22.0 vớiconfidencepenalty(mean/SEM5runs). Đây là **PERphonerecognitionkhôngphảiSM13detectionF1**. E3cũngcócấuhìnhkhôngtăngaccuracy, vàteacherLS cóthểlàmdistillationkémtốt. Không hứa gain tươngtự.

**Implementationtrapquantrọng:** targetSM13đangweightedCE[1,2.5,4], gapdùngsqrtinversefrequency. Khi smoothing phải khóa rõ:
- classweightsápvàomỗitermsofttargethaytheogoldclass;
- denominator/reduction của losstrain;
- heldCEdiagnostic vẫnđượctính theo goldhardlabels/originalweightsđểso sánh.

`CrossEntropyLoss(weight=...,label_smoothing=...)` khôngđượcmặcđịnhcoilàmọiđịnhnghĩaweightedsmoothingđềugiốngnhau. Cầntoyexpectedloss/gradienttests, khôngvừađổiweightingvừagọichỉthêmLS. Theo dõiS/O/Irecall vàFWR; minorityclass cóthểbịảnhhưởng.

### 3.2 Confidencepenalty

`L = L_weightedCE − beta·H(p)`.

Chốngoutputquápeaked bằngmộtpenaltyriêng, khôngđổiannotation. E4giảithíchLS/confidencepenaltyquahướngKL, khôngphảihai thứđồngnhất. βcầnspec/finitedevelopmentbudget; khôngtựgộp vớiLS/dropout2x/LRdecay. [E4]

### 3.3 R-Drop không phải chỉ tăngp

Giữnguyênp; cùngsamplequaheadhaimaskđộclập:

`L = (CE(p1,y)+CE(p2,y))/2 + alpha·(KL(p1||p2)+KL(p2||p1))/2`.

Ưuđiểm: dùngcache vàfrozenbackbonenhưcũ, épquyluậtclassifierổnđịnhquamasks. Nhượcđiểm: haiviewstăngheadcompute/memory,newalpha, cần faircostaccounting/pairedtwoCE-noKLcontrolnếumuốnquygaindokhốiconsistencychứkhôngchỉdoubleexposure. E5khôngcóbàiMDDtrong18datasetsđượckiểm, nêntransferlàgiảthuyết.

## 4. Earlystopping và giảmcapacity: rẻ nhưng khôngđượcdùngđểcherrypick

LocalmeanF1window11–20≈44.41%,41–50≈42.18%,291–300≈39.65%. Đâylàbằngchứnggiữtrainingdàikhônggiúpbềnvững, **khôngphảiquyềnchọnepochđẹpnhấttronghistoryrồigọiqualitygain**. [L2]

Lượtmodelselectionmới cầnTRAIN-derivedspeaker-disjointselectionpopulation, criteria/patience/tiebreak/thresholdruleđăngkýtrước; đầurađánhgiácuốikhácpopulationdùngchọn. Reused450TRAINheldchỉdevelopment; fullTRAIN1800→DEV_SEARCH599/commonmetricvẫnpendingvàDEV_SEARCHđãđượcreuse. Test/DEV_CALtiếptụcsealed. [E1,L4]

Smallerhead là diagnostic rõ: giảmwidths cònmộtnửa (target256/128→128/64,gap128/64→64/32), giữactivation/dropout/layers/features/loss. Architecturekhácthìkhôngthểinitialweightsequalityvớibaseline; same seed chỉkiểmsoátRNG, khôngmatchedtensorinitialization. NếugiảmfitF1nhưngheldF1tăng vàS/O/I/FWRqua safeguards, đó làevidencecapacityregularizationgiúp; nếuchỉfitvàheldcùnggiảm thìunderfit, khôngđượccoigiảmgapthànhcông.

## 5. Dữ liệu/augmentation: nơi có thể giải quyết giới hạn mạnh hơn

### 5.1 Đừng đánh đồng thêmdata với thêmCORRECTpseudo-label

HistoricalS0trial đãthử600câuthêmtừcùng12speakers;96.25%pseudo đượcgiữlàCORRECT. APkhôngtăngổnđịnh. Điềuđóchưakiểmchứngbenefitcủaspeakers/accentmới hoặcnhãnlỗithực. Nhữngtrials nàyprotocolkhácS1, khôngđượcdùngsốF1đósotrựctiếphiệntại. [L3]

Ưu tiênnghiêncứudata: bổsungerrorpositivesđa dạng/câncoverageS/O/I,bằnghumanannotationhoặcpipelineđãaudit; giữspeakerdisjointvàsourceallowlist/license. Labelnoise/ambiguityreviewởauthorizedTRAINtrướckhithêmđộphứctạp; khôngâmthầmchuyểngoldhoặcrelabelhardcases.

### 5.2 MDD-specificaugmentation

**Fu2021[E7]** chỉrareferenceconditioningcóthểkhiếnmodelcopycanonicalphonesvàrecallthấp; paperthửthay10–20%phoneprior bằngphone/vowel-consonant/confusionpairs. MatchedphonemeattentionmodelFmeasure52.51→56.08 ởVC10%;49.29làCNNRNNCTCbaselinekháckiếntrúc. PaperdùngTIMIT+L2ARCTIC/hierarchicalmetrics, khôngchứngminhgiảipháptrựctiếpchoSM13.

**Nguycơ chuyểnmù:** SM13referencevừađịnhnghĩaintendedphonevừaNtargets/N+1gaps; thayreferenceS/O/Isemantics vàeventmappingcóthểđổi. Khôngđượcđổirefnhưnggiữlabels/IDs/alignmentscũ. Nếuchỉmaskreferenceđểngănshortcut thìđólàreference-dropoutkhácmethod,nênđăngkýriêng. Hiệntạichưaquantifyreference-copyshortcuttrênSM13; zero-referenceOODablationkhôngđủcausalproof.

**SpeechBlender[E8]** tạoaudioerrorsbằngsmoothblendingtheophoneticunits/confusionpairs; officialSLaTE2023, preprint2022. PapercóArabicspeakerdisjoint6fit/5test,GOP/LSTM,EnglishpronunciationscorePCC. NóchoevidenceaugmentationcóthểgiúpMDDnhưngkhôngchứngminhinsertionexactgap/omissionlabels hợpvớiSM13. Cần donor/candidatechỉtừfit, confusionstatsfit-only,realisticlabelvalidation/predictedalignmentrerun, freshcacheidentity, khôngnhậnheld/testvàoaugmentation.

### 5.3 Không gọi mọi featuremask làSpecAugment

E9masktime/frequencytrênfilterbankinput. HiệnSM13cachelàpooledSSLrepresentation, khôngcòntime/frequencyaxesđúngnghĩa. Maskrandomcoordinate13×3072làaugmentationkhác, khôngthểtríchnhậngainASRpaperchono.

Noise/speed/pitch/phonemecorruptioncóthểđổiacousticcuehoặcboundaries; speechcontentđượcgiữkhôngđồngnghĩapronunciationerrorlabelđượcgiữ. Khôngápaggressivemaskinglêncácphonemehiếm màkhôngđolabelpreservation/coverage.

## 6. Fine-tune backbone / zero-oracle có giải quyết không?

- Fine-tunecóthểđổirepresentation, nhưngkhôngtựgiảmoverfit. E10choOODcounterexampletrênvision; khôngđượcquycon sốvisiongain/losschoMDD. LocalshortLoRAtrướcgiúpít/khôngổnđịnh, nênkhônghứachỉunfreezelàvượtplateau. [E10,L3]
- S2CROTTC-adaptedbackbone đãhọcTRAIN1800khôngđượcdùng450TRAINheldđểgọilàindependentOOF. Selected/finalparent/resource/protocolgatesvẫncần. [L4]
- Zero-oraclelàcontractđầuvào/lineage/khôngdùnggoldtiming-realization-errorlabelsởinference; nókhôngregularizeheadvàkhôngtạoerrorlabel/speakerdiversity. Textderivedreferenceauditcóíchchovalidity, khôngphảibằngchứngchữaoverfit.

## 7. Thử gì trước — đề xuất, KHÔNG tự launch

**Táchmodelselectionpolicykhỏicandidateablation.** Khiđánhgiácandidateởfixed50thìmọiarmđềufixed50,khôngchọnriêngepochđẹpcủamộtarm. Earlystopnếudùngtrongphasekếtiếpthìcùngrule/budget/tiebreakchohaiarm, khôngđổihistoricfixedresults.

| Đợt có điều kiện | Baseline | Candidate | Một biến thay đổi | Endpoint |
|---|---|---|---|---|
| A, đềxuấtưu tiên | Freshnormalizedrecipe | Same+LSε=.05 | Losssofttargetdefinitionđượclock/toytest |50/head,fullcurves,last10 |
| B, chỉnếucần | Freshnormalizedrecipe | Halfwidthhead | Capacityonly |50/head,cùngselectionpolicy |
| C, nếuA/Bkhôngđủvàcóbằngchứngđángthử | Freshnormalizedrecipe | RDropαlocktrước | SymmetricKLconsistency;costcontrol |50/head,cùngsample/optimizerupdates |

Khôngphảimatrix3đợttựđộng. Chỉđềxuất **mộtpairedtrialA** trước, khônggộpA+B+C. Pilotdropout2xđãcóprotocolkhácnhưngchưatrain; researchrecommendationkhôngtựchuyểnquyềnsangLS. Cầnuserchốtcanthiệp,trướcregistration/admission.

**Acceptance:** giữ6safeguardspreregisteredpilot-style(F1@.5fixed50≥+1pp,last10delta≥0,macroAPregression≤.5pp,eachS/O/IAPregression≤2pp,≥2/3speakersnotworse,FWRincrease≤.5/câu); thêmreportfit/heldevalCE,wrongconfidence≥.99,Brier/NLL,recallomission/insertionsupport. Coefficient/safeguardskhôngtunehậunghiệm. Outputmềmhơnhoặcgapnhỏhơn màheldF1xấukhôngphảiqualitygain.602historicalstates/protectedartifacts giữ nguyên.

**Resourceplanning(notmeasurement):** twofresh50/headarms≈0.25–1GPUhour/conservative≤1hproductionwindow, dựatrêncompletedcachedheadruns; phảiprofilethựctế. LS/smallerheadkhôngnewbackbonecache/APIcost; outputadmissiontạm1.5GiB/4GiBhomefloor cầnđotensors/receipts,predictions đểchốt. RDropcostcaohơn, cầnprofile/budgetriêng. Independent14GiB/var/tmpCROTTCexportfloor/sharedGPUlock/RAM16initial4runtimeGiB/VRAM512MiBkhôngđượcgiảm. Tạipreflighttrướcfree/var/tmpchưađạt14GiB; chưaadmitrunmới.

Khôngtạotraininglaunchscript hoặc giảdriverđãsẵnsàng: userhiệntạiyêucầuresearch. Khiđượccấpquyềncụthể,mớitạoconfig/driver/tests/checkpoint-verifier; chưađưaexecutioncommandchochứcnăngchưaimplement.

## 8. Nguồn / giới hạn

[E1] https://publikationen.bibliothek.kit.edu/7498 —earlystopping1998.
[E2] https://jmlr.csail.mit.edu/papers/v15/srivastava14a.html —dropout2014.
[E3] https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html —labelsmoothing2019.
[E4] https://arxiv.org/abs/1701.06548 —confidencepenalty2017,fullPDF§4.4.1/Table5.
[E5] https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html —RDrop2021.
[E6] https://arxiv.org/abs/1711.05101 —AdamW,ICLR2019.
[E7] https://arxiv.org/abs/2104.08428 —FuMDDaugmentationpreprint2021,§2.2/§3.1/§3.3.
[E8] https://www.isca-archive.org/slate_2023/elkheir23_slate.html —SpeechBlender2023,§2/§3.1/Table3–4.
[E9] https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html —SpecAugment2019.
[E10] https://arxiv.org/abs/2202.10054 —FT/OOD,ICLR2022.
[E11] https://proceedings.mlr.press/v70/guo17a.html —calibration,ICML2017.

Detailedpassages/sourcequality/localanchorsin`EVIDENCE.md`; exactqueries/rejectedleadsin`SEARCH_NOTES.md`.

**Giới hạn:** focusedreview,singlescreener,Googlegrounding+primarypublisher/arXivretrieval; khôngsystematicmeta-analysis/khôngđịnhlượngexpectedF1gain. KhôngcóopenedpaperchứngminhLS/RDrop/dropout2xtrênexactfrozenWavLMLarge/RMS/SM13protocolnày. Foundationalworksolderthan12monthsđượcdùngchomechanisms,khônglatestSOTAclaim. SourceclinicalMDDmis-expansion/wrongFuISCApage/notfoundOpenReviewđãloại; arXivfailedfetchđượcretrievequaPDFalternate. Khôngthaytraining/gold/referencepolicy/authorizationhaymởsealedsets.
