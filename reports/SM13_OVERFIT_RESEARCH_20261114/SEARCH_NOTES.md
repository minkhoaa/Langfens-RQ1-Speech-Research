# Research working notes — SM13 overfit

Scope: methods that could reduce the verified fit/held gap and improve held event detection F1 in frozen native WavLM Large + RMS-conditioned SM13 heads. Research only; no training authorization, test/DEV_CAL access, or recipe changes. Research date2026-11-14.

Questions before searching:
1. Which changes address overconfidence/memorization of small heads on fixed SSL representations, versus merely recalibrating a threshold?
2. What MDD-specific evidence exists for augmentation, mispronunciation simulation, label/data quality, speaker diversity, and task adaptation?
3. Which interventions can reuse the current cache (early stopping, capacity, loss/regularization) and which require audio/backbone recomputation?
4. Which popular options conflict with current event labels/alignment, or have already failed in local historical experiments?
5. What small, fair, preregistered comparison should precede more expensive fine-tuning?

Plan: primary papers/official proceedings and docs, approximately10–15 primary sources maximum; search original sources, open relevant pages, distinguish external evidence from untested transfer to current SM13. No broad automatic experiment grid. Rank by local mechanism match, evidence proximity, cost, and risk.

Local evidence:
- reports/SM13_DROPOUT_CAUSALITY_20261114/REPORT_VI.md: dropout runtime correct, strength not causal proof; pooledfitF1~99.61%/held40.88%, trueS/O confidence errors worsen.
- reports/SM13_NORMALIZED_DIAGNOSIS_20261114/AUDIT.json: fresh CPU fit/held/CE/confidence.
- experiments/sm13_three_directions_20261113/SYNTHESIS_VI.md: historical S0 pseudo-label/temporalCNN/LoRA trials no stable practical gain under a different protocol; do not claim these ideas are new or all refuted.
- GOAL_SM13.md: localonly, backbone/cache/test/lineage boundaries; no automatic finetune/sweeps/extension;14GiB CROTTC reserve. Prior dropout pairedpilot is constructionpreflightonly and resourceblocked.

## Executed searches (Google grounding,2026-11-14)

1. `mispronunciation detection diagnosis overfitting data augmentation self supervised speech small data label smoothing regularization WavLM`
2. `mispronunciation detection diagnosis synthetic errors data augmentation phoneme substitution omission insertion self supervised learning L2 ARCTIC`
3. `When Does Label Smoothing Help confidence penalty overfitting R Drop regularization dropout small datasets papers`
4. `early stopping But When Prechelt 1998 validation generalization neural networks SpecAugment Park 2019 fine tuning distort pretrained features out of distribution Kumar 2022`
5. `Decoupled Weight Decay Regularization Loshchilov Hutter AdamW R-Drop mispronunciation detection diagnosis speech recognition consistency regularization`
6. `"R-Drop: Regularized Dropout for Neural Networks" site:proceedings.neurips.cc`
7. `"A Full Text-Dependent End to End Mispronunciation" "isca-archive.org"`
8. `"R-Drop" "proceedings.neurips.cc/paper" "2021"`
9. `On Calibration of Modern Neural Networks Guo 2017 temperature scaling paper proceedings mlr`
10. `"Dropout: A Simple Way to Prevent Neural Networks from Overfitting" jmlr 2014`

Deduplicated bibliography:11primaryworks, including2MDDaugmentationpapers; preprintsandproceedings samework countedonce. No systematicdatabasehitcount orPRISMAclaim. Onlyoneprimarysearchengine used; arXiv/publisher retrieval viaread_url_content andfetch_content. Fulltexttargetedreads:earlystopping,confidencepenalty,labelsmoothing,RDrop,FuMDD,SpeechBlender. Officialabstractsreadforotherworks. Backwardleadsfromconfidencepenaltypaper toearlystopping/dropout/labelsmoothing; noexhaustiveforwardcitationsearch.

## Retrieval quality controls / rejected leads

- Groundingresponse3 mistakenly expandedMDD asMajorDepressiveDisorder; discardedclinicalclaims. MDDhereismispronunciationdetectionanddiagnosis;SM13internalname notpublicpaperidentifier.
- GuessedISCAURL`interspeech_2021/fu21b_interspeech.html` openedtoAISHELL4; rejectedaswrongpaper. Fu2021 usedverifiedarXivtextinstead. Noassertionthatnopeerreviewedversionexists.
- OpenReview`BJuysoF7hm`returnednotfound; notcited. AdamWusedverifiedarXivv3publishedICLR2019metadata.
- Severalread_url_content arXivcallsfailedHTTP500; fetch_content retrievedsamearXivabstracts/PDFs. Sourcesnotacceptedsolelyfromsearchsummary.
- SearchsummarySpeechBlendervenueyear2022correctedtoofficialSLaTE2023;2022ispreprintyear.
- Fu2021splitambiguityinsearchsummaryresolvedusingactualPDF§3.1:12fit/6dev/6test, notrandom20%holdout.
- SpeechBlendernumericclaimsarenotSM13effects; evidenceextractkeepsPCCseparatefromF1. NoestimationoffairmicroF1fromaggregatedpaperP/R.
- Temperature scalingdoesnotchangefixedmulticlassargmax,butSM13multiclasspS+pOorderingmaychange:avoidgenericclaimthatallAUC/APareunchanged.

FullsourceURLs/passages/limitationsarein`EVIDENCE.md`. Freshness note: boundedfocusedreview, notexhaustivecoverageoflatest2026literature; exactsearchresultcountsnotprovidedbygroundingtool. NoevidencefoundintheopenedsetofdirectRDrop/labelsmoothingoutcomesonthisexactfrozenWavLMLarge/RMS/SM13protocol; notacorpuswideabsenceclaim.
