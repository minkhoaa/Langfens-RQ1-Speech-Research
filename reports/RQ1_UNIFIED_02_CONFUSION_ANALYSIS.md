# Phân tích confusion và chất lượng quyết định

**Population:** DEV_SEARCH đã exposed, 19,755 target/20,354 exact gaps, 599 utterances. AP/P@K là ranking raw; các ma trận và P/R/F1 bên dưới gắn với policy CAL20 chọn từ DEV_CAL. Không trộn policy unrestricted với CAL20. `FWR` là false warnings per utterance; không dùng ngưỡng ≤0.20 làm điều kiện thành công.

## Ranking và top-K

P@K lấy K bằng số positives thật của từng lớp trên SEARCH (S=1,768; O=209; I=125), do đó Recall@K bằng số true positives trong top-K chia support. Insertion I là exact N+1 gap ranking.

| Arm | AP-S/O/I | P@support (Recall@support) S | O | I |
|---|---|---:|---:|---:|
| Gold-Only | .364602/.159433/.186715 | .4089 | .2105 | .2960 |
| H-002 target + GoldOnly gaps | .377810/.167294/.186715 | .4417 | .2392 | .2960 |
| U2 | .371360/.185318/.117992 | .4395 | .2584 | .1920 |
| U3 | .362413/.176117/.123822 | .4338 | .2536 | .2240 |
| G1 bottleneck128 | .371360/.185318/.125929 | .4395 | .2584 | .2080 |
| Balanced gap sampler | .371360/.185318/.125692 | .4395 | .2584 | .2160 |
| G2 legacy 6D gap_X only | .371360/.185318/.025938 | .4395 | .2584 | .0960 |
| G6 paired crop-edge sequence, no phone vectors | .371360/.185318/.108997 | .4395 | .2584 | .1760 |
| G6-REF same sequence + intended phones | .371360/.185318/.099673 | .4395 | .2584 | .1920 |
| G8 full crop context + intended phones | .371360/.185318/.116505 | .4395 | .2584 | .1920 |
| G6 continuous predicted-boundary W2V17 + intended phones | .371360/.185318/.080151 | .4395 | .2584 | .1280 |
| G9 G3 + CharSiU CTC6 S101 | .371360/.185318/.155957 | .4395 | .2584 | .2480 |
| G9 G3 + CharSiU CTC6 S202 | .362558/.170159/.132255 | .4344 | .2249 | .2240 |

U2 scores/target classes are held fixed in G1 and balanced rows; only the gap ranking changes. Top-K rates are descriptive, not selected operating thresholds. Gap positives near top of ranking remain low (U2 24/125 top-125, G1 26/125, balanced 27/125); the GoldOnly gap head ranks 37/125. Of 125 SEARCH insertion positives, 124 are internal; leading has 0/599 positives and trailing 1/599, so edge-specific rates/AP cannot be interpreted reliably.

## CAL20 operating decisions

### Class P/R/F1 and gap decisions

| Arm | S P/R/F1 | O P/R/F1 | I P/R/F1 | Macro3 Error F1 | Macro4 F1 | Binary MD AP/F1 | FWR |
|---|---|---|---|---:|---:|---:|---:|
| U2 | .4783/.0124/.0243 | .3299/.1531/.2092 | .3793/.0880/.1429 | .12542 | .33071 | .36332/.05755 | .17863 |
| G1 bottleneck128 | .4783/.0124/.0243 | .3299/.1531/.2092 | .3478/.1280/.1871 | .14018 | .34178 | .36332/.05755 | .19866 |
| Balanced sampler | .4783/.0124/.0243 | .3299/.1531/.2092 | .4186/.1440/.2143 | .14923 | .34856 | .36332/.05755 | .19032 |
| G6 continuous predicted-boundary W2V17 | .4783/.0124/.0243 | .3299/.1531/.2092 | .3250/.1040/.1576 | .13033 | .33439 | .36332/.05755 | .19366 |

G1 and balanced use unchanged U2 target scores, so identical target confusion and binary-MD metrics are expected. G6 continuous-boundary also fixes U2-S101 target scores; its insertion CAL20 P/R/F1=.325/.104/.158 and Macro3 F1=.13033, but raw AP-I falls to .08015 SEARCH. Balanced gives +.02381 Macro3 F1 vs U2 at its own CAL20 policy but CAL insertion AP-I falls from .13811 to .09404; no raw-ranking/calibration consistency or seed replication yet. Its SEARCH AP uplift is only +.00770 and exposed. These are screening observations, not validated improvements.

### Full confusion matrices

Matrices are **gold rows → predicted columns**, target classes C=CORRECT/S=SUBSTITUTION/O=OMISSION; gap rows show TP/FP/FN/TN.

**U2, unrestricted diagnostic** (not deployment policy; FWR=4.654):

| Gold \ Pred | C | S | O |
|---|---:|---:|---:|
| C | 15,300 | 1,846 | 632 |
| S | 634 | 1,043 | 91 |
| O | 66 | 30 | 113 |

Gap TP=31, FP=189, FN=94, TN=20,040. At this permissive point target S recall=.5899, O recall=.5407, insertion recall=.2480. Correct-class target warnings are extensive; many nominal errors are also misrouted to another error class.

**U2, CAL20:**

| Gold \ Pred | C | S | O |
|---|---:|---:|---:|
| C | 17,696 | 22 | 60 |
| S | 1,741 | 22 | 5 |
| O | 175 | 2 | 32 |

Gap TP=11, FP=18, FN=114, TN=20,211. Substitution recall collapses to 22/1,768=.0124 even though AP-S=.37136. This demonstrates operating-policy mismatch/threshold conservatism rather than no ranking signal. CAL20 is selected under a warning budget and is not a universal accuracy optimum.

**G1/balanced target confusion at CAL20** equals U2 because target scores and target threshold are unchanged. G1 gaps TP=16, FP=30, FN=109, TN=20,199; balanced gaps TP=18, FP=25, FN=107, TN=20,204. G6 CAL20 gaps TP=7, FP=10, FN=118, TN=20,219 (P=.411, R=.056, F1=.099); CAL20 Macro3 F1=.11067, Macro4=.31964, FWR=.16528. Its SEARCH AP-I is .10900, below U2; because it also removed reference-phone vectors, do not attribute this result to temporal context alone. G6-REF restores reference vectors: CAL20 gap TP=10, FP=23, FN=115, TN=20,206; insertion P/R/F1=.303/.080/.127, Macro3 F1=.12000, Macro4=.32664, FWR=.18698. CAL AP-I rises .06280→.10078 from G6 but SEARCH AP-I falls .10900→.09967, still below U2; one-seed result. G2 6D-only at CAL20 uses the same target matrix as U2; gap TP=2, FP=9, FN=123, TN=20,220 (P=.182, R=.016, F1=.029), consistent with very weak raw AP-I .02594.

**G3 acoustic+6D, per seed CAL20 policy** (same-seed U2 target scores; C/S/O target matrices are gold rows to predicted C,S,O):

| Seed | C gold→pred | S gold→pred | O gold→pred | Gap TP/FP/FN/TN | AP-I SEARCH | Macro3 AP SEARCH | CAL20 M3 F1 / FWR |
|---:|---|---|---|---|---:|---:|---:|
| 101 | (17696,22,60) | (1741,22,5) | (175,2,32) | 11/20/114/20209 | .13416 | .23028 | .12481 / .18197 |
| 202 | (17743,22,13) | (1756,11,1) | (198,0,11) | 29/93/96/20136 | .13572 | .22281 | .11368 / .21536 |
| 303 | (17632,96,50) | (1674,91,3) | (177,3,29) | 1/1/124/20228 | .12813 | .22426 | .10267 / .25543 |

G3 target raw scores are inherited from same-seed U2; CAL-selected target thresholds differ, so decision matrices vary by seed. Gap recall also varies sharply: seed303 predicts only 1 positive gap and misses 124/125. Mean G3 SEARCH AP-I=.13267 vs matched U2=.13900; no consistent ranking gain. G8 CAL20 gaps TP=17, FP=24, FN=108, TN=20,205; insertion P/R/F1=.415/.136/.205, but raw AP-I=.11651 and CAL AP-I=.07922, below U2. FWR is descriptive, not a hard gate.

**U2/U3 CAL20 matrices by matched seed** (gold rows→C,S,O predicted counts; gap TP/FP/FN/TN):

| Seed/model | C→pred | S→pred | O→pred | Gap TP/FP/FN/TN | Macro3 F1 | Macro4 F1 | FWR |
|---|---|---|---|---|---:|---:|---:|
| 101 U2 | (17696,22,60) | (1741,22,5) | (175,2,32) | 11/18/114/20211 | .12542 | .33071 | .17863 |
| 101 U3 | (17755,0,23) | (1766,0,2) | (194,0,15) | 7/13/118/20216 | .07234 | .29104 | .06344 |
| 202 U2 | (17684,66,28) | (1717,49,2) | (192,0,17) | 15/34/110/20195 | .11909 | .32592 | .21703 |
| 202 U3 | (17719,8,51) | (1753,9,6) | (185,0,24) | 3/0/122/20229 | .07416 | .29228 | .10851 |
| 303 U2 | (17632,96,50) | (1674,91,3) | (177,3,29) | 0/0/125/20229 | .09742 | .30967 | .25376 |
| 303 U3 | (17711,18,49) | (1743,20,5) | (180,1,28) | 0/0/125/20229 | .07153 | .29035 | .12187 |

U3 target routing is visibly threshold/policy-dependent: S recall is near zero in all three CAL20 points; its lower total FWR does not rescue error Macro F1. At seeds101/202 it still misses most exact-gap positives; at seed303 both U2/U3 predict zero insertion gaps. This is why raw AP and policy confusion remain separate. U3 unrestricted target matrix is C→(15589,1252,937), S→(770,848,150), O→(67,19,123); gap TP=35, FP=215, FN=90, TN=20,014. Its slightly higher insertion recall at that unrestricted point does not imply hierarchy caused the gap change.

## Interpretation

- **Ranking:** U2 raises O AP over H002 (+.0180), slightly loses S AP (−.00645), and loses I AP (−.06872). Gap-only substitution confirms the insertion ranking branch is weak.
- **Decision:** U2 CAL20 S recall is only 1.24%; the unrestricted choice raises target recall but costs FWR=4.65. This is a threshold-policy trade-off layered on top of AP capability.
- **Thresholds:** AP is invariant to threshold; class F1/confusion/FWR are not. Do not compare F1 from different threshold policies as if they were capability rankings.
- **Acoustic controls:** H002 reference-only retraining, acoustic-shuffle perturbation and zero-reference perturbation produce large target AP losses; insertion score is fixed GoldOnly in those controls, so they do not validate insertion acoustics.
- **G3 gap-input perturbations (inference only, seed101):** baseline AP-I=.134162; zero W2V blocks=.046932; deterministic acoustic shuffle=.011221; zero 6D gap_X=.083611; reference-only=.019925. CAL20 gap confusions: baseline TP=11/FP=20/FN=114/TN=20,209; zero-W2V TP=2/FP=2/FN=123/TN=20,227; zero-gap_X TP=7/FP=11/FN=118/TN=20,218; shuffled/ref-only TP=0/FP=0/FN=125/TN=20,229. Raw Macro3 AP baseline .230280; perturbations .201203/.189299/.213430/.192201 respectively. This supports sensitivity to acoustic/gap features but these are OOD perturbations, not retrained modality ablations.
- **G9 auxiliary posterior arm:** G3+local CharSiU CTC6 has CAL20 SEARCH gap TP=9/FP=14/FN=116/TN=20,215; insertion P/R/F1=.391/.072/.122; Macro3 AP=.237545/F1=.118343, FWR=.17195. It improves raw AP-I over G3 on exposed SEARCH but not CAL; second encoder and one seed limit interpretation.
