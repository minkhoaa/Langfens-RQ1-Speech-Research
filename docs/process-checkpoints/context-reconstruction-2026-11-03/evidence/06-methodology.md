# Forensic Reconstruction: LANGFENS Seam 6 — Speech Research Methodology, Architecture Decisions & Literature Evolution

**Target Seam:** SEAM 6 — Speech Research Documents, Methodological/Architectural Evolution & Scientific Literature  
**Authority Repositories & Examined Heads:**  
- Speech Research: `/home/khoa/Projects/KLTN/Dataset` (`prep/rq1-shared-base` @ `1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6`)  
- Backend Microservice: `/home/khoa/Projects/langfens/Project_Langfens_Microservice` (`generation-harness` @ `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48`)  
- Frontend Application: `/home/khoa/Projects/langfens/langfens-fe-app` (`autonomous-generation` @ `2d9fe94a54e65680eca56aafbe40e691018113da`)  
**Auditor:** Independent Read-Only Research Auditor  
**Date of Forensic Synthesis:** 2026-11-03 (Timezone normalized: Asia/Ho_Chi_Minh, +07:00)  

---

## 1. Scope & Primary Literature Inventory

The research repository maintains a tiered literature library structured into five pedagogical and technical pillars (`manifests/RESEARCH_LIBRARY.md`). The audit categorized all 12 key works across local availability, cryptographic hashes, and licensing constraints:

### 1.1 Accessible Full-Text Local Assets (8 Verified PDFs)
1. **Baevski et al. (2020)** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020 / arXiv:2006.11477v3. `references/papers/wav2vec_2.0.pdf` (740,863 B, SHA-256 `4a881708378143547a5c053f62090104afb49782a4c71d78441dde0a818758ff`). Pretrained backbone foundation (`facebook/wav2vec2-base-960h`, rev `22aad52...`, 94.4M params). [VERIFIED]
2. **Zhang et al. (2021)** *speechocean762: An Open-Source Non-native English Speech Corpus for Pronunciation Assessment*, Interspeech 2021 / arXiv:2104.01378v2. `references/papers/speechocean762.pdf` (512,281 B, SHA-256 `5edd602a0ee383e3dc20dcc6a462541dd6daf0c2f22c2b3d5d877770391bd657`). Multi-aspect scoring benchmark (accuracy, completeness, fluency, prosody). [VERIFIED]
3. **Zhao et al. (2018)** *L2-ARCTIC: A Non-native English Speech Corpus*, Interspeech 2018. `references/papers/zhao2018_l2_arctic.pdf` (721,794 B, SHA-256 `03a4b836bca32e6bdf3439bc363f8dbc1610d2186c577f492f4d6c243da5a022`). Corpus source (24 L2 speakers, 6 L1s: Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese). [VERIFIED]
4. **Peng et al. (2021)** *A Study on Fine-Tuning wav2vec2.0 Model for the Task of Mispronunciation Detection and Diagnosis*, Interspeech 2021. `references/papers/peng2021_wav2vec2_mdd.pdf` (1,442,794 B, SHA-256 `8c56fc608b97146b401aa86ddd5390b405e9809ed31ca3f4ac73d267f4aacce9`). Architectural blueprint for multi-task CTC fine-tuning on L2 speech. [VERIFIED]
5. **Gong et al. (2022)** *Transformer-Based Multi-Aspect Non-Native English Pronunciation Assessment (GOPT)*, ICASSP 2022 / arXiv:2205.03432v1. `references/papers/gong2022_gopt.pdf` (351,260 B, SHA-256 `44c5f27ea0977b40396e6fc538a1636074b1484c6b574b968997789b0ceb851c`). Transformer GOP on SpeechOcean762. [VERIFIED]
6. **Shahin et al. (2023/2025)** *Phonological Level wav2vec2-based Mispronunciation Detection and Diagnosis*, arXiv:2311.07037 / Speech Communication 2025. `references/papers/shahin2023_phonological_wav2vec2_mdd_preprint.pdf` (4,767,893 B, SHA-256 `202d722d4b0c15705c1bc0a5f11c924f13cec136705e4b05410ff32a79f4d456`). Multi-attribute phonological feedback (place, manner, voicing). [VERIFIED]
7. **Radford et al. (2023)** *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*, ICML 2023. `references/papers/radford2023_whisper.pdf` (823,459 B, SHA-256 `84496ed07ac01bf3d00866a1934f81173ddbba99f7488de59bb8ae3b28f0f85e`). Weak-supervision comparative encoder reference. [VERIFIED]
8. **Tu et al. (2025/2026)** *Mispronunciation Detection and Diagnosis Without Model Training: A Retrieval-Based Approach*, arXiv:2511.20107 / IEEE ICASSP 2026. `references/papers/tu2025_retrieval_mdd.pdf` (737,280 B, SHA-256 `2eb40576a3c2f4bf59f1471c8bb3c240711d0f13d9af31c19e0454532fac65d1`). Non-parametric kNN acoustic retrieval baseline. [VERIFIED]

### 1.2 Access-Restricted, Code-Inspected & External Citation Inventories (4 Documents)
1. **Witt & Young (2000)** *Phone-level Pronunciation Scoring and Assessment for Interactive Language Learning*, Speech Communication. DOI: `10.1016/S0167-6393(99)00044-8`. `references/papers/P02_witt2000_gop_citation.md` (`ACCESS_RESTRICTED` via Elsevier paywall). Mathematical formulation and bibliographic record preserved. [VERIFIED]
2. **Amrate & Tsai (2024)** *Computer-assisted pronunciation training: A systematic review*, ReCALL. DOI: `10.1017/S0958344024000181`. `references/papers/P09_amrate2024_capt_citation.md` (`ACCESS_RESTRICTED` via Cambridge Core paywall). 30-study CAPT pedagogical review preserved. [VERIFIED]
3. **Joint-Align (IEEE 420)**: Inspected in RQ-RESEARCH-01 Lane E. Full text blocked via IEEE Xplore paywall (`ACCESS_RESTRICTED`); detailed mechanism and compute claims excluded from implementation. [CORROBORATED]
4. **Cao et al. (2026)** *Segmentation-free GOP*: Inspected via public GitHub (`Ming-hao-Cao/Segmentation-free-GOP`). Code license was unspecified/unresolved; clean-room reimplemented under `ctc-union-language-v1` for P-SCORE. [CORROBORATED]

### 1.3 Dataset Documentation & Legal Provenance Ledgers
- **L2-ARCTIC Corpus:** Non-commercial educational research license (CC BY-NC 4.0). `references/licenses/CC-BY-NC-4.0.txt`. 24 non-native speakers, 3,599 valid WAV/TextGrid pairs (245 orphan audio exclusions).
- **SpeechOcean762 Corpus:** Open license (CC BY 4.0). `references/licenses/CC-BY-4.0.txt`. 5,000 utterances, 250 non-native speakers, multi-aspect 5-expert annotations.
- **CMU Pronouncing Dictionary:** CMUdict v0.7b BSD-style permissive license. `resources/CMUDict/cmudict.dict` (133,260 entries, 39 ARPAbet phoneme inventory).

---

## 2. Dated Chronological Evolution of Methodology & Decisions

| Timestamp (+07:00) | Research Phase / Commit | Core Goal | Decision & Rationale | Discarded Alternative | Action, Result & Limitations | Current Scientific Validity |
|---|---|---|---|---|---|:---:|
| **2026-10-23 15:54** | **RQ1-TRN-00** (`9382cca`) | Baseline paradigm selection | Prioritize Wav2Vec 2.0 multi-task MDD over classical GOP. Rationale: eliminate forced-alignment breakdown on non-native deletions ($D=3,420$, 45.8% in Vietnamese). | Classical Kaldi GOP as primary baseline | Comparative evaluation matrix published (`BASELINE_EXPERIMENT_DESIGN.md`). Avoided external Kaldi C++ dependencies. | **VALID** (Empirically verified) |
| **2026-10-23 21:46** | **RQ1-TRN-01A** (`5942338`) | Architecture specification | Implement 3-head `LangfensWav2Vec2MDD`: Head 1 CTC (40 cls), Head 2 Binary (1 logit via cross-attention), Head 3 39-class Substitution Diagnosis. | CTC-only or Binary-only architectures | Multi-task forward/backward smoke tests passed. Pretrained backbone (`facebook/wav2vec2-base-960h`) frozen. | **SUPERSEDED** (3rd head dropped) |
| **2026-10-23 23:52** | **RQ1-TRN-01B0** (`d63f25f`) | 100-utterance sanity run | Discovered severe diagnosis metric flaw: 87.6% of evaluated tokens were correct phones (`target = base_phone`), masking true substitution accuracy of 3.23% (4/124). | Retaining 1,000-token mixed diagnosis metric | Formal reconciliation (`SUBSTITUTION_DIAGNOSIS_METRIC_AUDIT.md`) enforced strict 124-token denominator for substitutions. | **VALID** (Strict accounting policy) |
| **2026-10-25 19:30** | **RQ1-TRN-01B1** (`73eae7e`) | CTC blank collapse mitigation | Mitigate CTC blank collapse (>98% blanks) via Variant D: cosine schedule warmup (100 updates), peak lr=1e-3, frozen backbone. | Trainable backbone or constant high lr | Decoded phoneme sequence length restored; greedy PER dropped to ~23%. Acoustic conditioning validated. | **VALID** (Baseline R training recipe frozen) |
| **2026-10-25 22:34** | **RQ1-TRN-02A** (`a6e8035`) | Fold 1 Baseline R training | Switch from 3-head to **Two-Head Contract (CTC + Binary only)**. Freeze Diagnosis head (`lambda_diag = 0.0`). Rationale: eliminate gradient conflict on tiny substitution support. | Training 39-class diagnosis head on full fold | Colab T4 training completed (11 epochs, 6,600 updates). Locked Epoch 6 / 3,600 updates `best.pt` (AP 0.316062). Diagnosis unreadiness codified. | **VALID** (Foundational production baseline) |
| **2026-10-26 16:01** | **RQ1-TRN-02A-EVAL** (`c4a658f`) | Zero-oracle deployment | Prohibit gold TextGrid timestamps as inference inputs. Formulate Protocol O (oracle tokens, AP 0.317021) vs Protocol Z (zero-oracle CMUdict G2P, AP 0.313530). | Relying on human TextGrid intervals at inference | DP Levenshtein alignment scored 9,919/9,956 tokens (99.63% coverage), proving architecture operates deployably with only raw audio + prompt text. | **VALID** (Mandatory deployment policy) |
| **2026-10-29 10:51** | **RQ1-TRN-02B** (`b088425`) | Prompt dropout ablation | Controlled acoustic ablation (C0 dropout $p=0.0$ vs C1 $p=0.25$) over 3,600 updates on Kaggle T4x2. Contrast: $\Delta = -0.007918$ (95% CI `[-0.018, +0.000]`). | Replacing Baseline R with C1 | Hypothesis NOT_SUPPORTED; prompt dropout degraded detection without expanding acoustic reliance. Baseline R retained as authoritative. | **VALID** (Negative causal result verified) |
| **2026-11-01 02:13** | **RQ1-TRN-03A** (`618e86d`) | 4-Fold cross-fold generalization | Train Folds 2, 3, and 4 Baseline R independently from pretrained init (3,600 updates each, Kaggle T4x2). Macro Binary AP 0.2792 +/- 0.0265 across 39,576 tokens. | Fine-tuning/warm-starting across folds | Provenance locked: dual checkpointing (`fixed_budget_checkpoint.pt` and `best.pt`), strict per-fold test speaker sealing ($S_{\text{train}} \cap S_{\text{test}} = \emptyset$). | **VALID** (Multi-fold baseline established) |
| **2026-11-02 13:53** | **RQ1-DIAG-01** (`b0ef963`) | Omission root-cause audit | Forensic audit of 39,576 tokens. Discovered severe omission chasm: Deletion AP collapses to 0.0305 (vs 0.2643 for substitutions), acoustic reliance collapses to +0.0017. | Blaming dataset size or annotations | Cross-attention mass is only 2.8% in-window (cycles over 6–7 global anchor frames >1.4s away). Omissions concentrated in codas (55.6%) and short duration (<80ms, 86.7%). | **VALID** (Established omission diagnostic baseline) |
| **2026-11-02 14:20** | **RQ1-DIAG-02** (`dd2dee9`) | Alignment causal probe | Claimed causal proof that cross-attention bypasses Wav2Vec2 omission evidence where CTC emitted 86% blanks. | Non-causal observation | Flawed forced-attention implementation bypassed learned MHA projections. | **SUPERSEDED** (Superseded by DIAG-02B) |
| **2026-11-02 15:08** | **RQ1-DIAG-02B** (`467c823`) | Counterfactual validity audit | Re-evaluated forced attention with projection-correct MHA. Disproved DIAG-02 causal claim: corrected drift was only +0.023 (not +0.25). Matched CTC blank AUC is only ~0.653. | Authorizing causal training from DIAG-02 | Issued **STRICT NO-GO for causal training authorization**. Matched controls also exhibit high blank posterior (0.7322 vs 0.8244 on omissions). | **VALID** (Rigorous scientific correction) |
| **2026-11-02 17:15** | **RQ-MODEL-01** (`baa4989`) | E1 Skip-MDD benchmark | Evaluated Experiment E1 (learnable acoustic skip token $\mathbf{s} \in \mathbb{R}^{768}$ + Gaussian monotonic prior) on Kaggle GPU. | Proceeding to cross-fold rollout | **E1 FAILED:** Omission AP degraded -7.90% (0.0350 to 0.0323); skip mass on deletions completely collapsed to 0.000000. STRICT NO-GO for cross-fold rollout. | **VALID** (Inductive bias collapse confirmed) |
| **2026-11-03 13:03** | **RQ-RESEARCH-01** (`0d268ff`) | Research closure & 19 candidates | Coordinated closure across 6 MDD families (Lanes A–F) using a 19-candidate rubric. Proved no candidate is an automatic winner. Selected two parallel pilots: P-SCORE and P-SUP. | Picking single candidate without empirical test | Rubric readiness: SF-SD 81, decoder control 79, compact subtype 75. Frozen Evaluation Contract V1 enacted across coordinator and child worktrees. | **VALID** (Research closure and pilot mandate) |
| **2026-11-03 17:51** | **RQ1-P-SCORE** (`6ee11a2`) | Frozen-weight graph benchmark | Executed P-SCORE (`GOP-CTC-SF-SD` vs `SF-S`) on CPU log-emissions cache. Result: Omission AP +0.0216 (PASS), Recall@FPR15 +18.25 pp (PASS), but Substitution AP -0.0243 (FAIL noninferiority). | Deploying P-SCORE as Baseline R replacement | Omission-substitution tradeoff proved: $\epsilon$-deletion shortcut bypasses acoustic energy on substitutions. Classified as INCONCLUSIVE_TRADEOFF / NO-GO FOR DEPLOYMENT. | **VALID** (Empirical tradeoff documented) |
| **2026-11-03 18:13** | **RQ1-P-SUP** (`0e6e380`) | Subtype supervision staging | Staged parameter-matched (266,667 params) P-SUP treatment (`T-OP` operation_ce) and control (`B-CONTROL` binary_collapsed) to private Kaggle T4x2 kernels (`--no-run`). | Unmatched capacity or heuristic edit tags | Immutable pinned SHA `df55f12`. Independent metadata confirms both executed 3,600 updates with 0 skips; scientific VAL comparison pending. | **VALID** (Awaiting scientific evaluation replay) |

---

## 3. Deep Architectural & Intent Evolution

### 3.1 Baseline Choice: Wav2Vec 2.0 Multi-Task vs. Classical GOP
- **Mathematical Formulations:**
  - *Classical GOP (Witt & Young 2000 [P02]):* Evaluates the phone log-likelihood ratio over acoustic observations $o$ within Viterbi segment boundaries $[t_s, t_e]$:
    $$\text{GOP}(p) = \frac{1}{|t_e - t_s + 1|} \sum_{t=t_s}^{t_e} \log \frac{P(o_t \mid s_p)}{\max_{q \in \mathcal{Q}} P(o_t \mid s_q)}$$
  - *Wav2Vec 2.0 Multi-Task MDD (Baevski et al. 2020; Peng et al. 2021):* Maps 16 kHz raw waveforms through a 7-layer convolutional feature encoder (stride 320x, 20 ms frames) and a 12-layer Transformer into latent features $\mathbf{H} \in \mathbb{R}^{T \times 768}$, optimized jointly via Connectionist Temporal Classification (CTC) and token-level binary cross-entropy:
    $$\mathcal{L}_{\text{total}} = \lambda_{\text{ctc}} \mathcal{L}_{\text{CTC}}(\mathbf{H}, \mathbf{Y}_{\text{realized}}) + \lambda_{\text{binary}} \mathcal{L}_{\text{BCE}}(\hat{\mathbf{y}}, \mathbf{Y}_{\text{binary}})$$
- **Theoretical Rationale:** Classical GOP strictly depends on external Viterbi forced alignment. In non-native speech containing frequent phoneme deletions ($D = 3,420$ in L2-ARCTIC; $1,566$ in the Vietnamese cohort alone, representing $45.8\%$ of all corpus deletions), forced alignment collapses by stretching neighboring phones over silent intervals or forcing arbitrary 10–20 ms boundaries. Wav2Vec 2.0 eliminates this vulnerability because its joint CTC loss learns continuous temporal correspondence internally without requiring external acoustic alignments or Praat boundaries.

### 3.2 The 3-Head to 2-Head Architecture Simplification
- **Initial 3-Head Design (RQ1-TRN-01A):** Included Head 1 (CTC realized recognition, 40 cls), Head 2 (Binary detection, 1 logit via cross-attention), and Head 3 (Substitution diagnosis, 39 cls).
- **The Metric Defect (RQ1-TRN-01B0):** Forensics revealed that `textgrid_parser.py` assigned `diagnosis_target = base_phone` to correct productions. Consequently, 87.6% of the 1,000 evaluated tokens were correct phonemes; true substitution diagnosis accuracy was an abysmal $3.23\%$ ($4 / 124$).
- **The Contractual Simplification (RQ1-TRN-02A):** Training a 39-class diagnosis head on sparse substitutions per batch (prevalence ~10%) caused severe multi-task gradient interference and exacerbated CTC blank collapse. The architecture was deliberately restricted to a **Two-Head Contract (CTC + Binary only)**, freezing the diagnosis head (`lambda_diag = 0.0`). The project committed never to claim acoustic diagnosis readiness until substitution supervision is rehabilitated.

### 3.3 Alignment & Time Oracle Restrictions: Protocol O vs. Protocol Z
- **The Time-Oracle Prohibition:** In `AGENTS.md` and `STANDARDIZATION_POLICY.md`, using gold TextGrid start/end timestamps ($t_{\text{start}}, t_{\text{end}}$) as inference inputs is strictly prohibited. An interactive CAPT system receives only raw audio and text prompts; supplying gold human boundaries would constitute an unrealistic time oracle.
- **Protocol O (Oracle-Canonical Benchmark):** Canonical phone sequence is extracted from TextGrid interval tiers (without timestamps), matching annotator dialect choices. Evaluated on 9,956 phone tokens. Used solely for historical literature comparability.
- **Protocol Z (Zero-Oracle Deployment):** Canonical phone sequence is synthesized purely from text via CMUdict G2P (v0.7b) and scored against gold labels via minimum-edit Levenshtein dynamic programming:
  $$\text{Text Prompt} \xrightarrow{\text{Tokenizer}} \{\text{word}_k\} \xrightarrow{\text{CMUdict Primary}} \{p_{k, 1}, \dots, p_{k, n}\} \xrightarrow{\text{Normalize}} \mathbf{P}_Z$$
  Protocol Z achieves 99.63% token coverage (9,919/9,956 tokens) and an AP of 0.313530 (a negligible drop of 0.35 pp vs Protocol O 0.317021), proving deployable zero-oracle validity.
- **Root Cause of the 37 Excluded Tokens in Protocol Z:**
  - 25 dictionary discrepancies (67.6%): 9 wh-aspirations (`/HH W/` in "what", "which" vs standard simplex `/W/`), 2 contraction expansions ("we'll" as `/W IY W IH L/`), and 14 weak vowel/schwa alternations.
  - 8 genuine gold acoustic deletions (21.6%): e.g. dropped initial `/HH/` in "had", coda rhotics `/R/` in "water", "short".
  - 4 phonetic substitutions (10.8%): derhotacization `/ER/` $\to$ `/EH/` and unclassified `<ERR>` tags.

### 3.4 Phoneme Omission & False-Negative Research: DIAG-01, DIAG-02, and DIAG-02B
- **Conceptual Disambiguation (DIAG-01):** Conflation between three terms was resolved:
  1. *CTC Levenshtein deletion* (~4.9% rate): ASR recognition gap where acoustic frames were present but missed by the CTC decoder.
  2. *Gold phoneme omission* (1.59% prevalence, 14.93% of errors): Non-native speaker deleted a canonical sound (`X,sil,d`).
  3. *Binary MDD false negative* (97.46% of omissions missed at $\tau=0.50$): Detection head failed to alert the learner.
- **Pipeline Asymmetry:** In `dataset.py`, `chronological_realized_phones` excludes deletions from CTC targets. CTC loss therefore provides **zero gradient signal** to penalize omissions against canonical prompts.
- **Cross-Attention Collapse:** Baseline R cross-attention queries are unconstrained by monotonic loss; mean in-window attention mass is only 2.8% (cycles over 6–7 global anchor frames >1.4s away). Omissions are acoustically blind (acoustic reliance gap collapses to $+0.0017$).
- **The DIAG-02 Causal Fallacy & DIAG-02B Correction:** DIAG-02 claimed CTC blank emissions (86%) proved Wav2Vec2 frames contain omission evidence bypassed by cross-attention. DIAG-02B audited the code and proved that DIAG-02's "forced attention" intervention bypassed learned value/output projections. When projection-correct MHA was evaluated, drift was only +0.023 (not +0.25). Furthermore, matched correct phones also exhibit high blank posterior (0.7322 vs 0.8244 on omissions; matched AUC ~0.653). DIAG-02B issued a **STRICT NO-GO for causal training authorization**.

### 3.5 E1 Failure & The 19-Candidate Research Rubric
- **Dataset Sufficiency Audit (RQ-MODEL-01):** Census proved training folds contain $>2,300$ omissions and $>9,300$ substitutions. Manual re-annotation was firmly rejected: omission blindness is architectural, not a data deficit.
- **Experiment E1 (Alignment-Aware Skip MDD):** Added a learnable acoustic skip token $\mathbf{s} \in \mathbb{R}^{768}$ to cross-attention key/values, a Gaussian monotonic distance prior $-\gamma (t/T - i/N)^2$, and an auxiliary skip weight feature in the binary MLP.
- **Kaggle GPU Benchmark Failure:** E1 degraded Omission AP by -7.90% (0.0350 to 0.0323) and Binary AP by -2.44% (0.3170 to 0.3093). Skip mass on deletions completely collapsed to **0.000000**! Without direct loss supervision on attention weights, queries preferred diffuse speech frames over the ungrounded static skip vector.
- **The 19-Candidate Research Rubric (RQ-RESEARCH-01):** Evaluated 19 candidates across 6 MDD families (Lanes A–F) using a standardized rubric:
  - *Lane A (CTC-GOP):* SF-SD (81), SF-S (74), SDI (68), SA (76).
  - *Lane B (Acoustic Realization):* CoCA-MDD (73), Fu CTC (71), Free-Phone DP (79).
  - *Lane C (Supervised Objectives):* Compact Subtype T-OP (75), Matched Binary Control (75), Focal BCE (70).
  - *Lane D (Alignment Mechanisms):* Monotonic Attention (69), Edit Lattices (66).
  - *Lane E (Joint Modeling):* Joint-Align (blocked), Text-Guided CTC (72).
  - *Lane F (Retrieval & Phonological):* kNN Retrieval (65), Phonological Attributes (67).
  Balanced readiness scores ranked SF-SD (81), decoder control (79), and compact subtype (75) at the top, but sensitivity testing showed rankings shifted by 3–5 points with single-criterion changes. This proved no candidate was an automatic winner and mandated prospective, parallel empirical pilots.

### 3.6 P-SCORE vs. P-SUP Rationale and Empirical Trade-Offs
- **P-SCORE (Post-Hoc Acoustic Graph Scoring):** Evaluates segmentation-free GOP (`GOP-CTC-SF-SD` vs `SF-S`) on frozen E0 Baseline R acoustic emissions.
  $$\mathcal{L}_{\text{denom}} = -\log P\left(\pi \in \mathcal{B}^{-1}\left(c_1 \cdots c_{j-1} \, \{p \in \mathcal{P} \cup \epsilon\} \, c_{j+1} \cdots c_J\right) \mid X\right)$$
  - *Empirical Findings on Aligned Val ($N=9,919$):* Omission AP improved $+73.5\%$ (+0.021615, PASS $\ge +0.020$); Recall@FPR15 jumped $+18.25\text{ pp}$ (43.07% vs 24.82%, PASS $\ge +0.030$).
  - *Trade-Off & Failure:* Substitution AP collapsed $-9.83\%$ (-0.024256, **FAIL** noninferiority $\ge -0.005$); Binary AP dropped $-0.009045$.
  - *Mechanism:* The $\epsilon$-deletion path in the denominator graph provides an acoustic shortcut that bypasses substituted acoustic segments, causing severe degradation on substitution detection. Classified as `INCONCLUSIVE_TRADEOFF`.
- **P-SUP (Compact Representation Learning):** Investigates whether direct 3-way subtype supervision (`T-OP` [correct, substitution, deletion]) during training overcomes the attention collapse without compromising substitutions, evaluated against an exactly capacity-matched (266,667 params) binary control (`B-CONTROL`):
  $$\mathcal{L}_{\text{task}} = -\log \frac{\exp(z_{y})}{\sum_{c=0}^2 \exp(z_c)} \quad \text{vs} \quad \mathcal{L}_{\text{control}} = \text{BCEWithLogits}(\text{logsumexp}(z_1, z_2) - z_0, \tilde{y})$$

### 3.7 Thesis Scope Demarcation: RQ1 vs. RQ2 Clarification
- **Primary Source Evidence:** Proposal document `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md` (Line 14, Sections 2.2(e), 2.3 RQ2, 3(b), 4(e), 4(g)).
- **Authoritative Resolution:** **RQ2 is EXCLUSIVELY educational learning content / exercise / test generation** (passage-grounded reading comprehension, grammar exercises with reference answers, multi-agent coordinator/generator/reviewer/validator architectures), NOT software-engineering test generation.
- **RQ1 Boundaries:** All speech processing, phoneme-level MDD fine-tuning, GOP scoring, SpeechOcean762 cross-corpus evaluation, and educational pronunciation feedback belong strictly to **RQ1**.

---

## 4. Latest Actual State & Historical Commitments

| Component / Subsystem | Historical Commitment / Registered Baseline | Latest Verified Actual State | Verification Evidence / Repository Reference | Status / Compliance |
|---|---|---|---|:---:|
| **Baseline R Checkpoint (E0)** | Epoch 6 / 3,600 updates `best.pt` (val AP 0.316062 Colab FP16) | Frozen, verified bit-identical on Drive & HF | Drive ID `10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I`; SHA-256 `a2445c8d...` | **VERIFIED** |
| **4-Fold Split Integrity** | 24 speakers, demographic balance, 0 test leakage | Verified across all 4 fold staging packages | Manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (`7d6bfdf6...`) | **VERIFIED** |
| **Test Set Isolation** | Active test partition $S_{\text{test}}^{(k)}$ strictly sealed | Sealed test audio/labels strictly excluded from all training packages | Staging archive `authorized_RQ1-TRN_fold1_train_val_only.zip` (`b750eb3b...`) | **VERIFIED** |
| **Multi-Fold Generalization** | Baseline R independently trained on Folds 2, 3, 4 | Completed 3,600 updates on Kaggle T4x2; Macro AP 0.2792 | Checkpoints mirrored to HF `minkhoaa/langfens-rq1-cross-fold-checkpoints` | **VERIFIED** |
| **E1 Skip-MDD Candidate** | Alignment-aware skip token for omission rescue | Failed benchmark (skip mass 0.000000); cross-fold rollout rejected | Benchmark report `manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json` | **CLOSED (NO-GO)** |
| **P-SCORE Candidate** | Segmentation-free GOP-CTC-SF-SD evaluation | Executed on CPU cache; confirmed omission-substitution tradeoff | Benchmark report `manifests/RQ1-WORKSPACE-PREP-01/P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md` | **COMPLETED (HOLD)** |
| **P-SUP Candidate** | Matched 266,667 param T-OP vs B-CONTROL training | Pinned to SHA `df55f12`; staged to Kaggle; independent metadata confirms 3,600 updates / 0 skips | Kernels `minkhoaa/langfens-rq1-psup-t-op-t4x2` & `b-control`; VAL evaluation pending | **AWAITING EVAL** |

---

## 5. Contradictions & Scientific Inconsistencies Discovered

1. **Diagnosis Metric Confounding (RQ1-TRN-01B0):** Initial reports claimed an 8.0% diagnosis accuracy on 1,000 tokens. The audit discovered that 87.6% of evaluated tokens were actually correct phonemes. On actual substitutions ($N=124$), accuracy was 3.23% (chance level). Corrected in `SUBSTITUTION_DIAGNOSIS_METRIC_AUDIT.md`.
2. **Forced-Attention Causal Fallacy (DIAG-02 vs. DIAG-02B):** DIAG-02 reported a dramatic +0.25 drift when forcing attention locally, interpreting this as proof that downstream MLPs collapse when forced to look at acoustic frames. DIAG-02B proved the implementation bypassed learned MHA projection weights and multiplied uniform weights by raw hidden states. Projection-correct MHA drift was only +0.023.
3. **CTC Blank Posterior Interpretation (DIAG-02 vs. DIAG-02B):** DIAG-02 asserted that an 86% blank argmax on omissions proved CTC frames contained rich omission signals bypassed by MDD. DIAG-02B demonstrated that duration-matched correct phonemes also exhibit a 73.2% mean blank posterior, yielding a modest matched AUC of only ~0.653.
4. **Historical Baseline AP Precision Divergence:** Baseline R Binary AP is reported as 0.316062 (Colab GPU FP16 AMP), 0.317021 (CPU FP32 replay Protocol O), 0.317012 (cross-fold pilot), and 0.313530 (CPU FP32 Protocol Z). Reconciled in `SCIENTIFIC_QA_AND_COMPARISON_REPORT.md` as arithmetic precision and protocol boundary differences, not data tampering.
5. **RQ2 Scope Misinterpretation:** Earlier child agent reports asserted RQ2 meant software testing (pytest generation) and claimed the IELTS Reading comprehension harness was off-scope. Auditing primary proposal `DC KLTN0127...md` confirmed RQ2 is educational exercise/test generation; the IELTS reading harness is directly aligned with RQ2 intent.

---

## 6. Claim & Evidence Registry

| Claim Identifier | Formulated Claim | Primary Source / Document Reference | Verified Evidence / Audit Artifact | Confidence Level |
|---|---|---|---|:---:|
| **CLM-01** | Classical GOP collapses on non-native deletions in L2-ARCTIC. | `manifests/RQ1-TRN-00/BASELINE_EXPERIMENT_DESIGN.md` | Witt & Young (2000); 1,566 deletions in Vietnamese cohort ($45.8\%$ of corpus deletions) cause boundary distortion. | **CORROBORATED** |
| **CLM-02** | Baseline R diagnosis head was uncalibrated and under-supervised. | `manifests/RQ1-TRN-01B0/reconciliation/SUBSTITUTION_DIAGNOSIS_METRIC_AUDIT.md` | Exact census on 32-utt train subset: 876 correct, 124 substitutions, 2 err tags. Substitution top-1 accuracy was $3.23\%$. | **VERIFIED** |
| **CLM-03** | Baseline R Fold 1 achieves Binary AP 0.316062 on Colab T4. | `manifests/RQ1-TRN-02A/checkpoints/01-two-head-colab-training-run.md` | Checkpoint `best.pt` metadata (SHA-256 `a2445c8d...`); reproduced bit-identically across CPU replays within FP16/FP32 bounds. | **VERIFIED** |
| **CLM-04** | Deployable MDD operates zero-oracle without TextGrid timestamps. | `manifests/RQ1-TRN-02A/evaluation_reconciliation/corrections/ORACLE_VS_ZERO_ORACLE_PROTOCOL.md` | Protocol Z CMUdict G2P aligns 9,919/9,956 tokens ($99.63\%$ coverage) with Binary AP 0.313530 (-0.0035 delta vs Protocol O). | **VERIFIED** |
| **CLM-05** | Prompt dropout ($p=0.25$) does not improve acoustic reliance. | `manifests/RQ1-TRN-02B/FINAL_THREE_MODEL_ASSESSMENT.md` | 1,000 cluster bootstrap resamples on Kaggle T4x2 runs; causal contrast $\Delta = -0.007918$ (95% CI `[-0.0183, +0.0004]`). | **VERIFIED** |
| **CLM-06** | Baseline R cross-attention temporally decouples from speech frames. | `manifests/RQ1-DIAG-01/checkpoints/01-phoneme-omission-and-false-negative-root-cause.md` | Mean in-window attention mass is $2.8\%$; peak frame temporal offset averages $1.415\text{ s}$ from true phone center. | **VERIFIED** |
| **CLM-07** | DIAG-02 causal proof of attention bypass is scientifically invalid. | `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md` | Interventions bypassed learned MHA projections; matched CTC blank AUC is only $\sim 0.653$ (controls also have $0.732$ blank posterior). | **VERIFIED** |
| **CLM-08** | L2-ARCTIC training sets are fully sufficient for omission modeling. | `manifests/RQ-MODEL-01/RQ_MODEL_01_REPORT.md` | Train census: Folds 1–4 contain $2,321$ to $2,522$ omissions and $>9,300$ substitutions across 16 speakers. | **VERIFIED** |
| **CLM-09** | Experiment E1 skip modeling completely collapses on omissions. | `manifests/RQ-MODEL-01/PROCESS_CHECKPOINT.md` | Kaggle GPU benchmark: Omission AP degraded $-7.90\%$; mean skip mass on deletions was strictly $0.000000$. | **VERIFIED** |
| **CLM-10** | Segmentation-free GOP-CTC-SF-SD exhibits omission-substitution tradeoff. | `manifests/RQ1-WORKSPACE-PREP-01/P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md` | Aligned Val ($N=9,919$): Omission AP $+73.5\%$ (+0.0216), but Substitution AP $-9.83\%$ (-0.0243, failing noninferiority $\ge -0.005$). | **VERIFIED** |
| **CLM-11** | RQ2 is educational test/exercise generation, not software tests. | Primary Proposal `DC KLTN0127...md` Line 14, Sec 2.3 & 3(b) | Explicit Vietnamese text specifies passage-grounded reading comprehension and grammar exercises with multi-agent review. | **VERIFIED** |

---

## 7. Unresolved Questions & Safe Operational Restart

### 7.1 Unresolved Research Questions
1. **Latest Distinct Omission Proposal Location:** An exhaustive filesystem search failed to locate an independent latest omission proposal beyond DIAG-01 and E1 reports. The operator must clarify whether a distinct written proposal exists or if P-SCORE/P-SUP represents the finalized proposal.
2. **P-SUP Scientific Evaluation on Validation Set:** Independent remote metadata confirms both P-SUP arms (`T-OP` and `B-CONTROL`) completed 3,600 updates on Kaggle T4x2 without skips. Standalone offline CPU evaluation replay on Fold 1 VAL (under Protocol O and Protocol Z) has not yet been executed to determine whether subtype supervision resolved the omission-substitution tradeoff.
3. **Paywalled Primary Literature (Witt & Young 2000, Amrate & Tsai 2024, Joint-Align IEEE 420):** Access remains paywall-restricted. Core mathematical formulations are preserved via bibliographic records, but full-text deep audits remain barred without institutional access.

### 7.2 Exact Safe Restart Procedure
To resume research operations without violating repository invariants or execution boundaries:
1. **Maintain Read-Only Discipline on Main Branches:** Current HEAD `1fa4a5b` on `prep/rq1-shared-base` must remain unmodified. Pre-existing untracked and modified files must be preserved intact.
2. **Awaiting Operator Authorization for P-SUP Evaluation Replay:**
   - Verify remote Hugging Face checkpoint availability for P-SUP Treatment (`T-OP`) and Control (`B-CONTROL`).
   - Execute standalone read-only offline evaluation script (`audit_scripts/eval_p_sup.py` or equivalent CPU replay) on Fold 1 VAL (`NJS`, `RRBI`, 300 TextGrids) using Protocol Z CMUdict alignments.
   - Compare `T-OP` vs `B-CONTROL` fixed-budget checkpoints against the prospective criteria frozen in `EVALUATION_CONTRACT_V1.md` ($\Delta \text{AP}_{\text{omit}} \ge +0.030$, $\Delta \text{AP}_{\text{sub}} \ge -0.005$).
3. **Strict Fallback Prohibition:** Maintain complete prohibition on local GPU/CPU model training; preserve per-fold test speaker sealing across all subsequent evaluations.
