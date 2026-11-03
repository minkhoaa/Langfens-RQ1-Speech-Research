// Langfens RQ1 — Parallel Research & Thesis-Aligned Method Selection
// Spawns six concurrent research lanes across research families A–F

const commonGuidance = `
AUTHORITATIVE CONTEXT & SCOPE:
- Read UIT Thesis Proposal: "../DC KLTN0127 23520772&23520757_TranThiHongYen (1).md" (authoritative boundary: reference-based phoneme-level MDD, substitutions, omissions/final consonant deletions, confusable pairs, acoustic representations, tiered feedback, Langfens Speaking web service).
- Read AGENTS.md: Remote GPU training only (T4/FP16), 4-fold split contract, sealed test isolation, dual checkpoint accounting (fixed-budget 3600-up vs best.pt), frozen backbone in eval.
- Core codebase seam: "src/langfens_prep/model.py", "losses.py", "alignment.py", "dataset.py", "g2p.py".
- Diagnostic evidence: "manifests/RQ1-DIAG-01/checkpoints/01-phoneme-omission-and-false-negative-root-cause.md", "manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md".
- Benchmark baseline: Baseline R (E0) val AP=0.317, Omission AP=0.035, PER=23.68%.
- E1 failure evidence: "manifests/RQ-MODEL-01/RQ_MODEL_01_REPORT.md" and "PROCESS_CHECKPOINT.md" (skip mass on deletions collapsed to 0.000000; Omission AP degraded -7.90%).
- Research Library: "manifests/RESEARCH_LIBRARY.md" and "references/papers/".

MANDATORY DELIVERABLE REQUIREMENTS (Items 1–10):
1. Review primary papers and available implementations.
2. Compare competing methods within this research family.
3. Verify academic claims, limitations, and reproducibility.
4. Explain deletion/substitution modeling and available supervision signals.
5. Map required changes to actual Langfens modules ("src/langfens_prep/") and data ("manifests/RQ1-SPLIT-01/").
6. Evaluate integration feasibility, inference latency, GPU training compute budget (< 30h T4 limit), and pedagogical feedback capability.
7. Identify thesis-scope violations, risks of drift into general foundation ASR, or over-simplification to deletion-only.
8. Propose candidate methods and minimal falsifiable pilot experiments.
9. Distinguish verified evidence from assumptions.
10. Produce a concise, thorough markdown report and write it directly to the designated file.
`;

const laneA_task = `
[LANE A: CTC/GOP and Segmentation-Free Deletion Scoring]
You are researching Family A: Goodness of Pronunciation (GOP), CTC posterior scoring, and segmentation-free deletion scoring.
Designated output file: "manifests/RQ-RESEARCH-01/LANE_A_CTC_GOP.md"

Focus areas:
- Classical GOP (Witt & Young, 2000; P02 citation in "references/papers/P02_witt2000_gop_citation.md").
- Neural/Transformer GOP (GOPT: Gong et al., 2022 in "references/papers/gong2022_gopt.pdf").
- CTC posterior scoring and blank token interpretation (Peng et al., 2021 in "references/papers/peng2021_wav2vec2_mdd.pdf").
- Crucial empirical check: DIAG-02B findings in "manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md" showed CTC blank frame posterior provides only modest matched discrimination (AUC ~0.653), not >84% confidence.
- Segmentation-free approaches: Can GOP score phoneme omissions when no acoustic interval exists? How does GOPT handle deleted tokens?
- Web service integration: Real-time inference latency for Langfens Speaking.

${commonGuidance}

Write your complete research report to "manifests/RQ-RESEARCH-01/LANE_A_CTC_GOP.md".
`;

const laneB_task = `
[LANE B: Acoustic Realization and Phoneme Recognition]
You are researching Family B: Direct phoneme recognition (free-phone ASR) and acoustic realization modeling.
Designated output file: "manifests/RQ-RESEARCH-01/LANE_B_ACOUSTIC_REALIZATION.md"

Focus areas:
- Free-phone CTC recognizer vs forced reference alignment (Peng et al., 2021; Baevski et al., 2020).
- Whisper phoneme decoding / encoder representations (Radford et al., 2023 in "references/papers/radford2023_whisper.pdf").
- L2-ARCTIC canonical vs realized transcription schema (Zhao et al., 2018 in "references/papers/zhao2018_l2_arctic.pdf"): annotations provide target phoneme, realized phoneme, and error type (sub, del, add).
- Levenshtein DP comparison between reference phoneme sequence and recognized phoneme sequence: does alignment-based deletion detection naturally solve omissions without frame-level boundary extraction?
- Why free-phone recognition suffers from high false-alarm rates on non-native accents (accented canonical vs phonemic substitution).
- Trade-offs: CTC blank collapse, PER floor (~22–24% on L2-ARCTIC), decoding latency, and feedback interpretability.

${commonGuidance}

Write your complete research report to "manifests/RQ-RESEARCH-01/LANE_B_ACOUSTIC_REALIZATION.md".
`;

const laneC_task = `
[LANE C: Deletion-Specific and Supervised Learning Objectives]
You are scouting and researching Family C: Deletion-specific loss formulations and supervised learning objectives.
Designated output file: "manifests/RQ-RESEARCH-01/LANE_C_SUPERVISED_OBJECTIVES.md"

Focus areas:
- Current loss contract in "src/langfens_prep/losses.py": "MaskedBCEWithLogitsLoss", "MaskedFocalBCEWithLogitsLoss", class-weighted BCE.
- Diagnostics from RQ1-TRN-03B: Arm 2A Focal loss (gamma=2.0) and Arm 2B Weighted loss (pos_weight=8.40) failed to improve 3600-update fixed-budget AP over Baseline R ("manifests/RQ1-TRN-03A/checkpoints/18-step18-scientific-comparison-correction.md" and "19-gate-c-scientific-closure.md").
- Multi-class reformulation: Formulating MDD as 3-class (0: Correct, 1: Substitution, 2: Deletion) or 4-class (including Insertion) vs binary BCE.
- Contrastive acoustic presence/absence loss, silence penalty, or boundary-aware loss.
- Available supervision signals: L2-ARCTIC TextGrid interval metadata (e.g., Praat interval duration == 0 for omissions vs silence gap between neighbors).
- Implementation mapping: Exact changes needed in "losses.py", "model.py", and dataset collation.

${commonGuidance}

Write your complete research report to "manifests/RQ-RESEARCH-01/LANE_C_SUPERVISED_OBJECTIVES.md".
`;

const laneD_task = `
[LANE D: Monotonic, Probabilistic and Edit-Aware Alignment]
You are scouting and researching Family D: Alignment mechanisms between prompt phonemes and acoustic representations.
Designated output file: "manifests/RQ-RESEARCH-01/LANE_D_ALIGNMENT_MECHANISMS.md"

Focus areas:
- Current cross-attention mechanism in "src/langfens_prep/model.py": Why unconstrained cross-attention fails on deletions (DIAG-01: 2.8% in-window attention mass, 1.4s peak temporal offset).
- Why E1 alignment-aware skip attention collapsed in "manifests/RQ-MODEL-01/RQ_MODEL_01_REPORT.md" (skip mass = 0.000000 on deletions; Omission AP degraded -7.90%).
- Monotonic attention alternatives: Monotonic Multihead Attention (MMA), Monotonic Chunkwise Attention (MoChA), Gaussian dynamic windowing.
- CTC forced alignment: Viterbi decoding over canonical prompt to get frame-to-phoneme boundaries without external tools (using CTC posteriors).
- Dynamic Time Warping (DTW) on acoustic embeddings or neural DP alignment.
- Trade-offs: Learnable alignment vs deterministic/training-free alignment, GPU training stability, and VRAM efficiency.

${commonGuidance}

Write your complete research report to "manifests/RQ-RESEARCH-01/LANE_D_ALIGNMENT_MECHANISMS.md".
`;

const laneE_task = `
[LANE E: Joint Acoustic-Recognition and Edit-Aware Modeling]
You are researching Family E: Joint end-to-end architectures modeling acoustics and edit operations simultaneously.
Designated output file: "manifests/RQ-RESEARCH-01/LANE_E_JOINT_MODELING.md"

Focus areas:
- Attention-based Encoder-Decoder (AED) models conditioned on canonical reference prompts (e.g., Listen, Attend and Spell variants for MDD; Lin et al.; Feng et al.).
- Neural Transducers (RNN-T / Conformer-Transducer) with explicit blank/null/deletion transitions.
- Edit-aware Sequence Tagging models (e.g., GECToR principles adapted to speech: predicting KEEP, REPLACE(ph), DELETE, INSERT(ph) tokens).
- How joint models handle deletion natively without requiring artificial skip embeddings or external forced aligners.
- Computational feasibility: Kaggle T4x2 / Colab T4 hardware limits (16 GB VRAM, 30 hours weekly quota), parameter footprint, convergence speed within ~3600-6000 updates.
- Risk of scope drift: Does training a full AED/Transducer violate the UIT thesis scope by turning into general speech recognition research?

${commonGuidance}

Write your complete research report to "manifests/RQ-RESEARCH-01/LANE_E_JOINT_MODELING.md".
`;

const laneF_task = `
[LANE F: Retrieval, Phonological and Hybrid Alternatives]
You are researching Family F: Alternative paradigms including retrieval-based MDD, phonological feature modeling, and hybrid pipelines.
Designated output file: "manifests/RQ-RESEARCH-01/LANE_F_RETRIEVAL_PHONOLOGICAL.md"

Focus areas:
- Retrieval-based MDD: Tu et al. (ICASSP 2026 / arXiv:2511.20107 in "references/papers/tu2025_retrieval_mdd.pdf") claiming 69.60% F1 training-free via kNN retrieval over native acoustic embeddings.
  Critique: Does retrieval work for omissions/deletions? What are index storage and search latency constraints?
- Phonological feature MDD: Shahin et al. (2023/2025 in "references/papers/shahin2023_phonological_wav2vec2_mdd_preprint.pdf") multi-attribute phonological feature detection (place, manner, voicing).
  Can articulatory/phonological feature absences directly detect consonant omissions (e.g., missing stop burst/closure)?
- Hybrid pipelines: External forced alignment (Montreal Forced Aligner - MFA or Kaldi) + SSL acoustic feature extraction + downstream classification/scoring.
- Educational corrective feedback: How well do retrieval and phonological representations support actionable feedback in Langfens Speaking (e.g., "unvoiced alveolar stop omitted at word coda")?

${commonGuidance}

Write your complete research report to "manifests/RQ-RESEARCH-01/LANE_F_RETRIEVAL_PHONOLOGICAL.md".
`;

const results = await runs.all([
  { key: "lane_a", agent: "researcher", task: laneA_task, output: "manifests/RQ-RESEARCH-01/LANE_A_CTC_GOP.md" },
  { key: "lane_b", agent: "researcher", task: laneB_task, output: "manifests/RQ-RESEARCH-01/LANE_B_ACOUSTIC_REALIZATION.md" },
  { key: "lane_c", agent: "scout", task: laneC_task, output: "manifests/RQ-RESEARCH-01/LANE_C_SUPERVISED_OBJECTIVES.md" },
  { key: "lane_d", agent: "scout", task: laneD_task, output: "manifests/RQ-RESEARCH-01/LANE_D_ALIGNMENT_MECHANISMS.md" },
  { key: "lane_e", agent: "researcher", task: laneE_task, output: "manifests/RQ-RESEARCH-01/LANE_E_JOINT_MODELING.md" },
  { key: "lane_f", agent: "researcher", task: laneF_task, output: "manifests/RQ-RESEARCH-01/LANE_F_RETRIEVAL_PHONOLOGICAL.md" }
]);

return results;
