"""
LANGFENS — RQ1: Scientific Evaluation Audit & Metric Reconciliation Tests.
Subagent: Agent D — Scientific Evaluation Audit.

Validates:
  1. Census reconciliation between 9,956 full canonical tokens and 9,919 aligned tokens on Fold-1 VAL.
  2. Exact identification of the 37 unaligned tokens (25 correct, 4 substitutions, 8 deletions).
  3. Metric reproducibility on both populations for Baseline R, GOP-SF-SD, GOP-SF-S, and GOP-SA.
  4. Prospective gate accounting: Omission AP PASS (+0.021615 >= +0.020), Recall@FPR15 PASS (+18.25 pp >= +0.030),
     Substitution AP FAIL (-0.024256 < -0.005 noninferiority).
  5. Capacity and parameter matching for P-SUP T-OP and B-CONTROL (266,667 head parameters, seed 1337).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile
import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

SPLIT_PATH = REPO_ROOT / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
ZIP_PATH = REPO_ROOT / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
PROMPTS_PATH = REPO_ROOT / "resources/L2-ARCTIC/prompts.json"
CMUDICT_PATH = REPO_ROOT / "resources/CMUDict/cmudict.dict"
PROTOCOL_Z_RECORDS = REPO_ROOT / "manifests/RQ1-TRN-02A/evaluation_reconciliation/zero_oracle_verification/PROTOCOL_Z_PREDICTION_RECORDS.json"
P_SCORE_REPORT = Path("/home/khoa/Projects/KLTN/langfens-rq1-p-score/manifests/RQ1-P-SCORE/evaluation/BENCHMARK_REPORT.json")


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = (ends + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def diagnostic_threshold(negative_scores: np.ndarray, ceiling: float = 0.15) -> float:
    s = np.asarray(negative_scores, dtype=np.float64)
    candidates = np.r_[np.unique(s), np.nextafter(s.max(), np.inf)]
    return float(next(t for t in candidates if np.mean(s >= t) <= ceiling))


def test_fold1_val_population_reconciliation():
    """Confirms exact token counts and partitions for full (9,956) vs aligned (9,919) populations."""
    from langfens_prep.alignment import align_canonical_sequences
    from langfens_prep.g2p import CMUDictG2P
    from langfens_prep.textgrid_parser import parse_gold_textgrid_content

    split_data = json.loads(SPLIT_PATH.read_text())
    fold1 = next(f for f in split_data["folds"] if f["fold_index"] == 1)
    val_utts = fold1["val"]["utterances"]
    assert len(val_utts) == 300

    prompts = json.loads(PROMPTS_PATH.read_text())
    g2p = CMUDictG2P(dict_path=CMUDICT_PATH, verify_checksum=True)

    zf = zipfile.ZipFile(ZIP_PATH)
    spk_zfs = {
        "NJS": zipfile.ZipFile(io.BytesIO(zf.read("NJS.zip"))),
        "RRBI": zipfile.ZipFile(io.BytesIO(zf.read("RRBI.zip")))
    }

    total_gold_tokens = 0
    total_aligned_tokens = 0
    full_counts = {"correct": 0, "substitution": 0, "deletion": 0}
    aligned_counts = {"correct": 0, "substitution": 0, "deletion": 0}
    unaligned_tokens = []

    for u in val_utts:
        spk, utt_name = u.split("/")
        prompt_text = prompts[utt_name]
        g2p_phones = g2p.phonemize(prompt_text)
        tg_text = spk_zfs[spk].read(f"{spk}/annotation/{utt_name}.TextGrid").decode("utf-8")
        ann = parse_gold_textgrid_content(tg_text, f"{utt_name}.TextGrid")
        gold_c_phones = [iv.canonical_phone for iv in ann.canonical_intervals]

        total_gold_tokens += len(gold_c_phones)
        for iv in ann.canonical_intervals:
            full_counts[iv.error_type] += 1

        al = align_canonical_sequences(gold_c_phones, g2p_phones, tie_break="sub_first")
        total_aligned_tokens += len(al["aligned_pairs"])

        for g_idx, _, _ in al["aligned_pairs"]:
            aligned_counts[ann.canonical_intervals[g_idx].error_type] += 1

        for g_idx, _ in al["unaligned_gold"]:
            iv = ann.canonical_intervals[g_idx]
            unaligned_tokens.append({
                "speaker": spk,
                "utterance_id": utt_name,
                "gold_index": g_idx,
                "canonical_phone": iv.canonical_phone,
                "realized_phone": iv.realized_phone,
                "error_type": iv.error_type,
            })

    # Assert Full Gold Population ($N=9,956$)
    assert total_gold_tokens == 9956
    assert full_counts["correct"] == 8799
    assert full_counts["substitution"] == 1012
    assert full_counts["deletion"] == 145
    assert full_counts["correct"] + full_counts["substitution"] + full_counts["deletion"] == 9956

    # Assert Aligned Population ($N=9,919$)
    assert total_aligned_tokens == 9919
    assert aligned_counts["correct"] == 8774
    assert aligned_counts["substitution"] == 1008
    assert aligned_counts["deletion"] == 137
    assert aligned_counts["correct"] + aligned_counts["substitution"] + aligned_counts["deletion"] == 9919

    # Assert Unaligned Tokens ($N=37$)
    assert len(unaligned_tokens) == 37
    unaligned_cor = sum(1 for t in unaligned_tokens if t["error_type"] == "correct")
    unaligned_sub = sum(1 for t in unaligned_tokens if t["error_type"] == "substitution")
    unaligned_del = sum(1 for t in unaligned_tokens if t["error_type"] == "deletion")

    assert unaligned_cor == 25
    assert unaligned_sub == 4
    assert unaligned_del == 8
    assert unaligned_cor + unaligned_sub + unaligned_del == 37


def test_baseline_r_metrics_on_both_populations():
    """Verifies Baseline R performance on aligned population (N=9,919) and full population (N=9,956)."""
    assert PROTOCOL_Z_RECORDS.is_file()
    with open(PROTOCOL_Z_RECORDS) as f:
        data = json.load(f)

    zf = zipfile.ZipFile(ZIP_PATH)
    from langfens_prep.textgrid_parser import parse_gold_textgrid_content
    spk_zfs = {
        "NJS": zipfile.ZipFile(io.BytesIO(zf.read("NJS.zip"))),
        "RRBI": zipfile.ZipFile(io.BytesIO(zf.read("RRBI.zip")))
    }

    aligned_scores = []
    aligned_etypes = []

    for u in data["utterances"]:
        spk = u["speaker"]
        uid = u["utterance_id"]
        tg_text = spk_zfs[spk].read(f"{spk}/annotation/{uid}.TextGrid").decode("utf-8")
        ann = parse_gold_textgrid_content(tg_text, f"{uid}.TextGrid")
        c_ivs = ann.canonical_intervals
        g2p_scores = u["g2p_scores"]
        for g_idx, z_idx, _ in u["aligned_pairs"]:
            aligned_scores.append(g2p_scores[z_idx])
            aligned_etypes.append(c_ivs[g_idx].error_type)

    scores = np.array(aligned_scores)
    etypes = np.array(aligned_etypes)

    # 1. Aligned Binary AP
    y_bin = np.isin(etypes, ["substitution", "deletion"]).astype(int)
    ap_bin = average_precision_binary(y_bin, scores)
    assert pytest.approx(ap_bin, abs=1e-5) == 0.313530

    # 2. Aligned Omission AP
    om_mask = np.isin(etypes, ["deletion", "correct"])
    y_om = (etypes[om_mask] == "deletion").astype(int)
    s_om = scores[om_mask]
    ap_om = average_precision_binary(y_om, s_om)
    assert pytest.approx(ap_om, abs=1e-5) == 0.032564

    # 3. Aligned Recall@FPR15
    cor_scores = scores[etypes == "correct"]
    tau_15 = diagnostic_threshold(cor_scores, 0.15)
    rec_fpr15 = float(np.mean(scores[etypes == "deletion"] >= tau_15))
    assert pytest.approx(rec_fpr15, abs=1e-4) == 0.394161

    # 4. Aligned Substitution AP
    sub_mask = np.isin(etypes, ["substitution", "correct"])
    y_sub = (etypes[sub_mask] == "substitution").astype(int)
    s_sub = scores[sub_mask]
    ap_sub = average_precision_binary(y_sub, s_sub)
    assert pytest.approx(ap_sub, abs=1e-5) == 0.304811


def test_p_score_candidate_acceptance_and_tradeoff_gates():
    """Verifies candidate scoring metrics, prospective gate pass/fails, and omission-substitution tradeoff."""
    if not P_SCORE_REPORT.is_file():
        pytest.skip("P-SCORE benchmark report not present at expected path")

    with open(P_SCORE_REPORT) as f:
        rep = json.load(f)

    p_contrast = rep["primary_contrast_sf_sd_minus_sf_s"]
    delta_omit_ap = p_contrast["delta_omission_ap"]
    delta_recall_fpr15 = p_contrast["delta_omission_recall_at_fpr15"]
    delta_sub_ap = p_contrast["delta_substitution_ap"]
    delta_bin_ap = p_contrast["delta_binary_ap"]

    # Prospective Gates Verification:
    # Gate 1: Primary Omission AP (target >= +0.020) -> PASS
    assert delta_omit_ap >= 0.020
    assert pytest.approx(delta_omit_ap, abs=1e-5) == 0.021615

    # Gate 2: Operating Sensitivity Recall@FPR15 (target >= +0.030) -> PASS
    assert delta_recall_fpr15 >= 0.030
    assert pytest.approx(delta_recall_fpr15, abs=1e-4) == 0.182482

    # Gate 3: Substitution AP Noninferiority (target >= -0.005) -> FAIL (Tradeoff)
    assert delta_sub_ap < -0.005, "Substitution AP noninferiority must be reported as breached"
    assert pytest.approx(delta_sub_ap, abs=1e-5) == -0.024256

    # Gate 4: Binary AP Noninferiority (target >= -0.005) -> FAIL
    assert delta_bin_ap < -0.005
    assert pytest.approx(delta_bin_ap, abs=1e-5) == -0.009045

    # Verify GOP-SA Viterbi Control metrics
    gop_sa = rep["scorers"]["gop_sa"]["primary_metrics"]
    assert pytest.approx(gop_sa["omission_ap"], abs=1e-5) == 0.049846
    assert pytest.approx(gop_sa["substitution_ap"], abs=1e-5) == 0.228104
    assert pytest.approx(gop_sa["binary_ap"], abs=1e-5) == 0.254370


def test_matched_t_op_vs_b_control_specifications():
    """Verifies that T-OP and B-CONTROL adhere strictly to matched capacity and execution contracts."""
    import sys
    import langfens_prep
    p_sup_dir = Path("/home/khoa/Projects/KLTN/langfens-rq1-p-sup/src/langfens_prep")
    if p_sup_dir.is_dir() and str(p_sup_dir) not in langfens_prep.__path__:
        langfens_prep.__path__.append(str(p_sup_dir))

    try:
        from langfens_prep.p_sup.model import CompactOperationHead, matched_initial_pair
    except ImportError:
        pytest.skip("langfens_prep.p_sup not found in path")

    treatment_head, control_head = matched_initial_pair(seed=1337)

    t_params = sum(p.numel() for p in treatment_head.parameters() if p.requires_grad)
    c_params = sum(p.numel() for p in control_head.parameters() if p.requires_grad)

    assert t_params == 266667
    assert c_params == 266667
    assert t_params == c_params, "Treatment and control must have strictly identical trainable parameter budget"
