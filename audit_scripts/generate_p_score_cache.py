#!/usr/bin/env python3
"""
Generate frozen E0 FP32 log-emissions cache for Fold-1 VAL partition (NJS, RRBI).
Strictly adheres to src/langfens_prep/p_score/cache.py schema and p-score-authorization-v1.
No training, no backward passes, no parameter updates.
"""

import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import torch
import torch.nn.functional as F

# Ensure local sources are discoverable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from langfens_prep.audio_loader import load_l2arctic_audio_from_zip
from langfens_prep.g2p import CMUDictG2P
from langfens_prep.model import LangfensWav2Vec2MDD
from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
from langfens_prep.p_score.cache import (
    CONTRACT_SHA256,
    E0_SHA256,
    PHONE_MAP,
    SPLIT_SHA256,
    authorize_cache,
    json_bytes,
    parse_json,
    sha256,
    validate_cache,
)


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("[INIT] Starting frozen E0 Fold-1 VAL emission cache generation...")

    # Paths and verification
    split_path = REPO_ROOT / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    if compute_sha256(split_path) != SPLIT_SHA256:
        raise ValueError(f"Split manifest SHA mismatch! Expected {SPLIT_SHA256}")

    ckpt_path = Path("/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-TRN-02A/recovered/best.pt")
    if compute_sha256(ckpt_path) != E0_SHA256:
        raise ValueError(f"E0 checkpoint SHA mismatch! Expected {E0_SHA256}")

    zip_path = Path("/home/khoa/Projects/KLTN/Dataset/datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing authorized dataset zip: {zip_path}")

    prompts_path = REPO_ROOT / "resources/L2-ARCTIC/prompts.json"
    cmudict_path = REPO_ROOT / "resources/CMUDict/cmudict.dict"

    # Load splits
    split_data = json.loads(split_path.read_text())
    fold1 = next(f for f in split_data["folds"] if f["fold_index"] == 1)
    val_utts = fold1["val"]["utterances"]
    val_spks = fold1["val"]["speakers"]
    assert len(val_utts) == 300, f"Expected 300 VAL utterances, got {len(val_utts)}"
    assert set(val_spks) == {"NJS", "RRBI"}, f"Expected NJS and RRBI, got {val_spks}"

    # Load prompts and G2P
    prompts = json.loads(prompts_path.read_text())
    g2p = CMUDictG2P(dict_path=cmudict_path, verify_checksum=True)

    # Load E0 model in eval mode on CPU
    print(f"[MODEL] Loading frozen E0 model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LangfensWav2Vec2MDD.from_pretrained(
        "facebook/wav2vec2-base-960h",
        revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing {missing}, unexpected {unexpected}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Cache provenance hashes
    git_commit = "5e4806e1ba03aa549b370f78b8a00938eda27ad1"
    audio_loader_sha = compute_sha256(REPO_ROOT / "src/langfens_prep/audio_loader.py")
    g2p_sha = compute_sha256(REPO_ROOT / "src/langfens_prep/g2p.py")
    manifest_sha = SPLIT_SHA256

    auth_id = "AUTH-RQ1-P-SCORE-E0-FOLD1-VAL"
    provenance = {
        "kind": "frozen-e0",
        "contract_sha256": CONTRACT_SHA256,
        "phone_map": list(PHONE_MAP),
        "model_sha256": E0_SHA256,
        "emission_source_commit": git_commit,
        "preprocessing_sha256": audio_loader_sha,
        "prompt_frontend_sha256": g2p_sha,
        "source_manifest_sha256": manifest_sha,
        "precision": "float32-log-probabilities",
        "runtime_inputs": "waveform_and_written_prompt_only",
        "source_authorization_id": auth_id,
    }

    records = []
    print(f"[INFERENCE] Generating log emissions for {len(val_utts)} utterances on CPU...")
    with torch.no_grad():
        for i, u_id in enumerate(val_utts):
            spk, utt_name = u_id.split("/")
            if spk not in {"NJS", "RRBI"}:
                raise ValueError(f"Safety violation: unexpected speaker {spk}")

            # 1. Load audio waveform
            wav, sr, _ = load_l2arctic_audio_from_zip(str(zip_path), spk, utt_name, target_sr=16000)
            wav_tensor = wav.unsqueeze(0)  # (1, T_samples)

            # 2. Extract acoustic features & run encoder
            fe = model.wav2vec2.feature_extractor(wav_tensor).transpose(1, 2)
            feat_len_tensor = model._get_feat_extract_output_lengths(torch.tensor([wav_tensor.shape[1]], dtype=torch.long))
            feat_len = int(feat_len_tensor.item())

            hidden, _ = model.wav2vec2.feature_projection(fe)
            enc_out = model.wav2vec2.encoder(hidden)
            hs = enc_out.last_hidden_state
            ctc_logits = model.ctc_head(hs)
            log_probs = F.log_softmax(ctc_logits, dim=-1)  # (1, T_frames, 40)

            # Crop to valid frames
            valid_emissions = log_probs[0, :feat_len].tolist()

            # 3. G2P prompt phonemization & word mapping
            prompt_text = prompts[utt_name]
            tokens = g2p.tokenize(prompt_text)
            prompt_phones = []
            word_indices = []
            for w_idx, tok in enumerate(tokens):
                tok_phones = g2p.word_to_phones(tok)
                prompt_phones.extend(tok_phones)
                word_indices.extend([w_idx] * len(tok_phones))

            prompt_phone_ids = [PHONE_TO_CTC_INDEX[p] for p in prompt_phones]

            record = {
                "speaker_id": spk,
                "utterance_id": utt_name,
                "source_speaker_id": spk,
                "source_utterance_id": utt_name,
                "condition": "matched",
                "prompt_phone_ids": prompt_phone_ids,
                "word_indices": word_indices,
                "log_emissions": valid_emissions,
                "valid_frames": feat_len,
                "recording_status": "ok",
                "failure_reason": None,
            }
            records.append(record)

            if (i + 1) % 50 == 0 or (i + 1) == len(val_utts):
                print(f"  Processed {i + 1}/{len(val_utts)} utterances (last: {u_id}, frames: {feat_len}, phones: {len(prompt_phone_ids)})")

    # Build cache structure
    cache = {
        "schema_version": "p-score-cache-v1",
        "provenance": provenance,
        "records": records,
    }

    # Validate cache internally
    print("[VALIDATION] Validating generated cache structure...")
    validated = validate_cache(cache, "frozen-cache")
    cache_bytes = json_bytes(cache)
    cache_hash = sha256(cache_bytes)
    print(f"[CACHE] Validated cache SHA-256: {cache_hash} ({len(cache_bytes)} bytes)")

    # Save cache file in private external cache directory
    cache_dir = Path("/home/khoa/Projects/KLTN/Dataset/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "rq1_p_score_fold1_val_emissions.json"
    cache_file.write_bytes(cache_bytes)
    print(f"[SAVE] Emission cache written to: {cache_file}")

    # Build authorization object
    authorization = {
        "schema_version": "p-score-authorization-v1",
        "authorization_id": auth_id,
        "scope": "frozen_emission_scoring_only",
        "contract_sha256": CONTRACT_SHA256,
        "cache_sha256": cache_hash,
        "provenance": provenance,
        "allowed_ids": val_utts,
        "conditions": ["matched"],
        "fit_speakers": ["BWC", "ERMS", "HKK", "HQTV", "SKA", "SVBI", "TNI", "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA"],
        "selection_speakers": ["MBMPS", "TLV"],
        "calibration_speakers": ["NCC", "THV"],
        "evaluation_speakers": ["NJS", "RRBI"],
        "parity_disposition": "independent_forward_scoring_only; no training or parameter modification",
    }

    auth_bytes = json_bytes(authorization)
    auth_hash = sha256(auth_bytes)
    print(f"[AUTH] Authorization SHA-256: {auth_hash}")

    # Verify authorization against cache
    authorize_cache(validated, cache_hash, authorization, split_path)
    print("[AUTH] authorize_cache check passed cleanly!")

    # Save authorization file in manifests/RQ1-P-SCORE/authorization/
    auth_dir = REPO_ROOT / "manifests/RQ1-P-SCORE/authorization"
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_file = auth_dir / "p_score_authorization_fold1_val.json"
    auth_file.write_bytes(auth_bytes)
    print(f"[SAVE] Authorization written to: {auth_file}")

    print("[SUCCESS] Frozen E0 Fold-1 VAL emission cache and authorization successfully created and verified!")


if __name__ == "__main__":
    main()
