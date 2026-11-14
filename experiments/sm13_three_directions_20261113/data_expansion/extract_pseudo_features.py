"""Extract fit-speaker-only 13-layer features for a fixed unlabeled L2 subset.

No error label is read or generated here. Prompt phones are references for
alignment/features only. Teacher probabilities are generated in train_eval.py.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, Wav2Vec2ForCTC

ROOT = Path("/home/khoa/Projects/New-Method-Research")
HERE = ROOT / "experiments/sm13_three_directions_20261113/data_expansion"
sys.path.insert(0, str(ROOT / "experiments/sm13_three_directions_20261113/shared"))
sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))
sys.path.insert(0, str(ROOT / "tools"))

from dataset_util import DatasetManager  # noqa: E402
from protocol import SplitProtocol, GROUPS  # noqa: E402
from lexicon import CMUDictLexicon  # noqa: E402

CHARSIU_MODEL = Path.home() / ".cache/huggingface/hub/models--charsiu--en_w2v2_fc_10ms/snapshots/e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f"
CHARSIU_TOKENIZER = Path.home() / ".cache/huggingface/hub/models--charsiu--tokenizer_en_cmu/snapshots/10507401aedf5e0aba164128535b49225ff95260"
W2V2_MODEL = "facebook/wav2vec2-base-960h"
MIRROR_REV = "cc02fd37197966e5de11ad737f7affd577c61992"
SPEAKERS = ("ABA", "ASI", "BWC", "EBVS", "HJK", "HKK", "HQTV", "LXC", "NJS", "PNV", "SKA", "SVBI")


def fit_folds(speaker: str) -> list[int]:
    return [fold for fold in range(4)
            if speaker not in GROUPS[fold] and speaker not in GROUPS[(fold + 1) % 4]]


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_centers(logits: np.ndarray, phones: list[str], tokenizer) -> tuple[np.ndarray, int]:
    vocab = tokenizer.get_vocab()
    sil = vocab["[SIL]"]
    ids = [vocab.get(p, vocab.get("[UNK]", sil)) for p in phones]
    lp = torch.log_softmax(torch.as_tensor(logits, dtype=torch.float64), -1).numpy()
    n, t = len(ids), lp.shape[0]
    labels = np.empty(2 * n, dtype=np.int64)
    sep = np.zeros(2 * n, bool)
    for i, phone_id in enumerate(ids):
        labels[2 * i] = phone_id
        labels[2 * i + 1] = sil
        sep[2 * i + 1] = True
    dp = np.full((t, 2 * n), -np.inf)
    back = np.full((t, 2 * n), -1, np.int32)
    dp[0, 0] = lp[0, labels[0]]
    if 2 * n > 1:
        dp[0, 1] = lp[0, labels[1]]
    for frame in range(1, t):
        for state in range(2 * n):
            opts = [(dp[frame - 1, state], state)]
            if state:
                opts.append((dp[frame - 1, state - 1], state - 1))
            if state > 1 and sep[state - 1]:
                opts.append((dp[frame - 1, state - 2], state - 2))
            value, previous = max(opts, key=lambda item: item[0])
            dp[frame, state] = value + lp[frame, labels[state]]
            back[frame, state] = previous
    state = int(np.argmax(dp[-1]))
    path = np.empty(t, np.int32)
    for frame in range(t - 1, -1, -1):
        path[frame] = state
        if frame:
            state = int(back[frame, state])
    centers = []
    for i in range(n):
        ix = np.flatnonzero(path == 2 * i)
        centers.append(float(ix.mean()) if ix.size else (i + 0.5) * t / max(1, n) - 0.5)
    return np.asarray(centers, np.float32), t


def extract_target_windows(hidden_states, centers: np.ndarray, n_frames: int) -> np.ndarray:
    layer_rows = []
    for layer in range(13):
        hidden = hidden_states[layer][0].float().cpu().numpy()
        scaled = np.rint(centers * (len(hidden) / max(1, n_frames))).astype(int)
        offsets = np.arange(-8, 9)
        chunks = hidden[np.clip(scaled[:, None] + offsets[None, :], 0, len(hidden) - 1)]
        middle = chunks[:, 3:14, :]
        layer_rows.append(np.concatenate((middle.mean(1), middle.std(1, ddof=0), middle[:, -1] - middle[:, 0]), axis=1))
    return np.stack(layer_rows, axis=1).astype(np.float32)


def speaker_candidates(spk: str, zf: zipfile.ZipFile, known_ids: set[str], lex: CMUDictLexicon,
                       supported_phones: set[str]):
    wav_paths = sorted(p for p in zf.namelist() if p.startswith(f"{spk}/wav/") and p.lower().endswith(".wav"))
    picks = []
    for wav_path in wav_paths:
        short_id = Path(wav_path).stem
        uid = f"{spk}_{short_id}"
        if uid in known_ids:
            continue
        txt_path = f"{spk}/transcript/{short_id}.txt"
        if txt_path not in zf.namelist():
            continue
        prompt = zf.read(txt_path).decode("utf-8", "replace").strip()
        phones = lex.sentence_to_phones(prompt)
        words = prompt.split()
        # Keep only prompts whose alphabetic tokens are all present in CMUdict.
        import re
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", prompt)
        if not phones or any(not lex.word_to_phones(w) for w in tokens) or any(p not in supported_phones for p in phones):
            continue
        picks.append((uid, wav_path, prompt, phones))
    picks.sort(key=lambda r: (hashlib.sha256(r[0].encode()).hexdigest(), r[0]))
    if len(picks) < 50:
        raise RuntimeError(f"{spk}: only {len(picks)} eligible nonmanifest WAVs, need fixed cap 50")
    return picks[:50]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = DatasetManager()
    split = SplitProtocol(dm).split(args.fold)
    known = {uid for uid, spk in dm.speakers.items() if spk in SPEAKERS}
    lex = CMUDictLexicon()
    tokenizer = AutoTokenizer.from_pretrained(str(CHARSIU_TOKENIZER), local_files_only=True)
    phone_inventory = sorted(p for p in tokenizer.get_vocab()
                             if p not in {"[SIL]", "[UNK]", "[PAD]", "<s>", "</s>"})
    refs_by_phone = {phone: i for i, phone in enumerate(phone_inventory)}
    if len(refs_by_phone) != 39:
        raise AssertionError(f"expected 39 CharSiu phones for shared reference schema, got {len(refs_by_phone)}")
    # Verify the tokenizer's sorted phone columns against the actual TRAIN
    # one-hot reference matrix, rather than relying on a matching dimension.
    train_ref_phone_to_columns: dict[str, set[int]] = {}
    cursor = 0
    manifest_rows = [json.loads(line) for line in
                     (ROOT / "data/derived/l2arctic/manifests/train.jsonl").read_text().splitlines() if line.strip()]
    if len(manifest_rows) != len(dm.order):
        raise AssertionError("TRAIN manifest changed while validating phone reference mapping")
    for row_index, record in enumerate(manifest_rows):
        uid = record["utterance_id"]
        if uid != dm.order[row_index]:
            raise AssertionError("DatasetManager and TRAIN manifest order mismatch")
        for target in record["target_phones"]:
            phone = target["target_phone"]
            column = int(np.argmax(dm.target_refs[cursor]))
            train_ref_phone_to_columns.setdefault(phone, set()).add(column)
            cursor += 1
    if cursor != len(dm.target_refs):
        raise AssertionError("TRAIN phone/reference row count mismatch")
    tokenizer_mapping = {phone: idx for phone, idx in refs_by_phone.items()}
    observed_mapping = {phone: next(iter(columns)) for phone, columns in train_ref_phone_to_columns.items()
                        if phone in tokenizer_mapping and len(columns) == 1}
    if observed_mapping != tokenizer_mapping:
        raise AssertionError(f"CharSiu phone columns differ from DatasetManager TRAIN references: {observed_mapping}")
    charsiu = None
    w2v2 = None
    feature_extractor = AutoFeatureExtractor.from_pretrained(W2V2_MODEL, local_files_only=True)
    if not feature_extractor.do_normalize:
        raise AssertionError("historical Wav2Vec2 pooled features require configured waveform normalization")
    all_features, all_refs, rows = [], [], []
    archive_receipts = {}
    start = time.perf_counter()
    out_dir = HERE / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spk in split.fit_speakers:
        speaker_cache = out_dir / f"speaker_{spk}_unlabeled_features.npz"
        speaker_events_path = out_dir / f"speaker_{spk}_event_manifest.jsonl"
        speaker_receipt_path = out_dir / f"speaker_{spk}_extraction_receipt.json"
        if speaker_cache.exists() and speaker_events_path.exists() and speaker_receipt_path.exists():
            cached_receipt = json.loads(speaker_receipt_path.read_text())
            if cached_receipt["first_fit_fold"] not in fit_folds(spk):
                raise AssertionError(f"cached feature source fold is not a fit fold for {spk}")
            cached_rows = [json.loads(line) for line in speaker_events_path.read_text().splitlines() if line.strip()]
            if len({r["utterance_id"] for r in cached_rows}) != 50:
                raise AssertionError(f"cached speaker {spk} does not contain exactly 50 sampled utterances")
            with np.load(speaker_cache) as z:
                all_features.append(z["target_features"])
                all_refs.append(z["target_refs"])
            rows.extend(cached_rows)
            archive_receipts[spk] = cached_receipt["source_archive"]
            print(f"fold={args.fold} speaker={spk}: reuse frozen cache from fit fold {cached_receipt['first_fit_fold']}", flush=True)
            continue
        archive = HERE / "raw_archives" / f"{spk}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"missing acquired source archive: {archive}")
        if charsiu is None:
            charsiu = Wav2Vec2ForCTC.from_pretrained(str(CHARSIU_MODEL), local_files_only=True).eval().to(device)
            w2v2 = AutoModel.from_pretrained(W2V2_MODEL, local_files_only=True).eval().to(device)
        archive_sha = sha256_file(archive)
        archive_receipts[spk] = {"path": str(archive.relative_to(ROOT)), "sha256": archive_sha,
                                 "source_revision": MIRROR_REV}
        with zipfile.ZipFile(archive) as zf:
            picks = speaker_candidates(spk, zf, known, lex, set(refs_by_phone))
            speaker_features, speaker_refs, speaker_rows = [], [], []
            for ordinal, (uid, wav_path, prompt, phones) in enumerate(picks, start=1):
                if any(p not in refs_by_phone for p in phones):
                    speaker_rows.append({"speaker_id": spk, "utterance_id": uid, "prompt": prompt,
                                 "status": "excluded_unsupported_reference_phone", "phones": phones})
                    continue
                audio_blob = zf.read(wav_path)
                samples, sr = sf.read(io.BytesIO(audio_blob), dtype="float32", always_2d=True)
                mono = samples.mean(axis=1)
                if sr != 16000:
                    # Match tools/parse_l2arctic.py exactly, including its
                    # PCM16 WAV quantization on save before feature extraction.
                    resampler = torchaudio.transforms.Resample(orig_freq=int(sr), new_freq=16000)
                    resampled = resampler(torch.from_numpy(mono).float().unsqueeze(0))
                    encoded = io.BytesIO()
                    torchaudio.save(encoded, resampled, 16000, format="wav")
                    encoded.seek(0)
                    mono, processed_sr = sf.read(encoded, dtype="float32", always_2d=False)
                    if processed_sr != 16000:
                        raise AssertionError(f"torchaudio output sample rate mismatch: {processed_sr}")
                # Match experiments/rq1_encoder_bench_01/features/extract_all_features.py
                # for Wav2Vec2. CharSiU's historical path receives the same
                # 16 kHz PCM16 waveform without this Wav2Vec2-specific normalization.
                processed = feature_extractor(mono, sampling_rate=16000, return_tensors="pt")
                w2v_wav = processed["input_values"].to(device)
                ctc_wav = torch.as_tensor(mono, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.inference_mode():
                    ctc_logits = charsiu(ctc_wav).logits[0].float().cpu().numpy()
                    centers, n_frames = align_centers(ctc_logits, phones, tokenizer)
                    out = w2v2(w2v_wav, output_hidden_states=True)
                    feat = extract_target_windows(out.hidden_states, centers, n_frames)
                refs = np.zeros((len(phones), 39), np.float32)
                for i, phone in enumerate(phones):
                    refs[i, refs_by_phone[phone]] = 1.0
                speaker_features.append(feat)
                speaker_refs.append(refs)
                for position, phone in enumerate(phones):
                    speaker_rows.append({
                        "dataset_id": "L2_ARCTIC_UNLABELED",
                        "speaker_id": spk,
                        "utterance_id": uid,
                        "position": position,
                        "intended_phone": phone,
                        "prompt": prompt,
                        "source_archive": str(archive.relative_to(ROOT)),
                        "source_archive_sha256": archive_sha,
                        "source_revision": MIRROR_REV,
                        "audio_member": wav_path,
                        "audio_sha256": sha256_bytes(audio_blob),
                        "source_sample_rate": int(sr),
                        "w2v2_feature_extractor": "facebook/wav2vec2-base-960h AutoFeatureExtractor; checkpoint do_normalize=True",
                        "charsiu_waveform": "same repository-resampled PCM16 waveform; Wav2Vec2 AutoFeatureExtractor normalization is not applied to CharSiU",
                        "alignment": "frozen CharSiu CTC prompt-phone forced alignment",
                        "status": "feature_extracted_unlabeled",
                    })
                if ordinal % 25 == 0:
                    print(f"fold={args.fold} speaker={spk} audio={ordinal}/50 phones={sum(len(x) for x in speaker_refs)}", flush=True)
        spk_features = np.concatenate(speaker_features, axis=0)
        spk_refs = np.concatenate(speaker_refs, axis=0)
        selected_ids = sorted({r["utterance_id"] for r in speaker_rows})
        if len(selected_ids) != 50:
            raise AssertionError(f"speaker {spk}: extracted {len(selected_ids)} of 50 sampled utterances")
        np.savez_compressed(speaker_cache, target_features=spk_features, target_refs=spk_refs)
        speaker_events_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in speaker_rows))
        speaker_receipt_path.write_text(json.dumps({
            "speaker_id": spk, "first_fit_fold": args.fold, "source_archive": archive_receipts[spk],
            "source_revision": MIRROR_REV, "selected_utterance_cap": 50,
            "selected_utterances": selected_ids,
            "feature_shape": list(spk_features.shape), "reference_shape": list(spk_refs.shape),
            "no_gold_error_labels_used": True,
        }, indent=2) + "\n")
        all_features.append(spk_features)
        all_refs.append(spk_refs)
        rows.extend(speaker_rows)

    feature_rows = sum(len(x) for x in all_features)
    ref_rows = sum(len(x) for x in all_refs)
    if feature_rows != ref_rows or len(rows) != feature_rows:
        raise AssertionError("cached acoustic and reference row counts diverged")
    receipt = {
        "fold": args.fold,
        "fit_speakers": list(split.fit_speakers),
        "calibration_speakers_excluded": list(split.calibration_speakers),
        "held_speakers_excluded": list(split.held_speakers),
        "selected_utterances_per_fit_speaker": 50,
        "source_repo": "https://huggingface.co/datasets/chikingsley/l2-arctic-release-v5.0",
        "source_revision": MIRROR_REV,
        "source_license": "CC-BY-NC-4.0 (mirror card)",
        "source_archives": archive_receipts,
        "selected_utterances": sorted({r["utterance_id"] for r in rows}),
        "feature_shape": [feature_rows, 13, 2304],
        "reference_shape": [ref_rows, 39],
        "features_stored_once_per_speaker": True,
        "feature_model": W2V2_MODEL,
        "feature_extractor": "AutoFeatureExtractor from facebook/wav2vec2-base-960h; do_normalize=True",
        "raw_resampling": "tools/parse_l2arctic.py torchaudio Resample; PCM16 WAV encode/decode; then AutoFeatureExtractor normalization",
        "feature_layers": list(range(13)),
        "phone_reference_columns_verified_against_train": True,
        "phone_reference_columns": refs_by_phone,
        "ctc_aligner": "charsiu/en_w2v2_fc_10ms (frozen)",
        "canonical_phones": "prompt transcript through repository CMUdict; intended text only",
        "no_gold_error_labels_used": True,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (out_dir / f"fold{args.fold}_extraction_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (out_dir / f"fold{args.fold}_event_manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
