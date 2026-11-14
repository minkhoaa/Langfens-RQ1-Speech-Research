#!/usr/bin/env python3
"""Audit TRAIN raw TextGrid labels from public v5 speaker ZIP mirrors.

Only HTTP byte-range requests for ZIP metadata and TRAIN TextGrids are made.
The archive and all evaluation splits are left untouched. This is a label
audit, not an independent listening judgment.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPO = "chikingsley/l2-arctic-release-v5.0"
BLOCK = 256 * 1024


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, speaker: str, size: int, session: requests.Session) -> None:
        self.url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{speaker}.zip"
        self.size = size
        self.session = session
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.calls = 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = (offset if whence == io.SEEK_SET else
                    self.pos + offset if whence == io.SEEK_CUR else self.size + offset)
        if not 0 <= self.pos <= self.size:
            raise ValueError(f"invalid ZIP offset: {self.pos}")
        return self.pos

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        offset = index * BLOCK
        count = min(BLOCK, self.size - offset)
        response = self.session.get(self.url, headers={"Range": f"bytes={offset}-{offset + count - 1}"}, timeout=60)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != count:
            raise IOError(f"short/non-range response at {offset}: HTTP {response.status_code}, {len(response.content)} != {count}")
        self.calls += 1
        self.bytes_fetched += len(response.content)
        self.cache[index] = response.content
        if len(self.cache) > 24:
            self.cache.popitem(last=False)
        return response.content

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.pos
        end = min(self.size, self.pos + size)
        parts = []
        while self.pos < end:
            block_index = self.pos // BLOCK
            block = self._block(block_index)
            within = self.pos % BLOCK
            take = min(end - self.pos, len(block) - within)
            parts.append(block[within:within + take])
            self.pos += take
        return b"".join(parts)


def phone_intervals(text: str) -> list[tuple[float, float, str]]:
    section = re.search(r'name\s*=\s*"phones?"([\s\S]*?)(?=\n\s*item\s*\[|\Z)', text)
    if not section:
        raise ValueError("phone tier absent")
    chunks = re.split(r'\n\s*intervals\s*\[\d+\]:', section.group(1))[1:]
    out = []
    for chunk in chunks:
        xmin = re.search(r'\bxmin\s*=\s*([0-9.]+)', chunk)
        xmax = re.search(r'\bxmax\s*=\s*([0-9.]+)', chunk)
        label = re.search(r'\btext\s*=\s*"(.*?)"', chunk)
        if xmin and xmax and label:
            out.append((float(xmin.group(1)), float(xmax.group(1)), label.group(1).strip()))
    return out


def target_labels(text: str) -> list[tuple[float, float, str]]:
    out = []
    for xmin, xmax, raw in phone_intervals(text):
        t = raw.strip()
        if not t or t.lower() in {"sil", "sp", "pause", "silence"}:
            continue
        parts = [p.strip() for p in t.split(",")]
        if len(parts) == 3 and parts[2].lower() == "a":
            continue
        out.append((xmin, xmax, raw))
    return out


def clean_phone(phone: str) -> str:
    """Match tools/parse_l2arctic.py exactly for audit comparisons."""
    return re.sub(r"[0-9]+$", "", phone.strip().upper().rstrip("*"))


def main() -> None:
    with (HERE / "AUDIO_REVIEW_QUEUE.csv").open(newline="") as f:
        queue = list(csv.DictReader(f))
    with (ROOT / "data/derived/l2arctic/manifests/train.jsonl").open() as f:
        manifest = {record["utterance_id"]: record for record in (json.loads(line) for line in f)}
    if len(manifest) != 1800:
        raise ValueError(f"unexpected TRAIN utterance count: {len(manifest)}")
    session = requests.Session()
    listing = session.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main?limit=100", timeout=30)
    listing.raise_for_status()
    sizes = {item["path"]: item["size"] for item in listing.json()}
    rows = []
    same_phone_rows = []
    all_slots = 0
    total_calls = total_bytes = 0
    for speaker in sorted({row["speaker"] for row in queue}):
        remote = HTTPRangeFile(speaker, sizes[f"{speaker}.zip"], session)
        with zipfile.ZipFile(remote) as archive:
            by_uid = {}
            for name in archive.namelist():
                if name.endswith(".TextGrid") and "annotation/" in name:
                    by_uid.setdefault(f"{speaker}_{Path(name).stem}", []).append(name)
            print(f"{speaker}: entries={len(archive.namelist())}, annotation IDs={len(by_uid)}", flush=True)
            slots_cache = {}
            for uid, rec in manifest.items():
                if rec["speaker_id"] != speaker:
                    continue
                candidates = by_uid.get(uid, [])
                if len(candidates) != 1:
                    raise ValueError(f"expected one TRAIN TextGrid for {uid}, got {candidates}")
                slots = target_labels(archive.read(candidates[0]).decode("utf-8-sig"))
                if len(slots) != rec["num_target_phones"]:
                    raise ValueError(f"slot count mismatch for {uid}: {len(slots)} != {rec['num_target_phones']}")
                slots_cache[uid] = (candidates[0], slots)
                for tp, (xmin, xmax, label) in zip(rec["target_phones"], slots):
                    all_slots += 1
                    parts = [p.strip() for p in label.split(",")]
                    if abs(float(tp["xmin"]) - xmin) > 1e-6 or abs(float(tp["xmax"]) - xmax) > 1e-6:
                        raise ValueError(f"raw interval/manifest mismatch: {uid} {tp['slot_idx']}")
                    if len(parts) == 3:
                        expected = {"s": "SUBSTITUTION", "d": "OMISSION"}.get(parts[2].lower(), "CORRECT")
                        raw_target = clean_phone(parts[0])
                        raw_realized = clean_phone(parts[1]) if expected == "SUBSTITUTION" else None
                    else:
                        expected = "CORRECT"
                        raw_target = clean_phone(label)
                        raw_realized = None
                    if tp["label"] != expected or tp["target_phone"] != raw_target:
                        raise ValueError(f"raw label/manifest mismatch: {uid} {tp['slot_idx']} {label}")
                    if expected == "SUBSTITUTION" and tp["realized_phone"] != raw_realized:
                        raise ValueError(f"raw realized/manifest mismatch: {uid} {tp['slot_idx']} {label}")
                    if tp["label"] == "SUBSTITUTION" and tp["target_phone"] == tp["realized_phone"]:
                        if len(parts) != 3 or parts[2].lower() != "s":
                            raise ValueError(f"raw substitution tag mismatch: {uid} {tp['slot_idx']} {label}")
                        if clean_phone(parts[0]) != tp["target_phone"] or clean_phone(parts[1]) != tp["realized_phone"]:
                            raise ValueError(f"raw phone normalization mismatch: {uid} {tp['slot_idx']} {label}")
                        same_phone_rows.append({
                            "event_key": f'L2_ARCTIC|{speaker}|{uid}|target|{tp["slot_idx"]}',
                            "speaker": speaker, "utterance": uid, "slot": tp["slot_idx"],
                            "raw_label": label, "raw_target": parts[0], "raw_realized": parts[1],
                            "realized_has_deviation_star": "*" in parts[1],
                            "raw_spelling_equal": parts[0] == parts[1],
                            "raw_interval_start_s": xmin, "raw_interval_end_s": xmax,
                            "textgrid_path_in_zip": candidates[0],
                        })
            for row in (r for r in queue if r["speaker"] == speaker):
                uid = row["utterance"]
                candidate, slots = slots_cache[uid]
                slot = int(row["slot"])
                if slot >= len(slots):
                    raise ValueError(f"slot {slot} absent in {uid}")
                xmin, xmax, label = slots[slot]
                manual_start = float(row["crop_start_s"])
                manual_end = float(row["crop_end_s"])
                if xmax < manual_start or xmin > manual_end:
                    raise ValueError(f"raw interval outside crop: {row['event_key']}: {xmin}, {xmax}")
                rows.append({**row, "textgrid_path_in_zip": candidate,
                             "raw_interval_start_s": xmin, "raw_interval_end_s": xmax,
                             "raw_phone_label": label})
        total_calls += remote.calls
        total_bytes += remote.bytes_fetched
        print(f"reviewed {len(rows)}/{len(queue)}; range calls={total_calls}", flush=True)

    with (HERE / "RAW_TEXTGRID_REVIEW.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (HERE / "RAW_SAME_PHONE_LABELS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(same_phone_rows[0]))
        writer.writeheader()
        writer.writerows(same_phone_rows)
    same = [r for r in rows if r["reason"].startswith("same_phone")]
    summary = {
        "queue_rows": len(rows), "same_phone_rows": len(same),
        "train_utterances_checked": len(manifest), "train_slots_checked": all_slots,
        "all_same_phone_rows": len(same_phone_rows),
        "all_same_phone_with_deviation_star": sum(r["realized_has_deviation_star"] for r in same_phone_rows),
        "all_same_phone_raw_spelling_equal": sum(r["raw_spelling_equal"] for r in same_phone_rows),
        "all_same_phone_without_star": sum(not r["realized_has_deviation_star"] for r in same_phone_rows),
        "same_phone_raw_label_counts": {label: sum(r["raw_phone_label"] == label for r in same)
                                        for label in sorted({r["raw_phone_label"] for r in same})},
        "range_calls": total_calls, "range_bytes_fetched": total_bytes,
        "source": f"https://huggingface.co/datasets/{REPO}/tree/main",
        "caveat": "Raw transcription audit only; no independent audio listening or relabeling.",
    }
    (HERE / "RAW_TEXTGRID_REVIEW_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
