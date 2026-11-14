#!/usr/bin/env python3
"""Create local context clips for independent human review of TRAIN events."""

from __future__ import annotations

import csv
import html
import random
from pathlib import Path

import soundfile as sf


HERE = Path(__file__).resolve().parent
PADDING_S = 0.25


def main() -> None:
    with (HERE / "RAW_TEXTGRID_REVIEW.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    random.Random(20261113).shuffle(rows)
    out = HERE / "review_clips"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.wav"):
        old.unlink()  # Only this script's generated review clips live in this directory.
    cache = {}
    index_rows = []
    for row in rows:
        path = row["audio_path"]
        if path not in cache:
            audio, sample_rate = sf.read(path)
            if sample_rate != 16000 or audio.ndim != 1:
                raise ValueError(f"unexpected audio format: {path}")
            cache[path] = audio
        audio = cache[path]
        start = max(0, int((float(row["crop_start_s"]) - PADDING_S) * 16000))
        end = min(len(audio), int((float(row["crop_end_s"]) + PADDING_S) * 16000))
        if end <= start:
            raise ValueError(f"empty clip: {row['event_key']}")
        filename = f"{len(index_rows) + 1:03d}.wav"
        sf.write(out / filename, audio[start:end], 16000, subtype="PCM_16")
        row["clip_path"] = f"review_clips/{filename}"
        row["clip_start_s"] = start / 16000
        row["clip_end_s"] = end / 16000
        index_rows.append(row)
    with (HERE / "AUDIO_REVIEW_WITH_CLIPS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    blind_fields = ["item_id", "clip_path", "prompt", "target_phone", "reviewer_1", "reviewer_2", "adjudication"]
    with (HERE / "AUDIO_REVIEW_BLIND.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=blind_fields)
        writer.writeheader()
        for i, row in enumerate(index_rows, 1):
            writer.writerow({"item_id": f"{i:03d}", "clip_path": row["clip_path"],
                             "prompt": row["prompt"], "target_phone": row["target_phone"],
                             "reviewer_1": "", "reviewer_2": "", "adjudication": ""})
    cards = []
    for i, row in enumerate(index_rows, 1):
        cards.append(
            '<article><h2>Review item ' + f"{i:03d}"
            + '</h2><audio controls preload="none" src="' + html.escape(row["clip_path"], quote=True)
            + '"></audio><p>' + html.escape(row["prompt"])
            + '</p><p>Target: <b>' + html.escape(row["target_phone"])
            + '</b></p><details><summary>Reveal original label after judging</summary><p>Raw annotation: <b>'
            + html.escape(row["raw_phone_label"]) + '</b> · Model score: '
            + html.escape(row["neural_s_score"]) + '</p><p>Event: <code>'
            + html.escape(row["event_key"]) + '</code></p></details></article>'
        )
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>SM13 TRAIN audio review</title>'
            '<style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}'
            'article{border-bottom:1px solid #bbb;padding:1rem 0}audio{width:100%}'
            'code{overflow-wrap:anywhere}</style><h1>SM13 TRAIN audio review</h1>'
            '<p>Listen and record two independent judgments in AUDIO_REVIEW_BLIND.csv. '
            'Reveal the original annotation only after those judgments. The clips are '
            'randomized and do not constitute judgments.</p>'
            + "\n".join(cards) + "</html>")
    (HERE / "AUDIO_REVIEW.html").write_text(page)
    print(f"Created {len(index_rows)} clips from {len(cache)} TRAIN recordings")


if __name__ == "__main__":
    main()
