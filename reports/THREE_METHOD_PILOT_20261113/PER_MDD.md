# PER-MDD local pilot

## Implementation

Implemented frame embedding nearest-neighbor retrieval in `research/three_method_pilot/per_mdd/` and ran it on the frozen three-method pilot manifest. This is the method described in §2.1–2.2 of the supplied PER-MDD preprint (arXiv:2511.20107v1): make one phone-labeled midpoint frame per aligned FIT realization span, compare every query frame to that pool by cosine similarity, take top 10, retain neighbors with similarity at least 0.7, vote by plurality, predict blank when none pass, then collapse adjacent repeated frame labels and remove blanks.

Pool labels are observed realized phones from FIT target spans and FIT insertion spans. Omitted target spans contribute no phone sample. The pool contains frame vectors, not class prototypes. The request records also contain reference sequences, so the loader drops the `reference_phones` field immediately for every role; retrieval, pool construction, and inference never inspect it. CAL and HELD annotation files are not loaded. The runner checks the frozen request IDs, FIT annotation IDs and TRAIN split, waveform SHA-256 values, and speaker-disjoint FIT/CAL/HELD groups before extraction.

## Backbone and fidelity

The preprint uses `facebook/hubert-large-ls960-ft`, an ASR-finetuned HuBERT-large checkpoint, with middle-frame pooling, top-k 10 and threshold 0.7. The first run used the locally cached `facebook/hubert-base-ls960` snapshot `dba3bb02fda4248b6e082697eee756de8fe8aa8a`. It is a pretrained self-supervised HuBERT-base model, not the paper's ASR-finetuned large model. This is a named backbone adaptation; the results should not be described as an exact reproduction of the paper's backbone.

A second run used the paper's original backbone, `facebook/hubert-large-ls960-ft`, pinned to revision `ece5fabbf034c1073acae96d5401b25be96709d8`. The CTC output head was discarded; the frozen HuBERT frame encoder supplied the embeddings. This run keeps the same 24-utterance FIT pool and 12 CAL/HELD queries as the pilot. The paper instead constructs its pool from 500 training recordings, so the backbone matches but the experiment scale and data protocol do not.

## Bounded run

- Manifest: `experiments/three_method_pilot_20261113/pilot.json` (SHA-256 `bd33048176672747e5f7b598e971f123d26943ff9c18fcb915a2cbc3c4ed438d`).
- FIT pool: 24 utterances, 723 realized-phone midpoint frame vectors.
- Query outputs: 6 CAL and 6 HELD utterances; official test data was not accessed.
- Model weights SHA-256: `062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad`.
- HuBERT-large weights SHA-256: `9cf43abec3f0410ad6854afa4d376c69ccb364b48ddddfd25c4c5aa16398eab0`.
- Extraction and retrieval elapsed time: 4.23 seconds for base and 7.11 seconds for large; 0 optimizer updates; GPU lock held nonblocking during both CUDA runs.
- Synthetic unit checks: 6 passed, covering thresholding, plurality tie resolution, blank and duplicate collapse, timestamp clipping, speaker isolation, and nonfinite vector rejection.

## Artifacts and use

- `experiments/three_method_pilot_20261113/per_mdd/recognized_phones.jsonl` contains the evaluator-facing recognized sequences (`utterance_id`, `role`, `recognized_phones`).
- `frame_retrieval.jsonl` retains per-frame labels, top-k cosine scores, vote counts, and collapsed sequences.
- `pool_provenance.jsonl` records each FIT-only pool vector's phone, source span, speaker, and frame index.
- `receipt.json` records input and model hashes, pool composition, parameters, compute duration, and isolation checks.
- `experiments/three_method_pilot_20261113/per_mdd_large/` contains the original-backbone run in the same artifact layout.

This is an implementation and bounded smoke pilot, not a paper reproduction or product acceptance result. The paper's backbone mismatch, tiny speaker-disjoint query set, and supplied corpus annotations limit conclusions; downstream sequence metrics and method comparison are reported by the pilot evaluator.
