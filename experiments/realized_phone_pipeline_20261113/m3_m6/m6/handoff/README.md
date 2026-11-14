# Frozen research prototype — not product accepted

Fixed M4 joint recognizer (seed303/fold0), FIT-derived finite CTC likelihood
decoder and CAL B=.20 policy. Local offline backbone and head included. This
is development-speaker seed replication, not fresh-speaker confirmation.
Training data provenance: L2-ARCTIC CC-BY-NC4.0 research data; no datasets or
credentials included. Attribute categories are a documented nominal profile,
not measurements of an individual learner's articulators. Scores are bounded
likelihood evidence, not calibrated full event posteriors. CTC emission frames
are not true phone boundaries. Arbitrary text-to-phone parity remains unsolved.

Install requirements.txt in a compatible environment. Supply request.json:
{"audio_path":"/absolute/path/learner.wav","reference_phones":["T","EH","S","T"],"speaker_id":"NEW_SPEAKER","utterance_id":"NEW_001"}

Run: python predict.py --bundle . --request request.json --output prediction.json
Input accepts waveform and supplied reference phones only. Gold labels,
realized transcripts and gold timestamps are rejected. Unsupported phones
(including SPN) fail explicitly; documented reference aliases are normalized.
Output has exactly N C/S/O events and N+1 insertion-gap events; multiple
greedy insertion candidates are retained. FWR .20 on calibration is not a
guarantee on a new speaker. Review model/decoder limitations before using it.

Fresh confirmation: python confirm.py --bundle . --manifest fresh_human_gold.json --output confirmation.json
Manifest must attest human_verified labels/speaker disjointness and provide
utterances with reference_phones, target_labels, gap_has_insertion plus request
IDs/audio. Reused corpus speakers and missing/malformed gold are rejected.
Attestation requires independent human provenance verification; it is not
proof merely because a JSON flag is set. Product acceptance still requires
user-agreed criteria. No fresh data or acceptance result is fabricated here.
