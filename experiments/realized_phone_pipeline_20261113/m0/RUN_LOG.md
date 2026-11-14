# M0 command record — 2026-11-13

1. Read GOAL, current registry/split specs, migration ledger, evaluator/schema, historical CTC/E4, soft alignment, focal and auxiliary/attribute reports. Open primary reference sources listed in REFERENCE_MATRIX.md. No historical test/dev_replication manifest/audio read.
2. `.venv/bin/python -m research.realized_phone_pipeline.audit`: initial run failed at Python `wave` reader (`unknown format: 3`). `soundfile.info` inspection independently showed all TRAIN WAVs are FLOAT, contradicting legacy PCM16 documentation. Replaced header reader with soundfile, recorded actual subtype and retained sample-rate/channel checks. No audio conversion performed.
3. Audit rerun succeeded: raw TRAIN parity 1800/1800, exact sequence eligible 1243, source archive hashes match, authorized audio and supplementary inventories exported. Later reruns added strict-subset support, annotation queue and scored-vs-physical SpeechOcean coverage.
4. Initial combined pytest invocation referenced a nonexistent test filename and collected no tests. Corrected to `research/evaluation_v3/test_evaluator_v3.py`; final tests and artifact verification are recorded in VALIDATION.log.
5. `.venv/bin/python -m pytest -q research/realized_phone_pipeline/test_contract.py research/evaluation_v3/test_evaluator_v3.py`: 24 passed.
6. PyTorch probe: version 2.6.0+cu124, CUDA unavailable, device_count 0. `nvidia-smi` unable to communicate with driver. No GPU job launched and no model training performed.

No new human annotation, model checkpoint or calibrated model policy was created. Source supervision is exported as corpus annotation, with uncertainty retained. The shared dirty workspace and all historical artifacts were preserved.
