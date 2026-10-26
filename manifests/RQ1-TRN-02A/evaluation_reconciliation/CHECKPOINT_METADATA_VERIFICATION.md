# Checkpoint metadata verification

Both recovered checkpoints deserialize as dictionaries with `model`, `optimizer`, `scheduler`, `scaler`, `epoch`, `successful_updates`, `micro_steps`, `best_ap`, `patience`, `history`, `rng`, and `metadata` fields.

| Item | `best.pt` | `last.pt` | Verification |
|---|---:|---:|---|
| Stored `epoch` | 5 | 10 | Driver stores zero-indexed epoch; corresponds to reported epochs 6 and 11 |
| `successful_updates` | 3,600 | 6,600 | Present; best checkpoint at end of epoch 6, last at end of epoch 11 |
| `best_ap` | 0.316062 | 0.316062 | Best-ever validation Binary AP, not last-epoch AP |
| Model state keys | 232 | 232 | Identical model-state structure |
| Tensor parameter count | 97,410,768 | 97,410,768 | Consistent with Wav2Vec2-base plus task heads |
| `metadata.git_sha` | `caa56783113d40eac438e5f458cdb313b8932218` | same | Matches supplied training execution SHA |
| `metadata.split_sha256` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | same | Matches contract manifest hash |
| Configuration | same | same | Fold 1, strict policy, frozen backbone, CTC + binary weights 1.0, pinned Wav2Vec2 revision |

The model architecture is defined by `src/langfens_prep/model.py` (`LangfensWav2Vec2MDD`), with the pretrained model recorded as `facebook/wav2vec2-base-960h` at revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`; checkpoint tensor count/key structure is compatible with that architecture. Full strict `load_state_dict` verification against the instantiated model has not been performed as part of this metadata-only check.

The checkpoint `epoch` fields are zero-based, whereas `history.json` and the saved evaluation report number epochs one-based. Thus `best.pt` field 5 corresponds to epoch 6 and `last.pt` field 10 to epoch 11. `best_ap` in `last.pt` is the best-ever AP; it is not evidence that last-epoch AP was 0.316062. Use epoch 11's history value as last-epoch AP when reporting it.

Split manifest SHA-256 independently recalculated as `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`. Exact byte comparisons confirm downloaded history and reports match Git copies. No metadata inconsistency was observed.
