# Superseded KD run artifacts

These preliminary KD runs are invalid for scientific acceptance. Student heads returned raw logits, but binary BCE treated selected logits as probabilities. Additionally, the KD padding mask admitted padded target/gap slots. In dual-I runs, insertion teacher losses were summed rather than reliability-averaged. Use only checkpoints/predictions/receipts created after the corrected loss/mask implementation. Copied preliminary checkpoint, prediction, evaluator, receipt, and available local logs are retained here. Gold-Only runs are not superseded by this note.
