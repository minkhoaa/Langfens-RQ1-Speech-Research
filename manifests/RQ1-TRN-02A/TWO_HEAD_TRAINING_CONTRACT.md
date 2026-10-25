# Two-head Fold 1 training contract

Implementation: `src/langfens_prep/model.py`, `audit_scripts/train_fold1_ctc_binary.py`.

- Objective weights are `lambda_ctc=1.0`, `lambda_binary=1.0`; total objective is `lambda_ctc * CTC + lambda_binary * masked BCE`.
- The training driver forwards CTC and binary labels only; it deliberately does not pass `target_diagnosis` or auxiliary diagnosis targets. The model's historical three-head architecture remains available.
- `configure_two_head_training()` freezes every Wav2Vec2 parameter and every diagnosis-head parameter. The exact trainable set comprises CTC head, cross-attention/canonical embeddings, and Binary MDD head. It returns precisely those parameters for optimizer construction. Frozen backbone remains in evaluation mode during `.train()` calls.
- CTC and Binary losses are checked for finiteness. Binary targets are restricted to `0`, `1`, or ignore index `-100`; existing CTC contract checks enforce target alphabet/blank/length constraints. Existing binary sequence-length and padding checks prohibit silent truncation.
- Driver checks absent diagnosis gradients, finite trainable gradients (GradScaler-detected overflow attempts are skipped and counted), and exact optimizer update accounting. Its optimizer never contains backbone or diagnosis parameters.
- Unit tests cover freezing, exact optimizer membership, eval-mode behavior, and AP implementation. Broader loss/shape contracts are exercised by the repository regression suite.
- The two observed-head micro-step updates on Colab changed both CTC and Binary parameters. Backbone and diagnosis parameters remained bit-identical.
