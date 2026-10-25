import pytest
import tempfile
import torch
from pathlib import Path
from audit_scripts.train_fold1_ctc_binary import (
    compute_levenshtein_edits,
    greedy_ctc_decode,
    checkpoint,
    average_precision_binary
)
from langfens_prep.model import LangfensWav2Vec2MDD


def test_greedy_ctc_decode_and_levenshtein_edits():
    # Sequence with repetitions and blanks
    frame_preds = [0, 1, 1, 0, 2, 2, 0, 3, 3, 3, 0]
    hyp = greedy_ctc_decode(frame_preds, blank_id=0)
    assert hyp == [1, 2, 3]

    ref = [1, 2, 4]  # 4 is substituted for 3
    total, s, d, i = compute_levenshtein_edits(ref, hyp)
    assert total == 1
    assert s == 1
    assert d == 0
    assert i == 0

    # Insertion test
    hyp_ins = [1, 2, 5, 4]
    total, s, d, i = compute_levenshtein_edits(ref, hyp_ins)
    assert total == 1
    assert s == 0
    assert d == 0
    assert i == 1

    # Deletion test
    hyp_del = [1, 4]
    total, s, d, i = compute_levenshtein_edits(ref, hyp_del)
    assert total == 1
    assert s == 0
    assert d == 1
    assert i == 0


def test_checkpoint_saves_and_restores_patience_and_skips():
    model = LangfensWav2Vec2MDD(
        hidden_dim=32, conv_dim=16, num_transformer_layers=1,
        num_attention_heads=4, intermediate_dim=64, dropout=0.1
    )
    trainable = model.configure_two_head_training()
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = Path(tmpdir) / "local"
        drive_dir = Path(tmpdir) / "drive"
        ckpt_path = local_dir / "last.pt"

        checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=3,
            updates=150,
            micro=300,
            best=0.654,
            history=[{"epoch": 1}],
            metadata={"test": True},
            patience=2,
            skipped_amp=5,
            consecutive_skips=1,
            drive_output_dir=drive_dir
        )

        assert ckpt_path.exists()
        assert (drive_dir / "last.pt").exists()
        assert ckpt_path.stat().st_size == (drive_dir / "last.pt").stat().st_size

        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded["epoch"] == 3
        assert loaded["successful_updates"] == 150
        assert loaded["patience"] == 2
        assert loaded["skipped_amp_attempts_total"] == 5
        assert loaded["consecutive_skips"] == 1
        assert loaded["best_ap"] == 0.654
