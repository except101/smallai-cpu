import json

import pytest
import torch

import smallai.training as training
from smallai.model import ModelConfig
from smallai.tokenizer import LocalTokenizer


def test_training_resume_matches_uninterrupted_run(local_data, tmp_path, monkeypatch):
    root, tokenizer = local_data
    cfg = ModelConfig(vocab_size=2048, hidden_size=16, intermediate_size=32, num_layers=1,
                      num_heads=2, num_kv_heads=1, max_seq_len=32, dropout=0.1)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    intermediate = tmp_path / "intermediate.pt"
    original_save = training.save_checkpoint

    def capture(path, checkpoint):
        original_save(path, checkpoint)
        if checkpoint["step"] == 2 and not intermediate.exists():
            original_save(intermediate, checkpoint)

    monkeypatch.setattr(training, "save_checkpoint", capture)
    settings = dict(stage="pretrain", data=root / "pretrain", tokenizer_path=tokenizer.path,
                    config_path=config_path, steps=4, block_size=16, batch_size=1, accumulation=2,
                    threads=2, eval_every=2, eval_batches=1, log_every=4, save_every=2)
    uninterrupted = training.train(output=tmp_path / "full", **settings)
    resumed = training.train(output=tmp_path / "resumed", resume=intermediate, **settings)
    a = training.load_checkpoint(uninterrupted["checkpoint"], tokenizer)
    b = training.load_checkpoint(resumed["checkpoint"], tokenizer)
    assert a["step"] == b["step"] == 4
    assert a["target_tokens"] == b["target_tokens"]
    for name, value in a["model"].items():
        assert torch.equal(value, b["model"][name]), name
    assert a["validation_loss"] == b["validation_loss"]
    with pytest.raises(ValueError, match="非空"):
        training.train(output=tmp_path / "full", **settings)
    with pytest.raises(ValueError, match="续训"):
        training.train(output=tmp_path / "bad_resume", resume=intermediate, **{**settings, "steps": 5})
    other_tokenizer = tmp_path / "other_tokenizer.json"
    other_tokenizer.write_bytes(tokenizer.path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="分词器"):
        training.load_checkpoint(intermediate, LocalTokenizer(other_tokenizer))
