import pytest
import torch
from pathlib import Path

from smallai.model import ModelConfig, SmallLM


def tiny(**overrides):
    values = dict(vocab_size=73, hidden_size=32, intermediate_size=64, num_layers=2,
                  num_heads=4, num_kv_heads=2, max_seq_len=64)
    values.update(overrides)
    return SmallLM(ModelConfig(**values)).eval()


def test_parameter_budget_and_shared_output_weights():
    cfg = ModelConfig()
    model = SmallLM(cfg)
    assert cfg.parameter_count() == 22_028_160
    assert sum(p.numel() for p in model.parameters()) == 22_028_160
    assert not hasattr(model, "lm_head")


@pytest.mark.parametrize("name", ["cpu_22m", "smoke"])
def test_installed_package_presets_match_workspace_configs(name):
    path = Path(__file__).resolve().parent.parent / "configs" / f"{name}.json"
    assert ModelConfig.load(name).to_dict() == ModelConfig.load(path).to_dict()


def test_future_tokens_cannot_change_past_logits():
    torch.manual_seed(5)
    model = tiny()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 40, 50, 60]])
    with torch.no_grad():
        a, b = model(first)["logits"], model(second)["logits"]
    torch.testing.assert_close(a[:, :3], b[:, :3], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("chunks", [[3, 1, 1, 1], [2, 2, 2]])
def test_cached_and_uncached_generation_match(chunks):
    model = tiny()
    tokens = torch.tensor([[1, 12, 23, 34, 45, 56]])
    with torch.no_grad():
        expected = model(tokens)["logits"]
        actual, past, start = [], None, 0
        for length in chunks:
            out = model(tokens[:, start:start + length], past_key_values=past, use_cache=True)
            actual.append(out["logits"])
            past = out["past_key_values"]
            start += length
    torch.testing.assert_close(torch.cat(actual, dim=1), expected, atol=1e-6, rtol=1e-5)
    assert past[0][0].shape == (1, 2, 6, 8)


def test_loss_predicts_next_token_and_ignores_masked_targets():
    model = tiny()
    ids = torch.tensor([[2, 5, 8, 11]])
    labels = torch.tensor([[-100, -100, 8, 11]])
    out = model(ids, labels)
    log_probabilities = out["logits"].log_softmax(-1)
    expected = -(log_probabilities[0, 1, 8] + log_probabilities[0, 2, 11]) / 2
    torch.testing.assert_close(out["loss"], expected)
    out["loss"].backward()
    assert torch.isfinite(model.embedding.weight.grad).all()
    with pytest.raises(ValueError, match="没有"):
        model(ids, torch.full_like(ids, -100))


def test_context_boundaries_are_enforced():
    model = tiny(max_seq_len=4)
    with pytest.raises(ValueError, match="上下文"):
        model(torch.ones(1, 5, dtype=torch.long))
    assert len(model.generate([1, 2, 3], max_new_tokens=50, temperature=0)) == 1
    with pytest.raises(ValueError, match="提示词"):
        model.generate([1, 2, 3, 4])
