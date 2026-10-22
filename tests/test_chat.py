from types import SimpleNamespace

import torch

import smallai.inference as inference


def test_chat_executes_tool_then_supplies_actual_result_to_next_turn(monkeypatch, capsys):
    model = SimpleNamespace(parameters=lambda: iter([torch.empty(1)]))
    monkeypatch.setattr(inference, "load_model", lambda *a: (model, None, {"step": 0}))
    invocations = []

    def reply(model, tokenizer, messages, tools, *args):
        invocations.append(list(messages))
        if len(invocations) == 1:
            return '<|tool_call|>{"name":"calculator","arguments":{"expression":"17 * 23"}}<|tool_end|>', messages
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["name"] == "calculator"
        assert messages[-1]["content"] == '{"result":391}'
        assert messages[-2]["tool_calls"][0]["id"] == messages[-1]["tool_call_id"]
        return "17 × 23 = 391。", messages

    monkeypatch.setattr(inference, "generate_reply", reply)
    inference.chat("local.pt", "tokenizer.json", prompt="请计算 17 * 23。")
    text = capsys.readouterr().out
    assert len(invocations) == 2
    assert '[工具结果] {"result":391}' in text
    assert "助手：17 × 23 = 391。" in text


def test_chat_displays_malformed_output_and_never_executes_it(monkeypatch, capsys):
    model = SimpleNamespace(parameters=lambda: iter([torch.empty(1)]))
    monkeypatch.setattr(inference, "load_model", lambda *a: (model, None, {"step": 0}))
    monkeypatch.setattr(inference, "generate_reply", lambda *a: ("<|tool_call|>{broken", a[2]))

    def unexpected_execution(call):
        raise AssertionError("Malformed output must not execute a tool")

    monkeypatch.setattr(inference, "execute_tool", unexpected_execution)
    inference.chat("local.pt", "tokenizer.json", prompt="计算。")
    text = capsys.readouterr().out
    assert "助手原始输出：<|tool_call|>{broken" in text
    assert "[本轮停止]" in text
