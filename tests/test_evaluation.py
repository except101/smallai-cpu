import json

import pytest

import smallai.inference as inference
from smallai.tools import compact_json


def test_evaluation_separates_format_arguments_execution_and_no_call(tmp_path, monkeypatch):
    records, responses = [], []
    for index, expression in enumerate(["1+2", "4+5", "3*4"]):
        records.append({"id": index, "messages": [{"role": "user", "content": expression}],
                        "expected_call": {"name": "calculator", "arguments": {"expression": expression}},
                        "expected_result": {"result": [3, 9, 12][index]}})
    for expression in ["1+2", "4+6"]:
        responses.append('<|tool_call|>' + compact_json({"name": "calculator", "arguments": {"expression": expression}}) + '<|tool_end|>')
    responses.extend(['<|tool_call|>{bad}<|tool_end|>', "你好", responses[0]])
    records += [{"id": index, "messages": [{"role": "user", "content": "你好"}], "expected_call": None}
                for index in [3, 4]]
    data = tmp_path / "eval.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    iterator = iter(responses)
    monkeypatch.setattr(inference, "load_model", lambda *a: (None, None, {"step": 7}))
    monkeypatch.setattr(inference, "generate_reply", lambda *a: (next(iterator), []))
    result = inference.evaluate("local.pt", "tokenizer.json", data, tmp_path / "report.json")
    assert result["format_accuracy"] == pytest.approx(2 / 3)
    assert result["tool_selection_accuracy"] == pytest.approx(2 / 3)
    assert result["exact_call_accuracy"] == pytest.approx(1 / 3)
    assert result["execution_accuracy"] == pytest.approx(1 / 3)
    assert result["no_call_accuracy"] == pytest.approx(0.5)
