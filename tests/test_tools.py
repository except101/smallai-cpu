from datetime import datetime

import pytest

from smallai.tools import calculate, execute_tool, parse_tool_call


@pytest.mark.parametrize("expression,expected", [("17 * 23", 391), ("(2 + 3) * 4", 20),
                                                 ("-7 / 2", -3.5), ("2 ** 5", 32)])
def test_real_calculations(expression, expected):
    assert execute_tool({"name": "calculator", "arguments": {"expression": expression}}) == {"result": expected}


@pytest.mark.parametrize("expression", ["__import__('os').getcwd()", "a + 1", "[1][0]", "True + 2",
                                        "1 / 0", "10 ** 99999999", "2 ** 0.5", "1e200", "" ])
def test_invalid_calculations_are_rejected(expression):
    with pytest.raises(ValueError):
        calculate(expression)


def test_only_complete_calls_are_accepted():
    assert parse_tool_call("正常回答。") is None
    valid = '<|tool_call|>{"name":"calculator","arguments":{"expression":"2+3"}}<|tool_end|>'
    assert parse_tool_call(valid)["name"] == "calculator"
    for text in [valid[:-4], "我先计算：" + valid, '<|tool_call|>{bad}<|tool_end|>',
                 '<|tool_call|>{"name":"calculator","arguments":"2+3"}<|tool_end|>']:
        with pytest.raises(ValueError):
            parse_tool_call(text)
    with pytest.raises(ValueError, match="未注册"):
        execute_tool({"name": "run_python", "arguments": {}})
    with pytest.raises(ValueError, match="expression"):
        execute_tool({"name": "calculator", "arguments": {"expression": "1+2", "extra": True}})


def test_current_time_is_read_from_clock_with_correct_zone():
    result = execute_tool({"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}})
    actual = datetime.fromisoformat(result["time"])
    assert actual.utcoffset().total_seconds() == 8 * 3600
    assert abs((datetime.now(actual.tzinfo) - actual).total_seconds()) < 3
