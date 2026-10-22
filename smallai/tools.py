"""Explicitly registered local tools. Model text is never executed as Python code."""

import ast
from datetime import datetime, timedelta, timezone
import json
import math
import operator


TOOL_START = "<|tool_call|>"
TOOL_END = "<|tool_end|>"
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator", "description": "计算四则运算和小整数幂。",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "算式，例如 17 * 23"}},
            "required": ["expression"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_current_time", "description": "查询当前时间。",
        "parameters": {"type": "object", "properties": {
            "timezone": {"type": "string", "enum": ["Asia/Shanghai", "UTC"]}},
            "required": ["timezone"], "additionalProperties": False}}},
]


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse_tool_call(text):
    text = text.strip()
    if TOOL_START not in text and TOOL_END not in text:
        return None
    if not text.startswith(TOOL_START) or not text.endswith(TOOL_END):
        raise ValueError("工具调用标记不完整，或混入了普通回答")
    payload = text[len(TOOL_START):-len(TOOL_END)]
    if len(payload) > 4096:
        raise ValueError("工具调用过长")
    try:
        call = json.loads(payload)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("工具参数不是合法 JSON") from error
    if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
            or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict)):
        raise ValueError("工具调用必须含 name 字符串和 arguments 对象")
    return call


def calculate(expression):
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 128:
        raise ValueError("算式必须是 1 到 128 字符的字符串")
    try:
        root = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise ValueError("算式语法错误") from error
    if sum(1 for _ in ast.walk(root)) > 64:
        raise ValueError("算式过于复杂")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}

    def evaluate(node, depth=0):
        if depth > 16:
            raise ValueError("算式嵌套过深")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = evaluate(node.operand, depth + 1)
            result = -result if isinstance(node.op, ast.USub) else result
        elif isinstance(node, ast.BinOp):
            left, right = evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                if type(right) is not int or abs(right) > 8 or abs(left) > 1e6:
                    raise ValueError("幂运算超出范围")
                result = left ** right
            elif type(node.op) in binary:
                result = binary[type(node.op)](left, right)
            else:
                raise ValueError("不支持这种运算")
        else:
            raise ValueError("仅支持数字、括号和算术运算符")
        if type(result) not in (int, float) or not math.isfinite(result) or abs(result) > 1e12:
            raise ValueError("运算结果超出范围")
        return result

    try:
        return evaluate(root.body)
    except (ZeroDivisionError, OverflowError) as error:
        raise ValueError("除零或数值溢出") from error


def execute_tool(call):
    name, arguments = call["name"], call["arguments"]
    if name == "calculator":
        if set(arguments) != {"expression"}:
            raise ValueError("calculator 需要且只接受 expression 参数")
        return {"result": calculate(arguments["expression"])}
    if name == "get_current_time":
        if set(arguments) != {"timezone"} or arguments["timezone"] not in ("Asia/Shanghai", "UTC"):
            raise ValueError("时区必须是 Asia/Shanghai 或 UTC")
        zone = timezone(timedelta(hours=8)) if arguments["timezone"] == "Asia/Shanghai" else timezone.utc
        return {"timezone": arguments["timezone"], "time": datetime.now(zone).isoformat(timespec="seconds")}
    raise ValueError(f"未注册的工具：{name}")
