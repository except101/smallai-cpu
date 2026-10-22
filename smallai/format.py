import json

from .tools import TOOL_END, TOOL_START, compact_json


DEFAULT_SYSTEM = "你是一个中文助手。回答清楚、简洁；缺少必要信息时先追问。"


def normalize_call(call):
    if not isinstance(call, dict):
        raise ValueError("tool_calls 中的元素必须是对象")
    function = call.get("function", call)
    if not isinstance(function, dict):
        raise ValueError("function 必须是对象")
    name, arguments = function.get("name"), function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError("function.arguments 不是合法 JSON") from error
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        raise ValueError("工具调用需要 name 和 arguments")
    return {"name": name, "arguments": arguments}


def tool_call_message(call, call_id="call_0"):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": call["name"], "arguments": compact_json(call["arguments"])}}]}


def encode_conversation(tokenizer, messages, tools=None, add_generation_prompt=False):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空数组")
    messages = [dict(message) for message in messages]
    if tools:
        if not isinstance(tools, list):
            raise ValueError("tools 必须是数组")
        instructions = ("可用工具：" + compact_json(tools) + "\n需要工具时，仅输出 "
                        + TOOL_START + '{"name":"工具名","arguments":{参数}}' + TOOL_END
                        + "。不需要工具时正常回答。工具结果由 tool 消息提供。")
        if messages[0].get("role") == "system":
            messages[0]["content"] = messages[0].get("content", "") + "\n" + instructions
        else:
            messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM + "\n" + instructions})
    ids, labels = [tokenizer.id("<|bos|>")], [-100]
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"不支持的消息角色：{role}")
        if role == "system" and index != 0:
            raise ValueError("system 消息只能出现在开头")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("消息 content 必须是字符串")
        header = [tokenizer.id(f"<|{role}|>")] + tokenizer.encode("\n")
        ids.extend(header)
        labels.extend([-100] * len(header))
        calls = message.get("tool_calls")
        if calls:
            if role != "assistant" or not isinstance(calls, list) or len(calls) != 1 or content:
                raise ValueError("第一版每个 assistant 消息支持一次工具调用，content 须为空")
            call = normalize_call(calls[0])
            body = [tokenizer.id(TOOL_START)] + tokenizer.encode(compact_json(call)) + [tokenizer.id(TOOL_END)]
        else:
            if role == "tool":
                content = compact_json({"name": message.get("name", ""), "content": content})
            body = tokenizer.encode(content)
        body.append(tokenizer.id("<|eos|>"))
        ids.extend(body)
        labels.extend(body if role == "assistant" else [-100] * len(body))
    if add_generation_prompt:
        header = [tokenizer.id("<|assistant|>")] + tokenizer.encode("\n")
        ids.extend(header)
        labels.extend([-100] * len(header))
    return ids, labels


def validate_sft_record(record):
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT 数据需要非空 messages 数组")
    pending, saw_user, saw_assistant = None, False, False
    tools = record.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools 必须是数组")
    names = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("tools 的每一项必须是对象")
        function = tool.get("function", tool)
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise ValueError("工具定义需要非空的 function.name")
        if function["name"] in names:
            raise ValueError("tools 中存在同名工具")
        names.add(function["name"])
    previous = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("messages 的每一项必须是对象")
        role = message.get("role")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("content 必须是字符串或 null")
        if role == "system":
            if index != 0 or not content:
                raise ValueError("system 消息必须非空且在开头")
        elif role == "user":
            if pending is not None or previous not in (None, "system", "assistant") or not content:
                raise ValueError("user 消息顺序错误或内容为空")
            saw_user = True
        elif role == "assistant":
            if not saw_user or pending is not None or previous not in ("user", "tool"):
                raise ValueError("assistant 消息顺序错误")
            saw_assistant = True
            if message.get("tool_calls"):
                calls = message["tool_calls"]
                if not isinstance(calls, list) or len(calls) != 1 or content:
                    raise ValueError("每轮只支持一个工具调用，且 content 须为空")
                call = normalize_call(calls[0])
                if call["name"] not in names:
                    raise ValueError(f"调用了 tools 未声明的工具：{call['name']}")
                pending = (call["name"], calls[0].get("id"))
            elif not content:
                raise ValueError("assistant 回答不能为空")
        elif role == "tool":
            if pending is None or message.get("name") != pending[0]:
                raise ValueError("tool 结果与前面的调用不匹配")
            if pending[1] and message.get("tool_call_id") != pending[1]:
                raise ValueError("tool_call_id 不匹配")
            pending = None
        else:
            raise ValueError(f"不支持的消息角色：{role}")
        previous = role
    if not saw_assistant or previous != "assistant":
        raise ValueError("SFT 对话应以 assistant 回答或工具调用结束")
