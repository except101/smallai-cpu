import json
from pathlib import Path
import time

from .data import iter_jsonl
from .format import DEFAULT_SYSTEM, encode_conversation, tool_call_message
from .tools import TOOL_SCHEMAS, compact_json, execute_tool, parse_tool_call
from .training import load_model


def fit_prompt(model, tokenizer, messages, tools, reserve=1):
    messages = list(messages)
    while True:
        ids, _ = encode_conversation(tokenizer, messages, tools, add_generation_prompt=True)
        if len(ids) + reserve <= model.config.max_seq_len:
            return ids, messages
        users = [index for index, message in enumerate(messages) if message["role"] == "user"]
        if len(users) < 2:
            raise ValueError("当前输入和工具定义超过上下文长度，请缩短输入或使用 /reset")
        # Remove a whole old user turn, including its tool calls and tool results.
        messages = messages[:users[0]] + messages[users[1]:]


def generate_reply(model, tokenizer, messages, tools, max_new_tokens=128, temperature=0.7):
    ids, messages = fit_prompt(model, tokenizer, messages, tools)
    banned = [tokenizer.id(token) for token in ("<|pad|>", "<|unk|>", "<|bos|>",
              "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>")]
    output = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                            eos_id=tokenizer.id("<|eos|>"), banned_ids=banned)
    return tokenizer.decode(output), messages


def chat(checkpoint, tokenizer_path, prompt=None, no_tools=False, max_new_tokens=128,
         temperature=0.7, threads=4):
    model, tokenizer, state = load_model(checkpoint, tokenizer_path, threads)
    tools = None if no_tools else TOOL_SCHEMAS
    messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
    print(f"已加载 {sum(p.numel() for p in model.parameters()):,} 参数模型，训练 step={state['step']}。")
    print("这是实验权重；回复质量取决于语料和训练量。/reset 清空历史，/exit 退出。")
    while True:
        try:
            question = prompt if prompt is not None else input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question == "/exit":
            return
        if question == "/reset":
            messages = messages[:1]
            if prompt is not None:
                return
            continue
        if not question:
            if prompt is not None:
                return
            continue
        messages.append({"role": "user", "content": question})
        try:
            for round_index in range(3):
                text, messages = generate_reply(model, tokenizer, messages, tools, max_new_tokens, temperature)
                try:
                    call = parse_tool_call(text) if tools else None
                except ValueError:
                    print("助手原始输出：" + text)
                    raise
                if call is None:
                    messages.append({"role": "assistant", "content": text})
                    print("助手：" + (text or "[本轮没有生成可显示的内容]"))
                    break
                if round_index == 2:
                    print("[已达到本轮工具调用上限，请重新提问。]")
                    break
                print(f"[工具调用] {compact_json(call)}")
                try:
                    result = execute_tool(call)
                except ValueError as error:
                    result = {"error": str(error)}
                print(f"[工具结果] {compact_json(result)}")
                call_id = f"call_{round_index}"
                messages.append(tool_call_message(call, call_id))
                messages.append({"role": "tool", "name": call["name"], "tool_call_id": call_id,
                                 "content": compact_json(result)})
        except (ValueError, KeyboardInterrupt) as error:
            print(f"[本轮停止] {error}")
            messages = messages[:1]
        if prompt is not None:
            return


def evaluate(checkpoint, tokenizer_path, data_path, output, threads=4, max_new_tokens=128, limit=None):
    model, tokenizer, state = load_model(checkpoint, tokenizer_path, threads)
    results = []
    started = time.perf_counter()
    for index, row in enumerate(iter_jsonl([data_path])):
        if limit is not None and index >= limit:
            break
        if "expected_call" not in row:
            raise ValueError("评测记录缺少 expected_call，不能把缺失字段当成无需工具")
        expected, call, error = row["expected_call"], None, None
        text, _ = generate_reply(model, tokenizer, row["messages"], row.get("tools"), max_new_tokens, 0)
        try:
            call = parse_tool_call(text)
        except ValueError as caught:
            error = str(caught)
        result = {"id": row.get("id", index), "output": text, "expected_call": expected,
                  "parsed_call": call, "parse_error": error}
        if expected is None:
            result["no_call_correct"] = call is None and error is None
        else:
            result["format_correct"] = call is not None and error is None
            result["selection_correct"] = call is not None and call["name"] == expected["name"]
            result["exact_call_correct"] = call == expected
            result["execution_correct"] = False
            if call is not None:
                try:
                    actual = execute_tool(call)
                    result["tool_result"] = actual
                    if "expected_result" in row:
                        result["execution_correct"] = actual == row["expected_result"]
                    else:
                        result["execution_correct"] = (call == expected and "error" not in actual)
                except ValueError as caught:
                    result["execution_error"] = str(caught)
        results.append(result)
        print(f"评测 {index + 1}: {result['id']}", flush=True)
    if not results:
        raise ValueError("评测数据为空")
    call_rows = [row for row in results if row["expected_call"] is not None]
    no_call_rows = [row for row in results if row["expected_call"] is None]

    def ratio(rows, key):
        return sum(bool(row[key]) for row in rows) / len(rows) if rows else None

    report = {"checkpoint": str(checkpoint), "step": state["step"], "cases": len(results),
              "call_cases": len(call_rows), "no_call_cases": len(no_call_rows),
              "format_accuracy": ratio(call_rows, "format_correct"),
              "tool_selection_accuracy": ratio(call_rows, "selection_correct"),
              "exact_call_accuracy": ratio(call_rows, "exact_call_correct"),
              "execution_accuracy": ratio(call_rows, "execution_correct"),
              "no_call_accuracy": ratio(no_call_rows, "no_call_correct"),
              "elapsed_seconds": time.perf_counter() - started,
              "scope": "first assistant tool decision only; final-answer quality and multi-turn success are not scored",
              "results": results}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: value for key, value in report.items() if key != "results"}
