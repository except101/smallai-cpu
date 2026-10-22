"""Bounded downloads of attributed public corpora; never downloads model weights."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import secrets
import sys
import threading
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .data import is_validation, prepare_dataset
from .format import encode_conversation, tool_call_message, validate_sft_record
from .tokenizer import LocalTokenizer, sha256_file, train_tokenizer
from .tools import TOOL_SCHEMAS, compact_json


SOURCES = {
    "web": {"dataset": "openbmb/Ultra-FineWeb", "config": "default", "split": "zh",
            "license": "apache-2.0", "language": "zh", "kind": "filtered web text"},
    "chat": {"dataset": "OpenAssistant/oasst2", "config": "default", "split": "ready_export",
             "license": "apache-2.0", "language": "zh", "kind": "human conversations",
             "file": "2023-11-05_oasst2_ready.messages.jsonl.gz"},
    "tools": {"dataset": "twinkle-ai/tw-function-call-reasoning-10k", "config": "default", "split": "train",
              "license": "cc-by-4.0", "language": "zh-Hant", "kind": "translated synthetic function calling",
              "file": "data/train-00000-of-00001.parquet"},
}
HAN = re.compile(r"[\u3400-\u9fff]")


class PreparationCancelled(ValueError):
    pass


def check_cancel(event):
    if event and event.is_set():
        raise PreparationCancelled("数据准备已取消，已下载的缓存可以复用")


def json_file(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class PublicClient:
    def __init__(self, cache, cancel=None, notice=None, vendor=None):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.cancel = cancel
        self.notice = notice
        self.vendor = vendor
        self.downloaded_bytes = 0
        self._lock = threading.Lock()

    def get_json(self, url, cache_key=None):
        target = self.cache / (hashlib.sha256(cache_key.encode()).hexdigest() + ".json") if cache_key else None
        if target and target.is_file():
            return json.loads(target.read_text(encoding="utf-8"))
        for attempt in range(6):
            delay = 1 + attempt * 2
            check_cancel(self.cancel)
            try:
                with urlopen(Request(url, headers={"User-Agent": "SmallAI-PublicData/0.1"}), timeout=30) as response:
                    payload = response.read(24 * 1024 * 1024 + 1)
                if len(payload) > 24 * 1024 * 1024:
                    raise ValueError("单次数据响应超过 24 MiB，请减小下载分页")
                value = json.loads(payload)
                with self._lock:
                    self.downloaded_bytes += len(payload)
                if target:
                    json_file(target, value)
                return value
            except HTTPError as error:
                if error.code in (401, 403):
                    raise ValueError("数据源需要授权，当前仅接入匿名可访问的数据源") from error
                if attempt == 5 or error.code not in (429, 500, 502, 503, 504):
                    raise ValueError(f"数据源请求失败：HTTP {error.code}") from error
                if error.code == 429:
                    try:
                        delay = min(180, max(15, int(error.headers.get("Retry-After", "0"))))
                    except ValueError:
                        delay = 0
                    delay = max(delay, min(120, 30 * (attempt + 1)))
                    if self.notice:
                        self.notice(phase="rate_limit", message=f"公开接口限流，等待 {delay} 秒后自动重试", current=None, total=None)
            except (URLError, TimeoutError, OSError) as error:
                if attempt == 5:
                    raise ValueError(f"下载失败，请稍后重试：{error}") from error
            if self.cancel:
                self.cancel.wait(delay)
            else:
                time.sleep(delay)
        raise ValueError("下载失败")

    def snapshot(self, source):
        dataset = source["dataset"]
        meta = self.get_json("https://huggingface.co/api/datasets/" + quote(dataset, safe="/"))
        if meta.get("gated") or meta.get("private"):
            raise ValueError(f"数据源不支持匿名下载：{dataset}")
        license_name = (meta.get("cardData") or {}).get("license")
        return {**source, "revision": meta["sha"],
                "url": "https://huggingface.co/datasets/" + dataset,
                "viewer_revision_pinned": "file" in source,
                "reported_license": license_name or source["license"]}

    def rows(self, source, max_scan):
        if str(source.get("file", "")).endswith(".parquet"):
            yield from self.parquet_rows(source, max_scan)
            return
        def page(offset):
            query = {"dataset": source["dataset"], "config": source["config"], "split": source["split"],
                     "offset": offset, "length": min(100, max_scan - offset)}
            url = "https://datasets-server.huggingface.co/rows?" + urlencode(query)
            key = source["revision"] + ":" + url
            return self.get_json(url, key)

        with ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0, max_scan, 200):
                check_cancel(self.cancel)
                offsets = list(range(start, min(start + 200, max_scan), 100))
                for offset, result in zip(offsets, pool.map(page, offsets)):
                    for item in result.get("rows", []):
                        check_cancel(self.cancel)
                        if item.get("truncated_cells"):
                            continue
                        yield item["row"], item.get("row_idx", offset)
                    if not result.get("rows") or offset + 100 >= result.get("num_rows_total", max_scan):
                        return

    def file(self, source, progress):
        target = self.cache / (source["revision"] + "_" + Path(source["file"]).name)
        if target.is_file():
            progress(target.stat().st_size, target.stat().st_size)
            return target
        url = "https://huggingface.co/datasets/" + source["dataset"] + "/resolve/" + source["revision"] + "/" + source["file"]
        temporary = target.with_suffix(target.suffix + ".partial")
        try:
            with urlopen(Request(url, headers={"User-Agent": "SmallAI-PublicData/0.1"}), timeout=30) as response, temporary.open("wb") as output:
                total = int(response.headers.get("Content-Length", 0))
                count = 0
                while block := response.read(512 * 1024):
                    check_cancel(self.cancel)
                    count += len(block)
                    if count > 120 * 1024 * 1024:
                        raise ValueError("会话数据下载超过 120 MiB 上限")
                    output.write(block)
                    progress(count, total)
            temporary.replace(target)
            self.downloaded_bytes += count
            return target
        except (URLError, TimeoutError, OSError) as error:
            raise ValueError(f"会话数据下载失败，请重试：{error}") from error
        finally:
            temporary.unlink(missing_ok=True)

    def parquet_rows(self, source, max_scan):
        try:
            import duckdb
        except ImportError:
            if self.vendor and Path(self.vendor).is_dir():
                sys.path.insert(0, str(self.vendor))
            try:
                import duckdb
            except ImportError as error:
                raise ValueError("缺少 Parquet 数据读取器，请安装项目的 data 可选依赖") from error
        def progress(count, total):
            if self.notice:
                self.notice(phase="tools_download", message="下载中文函数调用源文件", current=count,
                            total=total or None, unit="bytes")
        path = self.file(source, progress)
        temporary = self.cache / "reader_tmp"
        temporary.mkdir(exist_ok=True)
        connection = duckdb.connect(config={"threads": 1, "memory_limit": "256MB", "temp_directory": str(temporary)})
        try:
            cursor = connection.execute("SELECT id, query_zhtw, tools, answer FROM read_parquet(?)", [str(path)])
            columns = [item[0] for item in cursor.description]
            index = 0
            while rows := cursor.fetchmany(128):
                for values in rows:
                    check_cancel(self.cancel)
                    if index >= max_scan:
                        return
                    yield dict(zip(columns, values)), index
                    index += 1
        finally:
            connection.close()


def source_id(source, row_id):
    return {"dataset": source["dataset"], "revision_observed": source["revision"],
            "split": source["split"], "row_id": row_id}


def web_record(row, row_id, source, max_chars=4000):
    text = row.get("content", row.get("text"))
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize("NFC", text).replace("\x00", "").strip()
    if len(text) < 200 or len(HAN.findall(text)) < 100:
        return None
    if len(HAN.findall(text)) / max(len(text), 1) < 0.2:
        return None
    shortened = len(text) > max_chars
    if shortened:
        excerpt = text[:max_chars]
        endings = [excerpt.rfind(mark) for mark in ("。", "！", "？", "\n")]
        cut = max(endings)
        text = excerpt[:cut + 1] if cut >= max_chars // 2 else excerpt
    return {"text": text, "source": source_id(source, row_id), "excerpted": shortened,
            "upstream_source": row.get("source")}


def chat_records(messages, source, limit):
    by_id = {row["message_id"]: row for row in messages if row.get("message_id") and row.get("lang") == "zh"}
    candidates = {}
    for row in by_id.values():
        if row.get("role") != "assistant" or row.get("deleted") or row.get("review_result") is False:
            continue
        parent = by_id.get(row.get("parent_id"))
        if not parent or parent.get("role") != "prompter" or parent.get("parent_id") is not None or parent.get("deleted"):
            continue
        question, answer = parent.get("text"), row.get("text")
        if not isinstance(question, str) or not isinstance(answer, str) or len(answer.strip()) < 8:
            continue
        if len(question) + len(answer) > 2800:
            continue
        rank = row.get("rank") if row.get("rank") is not None else 999
        existing = candidates.get(parent["message_id"])
        if existing is None or rank < existing[0]:
            candidates[parent["message_id"]] = (rank, row, parent)
    selected = list(candidates.values())
    random.Random(42).shuffle(selected)
    results = []
    for _, row, parent in selected[:limit]:
        record = {"messages": [{"role": "user", "content": parent["text"].strip()},
                                {"role": "assistant", "content": row["text"].strip()}],
                  "source": source_id(source, row["message_id"])}
        # Availability of unrelated tools is an explicit augmentation, not a fabricated answer.
        text = parent["text"]
        if not re.search(r"计算|算出|等于|现在|几点|当前时间|今天日期", text):
            record["tools"] = TOOL_SCHEMAS
        validate_sft_record(record)
        results.append(record)
    return results


def normalize_schema(schema, depth=0):
    if not isinstance(schema, dict) or depth > 8:
        raise ValueError("工具参数结构无效")
    aliases = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object", "list": "array"}
    result = {}
    for key in ("type", "description", "enum", "default", "required", "additionalProperties"):
        if key in schema:
            result[key] = schema[key]
    if "type" in result:
        result["type"] = aliases.get(result["type"], result["type"]) if isinstance(result["type"], str) else result["type"]
        if result["type"] not in ("string", "integer", "number", "boolean", "object", "array", "null"):
            raise ValueError("不支持的工具参数类型")
    if isinstance(result.get("description"), str):
        result["description"] = result["description"][:180]
    if "properties" in schema:
        result["properties"] = {key: normalize_schema(value, depth + 1) for key, value in schema["properties"].items()}
    if "items" in schema:
        result["items"] = normalize_schema(schema["items"], depth + 1)
    return result


def function_record(row, row_id, source):
    query = row.get("query_zhtw")
    if not isinstance(query, str) or len(HAN.findall(query)) < 2:
        raise ValueError("non_chinese_query")
    calls = json.loads(row["answer"]) if isinstance(row.get("answer"), str) else row.get("answer")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("parallel_or_empty_calls")
    call = calls[0]
    if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
        raise ValueError("invalid_call")
    tools = json.loads(row["tools"]) if isinstance(row.get("tools"), str) else row.get("tools")
    if not isinstance(tools, list):
        raise ValueError("invalid_tools")
    normalized = []
    for tool in tools:
        function = tool.get("function", tool)
        raw = function.get("parameters", {})
        if raw.get("type") == "object" and "properties" in raw:
            parameters = normalize_schema(raw)
        else:
            properties = {key: normalize_schema(spec) for key, spec in raw.items()}
            parameters = {"type": "object", "properties": properties,
                          "required": [key for key, spec in raw.items() if "default" not in spec and not spec.get("optional")]}
        normalized.append({"type": "function", "function": {"name": function["name"],
                           "description": str(function.get("description", ""))[:300], "parameters": parameters}})
    target = next((index for index, tool in enumerate(normalized) if tool["function"]["name"] == call.get("name")), None)
    if target is None:
        raise ValueError("undeclared_tool")
    if len(normalized) > 3:
        candidates = [i for i in range(len(normalized)) if i != target]
        rng = random.Random(hashlib.sha256(query.encode()).hexdigest())
        keep = sorted([target] + rng.sample(candidates, 2))
        normalized = [normalized[index] for index in keep]
    record = {"tools": normalized, "messages": [{"role": "user", "content": query.strip()}, tool_call_message(call)],
              "source": source_id(source, row.get("id", row_id))}
    validate_sft_record(record)
    return record


def write_records(path, records):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_public_data(workspace, web_docs=5000, chat_rows=1500, tool_rows=1500,
                        progress=None, cancel=None, package_name=None):
    workspace = Path(workspace).resolve()
    for value in (web_docs, chat_rows, tool_rows):
        if type(value) is not int or not 20 <= value <= 20000:
            raise ValueError("每类数据目标必须在 20 到 20000 条之间")
    progress = progress or (lambda **value: print(json.dumps(value, ensure_ascii=False), flush=True))
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    name = package_name or f"public_zh_{stamp}_{secrets.token_hex(3)}"
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        raise ValueError("数据包名称无效")
    package = workspace / "data/public" / name
    if package.exists():
        raise ValueError("数据包已存在，请更换名称")
    package.mkdir(parents=True)
    client = PublicClient(workspace / "data/cache/public", cancel, notice=progress, vendor=workspace / "vendor")
    receipt = {"package": name, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
               "status": "preparing", "targets": {"web": web_docs, "chat": chat_rows, "tools": tool_rows},
               "sources": {}, "transformations": ["Chinese text selection", "web excerpts capped at 4000 characters",
                 "one best root answer per OpenAssistant prompt", "single tool call only; parallel calls skipped",
                 "reasoning fields omitted", "at most 3 available tools; declared target always preserved",
                 "tool parameter type normalization; descriptions shortened", "document-level deterministic train/validation split"]}
    try:
        for kind, source in SOURCES.items():
            progress(phase="sources", message="核对公开数据源", current=len(receipt["sources"]), total=3)
            receipt["sources"][kind] = client.snapshot(source)
        json_file(package / "sources.json", receipt)
        progress(phase="web", message="下载并筛选中文网页", current=0, total=web_docs)
        web, seen = [], set()
        for row, index in client.rows(receipt["sources"]["web"], min(60000, web_docs * 4)):
            record = web_record(row, index, receipt["sources"]["web"])
            if record:
                identity = hashlib.sha256(record["text"].encode()).digest()
                if identity not in seen:
                    seen.add(identity)
                    web.append(record)
                    if len(web) % 100 == 0:
                        progress(phase="web", message="下载并筛选中文网页", current=len(web), total=web_docs)
            if len(web) >= web_docs:
                break
        if len(web) < 20:
            raise ValueError("可用中文网页不足，请稍后重试")
        write_records(package / "corpus.jsonl", web)

        def download_progress(count, total):
            progress(phase="chat_download", message="下载 OpenAssistant 原始会话", current=count,
                     total=total or None, unit="bytes")
        archive = client.file(receipt["sources"]["chat"], download_progress)
        messages = []
        with gzip.open(archive, "rt", encoding="utf-8") as stream:
            for line in stream:
                check_cancel(cancel)
                row = json.loads(line)
                if row.get("lang") == "zh":
                    messages.append(row)
        chats = chat_records(messages, receipt["sources"]["chat"], chat_rows)
        if len(chats) < 20:
            raise ValueError("可用中文对话不足")
        write_records(package / "chat.jsonl", chats)
        progress(phase="chat", message="中文对话筛选完成", current=len(chats), total=len(chats))

        functions, rejected = [], Counter()
        progress(phase="tools", message="下载并转换中文函数调用", current=0, total=tool_rows)
        for row, index in client.rows(receipt["sources"]["tools"], 10000):
            try:
                record = function_record(row, index, receipt["sources"]["tools"])
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                rejected[str(error)[:120]] += 1
                continue
            functions.append(record)
            if len(functions) % 100 == 0:
                progress(phase="tools", message="下载并转换中文函数调用", current=len(functions), total=tool_rows)
            if len(functions) >= tool_rows:
                break
        if len(functions) < 20:
            raise ValueError("可用单次工具调用不足")
        write_records(package / "tools.jsonl", functions)

        # Tokenizer sees only training-side examples, never held-out SFT answers.
        training_texts = list(web)
        for record in chats + functions:
            identity = compact_json({"messages": record["messages"], "tools": record.get("tools", [])})
            if is_validation(identity):
                continue
            text = compact_json(record.get("tools", [])) + "\n" + "\n".join(
                message.get("content") or compact_json(message.get("tool_calls", [])) for message in record["messages"])
            training_texts.append({"text": text})
        token_text_path = package / "tokenizer_training.jsonl"
        write_records(token_text_path, training_texts)
        tokenizer_path = package / "tokenizer.json"
        progress(phase="tokenizer", message="从公开语料训练新分词器", current=None, total=None)
        train_tokenizer([token_text_path], tokenizer_path, vocab_size=16384)
        check_cancel(cancel)

        # Apply the model's actual context limit before balancing the tasks.
        tokenizer = LocalTokenizer(tokenizer_path)
        fit_chats = [r for r in chats if len(encode_conversation(tokenizer, r["messages"], r.get("tools"))[0]) <= 1024]
        fit_functions = [r for r in functions if len(encode_conversation(tokenizer, r["messages"], r.get("tools"))[0]) <= 1024]
        per_kind = min(len(fit_chats), len(fit_functions))
        if per_kind < 20:
            raise ValueError("适合当前上下文的对话或工具样本不足，请增加下载量")
        mixed = fit_chats[:per_kind] + fit_functions[:per_kind]
        random.Random(42).shuffle(mixed)
        write_records(package / "sft.jsonl", mixed)
        relative_tokenizer = tokenizer_path.relative_to(workspace).as_posix()
        common = {"source_type": "public", "package": name, "tokenizer_path": relative_tokenizer,
                  "provenance_file": (package / "sources.json").relative_to(workspace).as_posix(),
                  "sources": [{"dataset": source["dataset"], "license": source["license"], "url": source["url"]}
                              for source in receipt["sources"].values()]}
        progress(phase="encode", message="编码预训练数据", current=0, total=2)
        prepared_root = workspace / "data/prepared"
        pretrain = prepare_dataset("pretrain", [package / "corpus.jsonl"], tokenizer_path,
                                   prepared_root / (name + "_pretrain"),
                                   metadata={**common, "display_name": "公开中文网页 · 起步包",
                                             "sources": [common["sources"][0]]})
        check_cancel(cancel)
        progress(phase="encode", message="编码对话与工具调用数据", current=1, total=2)
        sft = prepare_dataset("sft", [package / "sft.jsonl"], tokenizer_path,
                             prepared_root / (name + "_sft"), block_size=1023,
                             metadata={**common, "display_name": "公开中文对话与工具 · 起步包",
                                       "sources": common["sources"][1:]})
        receipt.update(status="completed", accepted={"web": len(web), "chat": len(chats), "tools": len(functions),
                       "mixed_before_length_filter": len(mixed), "chat_within_context": len(fit_chats),
                       "tools_within_context": len(fit_functions)}, rejected_tools=dict(rejected),
                       tokenizer_sha256=sha256_file(tokenizer_path),
                       vocab_size=LocalTokenizer(tokenizer_path).vocab_size,
                       prepared={"pretrain": pretrain["splits"], "sft": sft["splits"],
                                 "sft_too_long_skipped": sft["too_long_skipped"]},
                       downloaded_bytes_this_run=client.downloaded_bytes,
                       output_sha256={filename: sha256_file(package / filename)
                                      for filename in ("corpus.jsonl", "chat.jsonl", "tools.jsonl", "sft.jsonl")},
                       outputs={"pretrain": (prepared_root / (name + "_pretrain")).relative_to(workspace).as_posix(),
                                "sft": (prepared_root / (name + "_sft")).relative_to(workspace).as_posix(),
                                "tokenizer": relative_tokenizer})
        json_file(package / "sources.json", receipt)
        progress(phase="completed", message="公开数据已准备好", current=2, total=2, result=receipt)
        return receipt
    except Exception as error:
        receipt.update(status="cancelled" if isinstance(error, PreparationCancelled) else "failed", error=str(error))
        json_file(package / "sources.json", receipt)
        raise
