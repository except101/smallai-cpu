import gzip
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import threading

from smallai.data import prepare_dataset
from smallai.format import validate_sft_record
import smallai.public_data as public
from smallai.tokenizer import LocalTokenizer
from smallai.training import load_checkpoint
from smallai.ui import TrainingManager
from smallai.ui import BusyError
from test_ui import wait_until


SOURCE = {"dataset": "test/public", "revision": "test-revision", "split": "train"}


def make_tool_row(index=0, parallel=False):
    names = [f"unused_{i}" for i in range(5)] + ["lookup"]
    tools = [{"name": name, "description": "查询条目", "parameters": {
        "code": {"type": "str", "description": "条目编号"},
        "limit": {"type": "int", "default": 5}}} for name in names]
    calls = [{"name": "lookup", "arguments": {"code": str(index), "limit": 2}}]
    return {"id": index, "query_zhtw": f"請查詢編號{index}的條目", "tools": json.dumps(tools),
            "answer": json.dumps(calls * (2 if parallel else 1)), "think": "Do not include this in training"}


def test_function_conversion_keeps_target_and_normalizes_schema():
    result = public.function_record(make_tool_row(), 0, SOURCE)
    validate_sft_record(result)
    assert len(result["tools"]) == 3
    target = next(tool["function"] for tool in result["tools"] if tool["function"]["name"] == "lookup")
    assert target["parameters"]["properties"]["code"]["type"] == "string"
    assert target["parameters"]["properties"]["limit"]["type"] == "integer"
    assert target["parameters"]["required"] == ["code"]
    assert "think" not in result
    call = result["messages"][-1]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"code": "0", "limit": 2}
    with pytest.raises(ValueError, match="parallel"):
        public.function_record(make_tool_row(parallel=True), 0, SOURCE)


def test_web_filter_and_excerpts_keep_provenance():
    text = ("这是一段用来检查中文网页筛选和句子边界的公开测试文字。" * 300)
    result = public.web_record({"content": text, "source": "test"}, 123, SOURCE, max_chars=400)
    assert result["excerpted"] and len(result["text"]) <= 400
    assert result["text"].endswith("。")
    assert result["source"]["row_id"] == 123
    assert public.web_record({"content": "English " * 200}, 0, SOURCE) is None


def test_chat_conversion_uses_one_best_root_answer():
    rows = [{"message_id": "q", "parent_id": None, "role": "prompter", "lang": "zh", "text": "如何安排学习计划？"},
            {"message_id": "a1", "parent_id": "q", "role": "assistant", "lang": "zh", "rank": 1, "text": "可以每天安排学习时间。"},
            {"message_id": "a0", "parent_id": "q", "role": "assistant", "lang": "zh", "rank": 0, "text": "先确定学习目标，再分配时间并定期复习。"},
            {"message_id": "q2", "parent_id": "a0", "role": "prompter", "lang": "zh", "text": "然后呢？"},
            {"message_id": "a2", "parent_id": "q2", "role": "assistant", "lang": "zh", "text": "这个回答缺少前面的上下文。"}]
    records = public.chat_records(rows, SOURCE, 100)
    assert len(records) == 1
    assert records[0]["source"]["row_id"] == "a0"
    assert records[0]["messages"][1]["content"] == rows[2]["text"]


def test_paged_download_uses_bounded_pages_and_skips_truncated_rows(tmp_path):
    class FakeClient(public.PublicClient):
        def get_json(self, url, cache_key=None):
            query = parse_qs(urlsplit(url).query)
            start, count = int(query["offset"][0]), int(query["length"][0])
            assert count <= 100
            return {"rows": [{"row_idx": i, "row": {"index": i}, "truncated_cells": ["text"] if i == 5 else []}
                             for i in range(start, start + count)], "num_rows_total": 230}
    source = {**SOURCE, "config": "default"}
    records = list(FakeClient(tmp_path).rows(source, 230))
    assert len(records) == 229
    assert [index for _, index in records][-1] == 229
    assert 5 not in [index for _, index in records]


def test_parquet_reader_uses_local_source_file(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    parquet = tmp_path / "tools.parquet"
    connection = duckdb.connect()
    connection.sql("SELECT 1 AS id, '中文查询' AS query_zhtw, '[]' AS tools, '[]' AS answer").write_parquet(str(parquet))
    connection.close()
    class LocalClient(public.PublicClient):
        def file(self, source, progress): return parquet
        def get_json(self, *args, **kwargs): raise AssertionError("Do not call the rows API for local Parquet")
    result = list(LocalClient(tmp_path / "cache").rows({**SOURCE, "file": "tools.parquet"}, 1))
    assert result[0][0]["query_zhtw"] == "中文查询"
    assert result[0][0]["id"] == 1


def test_data_job_cancellation_and_phase_units(ui_workspace, monkeypatch):
    import smallai.ui as ui
    def fake_prepare(workspace, progress, cancel, **kwargs):
        progress(phase="chat_download", message="download", current=100, total=200, unit="bytes")
        progress(phase="tools", message="convert", current=3, total=4)
        cancel.wait(10)
        raise public.PreparationCancelled("cancelled")
    monkeypatch.setattr(ui, "prepare_public_data", fake_prepare)
    manager = TrainingManager(ui_workspace)
    try:
        manager.prepare_public({})
        wait_until(lambda: manager.state()["data_job"].get("phase") == "tools")
        assert manager.state()["data_job"]["unit"] is None
        with pytest.raises(BusyError): manager.start({"model":"smoke", "steps":1})
        manager.cancel_public()
        wait_until(lambda: manager.state()["data_job"]["status"] == "cancelled")
    finally:
        manager.close()


def test_public_package_prepares_real_formats_without_touching_existing_tokenizer(tmp_path, monkeypatch):
    messages = []
    for i in range(60):
        messages.extend([
            {"message_id": f"q{i}", "parent_id": None, "role": "prompter", "lang": "zh", "text": f"学习主题{i}如何制定计划？"},
            {"message_id": f"a{i}", "parent_id": f"q{i}", "role": "assistant", "lang": "zh", "rank": 0,
             "text": f"主题{i}应该先确定目标，再进行练习并检查结果。"}])
    archive = tmp_path / "messages.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as stream:
        for row in messages:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    original = tmp_path / "data/tokenizer.json"
    original.parent.mkdir()
    original.write_text("preserve existing tokenizer", encoding="utf-8")

    class FakeClient:
        downloaded_bytes = 0
        def __init__(self, *args, **kwargs): pass
        def snapshot(self, source): return {**source, "revision": "test-revision", "url": "https://huggingface.co/datasets/" + source["dataset"]}
        def file(self, source, progress): return archive
        def rows(self, source, max_scan):
            for i in range(60):
                if source["dataset"] == public.SOURCES["web"]["dataset"]:
                    yield {"content": f"文档编号{i}。" + "这份中文文本用于测试公开数据准备流程。" * 20, "source": "fixture"}, i
                else:
                    row = make_tool_row(i)
                    row["tools"] = json.dumps(json.loads(row["tools"])[-1:])
                    yield row, i
    monkeypatch.setattr(public, "PublicClient", FakeClient)
    result = public.prepare_public_data(tmp_path, web_docs=60, chat_rows=60, tool_rows=60,
                                        package_name="test_public", progress=lambda **kw: None)
    assert result["status"] == "completed"
    assert original.read_text(encoding="utf-8") == "preserve existing tokenizer"
    assert result["prepared"]["sft_too_long_skipped"] == 0
    for stage in ("pretrain", "sft"):
        manifest = json.loads((tmp_path / result["outputs"][stage] / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_type"] == "public"
        assert (tmp_path / manifest["tokenizer_path"]).is_file()
        assert manifest["splits"]["train"]["documents"] > 0
        assert manifest["splits"]["val"]["documents"] > 0


def test_gui_uses_dataset_tokenizer_and_rejects_incompatible_sft(ui_workspace, local_data):
    root, tokenizer = local_data
    alternate = ui_workspace / "data/public/test/tokenizer.json"
    alternate.parent.mkdir(parents=True)
    alternate.write_bytes(tokenizer.path.read_bytes() + b"\n")
    new_tokenizer = LocalTokenizer(alternate)
    for stage in ("pretrain", "sft"):
        input_path = root / "demo" / ("corpus.jsonl" if stage == "pretrain" else "sft.jsonl")
        prepare_dataset(stage, [input_path], alternate, ui_workspace / "data/prepared" / ("public_" + stage),
                        metadata={"source_type": "public", "tokenizer_path": "data/public/test/tokenizer.json"})
    manager = TrainingManager(ui_workspace)
    try:
        job = manager.start({"model": "smoke", "steps": 1, "block_size": 64, "threads": 2,
                             "data": "data/prepared/public_pretrain"})
        wait_until(lambda: manager.state()["active_id"] is None)
        assert manager.state()["run"]["status"] == "completed"
        checkpoint = ui_workspace / job["output"] / "last.pt"
        assert load_checkpoint(checkpoint)["tokenizer_sha256"] == new_tokenizer.fingerprint
        assert manager.bootstrap()["datasets"][0]["source_type"] == "public"
        with pytest.raises(ValueError, match="不同分词器"):
            manager.start({"stage": "sft", "init_from": str(checkpoint.relative_to(ui_workspace)), "steps": 1})
    finally:
        manager.close()
