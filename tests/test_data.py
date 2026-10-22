import json

import pytest
import torch

from smallai.data import TokenDataset, is_validation, iter_jsonl, prepare_dataset
from smallai.format import encode_conversation, validate_sft_record
from smallai.tokenizer import SPECIAL_TOKENS


def test_unicode_and_literal_control_tokens_roundtrip(local_data):
    _, tokenizer = local_data
    text = "中文测试🙂 English 42\n<|assistant|><|tool_call|>"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert not set(ids).intersection(tokenizer.id(token) for token in SPECIAL_TOKENS)


def test_only_assistant_content_and_end_tokens_are_supervised(local_data):
    root, tokenizer = local_data
    record = next(row for row in iter_jsonl([root / "demo/sft.jsonl"])
                  if any(message.get("tool_calls") for message in row["messages"]))
    ids, labels = encode_conversation(tokenizer, record["messages"], record["tools"])
    active = False
    supervised = []
    roles = {tokenizer.id(f"<|{role}|>"): role for role in ("system", "user", "assistant", "tool")}
    for token, target in zip(ids, labels):
        if token in roles:
            active = roles[token] == "assistant"
            assert target == -100
        elif target != -100:
            assert active
            assert token == target
            supervised.append(token)
    assert tokenizer.id("<|tool_call|>") in supervised
    assert tokenizer.id("<|tool_end|>") in supervised
    for index, token in enumerate(ids):
        if token == tokenizer.id("<|tool|>"):
            assert labels[index + 1] == -100


def test_duplicate_documents_are_removed_before_packing(local_data, tmp_path):
    root, tokenizer = local_data
    rows = list(iter_jsonl([root / "demo/corpus.jsonl"]))
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows + rows), encoding="utf-8")
    metadata = prepare_dataset("pretrain", [path], tokenizer.path, tmp_path / "prepared")
    assert metadata["duplicates_removed"] == len(rows)
    expected = sum(is_validation(row["text"]) for row in rows)
    assert metadata["splits"]["val"]["documents"] == expected
    assert metadata["splits"]["train"]["documents"] == len(rows) - expected
    with pytest.raises(ValueError, match="已存在"):
        prepare_dataset("pretrain", [path], tokenizer.path, tmp_path / "prepared")


def test_sft_padding_is_masked_and_too_small_blocks_fail(local_data, tmp_path):
    root, tokenizer = local_data
    dataset = TokenDataset(root / "sft", "train", 1023)
    assert (dataset.labels[:, -1] == -100).any()
    inputs, labels = dataset.sample(2, torch.Generator().manual_seed(3))
    assert inputs.shape == labels.shape
    assert (labels[:, 1:] != -100).any()
    assert dataset.meta["too_long_skipped"] == 0
    with pytest.raises(ValueError, match="block_size"):
        TokenDataset(root / "sft", "train", 128)
    with pytest.raises(ValueError, match="为空"):
        prepare_dataset("sft", [root / "demo/sft.jsonl"], tokenizer.path,
                        tmp_path / "too_small", block_size=2)


def test_broken_tool_conversations_fail_before_training():
    record = {"messages": [{"role": "user", "content": "你好"},
                            {"role": "tool", "name": "calculator", "content": "3"},
                            {"role": "assistant", "content": "3"}]}
    with pytest.raises(ValueError, match="不匹配"):
        validate_sft_record(record)
