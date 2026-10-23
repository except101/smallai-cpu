"""Stream local documents into disk-backed datasets; split before token packing."""

import hashlib
import json
from pathlib import Path
import tempfile
import unicodedata

import numpy as np
import torch

from .format import encode_conversation, validate_sft_record
from .tokenizer import LocalTokenizer, sha256_file
from .tools import compact_json


def is_validation(text, fraction=0.1):
    if not 0 < fraction < 0.5:
        raise ValueError("验证集比例必须大于 0 且小于 0.5")
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return value / 2**64 < fraction


def iter_jsonl(paths):
    for path in map(Path, paths):
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("每行应为 JSON 对象")
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                yield row


def iter_corpus(paths):
    for path in map(Path, paths):
        if path.suffix.lower() == ".jsonl":
            for row in iter_jsonl([path]):
                text = row.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"{path} 的语料行需要 text 字符串")
                text = unicodedata.normalize("NFC", text.strip())
                if text:
                    yield text
        elif path.suffix.lower() == ".txt":
            paragraph = []
            with path.open(encoding="utf-8-sig") as stream:
                for line in stream:
                    if line.strip():
                        paragraph.append(line.rstrip())
                    elif paragraph:
                        yield unicodedata.normalize("NFC", "\n".join(paragraph).strip())
                        paragraph = []
                if paragraph:
                    yield unicodedata.normalize("NFC", "\n".join(paragraph).strip())
        else:
            raise ValueError(f"预训练语料仅支持 .jsonl 或 .txt：{path}")


def prepare_dataset(stage, paths, tokenizer_path, output, block_size=1023, val_fraction=0.1, metadata=None,
                    split_keys=None):
    if stage not in ("pretrain", "sft") or block_size < 2:
        raise ValueError("stage 或 block_size 无效")
    is_validation("check", val_fraction)
    tokenizer, output = LocalTokenizer(tokenizer_path), Path(output)
    if tokenizer.vocab_size > 65535:
        raise ValueError("本项目的 uint16 数据集最多支持 65535 个 token")
    if output.exists():
        raise ValueError(f"数据目录已存在：{output}。请指定新目录")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {split: {"documents": 0, "tokens": 0, "target_tokens": 0} for split in ("train", "val")}
    seen, duplicates, skipped = set(), 0, 0
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output.parent) as temporary:
        temp = Path(temporary)
        handles = {}
        try:
            for split in counts:
                handles[(split, "tokens")] = (temp / f"{split}.tokens.bin").open("wb")
                if stage == "sft":
                    handles[(split, "labels")] = (temp / f"{split}.labels.bin").open("wb")
            source = iter_corpus(paths) if stage == "pretrain" else iter_jsonl(paths)
            for row in source:
                if stage == "pretrain":
                    identity = row
                    ids = tokenizer.encode(row) + [tokenizer.id("<|eos|>")]
                    labels = ids
                else:
                    validate_sft_record(row)
                    identity = compact_json({"messages": row["messages"], "tools": row.get("tools", [])})
                    ids, labels = encode_conversation(tokenizer, row["messages"], row.get("tools"))
                    if len(ids) > block_size + 1:
                        skipped += 1
                        continue
                fingerprint = hashlib.sha256(identity.encode("utf-8")).digest()
                if fingerprint in seen:
                    duplicates += 1
                    continue
                seen.add(fingerprint)
                split_key = split_keys.get(identity, identity) if split_keys is not None else identity
                split = "val" if is_validation(split_key, val_fraction) else "train"
                counts[split]["documents"] += 1
                counts[split]["tokens"] += len(ids)
                counts[split]["target_tokens"] += sum(value != -100 for value in labels[1:])
                if stage == "sft":
                    padding = block_size + 1 - len(ids)
                    ids += [tokenizer.id("<|pad|>")] * padding
                    labels += [-100] * padding
                    np.asarray(labels, dtype="<i4").tofile(handles[(split, "labels")])
                np.asarray(ids, dtype="<u2").tofile(handles[(split, "tokens")])
        finally:
            for handle in handles.values():
                handle.close()
        if stage == "pretrain":
            for stats in counts.values():
                stats["target_tokens"] = max(0, stats["tokens"] - 1)
        for split, stats in counts.items():
            if not stats["documents"] or not stats["target_tokens"]:
                raise ValueError(f"{split} 集为空或没有训练目标。请增加不同样本；过长跳过 {skipped} 条")
        hashes = {path.name: sha256_file(path) for path in sorted(temp.glob("*.bin"))}
        meta = {"version": 1, "stage": stage, "tokenizer_sha256": tokenizer.fingerprint,
                "vocab_size": tokenizer.vocab_size, "block_size": block_size if stage == "sft" else None,
                "val_fraction": val_fraction, "splits": counts, "duplicates_removed": duplicates,
                "too_long_skipped": skipped, "files": hashes}
        if metadata:
            if any(key in meta or key == "fingerprint" for key in metadata):
                raise ValueError("附加元数据不能覆盖数据集格式字段")
            meta.update(metadata)
        meta["fingerprint"] = hashlib.sha256(compact_json(meta).encode("utf-8")).hexdigest()
        (temp / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        output.mkdir()
        for path in temp.iterdir():
            path.replace(output / path.name)
    return meta


class TokenDataset:
    def __init__(self, directory, split, block_size, verify=True):
        directory = Path(directory)
        self.meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.stage, self.block_size = self.meta["stage"], block_size
        if split not in ("train", "val"):
            raise ValueError("split 必须为 train 或 val")
        names = [f"{split}.tokens.bin"]
        if self.stage == "sft":
            names.append(f"{split}.labels.bin")
        if verify:
            for name in names:
                if sha256_file(directory / name) != self.meta["files"][name]:
                    raise ValueError(f"数据文件已改变，请重新准备数据：{name}")
        self.tokens = np.memmap(directory / names[0], dtype="<u2", mode="r")
        self.labels = None
        if self.stage == "sft":
            if block_size != self.meta["block_size"]:
                raise ValueError(f"SFT block_size 必须等于准备数据时的 {self.meta['block_size']}")
            self.tokens = self.tokens.reshape(-1, block_size + 1)
            self.labels = np.memmap(directory / names[1], dtype="<i4", mode="r").reshape(-1, block_size + 1)
        elif len(self.tokens) < block_size + 1:
            raise ValueError(f"{split} 仅有 {len(self.tokens)} token，请增加数据或减小 --block-size")

    def sample(self, batch_size, generator):
        if self.stage == "sft":
            indices = torch.randint(len(self.tokens), (batch_size,), generator=generator).tolist()
            inputs = np.stack([self.tokens[index] for index in indices]).astype(np.int64)
            labels = np.stack([self.labels[index] for index in indices]).astype(np.int64)
            last_target = int(np.nonzero(labels != -100)[1].max()) + 1
            inputs, labels = inputs[:, :last_target], labels[:, :last_target]
        else:
            starts = torch.randint(len(self.tokens) - self.block_size, (batch_size,), generator=generator).tolist()
            inputs = np.stack([self.tokens[start:start + self.block_size + 1] for start in starts]).astype(np.int64)
            labels = inputs.copy()
        return torch.from_numpy(inputs), torch.from_numpy(labels)
