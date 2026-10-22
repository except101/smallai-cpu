import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|system|>",
                  "<|user|>", "<|assistant|>", "<|tool|>", "<|tool_call|>", "<|tool_end|>"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalTokenizer:
    def __init__(self, path):
        self.path = Path(path)
        self.inner = Tokenizer.from_file(str(path))
        for token in SPECIAL_TOKENS:
            if self.inner.token_to_id(token) is None:
                raise ValueError(f"分词器缺少专用 token: {token}")
        # Ordinary text cannot insert role/control tokens. We append those IDs explicitly.
        self.inner.encode_special_tokens = True
        self.fingerprint = sha256_file(path)

    @property
    def vocab_size(self):
        return self.inner.get_vocab_size()

    def id(self, token):
        value = self.inner.token_to_id(token)
        if value is None:
            raise ValueError(f"未知专用 token: {token}")
        return value

    def encode(self, text):
        return self.inner.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.inner.decode(ids, skip_special_tokens=False)


def train_tokenizer(paths, output, vocab_size=16384, val_fraction=0.1):
    from .data import iter_corpus, is_validation

    if not 266 <= vocab_size <= 65535:
        raise ValueError("词表大小必须在 266 到 65535 之间")
    output = Path(output)
    if output.exists():
        raise ValueError(f"分词器已存在：{output}。请使用新路径，以免旧 checkpoint 失效")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    counts = {"training_documents": 0}

    def documents():
        for text in iter_corpus(paths):
            if not is_validation(text, val_fraction):
                counts["training_documents"] += 1
                yield text

    tokenizer.train_from_iterator(documents(), trainer)
    if not counts["training_documents"]:
        raise ValueError("没有可用于训练分词器的文本")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    meta = {**counts, "requested_vocab_size": vocab_size,
            "actual_vocab_size": tokenizer.get_vocab_size(), "val_fraction": val_fraction,
            "sha256": sha256_file(output), "pretrained_weights": False}
    output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    return meta
