import pytest
import torch
import shutil

from smallai.data import prepare_dataset
from smallai.demo import generate_demo
from smallai.tokenizer import LocalTokenizer, train_tokenizer


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def local_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("local_data")
    generate_demo(root / "demo")
    path = root / "tokenizer.json"
    train_tokenizer([root / "demo/corpus.jsonl"], path, vocab_size=2048)
    prepare_dataset("pretrain", [root / "demo/corpus.jsonl"], path, root / "pretrain")
    prepare_dataset("sft", [root / "demo/sft.jsonl"], path, root / "sft", block_size=1023)
    return root, LocalTokenizer(path)


@pytest.fixture
def ui_workspace(local_data, tmp_path):
    root, tokenizer = local_data
    (tmp_path / "data/prepared").mkdir(parents=True)
    shutil.copy2(tokenizer.path, tmp_path / "data/tokenizer.json")
    for stage in ("pretrain", "sft"):
        shutil.copytree(root / stage, tmp_path / "data/prepared" / stage)
    return tmp_path
