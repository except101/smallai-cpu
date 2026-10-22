import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

import torch

from .data import prepare_dataset
from .demo import generate_demo
from .inference import chat, evaluate
from .model import ModelConfig, SmallLM
from .tokenizer import train_tokenizer
from .tools import execute_tool, parse_tool_call
from .training import configure_cpu, train


ROOT = Path(__file__).resolve().parent.parent


def preset(name):
    local = ROOT / "configs" / f"{name}.json"
    return str(local) if local.is_file() else name


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def benchmark(config_path, steps=3, block_size=128, batch_size=1, threads=4):
    if steps < 1 or block_size < 2 or batch_size < 1:
        raise ValueError("基准测试参数无效")
    configure_cpu(threads)
    torch.manual_seed(42)
    cfg = ModelConfig.load(config_path)
    if block_size + 1 > cfg.max_seq_len:
        raise ValueError("block_size + 1 超过模型上下文")
    model = SmallLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    inputs = torch.randint(cfg.vocab_size, (batch_size, block_size + 1))
    durations = []
    for index in range(steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, inputs)["loss"]
        loss.backward()
        optimizer.step()
        if index:
            durations.append(time.perf_counter() - started)
    speed = steps * batch_size * block_size / sum(durations)
    return {"parameters": cfg.parameter_count(), "threads": threads, "block_size": block_size,
            "measured_steps": steps, "seconds_per_step": sum(durations) / steps,
            "target_tokens_per_second": speed,
            "estimated_hours_per_million_tokens": 1_000_000 / speed / 3600,
            "note": "短时 CPU 测量；长时间运行受温度、序列长度和其他程序影响。"}


def smoke(output):
    output = Path(output)
    if output.exists():
        raise ValueError(f"检查目录已存在，请换一个 --output：{output}")
    output.mkdir(parents=True)
    print("[1/6] 生成流程测试语料", flush=True)
    demo = output / "data"
    generate_demo(demo)
    print("[2/6] 从零训练 byte-level BPE 分词器", flush=True)
    tokenizer = output / "tokenizer.json"
    train_tokenizer([demo / "corpus.jsonl"], tokenizer, vocab_size=2048)
    print("[3/6] 准备预训练与 SFT 数据", flush=True)
    pretrain_data, sft_data = output / "pretrain_data", output / "sft_data"
    pretrain_meta = prepare_dataset("pretrain", [demo / "corpus.jsonl"], tokenizer, pretrain_data)
    sft_meta = prepare_dataset("sft", [demo / "sft.jsonl"], tokenizer, sft_data, block_size=1023)
    print("[4/6] CPU 随机初始化预训练（6 步）", flush=True)
    base = train(stage="pretrain", data=pretrain_data, tokenizer_path=tokenizer,
                 output=output / "pretrain", config_path=preset("smoke"), steps=6,
                 block_size=64, threads=4, eval_every=6, eval_batches=2, log_every=3)
    print("[5/6] 使用自己的预训练权重进行 SFT（6 步）", flush=True)
    sft = train(stage="sft", data=sft_data, tokenizer_path=tokenizer,
                output=output / "sft", init_from=base["checkpoint"], steps=6,
                block_size=1023, learning_rate=1e-4, threads=4,
                eval_every=6, eval_batches=2, log_every=3)
    print("[6/6] 验证工具执行，并检查模型生成与评测入口", flush=True)
    call = parse_tool_call('<|tool_call|>{"name":"calculator","arguments":{"expression":"17 * 23"}}<|tool_end|>')
    assert execute_tool(call) == {"result": 391}
    evaluation = evaluate(sft["checkpoint"], tokenizer, demo / "eval_tools.jsonl",
                          output / "tool_evaluation.json", limit=2, max_new_tokens=16)
    report = {"status": "passed", "pretrain": base, "sft": sft,
              "data": {"pretrain": pretrain_meta["splits"], "sft": sft_meta["splits"],
                       "sft_too_long_skipped": sft_meta["too_long_skipped"]},
              "tool_runtime_result": {"result": 391}, "model_evaluation": evaluation,
              "meaning": "程序流程检查通过；6 步训练不能证明聊天能力或工具调用能力。"}
    (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parser():
    main = argparse.ArgumentParser(description="SmallAI：在 CPU 上从随机权重训练中文实验模型")
    sub = main.add_subparsers(dest="command", required=True)
    command = sub.add_parser("doctor", help="检查 Python 和依赖")
    command = sub.add_parser("inspect", help="查看模型参数量和静态内存估算")
    command.add_argument("--config", default=preset("cpu_22m"))
    command = sub.add_parser("demo-data", help="生成自写的小型流程测试语料")
    command.add_argument("--output", default="data/demo")
    command = sub.add_parser("tokenizer", help="从本地语料训练自己的分词器")
    command.add_argument("--input", nargs="+", required=True)
    command.add_argument("--output", default="data/tokenizer.json")
    command.add_argument("--vocab-size", type=int, default=16384)
    command.add_argument("--val-fraction", type=float, default=0.1)
    command = sub.add_parser("prepare", help="划分和编码本地数据")
    command.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    command.add_argument("--input", nargs="+", required=True)
    command.add_argument("--tokenizer", default="data/tokenizer.json")
    command.add_argument("--output", required=True)
    command.add_argument("--block-size", type=int, default=1023)
    command.add_argument("--val-fraction", type=float, default=0.1)
    command = sub.add_parser("train", help="预训练、SFT 或恢复中断的训练")
    command.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    command.add_argument("--data", required=True)
    command.add_argument("--tokenizer", dest="tokenizer_path", default="data/tokenizer.json")
    command.add_argument("--output", required=True)
    command.add_argument("--config", dest="config_path", default=preset("cpu_22m"))
    command.add_argument("--steps", type=int, default=200)
    command.add_argument("--block-size", type=int, default=128)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--accumulation", type=int, default=1)
    command.add_argument("--learning-rate", type=float, default=3e-4)
    command.add_argument("--threads", type=int, default=4)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--init-from")
    command.add_argument("--resume")
    command.add_argument("--eval-every", type=int, default=25)
    command.add_argument("--eval-batches", type=int, default=4)
    command.add_argument("--save-every", type=int, default=50)
    command.add_argument("--log-every", type=int, default=10)
    command.add_argument("--stop-file", help=argparse.SUPPRESS)
    for name in ("chat", "evaluate"):
        command = sub.add_parser(name, help="本地对话" if name == "chat" else "检查首次工具调用判断")
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--tokenizer", dest="tokenizer_path", default="data/tokenizer.json")
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--max-new-tokens", type=int, default=128)
        if name == "chat":
            command.add_argument("--prompt", help="只回答一个问题后退出")
            command.add_argument("--no-tools", action="store_true")
            command.add_argument("--temperature", type=float, default=0.7)
        else:
            command.add_argument("--data", dest="data_path", required=True)
            command.add_argument("--output", default="outputs/tool_evaluation.json")
            command.add_argument("--limit", type=int)
    command = sub.add_parser("benchmark", help="短时测量 CPU 实际训练速度")
    command.add_argument("--config", dest="config_path", default=preset("cpu_22m"))
    command.add_argument("--steps", type=int, default=3)
    command.add_argument("--block-size", type=int, default=128)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--threads", type=int, default=4)
    command = sub.add_parser("smoke", help="从分词到工具评测，运行一次很小的完整流程")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    command.add_argument("--output", default=f"outputs/check_{stamp}")
    command = sub.add_parser("ui", help="打开本地可视化训练界面")
    command.add_argument("--port", type=int, default=8765)
    command.add_argument("--no-browser", action="store_true")
    command = sub.add_parser("public-data", help="下载并准备公开中文、对话和工具调用数据")
    command.add_argument("--web-docs", type=int, default=5000)
    command.add_argument("--chat-rows", type=int, default=1500)
    command.add_argument("--tool-rows", type=int, default=1500)
    command.add_argument("--name", dest="package_name")
    return main


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    command_parser = parser()
    if len(sys.argv) == 1:
        command_parser.print_help()
        return
    args = vars(command_parser.parse_args())
    command = args.pop("command")
    try:
        if command == "doctor":
            import numpy
            import tokenizers
            print_json({"python": sys.version, "python_path": sys.executable,
                        "torch": str(torch.__version__), "tokenizers": tokenizers.__version__,
                        "numpy": numpy.__version__, "device": "cpu", "pretrained_downloads": False})
        elif command == "inspect":
            cfg = ModelConfig.load(args["config"])
            print_json({"config": cfg.to_dict(), "parameters": cfg.parameter_count(),
                        "float32_weights_MiB": cfg.parameter_count() * 4 / 2**20,
                        "float32_training_state_MiB": cfg.parameter_count() * 16 / 2**20,
                        "memory_note": "不含激活值、KV cache、临时张量和 Python/PyTorch 自身开销。",
                        "vocab_note": "实际训练会使用分词器的真实词表；小语料可能使参数量低于此配置。"})
        elif command == "demo-data":
            print_json(generate_demo(args["output"]))
        elif command == "tokenizer":
            print_json(train_tokenizer(args["input"], args["output"], args["vocab_size"], args["val_fraction"]))
        elif command == "prepare":
            print_json(prepare_dataset(args["stage"], args["input"], args["tokenizer"], args["output"],
                                       args["block_size"], args["val_fraction"]))
        elif command == "train":
            result = train(**args)
            print_json(result)
            if result["interrupted"]:
                raise SystemExit(130)
        elif command == "chat":
            chat(**args)
        elif command == "evaluate":
            print_json(evaluate(**args))
        elif command == "benchmark":
            print_json(benchmark(**args))
        elif command == "smoke":
            print_json(smoke(args["output"]))
        elif command == "ui":
            from .ui import run_ui
            run_ui(Path.cwd(), port=args["port"], open_browser=not args["no_browser"])
        elif command == "public-data":
            from .public_data import prepare_public_data
            print_json(prepare_public_data(Path.cwd(), **args))
    except (ValueError, OSError) as error:
        command_parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
