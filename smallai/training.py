import json
import math
import os
from pathlib import Path
import time

import torch

from .data import TokenDataset
from .model import ModelConfig, SmallLM
from .tokenizer import LocalTokenizer


def configure_cpu(threads=4):
    if threads < 1:
        raise ValueError("CPU 线程数必须大于 0")
    torch.set_num_threads(threads)


def load_checkpoint(path, tokenizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("不支持的 checkpoint 格式")
    if tokenizer and checkpoint["tokenizer_sha256"] != tokenizer.fingerprint:
        raise ValueError("分词器与 checkpoint 不匹配；不能更换词表后继续使用旧权重")
    return checkpoint


def load_model(path, tokenizer_path, threads=4):
    configure_cpu(threads)
    tokenizer = LocalTokenizer(tokenizer_path)
    checkpoint = load_checkpoint(path, tokenizer)
    model = SmallLM(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, checkpoint


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
        write_checkpoint_metadata(path, value)
    finally:
        temporary.unlink(missing_ok=True)


def write_checkpoint_metadata(path, value):
    target = Path(path).with_suffix(".meta.json")
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    metadata = {"format_version": 1, "config": value["config"],
                "tokenizer_sha256": value["tokenizer_sha256"], "step": value["step"],
                "stage": value["settings"]["stage"]}
    temporary.write_text(json.dumps(metadata), encoding="utf-8")
    temporary.replace(target)
    return metadata


def checkpoint_metadata(path):
    sidecar = Path(path).with_suffix(".meta.json")
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return write_checkpoint_metadata(path, load_checkpoint(path))


@torch.inference_mode()
def validation_loss(model, dataset, batch_size, batches):
    model.eval()
    generator = torch.Generator().manual_seed(7001)
    total, count = 0.0, 0
    try:
        for _ in range(batches):
            inputs, labels = dataset.sample(batch_size, generator)
            n = int((labels[:, 1:] != -100).sum())
            total += float(model(inputs, labels)["loss"]) * n
            count += n
    finally:
        model.train()
    return total / count


def train(*, stage, data, tokenizer_path, output, config_path=None, steps=200,
          block_size=256, batch_size=1, accumulation=1, learning_rate=3e-4,
          threads=4, seed=42, init_from=None, resume=None, eval_every=25,
          eval_batches=4, save_every=50, log_every=10, stop_file=None):
    if stage not in ("pretrain", "sft"):
        raise ValueError("stage 必须是 pretrain 或 sft")
    if any(type(value) is not int or value < 1 for value in
           (steps, block_size, batch_size, accumulation, eval_every, eval_batches, save_every, log_every)):
        raise ValueError("步数、批大小和各间隔必须是正整数")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if init_from and resume:
        raise ValueError("--init-from 和 --resume 不能一起使用")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(f"输出目录非空：{output}。续训请加 --resume，重新训练请换输出目录")
    configure_cpu(threads)
    torch.manual_seed(seed)
    tokenizer = LocalTokenizer(tokenizer_path)
    training = TokenDataset(data, "train", block_size)
    validation = TokenDataset(data, "val", block_size)
    if training.stage != stage:
        raise ValueError("数据集阶段与 --stage 不一致")
    if tokenizer.fingerprint != training.meta["tokenizer_sha256"]:
        raise ValueError("数据集使用了不同分词器")
    source = load_checkpoint(resume or init_from, tokenizer) if resume or init_from else None
    if source:
        cfg = ModelConfig(**source["config"])
    elif config_path:
        cfg = ModelConfig.load(config_path)
        cfg.vocab_size = tokenizer.vocab_size
    else:
        raise ValueError("随机初始化需要 --config")
    if cfg.vocab_size != tokenizer.vocab_size or block_size + 1 > cfg.max_seq_len:
        raise ValueError("词表不匹配或 block_size + 1 超出模型上下文")
    model = SmallLM(cfg)
    if source:
        model.load_state_dict(source["model"])
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=learning_rate,
                                betas=(0.9, 0.95))
    settings = {"stage": stage, "steps": steps, "block_size": block_size,
                "batch_size": batch_size, "accumulation": accumulation,
                "learning_rate": learning_rate, "seed": seed,
                "data_fingerprint": training.meta["fingerprint"]}
    generator = torch.Generator().manual_seed(seed + 1)
    step, target_tokens, previous_seconds, last_validation = 0, 0, 0.0, None
    if resume:
        if source["settings"] != settings:
            raise ValueError("续训需保持原阶段、数据、总步数、批大小、学习率与 seed；新计划请用 --init-from")
        optimizer.load_state_dict(source["optimizer"])
        torch.set_rng_state(source["rng_state"])
        generator.set_state(source["sampler_state"])
        step, target_tokens = source["step"], source["target_tokens"]
        previous_seconds = source["elapsed_seconds"]
        last_validation = source.get("validation_loss")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{stage}: {parameters:,} 参数，CPU {threads} 线程，词表 {cfg.vocab_size}，起始 step={step}", flush=True)
    if not source:
        print("权重来源：随机初始化；未载入任何预训练模型。", flush=True)
    started = time.perf_counter()

    def checkpoint():
        save_checkpoint(output / "last.pt", {
            "format_version": 1, "config": cfg.to_dict(), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "settings": settings,
            "step": step, "target_tokens": target_tokens,
            "elapsed_seconds": previous_seconds + time.perf_counter() - started,
            "validation_loss": last_validation, "tokenizer_sha256": tokenizer.fingerprint,
            "rng_state": torch.get_rng_state(), "sampler_state": generator.get_state(),
            "initialization": "random" if source is None else str(resume or init_from),
        })

    interrupted = False
    try:
        while step < steps:
            if stop_file and Path(stop_file).is_file():
                interrupted = True
                print("收到停止请求，正在保存权重和续训状态。", flush=True)
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            micro_batches = [training.sample(batch_size, generator) for _ in range(accumulation)]
            counts = [int((labels[:, 1:] != -100).sum()) for _, labels in micro_batches]
            total_count, loss_value = sum(counts), 0.0
            warmup = max(1, min(20, steps // 10))
            if step < warmup:
                factor = (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, steps - warmup - 1)
                factor = 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * factor
            for (inputs, labels), count in zip(micro_batches, counts):
                loss = model(inputs, labels)["loss"]
                if not torch.isfinite(loss):
                    raise ValueError("训练 loss 非有限值；请检查数据或降低学习率")
                weighted = loss * (count / total_count)
                weighted.backward()
                loss_value += float(weighted.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            target_tokens += total_count
            record = {"step": step, "loss": loss_value, "lr": optimizer.param_groups[0]["lr"],
                      "grad_norm": float(norm), "target_tokens": target_tokens,
                      "elapsed_seconds": previous_seconds + time.perf_counter() - started}
            if step % eval_every == 0 or step == steps:
                last_validation = validation_loss(model, validation, batch_size, eval_batches)
                record["validation_loss"] = last_validation
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            if step % log_every == 0 or step == steps:
                speed = target_tokens / max(record["elapsed_seconds"], 1e-9)
                val = f", val={last_validation:.4f}" if "validation_loss" in record else ""
                print(f"step {step}/{steps}: loss={loss_value:.4f}{val}, {speed:.0f} target tok/s", flush=True)
            if step % save_every == 0:
                checkpoint()
    except KeyboardInterrupt:
        interrupted = True
        print("\n训练已中断，正在保存当前权重和续训状态。", flush=True)
    finally:
        checkpoint()
    result = {"checkpoint": str(output / "last.pt"), "step": step, "parameters": parameters,
              "target_tokens": target_tokens, "validation_loss": last_validation,
              "elapsed_seconds": previous_seconds + time.perf_counter() - started,
              "interrupted": interrupted}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
