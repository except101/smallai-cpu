"""A small decoder: tied embeddings, RMSNorm, RoPE, GQA and SwiGLU."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_layers: int = 10
    num_heads: int = 6
    num_kv_heads: int = 2
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_layers",
                     "num_heads", "num_kv_heads", "max_seq_len"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须为正整数")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads 必须能被 num_kv_heads 整除")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("RoPE 的每个注意力头维度必须是偶数")
        if not 0 <= self.dropout < 1 or self.norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("dropout、norm_eps 或 rope_theta 无效")

    @classmethod
    def load(cls, path):
        if str(path) == "cpu_22m":
            return cls()
        if str(path) == "smoke":
            return cls(vocab_size=2048, hidden_size=64, intermediate_size=128,
                       num_layers=2, num_heads=4, num_kv_heads=2)
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))

    def to_dict(self):
        return asdict(self)

    def parameter_count(self):
        d = self.hidden_size
        kv = self.num_kv_heads * (d // self.num_heads)
        block = 2 * d * d + 2 * d * kv + 3 * d * self.intermediate_size + 2 * d
        return self.vocab_size * d + self.num_layers * block + d


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        frequencies = cfg.rope_theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        angles = torch.outer(torch.arange(cfg.max_seq_len).float(), frequencies)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def rotate(self, x, start, length):
        cos, sin = self.cos[:, :, start:start + length], self.sin[:, :, start:start + length]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x, past=None, use_cache=False):
        batch, length, _ = x.shape
        start = 0 if past is None else past[0].shape[2]
        q = self.q_proj(x).view(batch, length, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.cfg.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.cfg.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotate(q, start, length), self.rotate(k, start, length)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        cache = (k, v) if use_cache else None
        repeats = self.cfg.num_heads // self.cfg.num_kv_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        mask = None
        causal = start == 0
        if start and length > 1:
            positions = torch.arange(start, start + length, device=x.device)
            mask = torch.arange(start + length, device=x.device)[None, :] <= positions[:, None]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, self.cfg.hidden_size)
        return self.out_proj(out), cache


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attention = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x, past=None, use_cache=False):
        attention, cache = self.attention(self.attn_norm(x), past, use_cache)
        x = x + attention
        normalized = self.ffn_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        return x, cache


class SmallLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.apply(self._initialize)
        for name, parameter in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * cfg.num_layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids 必须是非空的 [batch, length]")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("KV cache 层数不匹配")
        start = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if start + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("输入超过模型上下文长度")
        if labels is not None and past_key_values is not None:
            raise ValueError("训练时不能同时传入 KV cache")
        x, caches = self.embedding(input_ids), []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, cache = block(x, past, use_cache)
            caches.append(cache)
        logits = F.linear(self.norm(x), self.embedding.weight)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape or not (labels[:, 1:] != -100).any():
                raise ValueError("labels 形状错误或没有可学习的目标 token")
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "past_key_values": caches if use_cache else None}

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=128, temperature=0.7, top_k=40,
                 eos_id=None, banned_ids=()):
        if not prompt or max_new_tokens < 1 or temperature < 0 or top_k < 0:
            raise ValueError("生成参数无效")
        if len(prompt) >= self.config.max_seq_len:
            raise ValueError("提示词太长，请使用 /reset 清空历史或缩短输入")
        self.eval()
        device = self.embedding.weight.device
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        generated, cache = [], None
        for _ in range(min(max_new_tokens, self.config.max_seq_len - len(prompt))):
            out = self(ids, past_key_values=cache, use_cache=True)
            cache = out["past_key_values"]
            scores = out["logits"][:, -1, :].clone()
            if banned_ids:
                scores[:, list(banned_ids)] = -float("inf")
            if temperature == 0:
                token = scores.argmax(-1, keepdim=True)
            else:
                scores /= temperature
                if top_k:
                    threshold = torch.topk(scores, min(top_k, scores.shape[-1])).values[:, -1:]
                    scores[scores < threshold] = -float("inf")
                token = torch.multinomial(F.softmax(scores, dim=-1), 1)
            value = int(token.item())
            if value == eos_id:
                break
            generated.append(value)
            ids = token
        return generated
