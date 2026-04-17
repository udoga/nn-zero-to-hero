from dataclasses import dataclass
import os
from pathlib import Path
from typing import cast
from transformers import GPT2LMHeadModel
import tiktoken
import time
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class GPTConfig:
    ctx_length: int = 1024
    vocab_size: int = 50257
    block_count: int = 12
    head_count: int = 12
    emb_size: int = 768

    @classmethod
    def from_model(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        config_args = {
            "gpt2": dict(block_count=12, head_count=12, emb_size=768),
            "gpt2-medium": dict(block_count=24, head_count=16, emb_size=1024),
            "gpt2-large": dict(block_count=36, head_count=20, emb_size=1280),
            "gpt2-xl": dict(block_count=48, head_count=25, emb_size=1600),
        }[model_type]
        return cls(vocab_size=50257, ctx_length=1024, **config_args)


class ScaledInitLinear(nn.Linear):
    pass


class CausalSelfAttention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.emb_size % c.head_count == 0
        self.head_count = c.head_count
        self.emb_size = c.emb_size
        self.qkv_projection = nn.Linear(c.emb_size, 3 * c.emb_size)
        self.output_projection = ScaledInitLinear(c.emb_size, c.emb_size)
        self.register_buffer("tril", torch.tril(torch.ones(c.ctx_length, c.ctx_length)), persistent=False)

    def to_heads(self, x, B, T, C):
        return x.view(B, T, self.head_count, C // self.head_count).transpose(1, 2)

    def forward(self, x):
        B, T, C = x.size()                                                # (B, T, C)
        qkv = self.qkv_projection(x)                                      # (B, T, 3 * C)
        q, k, v = qkv.split(self.emb_size, dim=2)                         # (B, T, C) each
        q, k, v = [self.to_heads(part, B, T, C) for part in (q, k, v)]    # (B, HC, T, HS) each
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # (B, T, C) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)                  # (B, T, C)
        return self.output_projection(y)                                  # (B, T, C)


class FeedForward(nn.Sequential):
    def __init__(self, c: GPTConfig):
        super().__init__(
            nn.Linear(c.emb_size, 4 * c.emb_size),
            nn.GELU(approximate="tanh"),
            ScaledInitLinear(4 * c.emb_size, c.emb_size))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.pre_attention_norm = nn.LayerNorm(c.emb_size)
        self.attention = CausalSelfAttention(c)
        self.pre_feed_forward_norm = nn.LayerNorm(c.emb_size)
        self.feed_forward = FeedForward(c)

    def forward(self, x):
        x = x + self.attention(self.pre_attention_norm(x))
        x = x + self.feed_forward(self.pre_feed_forward_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.token_embedding = nn.Embedding(c.vocab_size, c.emb_size)
        self.position_embedding = nn.Embedding(c.ctx_length, c.emb_size)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.block_count)])
        self.final_norm = nn.LayerNorm(c.emb_size)
        self.unembedding = nn.Linear(c.emb_size, c.vocab_size, bias=False)
        self.token_embedding.weight = self.unembedding.weight
        self.apply(self.initialize_weights)

    def forward(self, token_ids, targets=None):
        B, T = token_ids.size()                                                       # (B, T)
        positions = torch.arange(0, T, device=token_ids.device)                       # (T)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)      # (B, T, C)
        for block in self.blocks:
            x = block(x)                                                              # (B, T, C)
        x = self.final_norm(x)                                                        # (B, T, C)
        logits = self.unembedding(x)                                                  # (B, T, vocab_size)
        loss = None if targets is None else self.get_loss(logits, targets)
        return logits, loss

    def get_loss(self, logits, targets):
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def generate(self, token_ids, new_token_count):
        for _ in range(new_token_count):
            ctx_token_ids = token_ids[:, -self.config.ctx_length:]                 # (B, T)
            logits, _ = self(ctx_token_ids)                                        # (B, T, vocab_size)
            next_token_logits = logits[:, -1, :]                                   # (B, vocab_size)
            next_token_probs = F.softmax(next_token_logits, dim=-1)                # (B, vocab_size)
            topk_probs, topk_indices = torch.topk(next_token_probs, k=50, dim=-1)  # (B, k) each
            next_token_indices = torch.multinomial(topk_probs, num_samples=1)      # (B, 1)
            next_token_ids = torch.gather(topk_indices, -1, next_token_indices)    # (B, 1)
            token_ids = torch.cat((token_ids, next_token_ids), dim=1)              # (B, T+1)
        return token_ids                                                           # (B, T+new_token_count)

    @classmethod
    def from_pretrained(cls, model_type):
        config = GPTConfig.from_model(model_type)
        local_model = GPT(config)
        local_dict = local_model.state_dict()
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_dict = hf_model.state_dict()
        transposed_keys = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(hf_dict) == len(local_dict)
        for hf_key in hf_dict.keys():
            local_key = cls.to_local_key(hf_key)
            transposed = any(hf_key.endswith(k) for k in transposed_keys)
            hf_weights = hf_dict[hf_key].t() if transposed else hf_dict[hf_key]
            assert hf_weights.shape == local_dict[local_key].shape
            local_dict[local_key].copy_(hf_weights)
        return local_model

    @classmethod
    def to_local_key(cls, hf_key):
        key = hf_key.removeprefix("transformer.")
        key = key.replace("wte.", "token_embedding.")
        key = key.replace("wpe.", "position_embedding.")
        key = key.replace("h.", "blocks.")
        key = key.replace("ln_1.", "pre_attention_norm.")
        key = key.replace("attn.c_attn.", "attention.qkv_projection.")
        key = key.replace("attn.c_proj.", "attention.output_projection.")
        key = key.replace("ln_2.", "pre_feed_forward_norm.")
        key = key.replace("mlp.c_fc.", "feed_forward.0.")
        key = key.replace("mlp.c_proj.", "feed_forward.2.")
        key = key.replace("ln_f.", "final_norm.")
        key = key.replace("lm_head.", "unembedding.")
        return key

    def get_param_groups(self, w_decay):
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        one_dim_params = [p for n, p in param_dict.items() if p.dim() < 2]
        multi_dim_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        return [{'params': multi_dim_params, 'weight_decay': w_decay},
                {'params': one_dim_params, 'weight_decay': 0.0}]

    def initialize_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear) and m.bias is not None:
            torch.nn.init.zeros_(m.bias)
        if isinstance(m, ScaledInitLinear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02 * (2 * self.config.block_count) ** -0.5)
        elif isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)


class DataLoader:
    def __init__(self, process_rank, process_count, token_ids, batch_size, ctx_length):
        self.process_count = process_count
        self.token_ids = token_ids
        self.batch_size = batch_size
        self.ctx_length = ctx_length
        self.initial_position = batch_size * ctx_length * process_rank
        self.position = self.initial_position

    def get_next_batch(self):
        buf = self.token_ids[self.position : self.position+self.batch_size*self.ctx_length+1]
        x = (buf[:-1]).view(self.batch_size, self.ctx_length)
        y = (buf[1:]).view(self.batch_size, self.ctx_length)
        self.update_position()
        return x, y

    def update_position(self):
        increment = self.batch_size * self.ctx_length * self.process_count
        self.position += increment
        if self.position + (increment + 1) > len(self.token_ids):
            self.position = self.initial_position


class GPTTrainer:
    def __init__(self, process_rank, process_count, data_loader, step_token_count, step_count):
        self.process_rank = process_rank
        self.process_count = process_count
        self.data_loader = data_loader
        self.step_token_count = step_token_count
        self.step_count = step_count
        self.microstep_count = self.get_microstep_count()

    def train(self, model, optimizer):
        model.train()
        for step in range(self.step_count):
            self.step_and_measure(model, step, optimizer)
        model.eval()

    def step_and_measure(self, model, step, optimizer):
        t0 = time.time()
        loss, norm, lr = self.step(model, step, optimizer)
        torch.cuda.synchronize()
        t1 = time.time()
        elapsed_ms = (t1 - t0)*1000
        tokens_per_sec = self.step_token_count * self.process_count / (t1 - t0)
        self.print_row(step, loss, norm, lr, elapsed_ms, tokens_per_sec)

    def print_row(self, step, loss, norm, lr, elapsed_ms, tokens_per_sec):
        if self.process_rank == 0:
            print(f"step: {step:4d} | loss: {loss:.6f} | norm: {norm.item():.4f} | lr: {lr:.4e} | "
                  f"elapsed: {elapsed_ms:.2f}ms | tokens/s: {tokens_per_sec:.2f}")

    def step(self, model, step, optimizer):
        optimizer.zero_grad()
        loss = self.get_accumulated_loss(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = self.get_lr(step)
        for g in optimizer.param_groups: g['lr'] = lr
        optimizer.step()
        return loss, norm, lr

    def get_accumulated_loss(self, model):
        loss = 0.0
        for microstep in range(self.microstep_count):
            loss += self.microstep(model, microstep)
        if self.process_count > 1:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def microstep(self, model, microstep):
        xb, yb = self.data_loader.get_next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        loss /= self.microstep_count
        if self.process_count > 1:
            model.require_backward_grad_sync = (microstep == self.microstep_count - 1)
        loss.backward()
        return loss.detach()

    def get_lr(self, step):
        max_lr = 6e-4
        min_lr = max_lr * 0.1
        warmup_steps = 10
        if step < warmup_steps:  return max_lr * (step+1) / warmup_steps
        if step > self.step_count: return min_lr
        decay_ratio = (step - warmup_steps) / (self.step_count - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    def get_microstep_count(self):
        microstep_token_count = self.data_loader.batch_size * self.data_loader.ctx_length * self.process_count
        assert self.step_token_count % microstep_token_count == 0
        return self.step_token_count // microstep_token_count


if __name__ == "__main__":
    process_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    process_count = int(os.environ.get('WORLD_SIZE', 1))
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    if process_rank == 0: print("Device:", device)

    if process_count > 1: torch.distributed.init_process_group(backend='nccl')
    torch.set_default_device(device)
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True)

    path = Path(__file__).resolve().parent.parent / 'data' / 'shakespeare.txt'
    text = path.read_text()
    encoder = tiktoken.get_encoding("gpt2")
    token_ids = torch.tensor(encoder.encode(text), dtype=torch.long)
    config = GPTConfig(ctx_length=1024, emb_size=768, block_count=12, head_count=12, vocab_size=50304)
    model = GPT(config)
    data_loader = DataLoader(process_rank, process_count, token_ids, batch_size=16, ctx_length=1024)
    trainer = GPTTrainer(process_rank, process_count, data_loader, step_token_count=524288, step_count=50)
    optimizer = torch.optim.AdamW(model.get_param_groups(w_decay=0.1), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    compiled_model = torch.compile(model)
    trained_model = DDP(compiled_model, device_ids=[local_rank]) if process_count > 1 else compiled_model
    trainer.train(trained_model, optimizer=optimizer)

    prompt = "Hello, I'm a language model,"
    input_token_ids = torch.tensor(encoder.encode(prompt), dtype=torch.long)
    output_token_ids = model.generate(input_token_ids.unsqueeze(0), new_token_count=30)[0]
    if process_rank == 0: print(encoder.decode(output_token_ids.tolist()))
    if process_count > 1: torch.distributed.destroy_process_group()
