from dataclasses import dataclass
from pathlib import Path
from typing import cast
from transformers import GPT2LMHeadModel
import tiktoken
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

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
            logits = self(ctx_token_ids)                                           # (B, T, vocab_size)
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

    def initialize_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear) and m.bias is not None:
            torch.nn.init.zeros_(m.bias)
        if isinstance(m, ScaledInitLinear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02 * (2 * self.config.block_count) ** -0.5)
        elif isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)


class ModelTrainer:
    def __init__(self, batch_size, ctx_length, token_ids):
        self.batch_size = batch_size
        self.ctx_length = ctx_length
        self.token_ids = token_ids
        self.position = 0

    def train(self, model, step_count, optimizer):
        model.train()
        for step in range(step_count):
            self.step(model, step, optimizer)
        model.eval()

    def step(self, model, step, optimizer):
        t0 = time.time()
        xb, yb = self.get_next_batch()
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        elapsed_ms = (t1 - t0)*1000
        tokens_per_sec = (xb.size(0) * xb.size(1)) / (t1 - t0)
        print(f"Step {step}, Loss: {loss.item()}, elapsed: {elapsed_ms:.2f}ms, tokens/s: {tokens_per_sec:.2f}")

    def get_next_batch(self):
        buf = self.token_ids[self.position : self.position+self.batch_size*self.ctx_length+1]
        x = (buf[:-1]).view(self.batch_size, self.ctx_length)
        y = (buf[1:]).view(self.batch_size, self.ctx_length)
        self.update_position()
        return x, y

    def update_position(self):
        self.position += self.batch_size * self.ctx_length
        if self.position + (self.batch_size * self.ctx_length + 1) > len(self.token_ids):
            self.position = 0

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    print("Device:", torch.get_default_device())
    torch.manual_seed(1337)
    if torch.cuda.is_available(): torch.cuda.manual_seed(1337)
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True)

    data_path = Path(__file__).resolve().parent.parent / 'data' / 'shakespeare.txt'
    text = data_path.read_text()
    encoder = tiktoken.get_encoding("gpt2")
    data = torch.tensor(encoder.encode(text), dtype=torch.long)
    trainer = ModelTrainer(batch_size=16, ctx_length=1024, token_ids=data)
    config = GPTConfig(ctx_length=1024, emb_size=768, block_count=12, head_count=12, vocab_size=encoder.n_vocab)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    compiled_model = torch.compile(model)
    trainer.train(compiled_model, step_count=50, optimizer=optimizer)

    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # prompt = "Hello, I'm a language model,"
    # input_token_ids = torch.tensor(encoder.encode(prompt), dtype=torch.long)
    # output_token_ids = model.generate(input_token_ids.unsqueeze(0), new_token_count=30)[0]
    # print(encoder.decode(output_token_ids.tolist()))
