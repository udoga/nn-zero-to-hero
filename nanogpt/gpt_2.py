from dataclasses import dataclass
from typing import cast
from transformers import GPT2LMHeadModel
import tiktoken

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    layer_count: int = 12
    head_count: int = 12
    embedding_size: int = 768

    @classmethod
    def from_model(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        config_args = {
            "gpt2": dict(layer_count=12, head_count=12, embedding_size=768),
            "gpt2-medium": dict(layer_count=24, head_count=16, embedding_size=1024),
            "gpt2-large": dict(layer_count=36, head_count=20, embedding_size=1280),
            "gpt2-xl": dict(layer_count=48, head_count=25, embedding_size=1600),
        }[model_type]
        return cls(vocab_size=50257, block_size=1024, **config_args)


class CausalSelfAttention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.embedding_size % c.head_count == 0
        self.head_count = c.head_count
        self.embedding_size = c.embedding_size
        self.qkv_projection = nn.Linear(c.embedding_size, 3 * c.embedding_size)
        self.output_projection = nn.Linear(c.embedding_size, c.embedding_size)
        self.register_buffer("tril", torch.tril(torch.ones(c.block_size, c.block_size)), persistent=False)

    def to_heads(self, x, B, T, C):
        return x.view(B, T, self.head_count, C // self.head_count).transpose(1, 2)

    def forward(self, x):
        B, T, C = x.size()                                              # (B, T, C)
        qkv = self.qkv_projection(x)                                    # (B, T, 3 * C)
        q, k, v = qkv.split(self.embedding_size, dim=2)                 # (B, T, C) each
        q, k, v = [self.to_heads(part, B, T, C) for part in (q, k, v)]  # (B, HC, T, HS) each
        A = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)            # (B, HC, T, T)
        A = A.masked_fill(cast(torch.Tensor, self.tril)[:T, :T] == 0, float("-inf"))        # (B, HC, T, T)
        A = F.softmax(A, dim=-1)                                        # (B, HC, T, T)
        y = A @ v                                                       # (B, HC, T, HS)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                # (B, T, C)
        return self.output_projection(y)                                # (B, T, C)


class FeedForward(nn.Sequential):
    def __init__(self, c: GPTConfig):
        super().__init__(
            nn.Linear(c.embedding_size, 4 * c.embedding_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * c.embedding_size, c.embedding_size),
        )


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.pre_attention_norm = nn.LayerNorm(c.embedding_size)
        self.attention = CausalSelfAttention(c)
        self.pre_feed_forward_norm = nn.LayerNorm(c.embedding_size)
        self.feed_forward = FeedForward(c)

    def forward(self, x):
        x = x + self.attention(self.pre_attention_norm(x))
        x = x + self.feed_forward(self.pre_feed_forward_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(c.vocab_size, c.embedding_size)
        self.position_embedding = nn.Embedding(c.block_size, c.embedding_size)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layer_count)])
        self.final_norm = nn.LayerNorm(c.embedding_size)
        self.unembedding = nn.Linear(c.embedding_size, c.vocab_size, bias=False)

    def forward(self, token_ids):
        B, T = token_ids.size()                                                       # (B, T)
        positions = torch.arange(0, T, device=token_ids.device)                       # (T)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)      # (B, T, C)
        for block in self.blocks:
            x = block(x)                                                              # (B, T, C)
        x = self.final_norm(x)                                                        # (B, T, C)
        return self.unembedding(x)                                                    # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, token_ids, new_token_count):
        for _ in range(new_token_count):
            context_size = self.position_embedding.num_embeddings
            context_token_ids = token_ids[:, -context_size:]                       # (B, T)
            logits = self(context_token_ids)                                       # (B, T, vocab_size)
            next_token_logits = logits[:, -1, :]                                   # (B, vocab_size)
            next_token_probs = F.softmax(next_token_logits, dim=-1)                # (B, vocab_size)
            next_token_ids = torch.multinomial(next_token_probs, num_samples=1)    # (B, 1)
            token_ids = torch.cat((token_ids, next_token_ids), dim=1)              # (B, T+1)
        return token_ids                                                           # (B, T+new_token_count)

    @classmethod
    def from_model(cls, model_type):
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


torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

model = GPT.from_model("gpt2")
encoder = tiktoken.get_encoding("gpt2")
prompt = "Hello, I'm a language model,"
input_token_ids = torch.tensor(encoder.encode(prompt), dtype=torch.long)
output_token_ids = model.generate(input_token_ids.unsqueeze(0), new_token_count=32)[0]
print(encoder.decode(output_token_ids.tolist()))
