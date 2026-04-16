import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class Head(nn.Module):
    def __init__(self, head_size, block_size, emb_size, dropout_rate):
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.tril: torch.Tensor
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        T = x.shape[1]
        k = self.key(x)                                          # (B, T, head_size)
        q = self.query(x)                                        # (B, T, head_size)
        A = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5           # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        A = A.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        A = F.softmax(A, dim=-1)                                 # (B, T, T)
        A = self.dropout(A)                                      # (B, T, T)
        v = self.value(x)                                        # (B, T, head_size)
        y = A @ v                                                # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, head_count, head_size, block_size, emb_size, dropout_rate):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, emb_size, dropout_rate) for _ in range(head_count)])
        self.proj = nn.Linear(head_size * head_count, emb_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Sequential):
    def __init__(self, emb_size):
        super().__init__(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size))

class Block(nn.Module):
    def __init__(self, head_count, block_size, emb_size, dropout_rate):
        super().__init__()
        head_size = emb_size // head_count
        self.attention = MultiHeadAttention(head_count, head_size, block_size, emb_size, dropout_rate)
        self.feed_forward = FeedFoward(emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class GPTChar(nn.Module):
    def __init__(self, vocab_size, block_size, emb_size, head_count, dropout_rate, layer_count):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(block_size, emb_size)
        self.blocks = nn.Sequential(*[Block(head_count, block_size, emb_size, dropout_rate) for _ in range(layer_count)])
        self.layer_norm = nn.LayerNorm(emb_size)
        self.unembedding = nn.Linear(emb_size, vocab_size)
        self.eval()

    def fit(self, train_data, batch_size, step_count, optimizer):
        self.train()
        for step in range(step_count):
            self.step(step, train_data, batch_size, optimizer)
        self.eval()

    def step(self, step, train_data, batch_size, optimizer):
        xb, yb = self.get_batch(train_data, batch_size)
        logits = self(xb)
        loss = self.get_loss(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0: print(f"Step {step}, Loss: {loss.item()}")

    def get_batch(self, data, batch_size):
        indices = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in indices])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in indices])
        return x, y

    def forward(self, token_ids):
        positions = torch.arange(token_ids.size(1))
        tok_emb = self.token_embedding(token_ids)    # (B, T, C)
        pos_emb = self.position_embedding(positions) # (T, C)
        x = tok_emb + pos_emb                        # (B, T, C)
        x = self.blocks(x)                           # (B, T, C)
        x = self.layer_norm(x)                       # (B, T, C)
        logits = self.unembedding(x)                 # (B, T, vocab_size)
        return logits

    def get_loss(self, logits, targets):
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def generate(self, token_ids, token_count):
        for _ in range(token_count):
            block_token_ids = token_ids[:, -self.block_size:]   # (B, T)
            logits = self(block_token_ids)                      # (B, T, vocab_size)
            logits = logits[:, -1, :]                           # (B, vocab_size), only the last token
            probs = F.softmax(logits, dim=-1)                   # (B, vocab_size)
            ids_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            token_ids = torch.cat((token_ids, ids_next), dim=1) # (B, T+1), appended column
        return token_ids

    def print_text(self, token_count, vocab):
        token_ids = torch.zeros((1, 1), dtype=torch.long)
        generated_token_ids = self.generate(token_ids, token_count)[0].cpu().tolist()
        print(''.join([vocab[i] for i in generated_token_ids]))


if __name__ == "__main__":
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    data_path = Path(__file__).resolve().parent.parent / 'data' / 'shakespeare.txt'
    text = data_path.read_text(encoding='utf-8')
    vocab = sorted(list(set(text)))
    data = torch.tensor([vocab.index(c) for c in text], dtype=torch.long)
    train_size = int(0.9 * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    model = GPTChar(len(vocab), block_size=8, emb_size=32, head_count=4, dropout_rate=0, layer_count=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.fit(train_data, batch_size=32, step_count=10000, optimizer=optimizer)
    model.print_text(token_count=500, vocab=vocab)
