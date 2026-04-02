import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, batch_token_ids, targets=None): # (B, T)
        logits = self.token_embedding_table(batch_token_ids) # (B, T, C)
        loss = None if targets is None else self.get_loss(logits, targets)
        return logits, loss

    def get_loss(self, logits, targets):
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        return F.cross_entropy(logits, targets)

    def generate(self, batch_token_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(batch_token_ids)
            logits = logits[:, -1, :] # focus only on the last time step, becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            batch_token_ids = torch.cat((batch_token_ids, idx_next), dim=1) # appending column, becomes (B, T+1)
        return batch_token_ids
