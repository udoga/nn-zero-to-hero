import torch
from pathlib import Path
from gpt import GPT

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

data_path = Path(__file__).resolve().parent.parent / 'data' / 'shakespeare.txt'
text = data_path.read_text(encoding='utf-8')
vocab = sorted(list(set(text)))
data = torch.tensor([vocab.index(c) for c in text], dtype=torch.long)
train_size = int(0.9 * len(data))
train_data, val_data = data[:train_size], data[train_size:]
model = GPT(len(vocab), block_size=8, emb_size=32, head_count=4, dropout_rate=0, layer_count=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.fit(train_data, batch_size=32, step_count=10000, optimizer=optimizer)
model.print_text(token_count=500, vocab=vocab)
