import torch
from bigram import BigramLanguageModel

def split_dataset(data, train_rate):
    n = int(train_rate * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(data, block_size, batch_size, generator):
    indices = torch.randint(len(data) - block_size, (batch_size,), device=data.device, generator=generator)
    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    return x, y

batch_size = 32
block_size = 8
learning_rate = 1e-3
train_rate = 0.9
num_steps = 10000
max_new_tokens = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

text = open('../data/shakespeare.txt', 'r', encoding='utf-8').read()
vocab = sorted(list(set(text)))
encode = lambda s: [vocab.index(c) for c in s]
decode = lambda l: ''.join([vocab[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long, device=device)
train_data, val_data = split_dataset(data, train_rate)
generator = torch.Generator().manual_seed(42)
model = BigramLanguageModel(len(vocab)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(num_steps):
    xb, yb = get_batch(train_data, block_size, batch_size, generator)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 1000 == 0: print(f"Step {step}, Loss: {loss.item()}")

batch_token_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(batch_token_ids, max_new_tokens=max_new_tokens)[0].cpu().tolist()))
