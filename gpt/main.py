import torch
from bigram import BigramLanguageModel

def split_dataset(data, train_rate):
    n = int(train_rate * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(data, block_size, batch_size):
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    return x, y

torch.manual_seed(1337)

text = open('../data/shakespeare.txt', 'r', encoding='utf-8').read()
vocab = sorted(list(set(text)))
encode = lambda s: [vocab.index(c) for c in s]
decode = lambda l: ''.join([vocab[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
train_data, val_data = split_dataset(data, 0.9)

model = BigramLanguageModel(len(vocab))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32
block_size = 8

for step in range(10000):
    xb, yb = get_batch(train_data, block_size, batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 1000 == 0: print(f"Step {step}, Loss: {loss.item()}")

batch_token_ids = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(batch_token_ids, max_new_tokens=500)[0].tolist()))
