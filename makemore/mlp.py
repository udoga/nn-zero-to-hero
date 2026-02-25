import torch
import torch.nn.functional as F

def create_dataset(block_size, chars, words):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for char in word + '.':
            ix = chars.index(char)
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

words = open('names.txt', 'r').read().splitlines()
chars = ['.'] + sorted(set(''.join(words)))
X, Y = create_dataset(3, chars, words)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g, requires_grad=True)
W1 = torch.randn((6, 100), generator=g, requires_grad=True)
b1 = torch.randn(100, generator=g, requires_grad=True)
W2 = torch.randn((100, 27), generator=g, requires_grad=True)
b2 = torch.randn(27, generator=g, requires_grad=True)
parameters = [C, W1, b1, W2, b2]

for i in range(5000):
    ix = torch.randint(0, X.shape[0], (32,))   # minibatch (32)
    emb = C[X[ix]]                             # (32, 3, 2)
    h = torch.tanh(emb.view(32, 6) @ W1 + b1)  # (32, 100)
    logits = h @ W2 + b2                       # (32, 27)
    loss = F.cross_entropy(logits, Y[ix])
    print(loss.item())
    loss.backward()
    for p in parameters:
        p.data += -0.01 * p.grad # type: ignore
        p.grad = None
