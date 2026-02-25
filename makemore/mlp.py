import random
import torch
import torch.nn.functional as F

def split_word_list(words):
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]

def create_dataset(words, chars, block_size=3):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for char in word + '.':
            target = chars.index(char)
            X.append(context)
            Y.append(target)
            context = context[1:] + [target]
    return torch.tensor(X), torch.tensor(Y)

class MLP:
    def __init__(self, vocab_size=27, embedding_dim=10, block_size=3, hidden_dim=200):
        g = torch.Generator().manual_seed(2147483647)
        self.C = torch.randn((vocab_size, embedding_dim), generator=g, requires_grad=True)
        self.W1 = torch.randn((block_size * embedding_dim, hidden_dim), generator=g, requires_grad=True)
        self.b1 = torch.randn(hidden_dim, generator=g, requires_grad=True)
        self.W2 = torch.randn((hidden_dim, vocab_size), generator=g, requires_grad=True)
        self.b2 = torch.randn(vocab_size, generator=g, requires_grad=True)
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

    def train(self, X_train, Y_train, epochs=10000, batch_size=32, lr=0.1):
        for _ in range(epochs):
            ix = torch.randint(0, X_train.shape[0], (batch_size,))
            loss = self.get_loss(X_train[ix], Y_train[ix])
            loss.backward()
            self.update_params(lr)

    def forward(self, X):
        emb = self.C[X]
        h = torch.tanh(emb.view(X.shape[0], -1) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def get_loss(self, X, Y):
        return F.cross_entropy(self.forward(X), Y)

    def update_params(self, lr):
        for p in self.parameters:
            p.data += -lr * p.grad # type: ignore
            p.grad = None

    def sample_name(self, chars):
        name = ''
        context = [0] * 3
        while True:
            logits = self.forward(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            index = torch.multinomial(probs, num_samples=1).item()
            name += chars[index]
            if index == 0: break
            context = context[1:] + [index]
        return name

words = open('names.txt', 'r').read().splitlines()
chars = ['.'] + sorted(set(''.join(words)))
train_words, dev_words, test_words = split_word_list(words)
X_train, Y_train = create_dataset(train_words, chars)
X_dev, Y_dev = create_dataset(dev_words, chars)
X_test, Y_test = create_dataset(test_words, chars)
mlp = MLP(block_size=3)
mlp.train(X_train, Y_train)
print("Train Loss:", mlp.get_loss(X_train, Y_train).item())
print("Dev Loss:", mlp.get_loss(X_dev, Y_dev).item())
print("Sample Names:", [mlp.sample_name(chars) for _ in range(10)])
