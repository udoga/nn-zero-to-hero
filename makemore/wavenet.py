import torch
import torch.nn.functional as F
from layers import Embedding, Flatten, Linear, BatchNorm1d, Tanh, Sequential

class WaveNet:
    def __init__(self, vocab_size=27, block_size=3, emb_dim=10, hidden_dim=200, seed=42):
        self.g = torch.Generator().manual_seed(seed)
        self.container = Sequential([
            Embedding(vocab_size, emb_dim),
            Flatten(),
            Linear(emb_dim * block_size, hidden_dim, generator=self.g, bias=False),
            BatchNorm1d(hidden_dim),
            Tanh(),
            Linear(hidden_dim, vocab_size, generator=self.g)])
        for p in self.container.parameters(): p.requires_grad = True
        self.loss_history = []
        self.calibrate_weights()

    @torch.no_grad()
    def calibrate_weights(self):
        last_layer = self.container.layers[-1]
        if isinstance(last_layer, Linear): last_layer.weight *= 0.1 # make less confident

    def train(self, X_train, Y_train, max_steps=200000, batch_size=32):
        for i in range(max_steps):
            self.step(i, X_train, Y_train, batch_size, max_steps)
        self.set_training(False)

    def step(self, i, X_train, Y_train, batch_size, max_steps):
        X_batch, Y_batch = self.make_batch(X_train, Y_train, batch_size)
        logits = self.forward(X_batch)
        loss = F.cross_entropy(logits, Y_batch)
        self.backward(loss)
        self.update(lr=0.1 if i < 150000 else 0.01)
        self.track(i, max_steps, loss)

    def make_batch(self, X, Y, batch_size):
        indices = torch.randint(0, X.shape[0], (batch_size,), generator=self.g)
        return X[indices], Y[indices]

    def forward(self, x):
        return self.container(x)

    def backward(self, loss):
        for p in self.container.parameters():
            p.grad = None
        loss.backward()

    def update(self, lr):
        for p in self.container.parameters():
            p.data -= lr * p.grad

    def track(self, i, max_steps, loss):
        if i % 10000 == 0: print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        self.loss_history.append(loss.log10().item())

    def set_training(self, training):
        for layer in self.container.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = training

    @torch.no_grad()
    def get_loss(self, X, Y):
        return F.cross_entropy(self.forward(X), Y)
