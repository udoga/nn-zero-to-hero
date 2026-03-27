import torch
import torch.nn.functional as F
from layers import Linear, BatchNorm1d, Tanh

class DeepNN:
    def __init__(self, vocab_size=27, block_size=3, n_embd=10, n_hidden=100, seed=2147483647):
        self.g = torch.Generator().manual_seed(seed)
        self.C = torch.randn((vocab_size, n_embd), generator=self.g)
        self.layers = [
            Linear(n_embd * block_size, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden, bias=False, generator=self.g), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, vocab_size, bias=False, generator=self.g), BatchNorm1d(vocab_size)]
        self.calibrate_weights()
        self.parameters = [self.C] + [p for layer in self.layers for p in layer.parameters()]
        for p in self.parameters: p.requires_grad = True

    @torch.no_grad()
    def calibrate_weights(self):
        self.layers[-1].gamma *= 0.1
        for layer in self.layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= 1.0 #5/3

    def train(self, X_train, Y_train, max_steps=200000, batch_size=32):
        for i in range(max_steps):
            indices = torch.randint(0, X_train.shape[0], (batch_size,), generator=self.g)
            logits = self.forward(X_train[indices])
            loss = F.cross_entropy(logits, Y_train[indices])
            loss.backward()
            self.update_params(lr=0.1 if i < 150000 else 0.01)
            if i % 10000 == 0: print(f"Step {i}, Loss: {loss.item()}")

    def forward(self, X):
        embeddings = self.C[X]
        x = embeddings.view(embeddings.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x

    def update_params(self, lr):
        for p in self.parameters:
            p.data -= lr * p.grad # type: ignore
            p.grad = None

    @torch.no_grad()
    def get_loss(self, X, Y):
        for layer in self.layers: layer.training = False
        return F.cross_entropy(self.forward(X), Y)
