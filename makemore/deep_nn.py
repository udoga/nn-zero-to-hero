import torch
import torch.nn.functional as F

class Linear:
    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_avg = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        x_avg = x.mean(0, keepdim=True) if self.training else self.running_avg
        x_var = x.var(0, keepdim=True) if self.training else self.running_var
        x_normalized = (x - x_avg) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_normalized + self.beta
        if self.training: self.update_buffers(x_avg, x_var)
        return self.out

    @torch.no_grad()
    def update_buffers(self, x_avg, x_var):
        self.running_avg = (1 - self.momentum) * self.running_avg + self.momentum * x_avg
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

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

    def train(self, X_train, Y_train, epochs=200000, batch_size=32):
        for i in range(epochs):
            indices = torch.randint(0, X_train.shape[0], (batch_size,), generator=self.g)
            logits = self.forward(X_train[indices])
            loss = F.cross_entropy(logits, Y_train[indices])
            loss.backward()
            self.update_params(lr=0.1 if i < 150000 else 0.01)
            if i % 10000 == 0: print(f"Epoch {i}, Loss: {loss.item()}")

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
