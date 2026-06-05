import torch

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
        dim = 0 if x.ndim == 2 else (0, 1)
        x_avg = x.mean(dim, keepdim=True) if self.training else self.running_avg
        x_var = x.var(dim, keepdim=True) if self.training else self.running_var
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

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]

class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
