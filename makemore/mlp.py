import torch
import torch.nn.functional as F

class MLP:
    def __init__(self, vocab_size=27, embedding_dim=10, block_size=3, hidden_dim=200):
        self.g = torch.Generator().manual_seed(2147483647)
        self.C = torch.randn((vocab_size, embedding_dim),               generator=self.g)
        self.W1 = torch.randn((block_size * embedding_dim, hidden_dim), generator=self.g) * (5/3) / ((block_size * embedding_dim)**0.5)
        self.b1 = torch.randn(hidden_dim,                               generator=self.g) * 0.01
        self.W2 = torch.randn((hidden_dim, vocab_size),                 generator=self.g) * 0.01
        self.b2 = torch.randn(vocab_size,                               generator=self.g) * 0
        self.bn_gain = torch.ones((1, hidden_dim))
        self.bn_bias = torch.zeros((1, hidden_dim))
        self.bn_avg = None
        self.bn_std = None
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2, self.bn_gain, self.bn_bias]
        for p in self.parameters: p.requires_grad = True

    def train(self, X_train, Y_train, epochs=200000, batch_size=32):
        for i in range(epochs):
            indices = torch.randint(0, X_train.shape[0], (batch_size,), generator=self.g)
            logits = self.forward(X_train[indices])
            loss = F.cross_entropy(logits, Y_train[indices])
            loss.backward()
            self.update_params(lr = 0.1 if i < 100000 else 0.01)
            if i % 10000 == 0: print(f"Epoch {i}, Loss: {self.get_loss(X_train, Y_train).item()}")
        self.calibrate_bn_stats(X_train)

    @torch.no_grad()
    def calibrate_bn_stats(self, X_train):
        emb = self.C[X_train]
        embcat = emb.view(emb.shape[0], -1)
        h_pre = embcat @ self.W1 + self.b1
        self.bn_avg = h_pre.mean(0, keepdim=True)
        self.bn_std = h_pre.std(0, keepdim=True)

    def forward(self, X):
        emb = self.C[X] # (batch_size, block_size, embedding_dim)
        embcat = emb.view(X.shape[0], -1) # (batch_size, block_size * embedding_dim)
        h_pre = embcat @ self.W1 + self.b1 # (batch_size, hidden_dim)
        bn_avg = h_pre.mean(dim=0, keepdim=True) if self.bn_avg is None else self.bn_avg # (1, hidden_dim)
        bn_std = h_pre.std(dim=0, keepdim=True)+1e-5 if self.bn_std is None else self.bn_std # (1, hidden_dim)
        h_normalized = self.bn_gain * (h_pre - bn_avg) / bn_std + self.bn_bias # (batch_size, hidden_dim)
        h = torch.tanh(h_normalized) # (batch_size, hidden_dim)
        logits = h @ self.W2 + self.b2 # (batch_size, vocab_size)
        return logits

    def update_params(self, lr):
        for p in self.parameters:
            p.data += -lr * p.grad # type: ignore
            p.grad = None

    @torch.no_grad()
    def get_loss(self, X, Y):
        return F.cross_entropy(self.forward(X), Y)
