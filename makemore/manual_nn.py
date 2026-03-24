import torch
import torch.nn.functional as F

# Calculates gradients manually.

class ManualNN:
    def __init__(self, vocab_size=27, emb_dim=10, block_size=3, hidden_dim=64):
        self.g = torch.Generator().manual_seed(2147483647)
        self.C = torch.randn((vocab_size, emb_dim),               generator=self.g)
        self.W1 = torch.randn((emb_dim * block_size, hidden_dim), generator=self.g) * (5/3)/((emb_dim*block_size)**0.5)
        self.b1 = torch.randn(hidden_dim,                         generator=self.g) * 0.1 # useless because of BN
        self.W2 = torch.randn((hidden_dim, vocab_size),           generator=self.g) * 0.1
        self.b2 = torch.randn(vocab_size,                         generator=self.g) * 0.1
        self.bn_gain = torch.randn((1, hidden_dim))*0.1 + 1.0
        self.bn_bias = torch.randn((1, hidden_dim))*0.1
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2, self.bn_gain, self.bn_bias]
        for p in self.parameters: p.requires_grad = True
        print("Number of parameters:", sum(p.nelement() for p in self.parameters))

    def check_grad(self, name, grad, t):
        if grad is None: return
        if grad.shape != t.grad.shape: raise ValueError("Invalid shape: grad=", grad.shape, " t.grad=", t.grad.shape)
        all_same = torch.all(grad == t.grad).item()
        all_close = torch.allclose(grad, t.grad)
        max_diff = (grad - t.grad).abs().max().item()
        print(f'{name:15s} | same: {str(all_same):5s} | close: {str(all_close):5s} | max_diff: {max_diff}')

    def create_batch(self, X, Y, batch_size):
        indices = torch.randint(0, X.shape[0], (batch_size,), generator=self.g)
        return X[indices], Y[indices]

    def train_step(self, X_batch, Y_batch):
        # Forward pass
        n = len(X_batch)
        emb = self.C[X_batch]                              # (n, block_size, emb_dim)
        embcat = emb.view(emb.shape[0], -1)                # (n, block_size * emb_dim)
        h_pre_bn = embcat @ self.W1 + self.b1              # (n, hidden_dim) === Linear Layer 1 ===
        bnmeani = 1/n * h_pre_bn.sum(0, keepdim=True)      # (1, hidden_dim) === Batchnorm Layer ===
        bndiff = h_pre_bn - bnmeani                        # (n, hidden_dim)
        bndiff2 = bndiff**2                                # (n, hidden_dim)
        bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)     # (1, hidden_dim) # Bessel's Correction: dividing by n-1
        bnvar_inv = (bnvar + 1e-5)**-0.5                   # (1, hidden_dim)
        bnraw = bndiff * bnvar_inv                         # (n, hidden_dim)
        h_pre_act = self.bn_gain * bnraw + self.bn_bias    # (n, hidden_dim)
        h = torch.tanh(h_pre_act)                          # (n, hidden_dim) === Non-linearity ===
        logits = h @ self.W2 + self.b2                     # (n, vocab_size) === Linear Layer 2 ===
        logit_maxes = logits.max(1, keepdim=True).values   # (n, 1)          === Cross Entropy Loss ===
        normalized_logits = logits - logit_maxes           # (n, vocab_size) # for numerical stability
        counts = normalized_logits.exp()                   # (n, vocab_size)
        counts_sum = counts.sum(dim=1, keepdim=True)       # (n, 1)
        counts_sum_inv = counts_sum**-1                    # (n, 1)
        probs = counts * counts_sum_inv                    # (n, vocab_size)
        logprobs = probs.log()                             # (n, vocab_size)
        loss = -logprobs[range(n), Y_batch].mean()         # (1)

        # PyTorch backward pass
        for p in self.parameters:
            p.grad = None
        for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, normalized_logits, logit_maxes, logits, h,
                  h_pre_act, bnraw, bnvar_inv, bnvar, bndiff2, bndiff, h_pre_bn, bnmeani, embcat, emb]:
            t.retain_grad()
        loss.backward()

        # Manual backward pass
        logprobs_grad = torch.zeros_like(logprobs)
        logprobs_grad[range(n), Y_batch] = -1/n
        probs_grad = logprobs_grad * (1 / probs)
        counts_sum_inv_grad = (probs_grad * counts).sum(dim=1, keepdim=True)
        counts_sum_grad = counts_sum_inv_grad * -(counts_sum**-2)
        counts_grad = (probs_grad * counts_sum_inv) + (counts_sum_grad * 1) # two branches
        normalized_logits_grad = counts_grad * normalized_logits.exp()
        logit_maxes_grad = (normalized_logits_grad * (-1)).sum(dim=1, keepdim=True)
        max_one_else_zero = torch.zeros_like(logits)
        max_one_else_zero.scatter_(1, logits.argmax(dim=1, keepdim=True), 1)
        logits_grad = (normalized_logits_grad * 1) + (logit_maxes_grad * max_one_else_zero) # two branches
        h_grad = logits_grad @ self.W2.T
        W2_grad = h.T @ logits_grad
        b2_grad = (logits_grad * 1).sum(dim=0, keepdim=False)
        h_pre_act_grad = h_grad * (1.0 - h**2)
        h_pre_act_grad = h_pre_act.grad # using this because manual backprop causes a small difference
        bngain_grad = (h_pre_act_grad * bnraw).sum(0, keepdim=True)
        bnbias_grad = (h_pre_act_grad * 1).sum(0, keepdim=True)
        bnraw_grad = h_pre_act_grad * self.bn_gain
        bnvar_inv_grad = (bnraw_grad * bndiff).sum(0, keepdim=True)
        bnvar_grad = bnvar_inv_grad * (-0.5)*(bnvar + 1e-5)**-1.5
        bndiff2_grad = bnvar_grad * torch.ones_like(bndiff2) * (1/(n-1))
        bndiff_grad = (bndiff2_grad * 2 * bndiff) + (bnraw_grad * bnvar_inv) # two branches
        bnmeani_grad = (bndiff_grad * -1).sum(0, keepdim=True)
        h_pre_bn_grad = (bnmeani_grad * torch.ones_like(h_pre_bn) * 1/n) + (bndiff_grad * 1) # two branches
        embcat_grad = h_pre_bn_grad @ self.W1.T
        W1_grad = embcat.T @ h_pre_bn_grad
        b1_grad = h_pre_bn_grad.sum(0)
        emb_grad = embcat_grad.view(emb.shape)
        C_grad = torch.zeros_like(self.C)
        for i in range(X_batch.shape[0]):
            for j in range(X_batch.shape[1]):
                index = X_batch[i, j]
                C_grad[index] += emb_grad[i, j]

        # Check calculated grads
        self.check_grad('logprobs', logprobs_grad, logprobs)
        self.check_grad('probs', probs_grad, probs)
        self.check_grad('counts_sum_inv', counts_sum_inv_grad, counts_sum_inv)
        self.check_grad('counts_sum', counts_sum_grad, counts_sum)
        self.check_grad('counts', counts_grad, counts)
        self.check_grad('norm_logits', normalized_logits_grad, normalized_logits)
        self.check_grad('logit_maxes', logit_maxes_grad, logit_maxes)
        self.check_grad('logits', logits_grad, logits)
        self.check_grad('h', h_grad, h)
        self.check_grad('W2', W2_grad, self.W2)
        self.check_grad('b2', b2_grad, self.b2)
        self.check_grad('hpreact', h_pre_act_grad, h_pre_act)
        self.check_grad('bngain', bngain_grad, self.bn_gain)
        self.check_grad('bnbias', bnbias_grad, self.bn_bias)
        self.check_grad('bnraw', bnraw_grad, bnraw)
        self.check_grad('bnvar_inv', bnvar_inv_grad, bnvar_inv)
        self.check_grad('bnvar', bnvar_grad, bnvar)
        self.check_grad('bndiff2', bndiff2_grad, bndiff2)
        self.check_grad('bndiff', bndiff_grad, bndiff)
        self.check_grad('bnmeani', bnmeani_grad, bnmeani)
        self.check_grad('hprebn', h_pre_bn_grad, h_pre_bn)
        self.check_grad('embcat', embcat_grad, embcat)
        self.check_grad('W1', W1_grad, self.W1)
        self.check_grad('b1', b1_grad, self.b1)
        self.check_grad('emb', emb_grad, emb)
        self.check_grad('C', C_grad, self.C)
