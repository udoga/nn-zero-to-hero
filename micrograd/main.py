from micrograd.value import Value
from micrograd.nn import MLP
from micrograd.helpers import draw_dot

def make_node_graph():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    draw_dot(L).view()

def make_neuron_graph():
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    e = (2*n).exp()
    o = (e - 1) / (e + 1)
    o.label = 'o'
    o.backward()
    draw_dot(o).view()

def make_torch_graph():
    import torch
    x1 = torch.Tensor([2.0]).double()               ; x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double()               ; x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double()              ; w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double()               ; w2.requires_grad = True
    b = torch.Tensor([6.8813735870195432]).double() ; b.requires_grad = True
    n = x1*w1 + x2*w2 + b
    o = torch.tanh(n)
    print(o)
    o.backward()
    print('---', x1.grad.item(), w1.grad.item(), x2.grad.item(), w2.grad.item())  # type: ignore[union-attr]

def make_neural_network():
    mlp = MLP(3, [4, 4, 1])
    inputs = [[2.0, 3.0, -1.0],
              [3.0, -1.0, 0.5],
              [0.5, 1.0, 1.0],
              [1.0, 1.0, -1.0]]
    targets = [1.0, -1.0, -1.0, 1.0]
    step_size = 0.01
    for _ in range(100):
        predictions = [mlp(x)[0] for x in inputs]
        loss = sum(((pred - target)**2 for pred, target in zip(predictions, targets)), Value(0.0))
        for p in mlp.parameters():
            p.grad = 0.0
        loss.backward()
        for p in mlp.parameters():
            p.data -= step_size * p.grad
        print("Loss:", round(loss.data, 4))
    print("Predictions:", [mlp(x)[0] for x in inputs])

make_neural_network()
