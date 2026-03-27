import random
import torch
from mlp import MLP
from deep_nn import DeepNN
from manual_nn import ManualNN
from wavenet import WaveNet
from torch.nn import functional as F
import matplotlib.pyplot as plt

def split_dataset(words, train_rate, dev_rate):
    random.seed(42)
    random.shuffle(words)
    n1 = int(train_rate * len(words))
    n2 = int((train_rate + dev_rate) * len(words))
    return words[:n1], words[n1:n2], words[n2:]

def create_samples(words, vocab, block_size):
    X, Y = [], []
    for word in words:
        add_samples(word, vocab, block_size, X, Y)
    return torch.tensor(X), torch.tensor(Y)

def add_samples(word, vocab, block_size, X, Y):
    context = [0] * block_size
    for char in word + '.':
        target = vocab.index(char)
        X.append(context)
        Y.append(target)
        context = context[1:] + [target]

def print_samples(X, Y, vocab, count=10):
    print("Samples:")
    for x,y in zip(X[:count], Y[:count]):
        print(''.join(vocab[i.item()] for i in x), '-->', vocab[y.item()])
    print("...\n")

def predict_next(model, context):
    logits = model.forward(torch.tensor([context]))
    probs = F.softmax(logits, dim=1)
    index = torch.multinomial(probs, num_samples=1).item()
    return index

def generate_name(model, vocab, block_size):
    name = '.' * block_size
    while True:
        context = [vocab.index(c) for c in name[-block_size:]]
        index = predict_next(model, context)
        name += vocab[index]
        if index == 0: break
    return name[block_size:]

def generate_names(model, vocab, block_size, count=10):
    return [generate_name(model, vocab, block_size) for _ in range(count)]

def plot_loss_history(loss_history):
    if loss_history is None: return
    plt.plot(torch.tensor(loss_history).view(-1, 1000).mean(1))
    plt.xlabel('Step')
    plt.ylabel('Log10(Loss)')
    plt.title('Training Loss History')
    plt.show()

words = open('names.txt', 'r').read().splitlines()
vocab = ['.'] + sorted(set(''.join(words)))
block_size = 3
train_words, dev_words, test_words = split_dataset(words, 0.8, 0.1)
X_train, Y_train = create_samples(train_words, vocab, block_size)
X_dev, Y_dev = create_samples(dev_words, vocab, block_size)
X_test, Y_test = create_samples(test_words, vocab, block_size)
print_samples(X_train, Y_train, vocab)

model = WaveNet(block_size=block_size) # or MLP(), DeepNN(), ManualNN()
model.train(X_train, Y_train, max_steps=30000)
print("Train Loss:", model.get_loss(X_train, Y_train).item())
print("Dev Loss:", model.get_loss(X_dev, Y_dev).item())
print("Sample Names:", generate_names(model, vocab, block_size, count=10))
plot_loss_history(model.loss_history)
