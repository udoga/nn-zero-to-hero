import random
import torch
from mlp import MLP
from deep_nn import DeepNN
from torch.nn import functional as F

def split_words(words):
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

def sample_name(model, chars):
    name = ''
    context = [0] * 3
    while True:
        logits = model.forward(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        index = torch.multinomial(probs, num_samples=1).item()
        name += chars[index]
        if index == 0: break
        context = context[1:] + [index]
    return name

words = open('names.txt', 'r').read().splitlines()
chars = ['.'] + sorted(set(''.join(words)))
train_words, dev_words, test_words = split_words(words)
X_train, Y_train = create_dataset(train_words, chars)
X_dev, Y_dev = create_dataset(dev_words, chars)
X_test, Y_test = create_dataset(test_words, chars)
model = DeepNN() # MLP()
model.train(X_train, Y_train)
print("Train Loss:", model.get_loss(X_train, Y_train).item())
print("Dev Loss:", model.get_loss(X_dev, Y_dev).item())
print("Sample Names:", [sample_name(model, chars) for _ in range(10)])
