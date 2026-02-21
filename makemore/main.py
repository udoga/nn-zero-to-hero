import torch
import matplotlib.pyplot as plt

def make_bigram_dict(words):
    b = {}
    for w in words:
        chars = ['<S>'] + list(w) + ['<E>']
        for c1, c2 in zip(chars, chars[1:]):
            b[(c1, c2)] = b.get((c1, c2), 0) + 1
    return sorted(b.items(), key=lambda pair: -pair[1])

def make_bigram_matrix(words):
    N = torch.zeros((28, 28), dtype=torch.int32)
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i for i, s in enumerate(chars)}
    stoi['<S>'] = 26
    stoi['<E>'] = 27
    for w in words:
        chars = ['<S>'] + list(w) + ['<E>']
        for c1, c2 in zip(chars, chars[1:]):
            N[stoi[c1], stoi[c2]] += 1
    return N

words = open("names.txt", 'r').read().splitlines()
plt.imshow(make_bigram_matrix(words))
plt.show()
