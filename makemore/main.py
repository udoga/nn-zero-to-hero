import torch
import matplotlib.pyplot as plt

def get_word_chars(word):
    return ['.'] + list(word) + ['.']

def get_all_chars(words):
    return ['.'] + sorted(list(set(''.join(words))))

def make_bigram_dict(words):
    result = {}
    for w in words:
        chars = get_word_chars(w)
        for c1, c2 in zip(chars, chars[1:]):
            result[(c1, c2)] = result.get((c1, c2), 0) + 1
    return sorted(result.items(), key=lambda pair: -pair[1])

def make_bigram_matrix(words, chars, stoi):
    matrix = torch.zeros((len(chars), len(chars)), dtype=torch.int32)
    for w in words:
        chars = get_word_chars(w)
        for c1, c2 in zip(chars, chars[1:]):
            matrix[stoi[c1], stoi[c2]] += 1
    return matrix

def show_bigram_matrix(matrix, itos):
    plt.figure(figsize=(16,16))
    plt.imshow(matrix, cmap='Blues')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            str = itos[i] + itos[j]
            plt.text(j, i, str, ha='center', va='bottom', color='gray')
            plt.text(j, i, round(matrix[i,j].item(), 2), ha='center', va='top', color='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def sample_name(generator, prob_matrix, itos):
    name = ''
    index = 0
    while True:
        row = prob_matrix[index, :]
        index = int(torch.multinomial(row, num_samples=1, replacement=True, generator=generator).item())
        name += itos[index]
        if index == 0: break
    return name

def sample_names(prob_matrix, itos, num_samples):
    generator = torch.Generator().manual_seed(2147483647)
    for _ in range(num_samples):
        print(sample_name(generator, prob_matrix, itos))

def find_avg_negative_log_likelihood(prob_matrix, words, stoi):
    total_nll = 0.0
    total_count = 0
    for w in words:
        chars = get_word_chars(w)
        for c1, c2 in zip(chars, chars[1:]):
            i1 = stoi[c1]
            i2 = stoi[c2]
            prob = prob_matrix[i1, i2]
            total_nll += -torch.log(prob)
            total_count += 1
    return total_nll / total_count

words = open("names.txt", 'r').read().splitlines()
chars = get_all_chars(words)
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
count_matrix = make_bigram_matrix(words, chars, stoi)
prob_matrix = count_matrix.float() / count_matrix.sum(dim=1, keepdim=True)
print(find_avg_negative_log_likelihood(prob_matrix, words, stoi))
