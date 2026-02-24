import torch
import matplotlib.pyplot as plt

def get_word_chars(word):
    return ['.'] + list(word) + ['.']

def get_all_chars(words):
    return ['.'] + sorted(list(set(''.join(words))))

def get_all_bigrams(words):
    bigrams = []
    for w in words:
        word_chars = get_word_chars(w)
        for c1, c2 in zip(word_chars, word_chars[1:]):
            bigrams.append((c1, c2))
    return bigrams

def get_bigram_counts(bigrams):
    result = {}
    for bigram in bigrams:
        result[bigram] = result.get(bigram, 0) + 1
    return sorted(result.items(), key=lambda pair: -pair[1])

def get_bigram_matrix(bigrams, all_chars):
    matrix = torch.zeros((len(all_chars), len(all_chars)), dtype=torch.int32)
    for c1, c2 in bigrams:
        matrix[all_chars.index(c1), all_chars.index(c2)] += 1
    return matrix

def show_bigram_matrix(matrix, all_chars):
    plt.figure(figsize=(16,16))
    plt.imshow(matrix, cmap='Blues')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, all_chars[i] + all_chars[j], ha='center', va='bottom', color='gray')
            plt.text(j, i, round(matrix[i,j].item(), 2), ha='center', va='top', color='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_sample_name(generator, prob_matrix, all_chars):
    name = ''
    index = 0
    while True:
        row = prob_matrix[index, :]
        index = int(torch.multinomial(row, num_samples=1, replacement=True, generator=generator).item())
        name += all_chars[index]
        if index == 0: break
    return name

def get_sample_names(prob_matrix, all_chars, num_samples):
    generator = torch.Generator().manual_seed(2147483647)
    names = []
    for _ in range(num_samples):
        names.append(get_sample_name(generator, prob_matrix, all_chars))
    return names

def get_avg_neg_log_likelihood(prob_matrix, words, all_chars):
    total_nll = 0.0
    total_count = 0
    for w in words:
        word_chars = get_word_chars(w)
        for c1, c2 in zip(word_chars, word_chars[1:]):
            prob = prob_matrix[all_chars.index(c1), all_chars.index(c2)]
            total_nll += -torch.log(prob)
            total_count += 1
    return total_nll / total_count

words = open("names.txt", 'r').read().splitlines()
all_chars = get_all_chars(words)
bigrams = get_all_bigrams(words)
bigram_counts = get_bigram_counts(bigrams)
count_matrix = get_bigram_matrix(bigrams, all_chars) + 1 # smoothing
prob_matrix = count_matrix.float() / count_matrix.sum(dim=1, keepdim=True)
# show_bigram_matrix(prob_matrix, all_chars)
# print(get_sample_names(prob_matrix, all_chars, 10))
print(get_avg_neg_log_likelihood(prob_matrix, ["andrejq"], all_chars))
