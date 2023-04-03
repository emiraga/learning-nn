from collections import defaultdict
import random
import string
import torch

allChars = '^$' + string.ascii_lowercase
stoi = {ch: i for i, ch in enumerate(allChars)}
itos = {i: ch for i, ch in enumerate(allChars)}
N = torch.zeros((28, 28), dtype=torch.int32)
words = open('names.txt', 'r').read().splitlines()
for word in words:
    chs = ['^'] + list(word) + ['$']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

print(N)
