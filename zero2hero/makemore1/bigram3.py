from collections import defaultdict
import matplotlib.pyplot as plt
import random
import string
import torch

allChars = '.' + string.ascii_lowercase
s2i = {ch: i for i, ch in enumerate(allChars)}
i2s = {i: ch for i, ch in enumerate(allChars)}
n = len(allChars)
# N = torch.zeros((n, n), dtype=torch.int32)
words = open('names.txt', 'r').read().splitlines()

# Create a training set of bigrams
xs, ys = [], []
for word in words[:1]:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = s2i[ch1]
        ix2 = s2i[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs, dtype=torch.int64)
ys = torch.tensor(ys, dtype=torch.int64)

print(xs, ys)

import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=n).float()
W = torch.randn((n, n), requires_grad=True)
print(xenc @ W)
