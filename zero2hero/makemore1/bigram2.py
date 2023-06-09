from collections import defaultdict
import matplotlib.pyplot as plt
import random
import string
import torch

allChars = '.' + string.ascii_lowercase
s2i = {ch: i for i, ch in enumerate(allChars)}
i2s = {i: ch for i, ch in enumerate(allChars)}
N = torch.zeros((len(allChars), len(allChars)), dtype=torch.int32)
words = open('names.txt', 'r').read().splitlines()
for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[s2i[ch1], s2i[ch2]] += 1

# plt.figure(figsize=(10,10))
# plt.imshow(N, cmap='Blues')
# for i in range(len(allChars)):
#     for j in range(len(allChars)):
#         chstr = i2s[i] + i2s[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color='gray', fontdict={"size": 5})
#         plt.text(j, i, str(N[i, j].item()), ha="center", va="top", color='gray', fontdict={"size": 6})
# plt.axis('off')
# plt.show()

# +1 for model smoothing
P = (N+1).float()
P /= P.sum(1, keepdim=True)
print(P)

g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        # p = torch.ones(len(allChars))
        # p = N[ix].float()
        # p = p / p.sum()
        p = P[ix]
        ix = int(torch.multinomial(p, num_samples=1, replacement=True, generator=g).item())
        if ix == 0:
            break
        else:
            out.append(i2s[ix])
    print(''.join(out))

neg_log_likelihood = 0.0
n = 0
for word in words[:3]:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = s2i[ch1]
        ix2 = s2i[ch2]
        prob = P[ix1, ix2]
        neg_log_likelihood -= torch.log(prob)
        n += 1
        print(f'{ch1},{ch2} = {prob:.4f}')
# Average NLL
print(neg_log_likelihood / n)
