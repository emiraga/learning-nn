from collections import defaultdict
import matplotlib.pyplot as plt
import random
import string
import torch

allChars = '.' + string.ascii_lowercase
s2i = {ch: i for i, ch in enumerate(allChars)}
i2s = {i: ch for i, ch in enumerate(allChars)}
N = torch.zeros((28, 28), dtype=torch.int32)
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


g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        p = N[ix].float()
        ix = int(torch.multinomial(p, num_samples=1, replacement=True, generator=g).item())
        if ix == 0:
            break
        else:
            out.append(i2s[ix])
    print(''.join(out))
