from collections import defaultdict
import random
import torch

words = open('names.txt', 'r').read().splitlines()
for word in words:
    pass

N = torch.zeros((28, 28), dtype=torch.int32)
print(N)
