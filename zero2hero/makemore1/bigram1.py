from collections import defaultdict
import random

words = open('names.txt', 'r').read().splitlines()
f = defaultdict(str)
for word in words:
    word = "^" + word + "$"
    for a, b in zip(word, word[1:]):
        f[a] += b

for _ in range(100):
    current_word = '^'
    while current_word[-1] != '$':
        current_word += random.choice(f[current_word[-1:]])
    current_word = current_word[1:-1]
    print(current_word, '***' if current_word in words else '')
