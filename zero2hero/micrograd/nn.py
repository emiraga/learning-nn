from micrograd import Value, draw_dot
import math
import random


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        out = sum((w * xi for w, xi in zip(self.w, x)), self.b)
        return out.tanh()
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# Simple examples of using MLP
n = Neuron(2)
# print(n([0.1, 0.3]))

l = Layer(2, 3)
# print(l([0.1, 0.3]))

m = MLP(2, [4, 3])
# print(m([0.1, 0.3]))

draw_dot(m([0.1, 0.3])[0]).render('graph.gv', view=False)

# Using training with examples
m = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

for _ in range(500):
    ypred = [m(x) for x in xs]
    loss = sum((ypred[i][0] - y) ** 2 for i, y in enumerate(ys))
    print(loss)
    for p in m.parameters():
        p.grad = 0.0
    loss.backward()
    for p in m.parameters():
        p.data -= 0.1 * p.grad

print(ypred)
print(ys)
