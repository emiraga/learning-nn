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

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

n = Neuron(2)
print(n([0.1, 0.3]))

l = Layer(2, 3)
print(l([0.1, 0.3]))

m = MLP(2, [4, 3])
print(m([0.1, 0.3]))

draw_dot(m([0.1, 0.3])[0]).render('graph.gv', view=False)
