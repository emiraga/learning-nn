import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

def f(x):
    return 3*x**2 - 4*x + 5

# xs = np.arange(-5, 5, 0.25)
# ys = f(xs)
# plt.plot(xs, ys)
# plt.show()

class Value:
    def __init__(self, data, _children=(), _op='') -> None:
        self.data = float(data)
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
    
    def __repr__(self) -> str:
        return f"Value({self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')


def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot


if __name__ == "__main__":
    a = Value(3.0)
    b = Value(-4.0)
    c = Value(5.0)
    d = a * b + c
    dot = draw_dot(d)
    dot.render('graph.gv', view=False)
