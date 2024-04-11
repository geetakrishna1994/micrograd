import random
from micrograd import Value


class Neuron:
    def __init__(self, nin, nonlin, use_bias=True):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.use_bias = use_bias
        if use_bias:
            self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        out = sum([wi * xi for wi, xi in zip(self.weights, x)])
        if self.use_bias:
            out += self.b
        if self.nonlin:
            out = out.relu()
        return out

    def parameters(self):
        _parameters = self.weights.copy()
        if self.use_bias:
            _parameters.append(self.b)
        return _parameters


class Layer:
    def __init__(self, nin, nout, nonlin, nbias=True):
        self.neurons = [Neuron(nin, nonlin, nbias) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        parameters = []
        for n in self.neurons:
            parameters.extend(n.parameters())
        return parameters


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin=i != len(nouts)-1)
                       for i in range(len(nouts))]

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out[0]

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.
