class Value:
    def __init__(self, data, parents={}, op=None):
        self.data = data
        self._parents = parents
        self._op = op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data + other.data, {self, other}, '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data - other.data, {self, other}, '-')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad * -1
        out._backward = _backward
        return out

    def __neg__(self):
        return Value(-1 * self.data)

    def __rsub__(self, other):
        return other + -1*self

    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data * other.data, {self, other}, '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data / other.data, {self, other}, '/')

        def _backward():
            self.grad += out.grad / other.data
            other.grad = out.grad * self.data * -1 / (other.data)**2
        out._backward = _backward
        return out

    def relu(self):
        out = Value(
            self.data if self.data > 0 else 0,
            {self},
            'ReLU'
        )

        def _backward():
            self.grad += out.grad if self.data > 0 else 0
        out._backward = _backward
        return out

    def backward(self):

        def create_graph(node, graph, visited):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    create_graph(parent, graph, visited)
                graph.append(node)
        graph = []
        visited = set()
        create_graph(self, graph, visited)
        self.grad = 1
        for node in reversed(graph):
            node._backward()


if __name__ == "__main__":
    a = Value(1)
    b = Value(-2)
    c = a + b
    c.backward()

    assert c.data == a.data + b.data
    assert a.grad == 1
    assert b.grad == 1

    d = a - b
    a.grad = 0
    b.grad = 0
    d.backward()

    assert d.data == a.data - b.data
    assert a.grad == 1
    assert b.grad == -1

    e = a*b
    a.grad = 0
    b.grad = 0
    e.backward()

    assert e.data == a.data * b.data
    assert a.grad == b.data
    assert b.grad == a.data

    f = c.relu()
    a.grad = 0
    b.grad = 0
    c.grad = 0
    f.backward()

    assert f.data == 0
    assert a.grad == 0
    assert b.grad == 0
