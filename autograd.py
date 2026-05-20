"""
Day 4 — Backpropagation Engine (Autograd)
Phase 1 — Foundations · 60-minute timed session

RULES:
  - Pure Python (NumPy optional for Stage 4 batch ops).
  - No PyTorch. No sklearn.
  - Read all 4 stages in README.md first.
  - Design architecture in comments BEFORE writing code.
  - print() is your only debugger.

ARCHITECTURE NOTES (fill in during your 5-minute design phase):
  -
  -
  -

CHAIN RULE PATTERN (memorize):
  Every op creates `out = Value(...)`, captures local grads in a closure,
  and `+=` them onto each parent's .grad inside `_backward`.

      out = Value(a.data + b.data, (a, b), "+")
      def _backward():
          a.grad += 1.0 * out.grad
          b.grad += 1.0 * out.grad
      out._backward = _backward

  backward(self):
      1. topo sort the DAG from self
      2. self.grad = 1.0
      3. for node in reversed(topo): node._backward()

  zero_grad(self):
      walk the DAG from self and reset every node's .grad to 0.0

  Numerical gradient check:
      (f(x + eps) - f(x - eps)) / (2 * eps)   should match .grad to < 1e-5
"""

from __future__ import annotations
from typing import Callable, Iterable, List, Set, Tuple, Union
import math
import random


Number = Union[int, float]


# -----------------------------------------------------------------------------
# STAGE 1 — VALUE NODE + BASIC OPS
# -----------------------------------------------------------------------------

class Value:
    """A scalar autograd node — like a tiny PyTorch tensor."""

    def __init__(self, data: Number, _children: Tuple["Value", ...] = (),
                 _op: str = "") -> None:
        # TODO: store self.data (float), self.grad = 0.0
        # TODO: store self._backward = lambda: None
        # TODO: store self._prev = set(_children), self._op = _op
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data},grad={self.grad})"

    # --- Stage 1 ops ---------------------------------------------------------

    def __add__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other),"+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other),"*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = _backward
        return out

    # Right-side variants so `2 + a` and `2 * a` also work.
    def __radd__(self, other: Union["Value", Number]) -> "Value":
        return self + other

    def __rmul__(self, other: Union["Value", Number]) -> "Value":
        return self * other

    # -------------------------------------------------------------------------
    # STAGE 2 — BACKWARD PASS
    # -------------------------------------------------------------------------

    def __neg__(self) -> "Value":
        """-self — build from __mul__ with -1."""
        return self * -1

    def __sub__(self, other: Union["Value", Number]) -> "Value":
        """self - other — build from __add__ and __neg__."""
        return self + (-other)

    def __rsub__(self, other: Union["Value", Number]) -> "Value":
        return Value(other) + (-self)

    def _topological_sort(self) -> List["Value"]:
        """DFS — append a node only AFTER recursing into all children."""
        topo = []
        visited = set()
        def visit(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    visit(child)
                topo.append(node)
        visit(self)
        return topo


    def backward(self) -> None:
        """Reverse-mode autodiff. Sets grads on every reachable node."""
        topo = self._topological_sort()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        """Walk the DAG and reset every reachable node's .grad to 0."""
        for node in self._topological_sort():
            node.grad = 0.0

    # -------------------------------------------------------------------------
    # STAGE 3 — FULL OPERATION LIBRARY
    # -------------------------------------------------------------------------

    def __pow__(self, n: Number) -> "Value":
        """self ** n for scalar n. Grad: n * self.data ** (n - 1)."""
        out = Value(self.data ** n,(self,), f"**{n}")
        def _backward():
            self.grad += n * (self.data ** (n-1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other: Union["Value", Number]) -> "Value":
        """self / other — build from __mul__ and __pow__(-1)."""
        return self*other ** -1 

    def __rtruediv__(self, other: Union["Value", Number]) -> "Value":
        return other * self ** -1

    def relu(self) -> "Value":
        """max(0, x). Grad flows where data > 0, zero elsewhere."""
        out = Value(max(0,self.data), (self,),"relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> "Value":
        """e^x. Grad: out.data * out.grad."""
        out = Value(math.exp(self.data),(self,),"exp")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self) -> "Value":
        """ln(x). Grad: (1/x) * out.grad. Assumes x > 0."""
        out = Value(math.log(self.data),(self,),"log")
        def _backward():
            self.grad += (1/self.data)*out.grad
        out._backward = _backward
        return out


# Convenience wrappers — implement once Value is up.
def sigmoid(x: Value) -> Value:
    """Compose sigmoid from exp and division. No special-case backward."""
    return Value(1.0)/(Value(1.0)+(-x).exp())

def tanh(x: Value) -> Value:
    """Compose tanh from exp. (exp(2x) - 1) / (exp(2x) + 1)."""
    e2x = (Value(2.0)*x).exp()
    return (e2x - Value(1.0))/(e2x+Value(1.0))


# -----------------------------------------------------------------------------
# STAGE 4 — TRAIN A NETWORK WITH YOUR ENGINE (bonus)
# -----------------------------------------------------------------------------

class Neuron:
    """A single neuron — list of Value weights + bias."""

    def __init__(self, n_in: int, activation: str = "relu", seed: int = 0):
        # TODO: self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        # TODO: self.b = Value(0.0)
        # TODO: self.activation = activation  # "relu" | "sigmoid" | "tanh" | "linear"
        raise NotImplementedError

    def __call__(self, x: List[Union[Value, Number]]) -> Value:
        """Return activation(sum(wi * xi) + b)."""
        raise NotImplementedError

    def parameters(self) -> List[Value]:
        raise NotImplementedError


class Layer:
    """A list of Neurons sharing the same input."""

    def __init__(self, n_in: int, n_out: int, activation: str = "relu"):
        raise NotImplementedError

    def __call__(self, x: List[Union[Value, Number]]) -> List[Value]:
        raise NotImplementedError

    def parameters(self) -> List[Value]:
        raise NotImplementedError


class MLP:
    """Multi-layer perceptron — list of Layers."""

    def __init__(self, n_in: int, layer_dims: List[int],
                 activations: List[str]):
        # TODO: build self.layers using Layer(n_in, layer_dims[0], activations[0]),
        # TODO: then Layer(layer_dims[i-1], layer_dims[i], activations[i]).
        raise NotImplementedError

    def __call__(self, x: List[Union[Value, Number]]) -> Union[Value, List[Value]]:
        """Returns a single Value if the last layer has one neuron, else list."""
        raise NotImplementedError

    def parameters(self) -> List[Value]:
        raise NotImplementedError


def train(model: MLP, X: List[List[Number]], y: List[Number],
          lr: float = 0.05, epochs: int = 200,
          loss: str = "mse") -> List[float]:
    """
    Manual training loop using ONLY your engine.

    Each epoch:
      1. zero grads on all params
      2. forward-pass every (x, y_i), accumulate loss as a Value
      3. loss.backward()
      4. for p in params: p.data -= lr * p.grad

    `loss`: 'mse' for sum-of-squared-errors, 'bce' for binary cross-entropy.
    Returns the loss-per-epoch list.
    """
    raise NotImplementedError


# -----------------------------------------------------------------------------
# MAIN — quick smoke run when executing this file directly
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # TODO: once Stage 1+2 are in, sanity-check c = a*b + a yields a.grad = 5.
    pass
