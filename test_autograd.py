"""
Edge case tests for Day 4 — Backpropagation Engine (Autograd).

Per the 30-Day Mantra: tests are written BEFORE you implement.
Run with: python test_autograd.py

Tests are organized by stage. Each test prints PASS/FAIL — print() is your debugger.
Comment out tests for stages you haven't reached yet.

NUMERICAL GRADIENT CHECK
    For any scalar f and small eps:
        grad_num = (f(x + eps) - f(x - eps)) / (2 * eps)
    We compare against your engine's .grad with:
        rel_err = |grad_num - grad_analytical| / max(1, |grad_num| + |grad_analytical|)
    rel_err < 1e-5 means the chain rule is wired correctly.
"""

import math
import random
from autograd import (
    Value,
    sigmoid,
    tanh,
    Neuron,
    Layer,
    MLP,
    train,
)


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def _grad_check(forward_fn, inputs, target_idx: int, eps: float = 1e-6) -> float:
    """
    Numerical-vs-analytical gradient check for inputs[target_idx].

    forward_fn: a function that takes a *fresh* list of Value inputs (so we can
                rebuild the graph cleanly each time) and returns the output Value.
    inputs:     list of float scalars to wrap into Value nodes.
    target_idx: which input we want the gradient w.r.t.

    Returns the relative error.
    """
    # Analytical via the engine.
    vals = [Value(x) for x in inputs]
    out = forward_fn(vals)
    out.backward()
    grad_analytical = vals[target_idx].grad

    # Numerical via central differences.
    def f_at(perturb):
        perturbed = list(inputs)
        perturbed[target_idx] = inputs[target_idx] + perturb
        vs = [Value(x) for x in perturbed]
        return forward_fn(vs).data

    grad_num = (f_at(+eps) - f_at(-eps)) / (2 * eps)

    denom = max(1.0, abs(grad_num) + abs(grad_analytical))
    return abs(grad_num - grad_analytical) / denom


# -----------------------------------------------------------------------------
# STAGE 1 TESTS — Value node + basic ops
# -----------------------------------------------------------------------------

def test_stage_1():
    print("\n--- Stage 1: Value Node + Basic Ops ---")

    # Construction.
    v = Value(3.0)
    _check("Value stores data", v.data == 3.0)
    _check("Value initial grad is 0", v.grad == 0.0)
    _check("repr is a string", isinstance(repr(v), str))

    # Add forward.
    a, b = Value(2.0), Value(5.0)
    c = a + b
    _check("add forward: 2 + 5 == 7", c.data == 7.0, f"got {c.data}")
    _check("add records both parents",
           {id(a), id(b)} <= {id(p) for p in c._prev})

    # Mul forward.
    d = a * b
    _check("mul forward: 2 * 5 == 10", d.data == 10.0, f"got {d.data}")

    # Mixed-type ops (scalar on the right).
    e = a + 1.5
    _check("a + 1.5 works (other is float)",
           e.data == 3.5, f"got {e.data}")
    f = a * 4
    _check("a * 4 works (other is int)",
           f.data == 8.0, f"got {f.data}")

    # Right-side scalar ops.
    g = 2 + a
    _check("2 + a works (__radd__)",
           g.data == 4.0, f"got {g.data}")
    h = 3 * a
    _check("3 * a works (__rmul__)",
           h.data == 6.0, f"got {h.data}")


# -----------------------------------------------------------------------------
# STAGE 2 TESTS — Backward + topo sort
# -----------------------------------------------------------------------------

def test_stage_2():
    print("\n--- Stage 2: Backward Pass ---")

    # The canonical micrograd sanity test.
    #   c = a * b + a;  dc/da = b + 1;  dc/db = a
    a, b = Value(3.0), Value(4.0)
    c = a * b + a
    c.backward()
    _check("c = a*b + a -> a.grad == b + 1 == 5", a.grad == 5.0,
           f"got {a.grad}")
    _check("c = a*b + a -> b.grad == a == 3", b.grad == 3.0,
           f"got {b.grad}")

    # Reusing a node multiple times — grads must accumulate (+=).
    x = Value(2.0)
    y = x + x + x         # y = 3x, dy/dx = 3
    y.backward()
    _check("reused node accumulates: y = x + x + x -> x.grad == 3",
           x.grad == 3.0, f"got {x.grad}")

    # __neg__ and __sub__.
    p, q = Value(7.0), Value(2.0)
    r = p - q             # r = 5; dr/dp = 1, dr/dq = -1
    r.backward()
    _check("sub forward: 7 - 2 == 5", r.data == 5.0)
    _check("sub backward: p.grad == 1, q.grad == -1",
           p.grad == 1.0 and q.grad == -1.0,
           f"p={p.grad}, q={q.grad}")

    # zero_grad should reset everywhere.
    p.zero_grad()
    _check("zero_grad resets p.grad and q.grad",
           p.grad == 0.0 and q.grad == 0.0,
           f"p={p.grad}, q={q.grad}")

    # Topo sort returns a list with the output node last.
    a, b = Value(1.0), Value(2.0)
    c = a * b
    topo = c._topological_sort()
    _check("_topological_sort puts output node last",
           topo[-1] is c, f"got {[t._op for t in topo]}")
    _check("_topological_sort covers all reachable nodes",
           {id(a), id(b), id(c)} <= {id(t) for t in topo})

    # Numerical check on a small expression: f(a, b) = (a + b) * a - b
    def f_small(vs):
        a, b = vs
        return (a + b) * a - b

    rel_a = _grad_check(f_small, [1.5, -0.7], target_idx=0)
    rel_b = _grad_check(f_small, [1.5, -0.7], target_idx=1)
    _check("grad-check (a+b)*a - b w.r.t. a < 1e-5",
           rel_a < 1e-5, f"rel_err={rel_a:.2e}")
    _check("grad-check (a+b)*a - b w.r.t. b < 1e-5",
           rel_b < 1e-5, f"rel_err={rel_b:.2e}")


# -----------------------------------------------------------------------------
# STAGE 3 TESTS — Full operation library
# -----------------------------------------------------------------------------

def test_stage_3():
    print("\n--- Stage 3: Full Operation Library ---")

    # Pow.
    a = Value(3.0)
    b = a ** 2                          # 9; db/da = 2a = 6
    b.backward()
    _check("a ** 2 forward: 3^2 == 9", b.data == 9.0, f"got {b.data}")
    _check("a ** 2 backward: a.grad == 2*a == 6",
           math.isclose(a.grad, 6.0), f"got {a.grad}")

    # Div.
    a, b = Value(6.0), Value(2.0)
    c = a / b                           # 3
    c.backward()
    _check("a / b forward: 6 / 2 == 3", math.isclose(c.data, 3.0))
    # da: 1/b = 0.5;   db: -a / b^2 = -1.5
    _check("a / b backward: a.grad == 1/b == 0.5",
           math.isclose(a.grad, 0.5), f"got {a.grad}")
    _check("a / b backward: b.grad == -a/b^2 == -1.5",
           math.isclose(b.grad, -1.5), f"got {b.grad}")

    # ReLU at a positive input.
    x = Value(2.0)
    y = x.relu()
    y.backward()
    _check("relu(2) == 2", y.data == 2.0)
    _check("relu'(2) flows: x.grad == 1", x.grad == 1.0, f"got {x.grad}")

    # ReLU at a negative input — grad is killed.
    x = Value(-1.0)
    y = x.relu()
    y.backward()
    _check("relu(-1) == 0", y.data == 0.0)
    _check("relu'(-1) kills grad: x.grad == 0", x.grad == 0.0)

    # exp.
    x = Value(1.0)
    y = x.exp()                         # e
    y.backward()
    _check("exp(1) == e", math.isclose(y.data, math.e, rel_tol=1e-9))
    _check("exp' == exp: x.grad == e",
           math.isclose(x.grad, math.e, rel_tol=1e-9), f"got {x.grad}")

    # log.
    x = Value(2.0)
    y = x.log()                         # ln 2
    y.backward()
    _check("log(2) == ln(2)", math.isclose(y.data, math.log(2.0)))
    _check("log' == 1/x: x.grad == 0.5",
           math.isclose(x.grad, 0.5), f"got {x.grad}")

    # Composition: sigmoid built from exp + div should match the math.
    s = sigmoid(Value(0.0))
    _check("sigmoid(0) == 0.5", math.isclose(s.data, 0.5))
    s.backward()
    # NOTE: this test relies on sigmoid being a fresh node — no need to
    # check input grad here; that's covered by the numerical sweep below.

    # tanh.
    t = tanh(Value(0.0))
    _check("tanh(0) == 0", math.isclose(t.data, 0.0, abs_tol=1e-9))

    # Numerical sweep across all the ops.
    test_inputs = [(-2.3, 1.7), (0.8, 0.3), (1.1, 2.2)]

    def f_pow(vs):
        return vs[0] ** 3 + vs[1] * vs[0]

    def f_div(vs):
        return (vs[0] + 1.0) / (vs[1] + 2.0)

    def f_exp_log(vs):
        return (vs[0].exp() + vs[1].log()) * vs[0]

    def f_sigmoid(vs):
        return sigmoid(vs[0] * vs[1] + vs[0])

    def f_tanh(vs):
        return tanh(vs[0] + vs[1] * 0.5)

    funcs = [("pow + mul", f_pow), ("div", f_div),
             ("exp + log + mul", f_exp_log),
             ("sigmoid composition", f_sigmoid),
             ("tanh composition", f_tanh)]

    for name, fn in funcs:
        max_err = 0.0
        for inp in test_inputs:
            for idx in (0, 1):
                # log() needs positive input — push inputs positive for that case
                if name == "exp + log + mul" and inp[1] <= 0:
                    inp = (inp[0], abs(inp[1]) + 0.5)
                err = _grad_check(fn, list(inp), target_idx=idx)
                max_err = max(max_err, err)
        _check(f"grad-check {name} < 1e-5",
               max_err < 1e-5, f"max rel_err={max_err:.2e}")


# -----------------------------------------------------------------------------
# STAGE 4 TESTS — Neuron / Layer / MLP + XOR (bonus)
# -----------------------------------------------------------------------------

def test_stage_4():
    print("\n--- Stage 4: Neuron / Layer / MLP + XOR ---")

    random.seed(0)

    # Neuron — one call, output is a Value.
    n = Neuron(n_in=3, activation="relu", seed=0)
    out = n([1.0, -2.0, 0.5])
    _check("Neuron(__call__) returns a Value", isinstance(out, Value))

    params = n.parameters()
    _check("Neuron.parameters has n_in weights + 1 bias",
           len(params) == 4, f"got {len(params)}")

    # Layer — n_out outputs.
    layer = Layer(n_in=3, n_out=2, activation="relu")
    outs = layer([0.1, 0.2, 0.3])
    _check("Layer(__call__) returns n_out Values",
           isinstance(outs, list) and len(outs) == 2,
           f"got {type(outs)}, len {len(outs) if isinstance(outs, list) else 'N/A'}")
    _check("Layer.parameters has n_in*n_out + n_out entries",
           len(layer.parameters()) == 3 * 2 + 2,
           f"got {len(layer.parameters())}")

    # MLP — [2, 4, 1] should produce a single Value output.
    mlp = MLP(n_in=2, layer_dims=[4, 1], activations=["relu", "sigmoid"])
    y = mlp([0.5, -0.5])
    _check("MLP with final layer of 1 neuron returns single Value",
           isinstance(y, Value), f"got {type(y)}")

    # Train on XOR.
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]
    mlp = MLP(n_in=2, layer_dims=[4, 4, 1],
              activations=["tanh", "tanh", "sigmoid"])
    losses = train(mlp, X, y, lr=0.1, epochs=200, loss="mse")
    _check("train returns a non-empty loss list",
           isinstance(losses, list) and len(losses) > 0,
           f"len={len(losses) if isinstance(losses, list) else 'N/A'}")
    _check("XOR training loss decreases meaningfully",
           losses[-1] < losses[0] * 0.5,
           f"{losses[0]:.4f} -> {losses[-1]:.4f}")

    # Predictions should match labels.
    preds = []
    for x in X:
        p = mlp(x)
        preds.append(1 if p.data > 0.5 else 0)
    _check("XOR predictions correct (all 4)",
           preds == [int(t) for t in y],
           f"preds={preds}, y={y}")


# -----------------------------------------------------------------------------
# RUN ALL
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Day 4 — Autograd Engine · Edge Case Tests")
    print("=" * 70)
    test_stage_1()
    test_stage_2()
    test_stage_3()
    test_stage_4()
    print("\nDone.")
