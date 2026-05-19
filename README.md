# Day 4 — Backpropagation Engine (Autograd)

**Phase 1 — Foundations** · 60-minute timed session · NumPy optional, but the engine itself is pure Python `Value` nodes

---

## The Problem

Build a **micrograd-style automatic differentiation engine** — PyTorch autograd, from scratch. Every `Value` node stores `.data`, `.grad`, and a `._backward` closure. Operations on `Value` instances dynamically build a computation DAG; calling `.backward()` on the output node walks that DAG in reverse topological order and propagates gradients via the chain rule.

By the end of the hour, you should be able to train logistic regression — or even XOR — using **only your engine**, and your gradients should match a numerical estimate to better than `1e-5`. After this day, PyTorch's autograd is no longer magic.

---

## Rules (from the 30-Day Plan)

1. Read all 4 stages **before** writing any code.
2. Design your architecture in **comments first** inside `autograd.py`.
3. Write your own **edge case tests before** implementing — see `test_autograd.py`.
4. **`print()` is your only debugger.** No notebooks during the timed session.
5. **3 stages clean beats 4 stages with NaN losses.** Build intuition, not just code.

---

## Time Budget — 60 Minutes (compressed from the doc's 120)

| Activity | Clock | Duration | Note |
|---|---|---|---|
| Read problem + design architecture | 0:00–0:05 | 5 min | NEVER skip this |
| Stage 1 — Value node + add/mul | 0:05–0:20 | 15 min | Get the closure pattern right |
| Stage 2 — Backward + topo sort | 0:20–0:35 | 15 min | The heart of the engine |
| Stage 3 — Full op library | 0:35–0:50 | 15 min | pow, relu, exp, log, div |
| Stage 4 — Neuron/Layer/MLP + XOR | 0:50–1:00 | 10 min | Trains using only your engine |

If you reach Stage 3 cleanly by minute 50 you're on track. Stage 4 is bonus.

---

## Data Setup

No external data needed for Stages 1–3. Build and verify with hand-calculated examples:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b           # data = 6
c.backward()
# Expected: a.grad == 3, b.grad == 2 (chain rule on c = a*b)
```

For Stage 4 use the XOR dataset from Day 3:

```python
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]
```

---

## Architecture Hints

```python
class Value:
    data: float
    grad: float                 # init 0.0
    _backward: Callable[[], None]   # init lambda: None
    _prev: set[Value]           # parents in the DAG
    _op: str                    # debug label
```

**Chain rule pattern.** Each operation creates a new `out = Value(...)` and defines an `_backward` closure that *adds* the appropriate local gradient to each parent's `.grad`. The `+=` is critical — a node can be used multiple times in an expression, and its grads must accumulate.

**Topological sort.** DFS from the output node, marking visited; append a node to the order *after* recursing into all children. Reverse the result. Then call each node's `_backward` in that order.

**Numerical gradient check.** For any scalar function `f(x)` and small `eps`:

```
grad_num = (f(x + eps) - f(x - eps)) / (2 * eps)
```

Compare to your analytical `.grad` — relative error should be `< 1e-5`.

---

## Stage 1 — Value Node + Basic Ops (15 min)

**Class:** `Value`

**Methods:**
- `__init__(data, _children=(), _op="")`
- `__add__(self, other) -> Value` — `out.grad` flows to both parents (each `+= out.grad`)
- `__mul__(self, other) -> Value` — each parent gets `+= out.grad * other_parent.data`
- `__repr__(self) -> str`

Don't forget `__radd__` / `__rmul__` so you can write `2 * a` as well as `a * 2`.
Wrap non-`Value` operands (int/float) into `Value` inside the op so the graph stays homogeneous.

---

## Stage 2 — Backward Pass (15 min)

**Methods:**
- `backward(self) -> None` — triggers full reverse-mode autodiff
- `zero_grad(self) -> None` — walks the graph and resets every `.grad` to 0
- `_topological_sort() -> List[Value]` — DFS, append after visiting children
- `__neg__(self) -> Value` — `-self` (build from `__mul__` with `-1`)
- `__sub__(self, other) -> Value` — `self + (-other)`

`backward()`: set `self.grad = 1.0`, then iterate the reverse topo order calling each node's `_backward`.

**Sanity test:** `a = Value(3); b = Value(4); c = a * b + a; c.backward()` should give `a.grad == 5` and `b.grad == 3`. (Because `dc/da = b + 1 = 5` and `dc/db = a = 3`.)

---

## Stage 3 — Full Operation Library (15 min)

**Methods:**
- `__pow__(self, n) -> Value` — only `n` as a scalar; grad: `n * self.data^(n-1)`
- `relu(self) -> Value` — `max(0, x)`; grad flows where `data > 0`
- `exp(self) -> Value` — `e^x`; grad: `out.data * out.grad`
- `log(self) -> Value` — `ln(x)`; grad: `(1/x) * out.grad`
- `__truediv__(self, other) -> Value` — build from `__mul__` and `__pow__(-1)`
- `__rsub__`, `__rtruediv__` so scalars work on both sides

`sigmoid` and `tanh` should fall out of these primitives — **no special-case `_backward`**.
Verify every op against the numerical gradient. If a single op fails the check, the rest of the engine is built on sand.

---

## Stage 4 — Train a Network with Your Engine (10 min, bonus)

**Classes:**
- `Neuron` — stores a list of `Value` weights + bias. `__call__(x) -> Value` returns `activation(w·x + b)`.
- `Layer` — list of `Neuron`s. `__call__(x) -> List[Value]`.
- `MLP` — list of `Layer`s. `__call__(x) -> Value` (single output) or `List[Value]`. `parameters() -> List[Value]`.
- `train(model, X, y, lr, epochs) -> List[float]` — manual training loop using only your engine.

Loss for XOR: BCE (`-y*log(p) - (1-y)*log(1-p)`) built from `log` / `exp`, **or** sum of squared errors `(pred - y)**2`.
Each epoch:
1. Zero grads on all parameters.
2. Forward pass for every example, accumulate loss.
3. Call `loss.backward()`.
4. SGD step: `for p in params: p.data -= lr * p.grad`.

Target: XOR loss < 0.01 with a `[2, 4, 1]` MLP. If your engine matches Day 3's behavior, you've reproduced PyTorch from first principles.

---

## Debrief Questions (fill in `notes.md` after)

- Does your engine match numerical gradients on all operations (max rel err < 1e-5)?
- Did `zero_grad` matter? What happens if you forget to call it before each epoch?
- Could you express `sigmoid` and `tanh` purely as compositions, with no special backward?
- Train-time on XOR: how does it compare to Day 3's matrix-based approach? Why slower?
- What would you add to make it faster? (Hint: batching, broadcasting, fused ops.)
- Time log: minutes per stage.

---

## Reference — Days 1, 2, 3

The class skeleton, time-log workflow, and test-first style mirror the previous days (`../Day01_Linear_Regression_Engine/`, `../Day02_Binary_Classifier/`, `../Day03_Neural_Network/`).

The mental shift today: in Days 1–3 you derived gradients **analytically** and wrote them out by hand. Today the gradients are produced **by composition** — each operation knows how its local gradient flows, and the engine assembles the chain rule by walking the DAG. The `(1/m) * X.T @ (p - y)` you wrote in Day 2 falls out automatically here once you compose `log`, `exp`, `mul`, and `sum`.

---

## Files in This Folder

- `autograd.py` — your implementation (start here)
- `test_autograd.py` — edge case tests, run as you implement each stage
- `notes.md` — time log and debrief answers
- `README.md` — this file
