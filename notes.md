# Day 4 — Notes & Debrief

**Date:** 2026-05-19
**Problem:** Backpropagation Engine (Autograd)
**Phase:** 1 — Foundations

---

## Architecture Design (write during your 5-min design phase)

> Sketch the Value layout, what `_backward` closures capture, and the order you'll implement methods. Do this BEFORE writing any code.



---

## Time Log (fill in as you go)

| Stage | Planned | Actual Start | Actual End | Duration | Notes |
|---|---|---|---|---|---|
| Read + design | 5 min | 0:00 | | | |
| Stage 1 — Value + add/mul | 15 min | 0:05 | | | |
| Stage 2 — backward + topo sort | 15 min | 0:20 | | | |
| Stage 3 — pow/relu/exp/log/div | 15 min | 0:35 | | | |
| Stage 4 — Neuron/Layer/MLP + XOR (bonus) | 10 min | 0:50 | | | |

**Where did time leak?**

---

## Debrief Questions

**1. Does your engine match numerical gradients on all operations (max rel err < 1e-5)?**



**2. Did `zero_grad` matter? What happens if you forget to call it before each epoch?**



**3. Could you express `sigmoid` and `tanh` purely as compositions, with no special backward?**



**4. Train-time on XOR: how does it compare to Day 3's matrix-based approach? Why slower?**



**5. What would you add to make it faster?**



---

## Things to Drill Tomorrow

- [ ]
- [ ]
- [ ]
