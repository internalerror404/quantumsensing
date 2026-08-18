# Review reproduction scripts

Measurements cited in `../REVIEW.md`. Run from `experiments/src`:

```bash
cd experiments/src
PYTHONPATH=. python3 ../../review/selector_fix_probe.py
```

- `selector_fix_probe.py` — retrains the DeepSets selector with the five fixes from REVIEW.md §B1
  (straight-through hard top-K, scale-relative softmin tau, real annealing, float64, quadrature
  order 512) and evaluates it against the deterministic baselines on 30 held-out tasks.
  ~13 s CPU. Takes the policy from 0.015x to 1.37x angular_spread.
