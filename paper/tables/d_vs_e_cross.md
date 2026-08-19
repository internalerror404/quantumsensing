# D-optimal vs E-optimal: cross-objective comparison (representative cells, seed 2026, order 1024)

Each design is scored on BOTH objectives plus the posterior metrics — scoring a design only on its own objective would be circular. Whitened convention throughout; unit prior.

| M | d | K | design | λ_min (E obj) | logdet (D obj) | RMSE | worst σ | select time |
|---|---|---|---|---|---|---|---|---|
| 256 | 12 | 18 | greedy-D | 1.650 | **38.6** | 0.263 | 0.614 | 1 ms |
| 256 | 12 | 18 | relaxed-E | **3.382** | 29.9 | 0.324 | 0.478 | 1.1 s |
| 1024 | 12 | 18 | greedy-D | 0.477 | **23.6** | 0.406 | 0.823 | 1 ms |
| 1024 | 12 | 18 | relaxed-E | **1.413** | 17.2 | 0.466 | 0.644 | 5.7 s |
| 4096 | 16 | 24 | greedy-D | 0.175 | **16.6** | 0.545 | 0.923 | 5 ms |
| 4096 | 16 | 24 | relaxed-E | **0.642** | 8.8 | 0.619 | 0.780 | 40.5 s |

Reading: relaxed-E wins the minimum eigenvalue (its objective) by 2–4x and the worst-direction posterior; greedy-D wins log-determinant (its objective) and average RMSE, at three to four orders of magnitude lower latency. Neither dominates: the choice is objective-driven, and greedy-D remains the deployment method with relaxed-E as the registered E-objective reference.
