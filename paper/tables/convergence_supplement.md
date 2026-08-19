# Numerical-convergence supplement

Consolidates every quadrature and order-stability measurement in the package. All entries are drawn programmatically from the canonical JSON reports; nothing is asserted without a measured value.

## 1. Compact-gauge cancellation vs quadrature order (gate 2, single ray)

| order | max abs A(2 d^s V) |
|---|---|
| 128 | 6.04e-08 |
| 256 | 2.90e-08 |
| 512 | 1.05e-10 |
| 1024 | 4.07e-14 |

Gate 3 integrates the same direction on all 288 rays at the package order (1024): prefix max 1.21e-12 against the 1e-10 tolerance.

## 2. Packet-variance quadrature (gate 5)

Var(nu) computed from psi_s by quadrature matches 1/(4 s^2) to relative error 6.7e-16 across the 17-point width grid; log-log slope -2.0000000000.

## 3. Endpoint identity with a non-polynomial profile (gate 6)

Relative error 1.51e-14 at order 1024 against the 1e-8 threshold.

## 4. Order-512 vs order-1024 design stability (scaling study, boundary cells)

| M | d | K | subsets identical (5 seeds) | max rel lambda_min shift |
|---|---|---|---|---|
| 256 | 6 | 9 | yes | 2.2e-08 |
| 1024 | 12 | 18 | yes | 1.5e-08 |
| 4096 | 16 | 32 | yes | 2.9e-08 |

## 5. Static-gauge delay residual vs order (joint experiment, narrow 0.30-radius bumps)

| order | max abs delay residual |
|---|---|
| 1024 | 1.4e-08 |
| 2048 | 2.0e-11 |
| 4096 | 4.4e-15 |

Selected order 2048 against the 1e-10 threshold; redshift response exactly 0.0. Supplementary measurement (recorded in STATUS): the same residual reaches 5.8e-17 at order 8192.

## 6. Static-redshift experiment order stability

Order-512 and order-1024 runs produce identical reports up to the delay-noise floor (max |A| 3.63e-18 vs 3.41e-18); identical selected links; identical linearization slope 2.0001685.

## 7. Scale separation in the no-quotient control

| scale | value |
|---|---|
| exact conformal zeros | 1.2e-18 |
| quadrature potential-gauge floor (order 1024) | 2.4e-10 |
| smallest physical singular value | 6.9e-02 |

Six orders of separation between the gauge floor and the physical spectrum; the rank threshold is pinned between them.
