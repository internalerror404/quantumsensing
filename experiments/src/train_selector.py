"""Amortized learned null-channel selector.

The network learns a task-to-design map. Each task supplies a candidate sensitivity
matrix J, per-ray QFI weights q, geometric ray features, and an availability mask.
Training uses a straight-through hard top-K so the forward pass matches evaluation;
the linear-Gaussian MAP estimator remains the reconstruction baseline so that gains
are attributable to design, not a black-box inverse model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from light_ray import candidate_rays, make_physical_modes, mode_matrix

D_MAX = 12
BLOCK_MAX = 0.20


class DeepSetSelector(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,M,F]
        h = self.embed(x)
        global_context = h.mean(dim=1, keepdim=True).expand_as(h)
        return self.score(torch.cat((h, global_context), dim=-1)).squeeze(-1)


def angular_sector_mask(directions: np.ndarray, rng: np.random.Generator,
                        max_fraction: float = BLOCK_MAX) -> np.ndarray:
    """Block contiguous solid-angle sectors, not independent random rays.

    Physical obstruction -- Sun avoidance, occultation, a horizon -- removes a
    band of directions. Independent per-ray thinning leaves angular coverage
    essentially intact and almost never threatens rank, which makes the
    full-rank gate nearly free and hides the case the design problem is about.
    """
    m = directions.shape[0]
    target = rng.uniform(0.0, max_fraction)
    blocked = np.zeros(m, dtype=bool)
    if target <= 0.0:
        return blocked
    for _ in range(4):
        if blocked.mean() >= target:
            break
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        # Half-angle chosen so one sector covers roughly the remaining budget.
        remaining = max(target - blocked.mean(), 1e-3)
        cos_half = 1.0 - 2.0 * min(remaining * rng.uniform(0.6, 1.4), 0.5)
        blocked |= (directions @ axis) >= cos_half
    return blocked


def generate_tasks(base_j: np.ndarray, ray_features: np.ndarray, q_base: np.ndarray,
                   directions: np.ndarray, batch: int, rng: np.random.Generator,
                   d: int | None = None, k: int | None = None):
    """Sample a batch of design tasks.

    Randomizes, per task: the metric-family subspace and its dimension d, the
    orthogonal mixing and log scaling, packet widths, angular sector blocking,
    and candidate-pool density.

    The orthogonal mixing alone is not a task distribution. logdet(G^T F_S G) =
    logdet(F_S) + 2 logdet(G), and the second term does not depend on S, so the
    D-optimal ranking is *exactly* invariant under it -- measured spread 4e-12
    across 200 subsets. Diversity has to come from the pool and the budget,
    which is what actually changes which rays are worth measuring.
    """
    m, d_full = base_j.shape
    if d is None:
        d = int(rng.choice([6, 8, 10, 12]))
    if k is None:
        k = int(round(d * rng.choice([1.0, 1.25, 1.5, 2.0])))
    feat_dim = D_MAX + 1 + ray_features.shape[1] + 3
    js = np.zeros((batch, m, D_MAX), dtype=np.float32)
    qs = np.empty((batch, m), dtype=np.float32)
    avail = np.ones((batch, m), dtype=bool)
    feats = np.zeros((batch, m, feat_dim), dtype=np.float32)
    for b in range(batch):
        modes = rng.choice(d_full, size=d, replace=False)
        mix, _ = np.linalg.qr(rng.normal(size=(d, d)))
        scales = np.exp(rng.uniform(-0.35, 0.35, size=d))
        j = base_j[:, modes] @ mix @ np.diag(scales)
        j = j / np.maximum(np.linalg.norm(j, axis=0), 1e-8)

        q = q_base * np.exp(rng.normal(scale=0.35, size=m))
        blocked = angular_sector_mask(directions, rng)
        # Candidate-pool density: thin the surviving pool.
        density = rng.uniform(0.5, 1.0)
        blocked |= rng.random(m) > density
        if blocked.all():
            blocked[:] = False
        q = np.where(blocked, 1e-8, q)

        js[b, :, :d] = j
        qs[b] = q
        avail[b] = ~blocked
        feats[b] = np.concatenate((
            np.pad(j, ((0, 0), (0, D_MAX - d))),
            np.log(q + 1e-12)[:, None],
            ray_features,
            np.full((m, 1), d / D_MAX),
            np.full((m, 1), k / d),
            (~blocked).astype(np.float32)[:, None],
        ), axis=1).astype(np.float32)
    return js, qs, feats, avail, d, k


def soft_design_loss(j: torch.Tensor, q: torch.Tensor, logits: torch.Tensor, k: int,
                     objective: str, temperature: float, ridge: float = 1e-8) -> torch.Tensor:
    """Straight-through hard top-K allocation.

    The forward pass evaluates the design that will actually be measured; the
    backward pass flows through the soft allocation. With a plain soft
    allocation the two disagree badly: measured, the top-K rays held 8.41 of a
    K=16 photon budget, and lambda_min fell by 1000x on rounding.
    """
    soft = k * torch.softmax(logits / temperature, dim=1)
    topk = torch.topk(logits, k, dim=1).indices
    hard = torch.zeros_like(soft).scatter_(1, topk, 1.0)
    weights = hard + soft - soft.detach()

    f = torch.einsum("bmi,bm,bmj->bij", j, weights * q, j)
    eye = torch.eye(f.shape[-1], device=f.device, dtype=f.dtype).expand_as(f)
    f = f + ridge * eye
    if objective == "D":
        core = -torch.linalg.slogdet(f).logabsdet.mean()
    elif objective == "E":
        eig = torch.linalg.eigvalsh(f)
        # Scale-relative temperature. A fixed tau=0.03 against eigenvalues of
        # order 1-30 underflows to the hard min (measured agreement 1.3e-13),
        # so the "smooth" min was not smooth and gradients flowed through a
        # single eigenvector.
        tau = (0.02 * eig[:, -1:]).detach().clamp(min=1e-8)
        core = -(-tau[:, 0] * torch.logsumexp(-eig / tau, dim=1)).mean()
    else:
        raise ValueError("objective must be D or E")
    cap = torch.relu(soft - 1.0).square().mean()
    return core + 0.5 * cap


def rounded_metrics(j: np.ndarray, q: np.ndarray, logits: np.ndarray, k: int,
                    d: int, rank_tol: float = 1e-10) -> dict:
    idx = np.argsort(logits)[-k:]
    js = j[idx][:, :d]
    f = js.T @ (q[idx, None] * js)
    vals = np.linalg.eigvalsh(0.5 * (f + f.T))
    # Same convention as task1_verify: eigenvalue threshold is the squared
    # singular-value tolerance, with a roundoff floor.
    threshold = max(rank_tol ** 2 * max(float(vals[-1]), 1.0),
                    100.0 * np.finfo(float).eps * max(float(vals[-1]), 1.0) * d)
    return {
        "indices": np.sort(idx).tolist(),
        "rank": int(np.count_nonzero(vals > threshold)),
        "dimension": d,
        "full_rank": bool(np.count_nonzero(vals > threshold) == d),
        "min_eigenvalue": float(vals[0]),
        "logdet": float(np.linalg.slogdet(f + 1e-12 * np.eye(d))[1]),
    }


def select(model: nn.Module, feats: np.ndarray, avail: np.ndarray, k: int) -> np.ndarray:
    """Hard top-K over available rays only."""
    with torch.no_grad():
        logits = model(torch.from_numpy(feats)).numpy()
    logits = np.where(avail, logits, -np.inf)
    return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/selector_checkpoint.pt"))
    parser.add_argument("--report", type=Path, default=Path("results/selector_report.json"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--k", type=int, default=0, help="0 = randomize K/d per task")
    parser.add_argument("--objective", choices=["D", "E"], default="E")
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(args.seed)
    rays = candidate_rays(direction_count=96, offsets_per_direction=4)
    base_j = mode_matrix(rays, make_physical_modes())
    base_j = base_j / np.maximum(np.linalg.norm(base_j, axis=0), 1e-12)
    widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
    q_base = 1.0 / np.square(widths)
    ray_features = np.array([np.concatenate((r.theta, r.offset)) for r in rays], dtype=np.float32)
    directions = np.array([r.theta for r in rays], dtype=float)

    fixed_k = args.k if args.k > 0 else None
    _, _, probe, _, _, _ = generate_tasks(base_j, ray_features, q_base, directions, 1, rng)
    input_dim = probe.shape[-1]
    model = DeepSetSelector(input_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    losses = []
    for step in range(args.steps):
        js, qs, feats, avail, d, k = generate_tasks(
            base_j, ray_features, q_base, directions, args.batch, rng, k=fixed_k)
        # float64 throughout: lambda_min of a 12x12 Gram in float32 is not
        # accurate enough to optimize against, and these matrices are tiny.
        jt = torch.from_numpy(js[:, :, :d]).double()
        qt = torch.from_numpy(qs).double()
        logits = model(torch.from_numpy(feats)).double()
        logits = logits.masked_fill(~torch.from_numpy(avail), -1e30)
        # Anneal to a genuinely sharp allocation. max(0.25, 1.5*0.999^step) is
        # still 1.168 at step 250 and 0.334 at step 1500.
        temperature = max(0.05, 1.0 * (0.5 ** (4.0 * step / max(args.steps, 1))))
        loss = soft_design_loss(jt, qt, logits, k, args.objective, temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim,
                "objective": args.objective, "d_max": D_MAX}, args.output)

    heldout = []
    for _ in range(8):
        js, qs, feats, avail, d, k = generate_tasks(
            base_j, ray_features, q_base, directions, 1, rng, k=fixed_k)
        logits = select(model, feats, avail, k)[0]
        heldout.append(rounded_metrics(js[0], qs[0], logits, k, d))
    report = {
        "seed": args.seed,
        "objective": args.objective,
        "steps": args.steps,
        "batch": args.batch,
        "k_policy": "fixed" if fixed_k else "randomized K/d in {1.0,1.25,1.5,2.0}",
        "final_loss": losses[-1],
        "loss_tail_mean": float(np.mean(losses[-50:])),
        "heldout": heldout,
        "heldout_full_rank_rate": float(np.mean([h["full_rank"] for h in heldout])),
        "status": "engineering sanity run only; not a paper result without the preregistered multi-seed evaluation",
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
