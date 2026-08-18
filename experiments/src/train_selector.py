"""Amortized learned null-channel selector.

The network learns a task-to-design map. Each task supplies a candidate sensitivity
matrix J, per-ray QFI weights q, and geometric ray features. Training uses a soft
allocation; evaluation rounds to K distinct rays. The linear-Gaussian MAP estimator
remains the reconstruction baseline so that gains are attributable to design, not a
black-box inverse model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from light_ray import candidate_rays, make_physical_modes, mode_matrix


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


def generate_tasks(base_j: np.ndarray, ray_features: np.ndarray, q_base: np.ndarray,
                   batch: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, d = base_j.shape
    js = np.empty((batch, m, d), dtype=np.float32)
    qs = np.empty((batch, m), dtype=np.float32)
    feats = np.empty((batch, m, d + 1 + ray_features.shape[1]), dtype=np.float32)
    for b in range(batch):
        mix, _ = np.linalg.qr(rng.normal(size=(d, d)))
        scales = np.exp(rng.uniform(-0.35, 0.35, size=d))
        j = base_j @ mix @ np.diag(scales)
        # Randomly block up to 20% of candidate rays and vary packet widths.
        q = q_base * np.exp(rng.normal(scale=0.35, size=m))
        blocked = rng.random(m) < rng.uniform(0.0, 0.20)
        q[blocked] = 1e-8
        col_norm = np.linalg.norm(j, axis=0)
        j = j / np.maximum(col_norm, 1e-8)
        js[b] = j.astype(np.float32)
        qs[b] = q.astype(np.float32)
        feats[b] = np.concatenate((j, np.log(q + 1e-12)[:, None], ray_features), axis=1).astype(np.float32)
    return js, qs, feats


def soft_design_loss(j: torch.Tensor, q: torch.Tensor, logits: torch.Tensor, k: int,
                     objective: str, temperature: float, ridge: float = 1e-5) -> torch.Tensor:
    weights = k * torch.softmax(logits / temperature, dim=1)
    weighted = weights * q
    f = torch.einsum("bmi,bm,bmj->bij", j, weighted, j)
    eye = torch.eye(f.shape[-1], device=f.device, dtype=f.dtype).expand_as(f)
    f = f + ridge * eye
    eig = torch.linalg.eigvalsh(f)
    if objective == "D":
        core = -torch.logdet(f).mean()
    elif objective == "E":
        tau = 0.03
        smooth_min = -tau * torch.logsumexp(-eig / tau, dim=1)
        core = -smooth_min.mean()
    else:
        raise ValueError("objective must be D or E")
    # Discourage assigning more than one photon-equivalent to a single ray before top-K rounding.
    cap = torch.relu(weights - 1.0).square().mean()
    return core + 0.5 * cap


def rounded_metrics(j: np.ndarray, q: np.ndarray, logits: np.ndarray, k: int) -> dict:
    idx = np.argsort(logits)[-k:]
    f = j[idx].T @ (q[idx, None] * j[idx])
    vals = np.linalg.eigvalsh(f)
    return {
        "indices": np.sort(idx).tolist(),
        "rank": int(np.linalg.matrix_rank(f, tol=1e-9 * max(float(vals[-1]), 1.0))),
        "min_eigenvalue": float(vals[0]),
        "logdet": float(np.linalg.slogdet(f + 1e-12 * np.eye(f.shape[0]))[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/selector_checkpoint.pt"))
    parser.add_argument("--report", type=Path, default=Path("results/selector_report.json"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--objective", choices=["D", "E"], default="E")
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rays = candidate_rays(direction_count=96, offsets_per_direction=4)
    base_j = mode_matrix(rays, make_physical_modes(), order=192)
    base_j = base_j / np.maximum(np.linalg.norm(base_j, axis=0), 1e-12)
    widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
    q_base = 1.0 / np.square(widths)
    ray_features = np.array([np.concatenate((r.theta, r.offset)) for r in rays], dtype=np.float32)

    input_dim = base_j.shape[1] + 1 + ray_features.shape[1]
    model = DeepSetSelector(input_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    losses = []
    for step in range(args.steps):
        js, qs, feats = generate_tasks(base_j, ray_features, q_base, args.batch, rng)
        jt = torch.from_numpy(js)
        qt = torch.from_numpy(qs)
        xt = torch.from_numpy(feats)
        temperature = max(0.25, 1.5 * (0.999 ** step))
        logits = model(xt)
        loss = soft_design_loss(jt, qt, logits, args.k, args.objective, temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim, "objective": args.objective}, args.output)

    # One held-out task sanity report.
    js, qs, feats = generate_tasks(base_j, ray_features, q_base, 1, rng)
    with torch.no_grad():
        logits = model(torch.from_numpy(feats))[0].numpy()
    report = {
        "seed": args.seed,
        "objective": args.objective,
        "steps": args.steps,
        "batch": args.batch,
        "selected_rays": args.k,
        "final_loss": losses[-1],
        "loss_tail_mean": float(np.mean(losses[-50:])),
        "heldout": rounded_metrics(js[0], qs[0], logits, args.k),
        "status": "engineering sanity run only; not a paper result without the preregistered multi-seed evaluation",
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
