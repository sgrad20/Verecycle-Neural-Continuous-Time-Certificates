import sys
from pathlib import Path
import csv
from datetime import datetime
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

#  .\.venv\Scripts\activate     then  python experiments_ct/run_ct_verecycle_pendulum.py
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import stochastic_rsa as rsa
from rl_agent.policy import TanhPolicy
from controlled_sde import InvertedPendulum
from stochastic_rsa.continuous_vere_cycle import run_continuous_VeRecycle


@dataclass
class Box:
    low: np.ndarray
    high: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.low.shape[0])

    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        u = torch.rand((n, self.dimension), device=device)
        low = torch.tensor(self.low, dtype=torch.float32, device=device)
        high = torch.tensor(self.high, dtype=torch.float32, device=device)
        return low + (high - low) * u

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        low = torch.tensor(self.low, dtype=x.dtype, device=x.device)
        high = torch.tensor(self.high, dtype=x.dtype, device=x.device)
        return (x >= low).all(dim=-1) & (x <= high).all(dim=-1)


@dataclass
class MultiBox:
    sets: list

    @property
    def dimension(self) -> int:
        return self.sets[0].dimension

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        inside = self.sets[0].contains(x)
        for s in self.sets[1:]:
            inside = inside | s.contains(x)
        return inside


def sample_Xq_box(
    center_low: np.ndarray,
    center_high: np.ndarray,
    half_width: np.ndarray,
) -> Box:
    c = np.random.uniform(center_low, center_high).astype(np.float32)
    low = (c - half_width).astype(np.float32)
    high = (c + half_width).astype(np.float32)
    return Box(low=low, high=high)


def make_grid_centers(region: Box, mesh_size: float) -> tuple[torch.Tensor, float]:
    low = region.low.astype(np.float32)
    high = region.high.astype(np.float32)
    dim = low.shape[0]

    width = float(mesh_size)
    num = np.ceil((high - low) / width).astype(int)
    num = np.maximum(num, 1)

    axes = [
        np.linspace(low[i] + 0.5 * width, high[i] - 0.5 * width, int(num[i]))
        for i in range(dim)
    ]
    mesh = np.meshgrid(*axes, indexing="xy")
    pts = np.stack([m.reshape(-1) for m in mesh], axis=1)
    centers = torch.tensor(pts, dtype=torch.float32)

    half_width = 0.5 * width
    return centers, half_width


def ibp_bounds_on_box(
    verifier: BoundedModule,
    box: Box,
    mesh_size: float,
    batch_size: int = 2000,
) -> tuple[float, float]:
    """
    Returns (inf_lb, sup_ub) for the certificate network on the box
    using IBP over grid cells.
    """
    centers, eps = make_grid_centers(box, mesh_size)

    inf_lb = float("inf")
    sup_ub = float("-inf")

    with torch.no_grad():
        n = centers.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            loc = centers[start:end]

            bounded = BoundedTensor(
                loc,
                PerturbationLpNorm(
                    x_L=loc - eps,
                    x_U=loc + eps,
                ),
            )

            lb, ub = verifier.compute_bounds(bounded, method="IBP")
            lb = lb.squeeze(-1)
            ub = ub.squeeze(-1)

            inf_lb = min(inf_lb, float(torch.min(lb).item()))
            sup_ub = max(sup_ub, float(torch.max(ub).item()))

    return inf_lb, sup_ub


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)

    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


class LocalDisruptionPendulum(InvertedPendulum):
    """
    Wrapper that changes dynamics only inside X_q
    by scaling diffusion by noise_scale.
    """
    def __init__(self, policy_net, Xq: Box, noise_scale: float = 1.5):
        super().__init__(policy_net)
        self.Xq = Xq
        self.noise_scale = noise_scale

    def g(self, t, y):
        g = super().g(t, y)
        inside = self.Xq.contains(y)
        if inside.any():
            g = g.clone()
            g[inside] = self.noise_scale * g[inside]
        return g


def simulate_reach_avoid(
    sde,
    X0: Box,
    XT: Box,
    XU,
    N: int = 5000,
    T: float = 5.0,
    dt: float = 0.01,
    device: str = "cpu",
) -> tuple[float, tuple[float, float]]:
    """
    Euler-Maruyama simulation.
    Assumes sde has methods f(t, y) and g(t, y).
    """
    x = X0.sample(N, device=device)
    t = 0.0

    reached = torch.zeros(N, dtype=torch.bool, device=device)
    failed = torch.zeros(N, dtype=torch.bool, device=device)

    steps = int(T / dt)
    sqrt_dt = math.sqrt(dt)

    for _ in range(steps):
        alive = ~(reached | failed)
        if not alive.any():
            break

        xa = x[alive]
        ta = torch.full((xa.shape[0],), float(t), dtype=xa.dtype, device=device)

        with torch.no_grad():
            f = sde.f(ta, xa)
            g = sde.g(ta, xa)

            if g.dim() == 2:
                noise = torch.randn_like(g)
                dx = f * dt + g * sqrt_dt * noise
            else:
                noise = torch.randn((xa.shape[0], g.shape[2]), device=device, dtype=xa.dtype)
                dx = f * dt + torch.bmm(g, noise.unsqueeze(-1)).squeeze(-1) * sqrt_dt

        xa_next = xa + dx
        x[alive] = xa_next

        reached_now = XT.contains(xa_next)
        failed_now = XU.contains(xa_next)

        reached[alive] |= reached_now
        failed[alive] |= failed_now

        t += dt

    successes = int(reached.sum().item())
    p_hat = successes / N
    ci = wilson_ci(successes, N)
    return p_hat, ci


def theorem_curve(alpha_RA: float, beta_RA: float, m_vals: np.ndarray) -> np.ndarray:
    """
    The theorem-consistent reclaimed curve:
        min(1 - alpha/beta, max(0, 1 - alpha/m))
    """
    return np.minimum(
        1.0 - alpha_RA / beta_RA,
        np.maximum(0.0, 1.0 - alpha_RA / m_vals),
    )


def main():
    device = "cpu"
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_path = repo_root / "certified.pt"
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    policy_net = TanhPolicy(2, 1, 64, device=device)
    policy_net.load_state_dict(ckpt["policy"])
    policy_net.eval()

    cert_net = rsa.CertificateModule(device=device)
    cert_net.load_state_dict(ckpt["certificate"])
    cert_net.eval()

    dummy = torch.zeros((1, 2), dtype=torch.float32)
    verifier = BoundedModule(cert_net, dummy, bound_opts={"conv_mode": "matrix"})

    X0 = Box(
        low=np.array([-1.0, 3 * math.pi / 4], dtype=np.float32),
        high=np.array([1.0, 5 * math.pi / 4], dtype=np.float32),
    )
    XT = Box(
        low=np.array([-4.0, -math.pi / 2], dtype=np.float32),
        high=np.array([4.0, math.pi / 2], dtype=np.float32),
    )
    XU_1 = Box(
        low=np.array([-20.0, -2 * math.pi], dtype=np.float32),
        high=np.array([-10.0, -3 * math.pi / 2], dtype=np.float32),
    )
    XU_2 = Box(
        low=np.array([10.0, 3 * math.pi / 2], dtype=np.float32),
        high=np.array([20.0, 2 * math.pi], dtype=np.float32),
    )
    XU = MultiBox([XU_1, XU_2])

    Xq = Box(
        low=np.array([-9.0, -2.0], dtype=np.float32),
        high=np.array([-7.0, -1.0], dtype=np.float32),
    )
    disrupted_region = MultiBox([Xq])

    mesh_sizes = [0.05, 0.02, 0.01, 0.005]
    batch_size = 2000

    # ------------------------------------------------------------
    # Experiment 1 — Mesh sweep
    # ------------------------------------------------------------
    print("\n=== Mesh sweep: runtime vs tightness ===")
    print("mesh\talpha_RA\tbeta_RA\tm_lb\teps_orig\teps_rec\ttime_s\tformula_diff")

    results = []

    for mesh_size in mesh_sizes:
        # alpha_RA = sup UB on X0
        _, alpha_RA = ibp_bounds_on_box(
            verifier, X0, mesh_size=mesh_size, batch_size=batch_size
        )

        # beta_RA = inf LB on unsafe union
        beta_RA = float("inf")
        for box in XU.sets:
            lb, _ = ibp_bounds_on_box(
                verifier, box, mesh_size=mesh_size, batch_size=batch_size
            )
            beta_RA = min(beta_RA, lb)

        eps_orig = 1.0 - (alpha_RA / beta_RA)

        # VeRecycle
        t0 = time.perf_counter()
        eps_reclaimed, m_lb = run_continuous_VeRecycle(
            verifier=verifier,
            alpha_RA=alpha_RA,
            beta_RA=beta_RA,
            disrupted_region=disrupted_region,
            mesh_size=mesh_size,
            batch_size=1000,
        )
        t1 = time.perf_counter()
        verecycle_time = t1 - t0

        # Algebraic consistency check only
        eps_formula = min(
            1.0 - alpha_RA / beta_RA,
            max(0.0, 1.0 - alpha_RA / m_lb) if m_lb > 0 else 0.0,
        )
        eps_formula = max(min(eps_formula, 1.0), 0.0)
        absdiff = abs(eps_reclaimed - eps_formula)

        print(
            f"{mesh_size:.4f}\t"
            f"{alpha_RA:.6f}\t{beta_RA:.6f}\t{m_lb:.6f}\t"
            f"{eps_orig:.6f}\t{eps_reclaimed:.6f}\t"
            f"{verecycle_time:.4f}\t{absdiff:.1e}"
        )

        results.append((
            mesh_size, alpha_RA, beta_RA, m_lb,
            eps_orig, eps_reclaimed, verecycle_time, absdiff
        ))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mesh_csv_path = RESULTS_DIR / f"ct_verecycle_mesh_sweep_{timestamp}.csv"

    with open(mesh_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mesh_size", "alpha_RA", "beta_RA", "m_lb",
            "eps_orig", "eps_reclaimed", "verecycle_time_s", "formula_absdiff",
            "Xq_low0", "Xq_low1", "Xq_high0", "Xq_high1"
        ])
        for (mesh_size, alpha_RA, beta_RA, m_lb, eps_orig, eps_reclaimed, verecycle_time, absdiff) in results:
            writer.writerow([
                mesh_size, alpha_RA, beta_RA, m_lb,
                eps_orig, eps_reclaimed, verecycle_time, absdiff,
                float(Xq.low[0]), float(Xq.low[1]), float(Xq.high[0]), float(Xq.high[1])
            ])

    print(f"\nSaved mesh sweep CSV to: {mesh_csv_path}")

    # ------------------------------------------------------------
    # Experiment 2 — Random X_q sweep
    # ------------------------------------------------------------
    np.random.seed(0)

    mesh_size = mesh_sizes[-1]
    n_regions = 20

    center_low = np.array([-12.0, -4.0], dtype=np.float32)
    center_high = np.array([-4.0, 0.0], dtype=np.float32)
    half_width = np.array([1.0, 0.5], dtype=np.float32)

    print("\n=== Random X_q sweep (fixed mesh) ===")
    print("i\tXq_low\t\tXq_high\t\tm_lb\teps_rec\ttime_s")

    sweep_rows = []

    # alpha_RA and beta_RA do not depend on X_q
    _, alpha_RA = ibp_bounds_on_box(
        verifier, X0, mesh_size=mesh_size, batch_size=batch_size
    )

    beta_RA = float("inf")
    for box in XU.sets:
        lb, _ = ibp_bounds_on_box(
            verifier, box, mesh_size=mesh_size, batch_size=batch_size
        )
        beta_RA = min(beta_RA, lb)

    for i in range(n_regions):
        Xq_i = sample_Xq_box(center_low, center_high, half_width)
        disrupted_i = MultiBox([Xq_i])

        t0 = time.perf_counter()
        eps_reclaimed, m_lb = run_continuous_VeRecycle(
            verifier=verifier,
            alpha_RA=alpha_RA,
            beta_RA=beta_RA,
            disrupted_region=disrupted_i,
            mesh_size=mesh_size,
            batch_size=1000,
        )
        t1 = time.perf_counter()
        verecycle_time = t1 - t0

        eps_formula = min(
            1.0 - alpha_RA / beta_RA,
            max(0.0, 1.0 - alpha_RA / m_lb) if m_lb > 0 else 0.0,
        )
        eps_formula = max(min(eps_formula, 1.0), 0.0)
        absdiff = abs(eps_reclaimed - eps_formula)

        print(
            f"{i}\t"
            f"[{Xq_i.low[0]:.2f},{Xq_i.low[1]:.2f}]\t"
            f"[{Xq_i.high[0]:.2f},{Xq_i.high[1]:.2f}]\t"
            f"{m_lb:.4f}\t{eps_reclaimed:.4f}\t{verecycle_time:.4f}"
        )

        sweep_rows.append((
            i,
            float(Xq_i.low[0]), float(Xq_i.low[1]),
            float(Xq_i.high[0]), float(Xq_i.high[1]),
            float(alpha_RA), float(beta_RA),
            float(m_lb), float(eps_reclaimed),
            float(verecycle_time), float(absdiff)
        ))

    eps_vals = np.array([r[8] for r in sweep_rows], dtype=np.float64)
    time_vals = np.array([r[9] for r in sweep_rows], dtype=np.float64)

    print("\n=== Summary (Random X_q sweep) ===")
    print(f"mesh_size = {mesh_size}")
    print(
        f"eps_reclaimed: mean={eps_vals.mean():.6f}, "
        f"std={eps_vals.std(ddof=1):.6f}, "
        f"min={eps_vals.min():.6f}, max={eps_vals.max():.6f}"
    )
    print(
        f"time_s: mean={time_vals.mean():.4f}, "
        f"std={time_vals.std(ddof=1):.4f}"
    )

    random_csv_path = RESULTS_DIR / f"ct_verecycle_randomXq_{timestamp}.csv"

    with open(random_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "i",
            "Xq_low0", "Xq_low1", "Xq_high0", "Xq_high1",
            "mesh_size",
            "alpha_RA", "beta_RA",
            "m_lb", "eps_reclaimed",
            "verecycle_time_s", "formula_absdiff"
        ])
        for r in sweep_rows:
            i, xl0, xl1, xh0, xh1, a, b, m_lb, eps, t, d = r
            writer.writerow([i, xl0, xl1, xh0, xh1, mesh_size, a, b, m_lb, eps, t, d])

    print(f"\nSaved random-X_q sweep CSV to: {random_csv_path}")

    # ------------------------------------------------------------
    # Experiment 3 — Monte Carlo sanity check
    # ------------------------------------------------------------
    print(f"\nUsing mesh_size={mesh_size} for Monte Carlo comparison...\n")

    _, alpha_RA = ibp_bounds_on_box(
        verifier, X0, mesh_size=mesh_size, batch_size=batch_size
    )

    beta_RA = float("inf")
    for box in XU.sets:
        lb, _ = ibp_bounds_on_box(
            verifier, box, mesh_size=mesh_size, batch_size=batch_size
        )
        beta_RA = min(beta_RA, lb)

    eps_reclaimed, m_lb = run_continuous_VeRecycle(
        verifier=verifier,
        alpha_RA=alpha_RA,
        beta_RA=beta_RA,
        disrupted_region=disrupted_region,
        mesh_size=mesh_size,
        batch_size=1000,
    )

    print(f"[MC compare] eps_reclaimed (original X_q) = {eps_reclaimed:.6f}")
    print("Running Monte Carlo on disrupted dynamics...")

    sde_new = LocalDisruptionPendulum(policy_net, Xq=Xq, noise_scale=1.2)
    p_hat, ci = simulate_reach_avoid(
        sde=sde_new,
        X0=X0,
        XT=XT,
        XU=XU,
        N=3000,
        T=5.0,
        dt=0.01,
        device=device,
    )

    print(f"MC p_hat = {p_hat:.6f}, 95% CI = [{ci[0]:.6f}, {ci[1]:.6f}]")
    print("Check:")
    print(f"  p_hat >= eps_reclaimed ?  {p_hat >= eps_reclaimed}")


if __name__ == "__main__":
    main()