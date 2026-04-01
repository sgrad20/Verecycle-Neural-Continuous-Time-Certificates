import sys
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math
import time
import numpy as np
import numpy.random as npr
import torch
import matplotlib.pyplot as plt

import controlled_sde
from rl_agent import TanhPolicy
import stochastic_rsa as rsa
from stochastic_rsa.continuous_vere_cycle import run_continuous_VeRecycle
from auto_LiRPA import BoundedModule

torch.set_default_dtype(torch.float32)
torch.use_deterministic_algorithms(True)

DEVICE = torch.device("cpu")
RESULTS_DIR = REPO_ROOT / "experiments_ct" / "results_debug_v"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class SimpleRegion:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class SimpleRegionUnion:
    def __init__(self, regions):
        self.sets = regions


# ---------------------------------------------------------------------
# Pendulum setup
# ---------------------------------------------------------------------
global_bounds = np.array([[[-20.0, -2 * np.pi], [20.0, 2 * np.pi]]], dtype=np.float32)
initial_bounds = np.array([[[-1.0, 3 / 4 * np.pi], [1.0, 5 / 4 * np.pi]]], dtype=np.float32)
target_bounds = np.array([[[-4.0, -np.pi / 2], [4.0, np.pi / 2]]], dtype=np.float32)
unsafe_bounds = np.array(
    [
        [[-20.0, -2 * np.pi], [-10.0, -3 / 2 * np.pi]],
        [[10.0, 3 / 2 * np.pi], [20.0, 2 * np.pi]],
    ],
    dtype=np.float32,
)

interest_set = rsa.AABBSet(global_bounds, DEVICE)
initial_set = rsa.AABBSet(initial_bounds, DEVICE)
target_set = rsa.AABBSet(target_bounds, DEVICE)
unsafe_set = rsa.AABBSet(unsafe_bounds, DEVICE)

# Match your main experiment setup if you want comparable values
spec = rsa.Specification(
    interest_set,
    initial_set,
    unsafe_set,
    target_set,
    0.9,
    0.0,
)

# ---------------------------------------------------------------------
# Monte Carlo horizon
# ---------------------------------------------------------------------
DT = 0.1
N_STEPS = 20
T_SIZE = N_STEPS + 1
TS = torch.linspace(0.0, DT * N_STEPS, T_SIZE, device=DEVICE)


def load_everything():
    ckpt_path = REPO_ROOT / "certified.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    policy = TanhPolicy(2, 1, 64, device=DEVICE)
    policy.load_state_dict(ckpt["policy"])
    policy.requires_grad_(False)
    policy.eval()

    sde = controlled_sde.InvertedPendulum(policy)

    net = rsa.CertificateModule(device=DEVICE)
    net.load_state_dict(ckpt["certificate"])
    net.eval()

    certificate = rsa.SupermartingaleCertificate(sde, spec, net, DEVICE)
    return policy, sde, net, certificate


def make_verifier(net):
    dummy_input = torch.zeros(1, 2, dtype=torch.float32, device=DEVICE)
    verifier = BoundedModule(net, (dummy_input,), device=DEVICE)
    return verifier


def estimate_alpha(net, initial_set, n_samples=5000):
    xs = initial_set.sample(n_samples)
    with torch.no_grad():
        vals = net(xs).cpu().numpy().reshape(-1)
    return float(vals.max())


def estimate_beta(net, unsafe_set, n_samples=5000):
    xs = unsafe_set.sample(n_samples)
    with torch.no_grad():
        vals = net(xs).cpu().numpy().reshape(-1)
    return float(vals.min())


def in_any_box(x_np: np.ndarray, bounds_np: np.ndarray) -> bool:
    for box in bounds_np:
        low = box[0]
        high = box[1]
        if np.all(x_np >= low) and np.all(x_np <= high):
            return True
    return False


def in_target(x_np: np.ndarray) -> bool:
    return in_any_box(x_np, target_bounds)


def in_unsafe(x_np: np.ndarray) -> bool:
    return in_any_box(x_np, unsafe_bounds)


def rollout_once_original(sde, x0_np: np.ndarray) -> bool:
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    path = sde.sample(x0, TS, method="srk").squeeze(0).detach().cpu().numpy()

    for x in path:
        if in_unsafe(x):
            return False
        if in_target(x):
            return True

    return False


def monte_carlo_original(sde, n_samples=20, seed=0):
    npr.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    successes = 0
    for i in range(n_samples):
        x0 = np.random.uniform(initial_bounds[0, 0], initial_bounds[0, 1]).astype(np.float32)
        if rollout_once_original(sde, x0):
            successes += 1
        print(f"MC rollout {i + 1}/{n_samples}", flush=True)

    p = successes / n_samples
    radius = 1.96 * math.sqrt(max(p * (1.0 - p), 1e-12) / n_samples)
    lo = max(0.0, p - radius)
    hi = min(1.0, p + radius)
    return p, lo, hi


def cert_value_raw(net, x_np: np.ndarray) -> float:
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        v = net(x).squeeze().item()
    return float(v)


def estimate_m_on_box_grid(net, low, high, nx=31, ny=31):
    xs = np.linspace(low[0], high[0], nx, dtype=np.float32)
    ys = np.linspace(low[1], high[1], ny, dtype=np.float32)

    m = float("inf")
    argmin = None

    for x in xs:
        for y in ys:
            pt = np.array([x, y], dtype=np.float32)
            v = cert_value_raw(net, pt)
            if v < m:
                m = v
                argmin = pt.copy()

    return m, argmin


def make_disrupted_region(low, high):
    return SimpleRegionUnion([SimpleRegion(low, high)])


def classify_regime(m, alpha_ra, beta_ra):
    if m <= alpha_ra:
        return "ZERO-CASE (m <= alpha)"
    if m < beta_ra:
        return "TRADEOFF-CASE (alpha < m < beta)"
    return "SATURATION-CASE (m >= beta)"


def make_plots(rows, out_dir: Path):
    if not rows:
        return

    names = [row["name"] for row in rows]
    m_lb = [row["m_lb_ibp"] for row in rows]
    m_grid = [row["m_grid_min_raw"] for row in rows]
    vr = [row["verecycle_bound"] for row in rows]

    x = np.arange(len(rows))

    plt.figure(figsize=(10, 5))
    plt.bar(x, m_lb)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("m_lb_ibp")
    plt.title("IBP lower bound over changed region")
    plt.tight_layout()
    plt.savefig(out_dir / "m_lb_ibp.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, m_lb, width=0.4, label="m_lb_ibp")
    plt.bar(x + 0.2, m_grid, width=0.4, label="m_grid_min_raw")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Certificate value")
    plt.title("IBP lower bound vs raw grid minimum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "m_lb_vs_m_grid.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, vr)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("VeRecycle bound")
    plt.title("Reclaimed VeRecycle bounds")
    plt.tight_layout()
    plt.savefig(out_dir / "verecycle_bounds.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(m_lb, vr, s=80)
    for i, name in enumerate(names):
        plt.text(m_lb[i], vr[i], name, fontsize=8, va="bottom", ha="right")
    plt.xlabel("m_lb_ibp")
    plt.ylabel("VeRecycle bound")
    plt.title("Local lower bound vs VeRecycle guarantee")
    plt.tight_layout()
    plt.savefig(out_dir / "m_lb_vs_verecycle.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    _, sde, net, _ = load_everything()
    verifier = make_verifier(net)

    alpha_ra = estimate_alpha(net, initial_set)
    beta_ra = estimate_beta(net, unsafe_set)
    rho_orig = 1.0 - alpha_ra / beta_ra

    print("\n=== ESTIMATED CERTIFICATE PARAMETERS ===")
    print(f"alpha (max over X0)       = {alpha_ra:.6f}")
    print(f"beta  (min over unsafe)   = {beta_ra:.6f}")
    print(f"alpha/beta                = {alpha_ra / beta_ra:.6f}")
    print(f"implied reach-avoid prob  = {rho_orig:.6f}")
    print()

    print("=" * 80)
    print("CERTIFICATE / ORIGINAL-MC DEBUG")
    print("=" * 80)
    print(f"alpha_RA            = {alpha_ra:.6f}")
    print(f"beta_RA             = {beta_ra:.6f}")
    print(f"original bound      = {rho_orig:.6f}")
    print()

    p_mc, lo_mc, hi_mc = monte_carlo_original(sde, n_samples=20, seed=0)
    print("Original dynamics Monte Carlo")
    print(f"  MC success        = {p_mc:.6f}")
    print(f"  95% CI            = [{lo_mc:.6f}, {hi_mc:.6f}]")
    print()

    xq_cases = [
        {
            "name": "xq_far_right_small",
            "low": np.array([11.0, 3.5], dtype=np.float32),
            "high": np.array([12.5, 4.5], dtype=np.float32),
        },
        {
            "name": "xq_far_right_mid",
            "low": np.array([9.0, 3.0], dtype=np.float32),
            "high": np.array([12.0, 5.0], dtype=np.float32),
        },
        {
            "name": "xq_mid_right_upper",
            "low": np.array([4.0, 2.0], dtype=np.float32),
            "high": np.array([7.0, 3.5], dtype=np.float32),
        },
        {
            "name": "xq_transition_7_2p0_to_8_3p0",
            "low": np.array([7.0, 2.0], dtype=np.float32),
            "high": np.array([8.0, 3.0], dtype=np.float32),
        },
        {
            "name": "xq_transition_7p5_2p3_to_9_3p3",
            "low": np.array([7.5, 2.3], dtype=np.float32),
            "high": np.array([9.0, 3.3], dtype=np.float32),
        },
        {
            "name": "xq_transition_8_2p5_to_9p5_3p5",
            "low": np.array([8.0, 2.5], dtype=np.float32),
            "high": np.array([9.5, 3.5], dtype=np.float32),
        },
        {
            "name": "xq_transition_8p5_2p7_to_10_3p8",
            "low": np.array([8.5, 2.7], dtype=np.float32),
            "high": np.array([10.0, 3.8], dtype=np.float32),
        },
        {
            "name": "xq_transition_9_3_to_11_4",
            "low": np.array([9.0, 3.0], dtype=np.float32),
            "high": np.array([11.0, 4.0], dtype=np.float32),
        },
        {
            "name": "xq_borderline_nonzero",
            "low": np.array([4.5, 2.1], dtype=np.float32),
            "high": np.array([6.0, 3.0], dtype=np.float32),
        }
    ]

    print("=" * 80)
    print("VeRecycle REGIME DEBUG")
    print("=" * 80)

    rows = []
    for case in xq_cases:
        name = case["name"]
        low = case["low"]
        high = case["high"]

        disrupted_region = make_disrupted_region(low, high)

        eps_rec, m_lb_ibp = run_continuous_VeRecycle(
            verifier=verifier,
            alpha_RA=alpha_ra,
            beta_RA=beta_ra,
            disrupted_region=disrupted_region,
            mesh_size=0.05,
            batch_size=1000,
        )

        m_grid, argmin_grid = estimate_m_on_box_grid(net, low, high, nx=31, ny=31)
        regime = classify_regime(m_lb_ibp, alpha_ra, beta_ra)

        row = {
            "name": name,
            "low_0": float(low[0]),
            "low_1": float(low[1]),
            "high_0": float(high[0]),
            "high_1": float(high[1]),
            "alpha_ra": alpha_ra,
            "beta_ra": beta_ra,
            "original_bound": rho_orig,
            "m_lb_ibp": float(m_lb_ibp),
            "m_grid_min_raw": float(m_grid),
            "m_grid_argmin_0": float(argmin_grid[0]),
            "m_grid_argmin_1": float(argmin_grid[1]),
            "verecycle_bound": float(eps_rec),
            "regime": regime,
            "mc_original": p_mc,
            "mc_original_ci_low": lo_mc,
            "mc_original_ci_high": hi_mc,
        }
        rows.append(row)

        print("-" * 80)
        print(f"name                : {name}")
        print(f"low                 : {low}")
        print(f"high                : {high}")
        print(f"m_lb_ibp            : {m_lb_ibp:.6f}")
        print(f"m_grid_min_raw      : {m_grid:.6f}")
        print(f"m_grid_argmin       : {argmin_grid}")
        print(f"regime              : {regime}")
        print(f"VeRecycle bound     : {eps_rec:.6f}")
        print()

    rows.sort(key=lambda r: r["m_lb_ibp"])

    csv_path = out_dir / "debug_v_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_plots(rows, out_dir)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved plots to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()