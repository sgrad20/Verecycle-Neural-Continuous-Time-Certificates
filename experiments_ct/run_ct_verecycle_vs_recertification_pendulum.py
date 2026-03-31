import sys
import csv
import math
import time
import traceback
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
RESULTS_DIR = REPO_ROOT / "experiments_ct" / "results_ct_pendulum_final_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Experiment knobs
# ---------------------------------------------------------------------
# Original default values:
# MC_SAMPLES = 100
# RECERT_TIMEOUT_S = 500
# Temporary quick-test settings:
# MC_SAMPLES = 10
# RECERT_TIMEOUT_S = 100000
MC_SAMPLES = 500
RECERT_TIMEOUT_S = 300
RECERT_VERIFY_EVERY_N = 1000
RECERT_VERIFIER_MESH = 400
RECERT_BATCH_SIZE = 128
RECERT_LR = 5e-4
RECERT_VERIFICATION_SLACK = 4.0
RECERT_REGULARIZER_LAMBDA = 1e-1
RECERT_ZETA = 1.0
RECERT_MAX_DEPTH = 4
RECERT_N_EPOCHS = 1000000
REACH_AVOID_PROBABILITY = 0.9
STAY_PROBABILITY = 0.0

# ---------------------------------------------------------------------
# Pendulum specification: same as your training script
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

spec = rsa.Specification(
    interest_set,
    initial_set,
    unsafe_set,
    target_set,
    REACH_AVOID_PROBABILITY,
    STAY_PROBABILITY,
)

# ---------------------------------------------------------------------
# Time horizon for Monte Carlo
# ---------------------------------------------------------------------
DT = 0.1
N_STEPS = 20
T_SIZE = N_STEPS + 1
TS = torch.linspace(0.0, DT * N_STEPS, T_SIZE, device=DEVICE)


# ---------------------------------------------------------------------
# Helper region classes for CT-VeRecycle
# ---------------------------------------------------------------------
class SimpleRegion:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class SimpleRegionUnion:
    def __init__(self, regions):
        self.sets = regions


def make_disrupted_region(low, high):
    return SimpleRegionUnion([SimpleRegion(low, high)])


@dataclass
class Scenario:
    name: str
    low: np.ndarray
    high: np.ndarray
    change_type: str
    diffusion_scale: float = 1.0
    drift_bias: tuple = (0.0, 0.0)
    description: str = ""


class LocalChangedPendulum:
    """
    Localized dynamics modification used for Monte Carlo and re-certification.

    For this repo:
    - self.base.f expects (t, x)
    - self.base.g expects (t, x)
    - both internally compute the control from the policy
    """

    def __init__(self, base_policy, scenario: Scenario):
        self.policy = base_policy
        self.base = controlled_sde.InvertedPendulum(base_policy)
        self.scenario = scenario

    def _inside_xq(self, x: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.scenario.low, dtype=x.dtype, device=x.device)
        high = torch.as_tensor(self.scenario.high, dtype=x.dtype, device=x.device)
        return torch.logical_and(x >= low, x <= high).all(dim=-1)

    def f(self, t, x: torch.Tensor) -> torch.Tensor:
        drift = self.base.f(t, x)
        inside = self._inside_xq(x)

        if not inside.any():
            return drift

        drift = drift.clone()
        if self.scenario.change_type == "drift_bias":
            bias = torch.as_tensor(
                self.scenario.drift_bias,
                dtype=drift.dtype,
                device=drift.device,
            )
            drift[inside] = drift[inside] + bias
        elif self.scenario.change_type != "diffusion_scale":
            raise ValueError(f"Unsupported change_type: {self.scenario.change_type}")

        return drift

    def g(self, t, x: torch.Tensor) -> torch.Tensor:
        diffusion = self.base.g(t, x)
        inside = self._inside_xq(x)

        if not inside.any():
            return diffusion

        diffusion = diffusion.clone()
        if self.scenario.change_type == "diffusion_scale":
            diffusion[inside] = diffusion[inside] * float(self.scenario.diffusion_scale)
        elif self.scenario.change_type != "drift_bias":
            raise ValueError(f"Unsupported change_type: {self.scenario.change_type}")

        return diffusion

    def sample(self, x0, ts, method="srk"):
        if method != "srk":
            raise ValueError("This wrapper only supports method='srk'.")

        x = x0.clone().to(DEVICE)
        out = [x.clone()]

        for i in range(1, ts.shape[0]):
            dt = float(ts[i] - ts[i - 1])
            sqrt_dt = math.sqrt(dt)

            x_next = []
            for b in range(x.shape[0]):
                xb = x[b:b + 1]

                with torch.no_grad():
                    drift = self.f(None, xb).reshape(-1).to(dtype=torch.float32, device=DEVICE)
                    diffusion = self.g(None, xb).reshape(-1).to(dtype=torch.float32, device=DEVICE)

                noise = torch.randn_like(diffusion)
                xn = xb.squeeze(0) + drift * dt + diffusion * sqrt_dt * noise
                x_next.append(xn)

            x = torch.stack(x_next, dim=0)
            out.append(x.clone())

        return torch.stack(out, dim=0)

    def close(self):
        if hasattr(self.base, "close"):
            self.base.close()

    def __getattr__(self, name):
        return getattr(self.base, name)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_certified_checkpoint():
    ckpt_path = REPO_ROOT / "certified.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    policy = TanhPolicy(2, 1, 64, device=DEVICE)
    policy.load_state_dict(ckpt["policy"])
    policy.requires_grad_(False)
    policy.eval()

    net = rsa.CertificateModule(device=DEVICE)
    net.load_state_dict(ckpt["certificate"])
    net.eval()

    base_sde = controlled_sde.InvertedPendulum(policy)
    certificate = rsa.SupermartingaleCertificate(base_sde, spec, net, DEVICE)

    return policy, net, base_sde, certificate


def make_verifier(net):
    dummy_input = torch.zeros(1, 2, dtype=torch.float32, device=DEVICE)
    return BoundedModule(net, (dummy_input,), device=DEVICE)


# ---------------------------------------------------------------------
# Certificate statistics
# ---------------------------------------------------------------------
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


def cert_value_raw(net, x_np: np.ndarray) -> float:
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        v = net(x).squeeze().item()
    return float(v)


def estimate_m_on_box_grid(net, low, high, nx=41, ny=41):
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


# ---------------------------------------------------------------------
# Reach-avoid Monte Carlo
# ---------------------------------------------------------------------
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


def rollout_once(sde, x0_np: np.ndarray) -> bool:
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    path = sde.sample(x0, TS, method="srk").squeeze(0).detach().cpu().numpy()

    for x in path:
        if in_unsafe(x):
            return False
        if in_target(x):
            return True
    return False


def monte_carlo_reach_avoid(sde, n_samples=MC_SAMPLES, seed=0, show_progress=False):
    npr.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    successes = 0
    for i in range(n_samples):
        x0 = np.random.uniform(initial_bounds[0, 0], initial_bounds[0, 1]).astype(np.float32)
        if rollout_once(sde, x0):
            successes += 1
        if show_progress and ((i + 1) % 20 == 0 or i == n_samples - 1):
            print(f"    MC {i + 1}/{n_samples}")

    p = successes / n_samples
    radius = 1.96 * math.sqrt(max(p * (1.0 - p), 1e-12) / n_samples)
    lo = max(0.0, p - radius)
    hi = min(1.0, p + radius)
    return p, lo, hi


# ---------------------------------------------------------------------
# Re-certification worker with timeout
# ---------------------------------------------------------------------
def scenario_to_payload(scenario: Scenario):
    return {
        "name": scenario.name,
        "low": scenario.low.tolist(),
        "high": scenario.high.tolist(),
        "change_type": scenario.change_type,
        "diffusion_scale": float(scenario.diffusion_scale),
        "drift_bias": tuple(float(x) for x in scenario.drift_bias),
        "description": scenario.description,
    }


def payload_to_scenario(payload):
    return Scenario(
        name=payload["name"],
        low=np.asarray(payload["low"], dtype=np.float32),
        high=np.asarray(payload["high"], dtype=np.float32),
        change_type=payload["change_type"],
        diffusion_scale=float(payload["diffusion_scale"]),
        drift_bias=tuple(payload["drift_bias"]),
        description=payload["description"],
    )


def _recert_worker(policy_state_dict, certificate_state_dict, scenario_payload, queue):
    start = time.time()
    try:
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)

        device = torch.device("cpu")
        scenario = payload_to_scenario(scenario_payload)

        local_interest_set = rsa.AABBSet(global_bounds, device)
        local_initial_set = rsa.AABBSet(initial_bounds, device)
        local_target_set = rsa.AABBSet(target_bounds, device)
        local_unsafe_set = rsa.AABBSet(unsafe_bounds, device)
        local_spec = rsa.Specification(
            local_interest_set,
            local_initial_set,
            local_unsafe_set,
            local_target_set,
            REACH_AVOID_PROBABILITY,
            0.0,
        )

        policy = TanhPolicy(2, 1, 64, device=device)
        policy.load_state_dict(policy_state_dict)
        policy.requires_grad_(False)
        policy.eval()

        modified_sde = LocalChangedPendulum(policy, scenario)

        torch.manual_seed(0)
        npr.seed(0)
        np.random.seed(0)

        net = rsa.CertificateModule(device=device)
        net.load_state_dict(certificate_state_dict)
        net.train(True)

        certificate = rsa.SupermartingaleCertificate(modified_sde, local_spec, net, device)

        result = certificate.train(
            n_epochs=RECERT_N_EPOCHS,
            batch_size=RECERT_BATCH_SIZE,
            lr=RECERT_LR,
            verify_every_n=RECERT_VERIFY_EVERY_N,
            verifier_mesh_size=RECERT_VERIFIER_MESH,
            zeta=RECERT_ZETA,
            regularizer_lambda=RECERT_REGULARIZER_LAMBDA,
            verification_slack=RECERT_VERIFICATION_SLACK,
            max_depth=RECERT_MAX_DEPTH,
        )
        elapsed = time.time() - start
        modified_sde.close()

        certified = bool(result[0]) if isinstance(result, (tuple, list)) and len(result) >= 1 else bool(result)
        epoch = int(result[1]) if isinstance(result, (tuple, list)) and len(result) >= 2 else -1

        alpha_ra = float("nan")
        beta_ra = float("nan")
        rho = float("nan")
        if certified:
            alpha_ra = estimate_alpha(net, local_initial_set)
            beta_ra = estimate_beta(net, local_unsafe_set)
            if beta_ra > 0.0:
                rho = max(1.0 - alpha_ra / beta_ra, 0.0)
            else:
                rho = 0.0

        queue.put(
            {
                "status": "certified" if certified else "not_certified",
                "rho": rho,
                "alpha_ra": alpha_ra,
                "beta_ra": beta_ra,
                "epoch": epoch,
                "elapsed": elapsed,
                "history": [
                    {
                        "epoch": epoch,
                        "rho": rho,
                    }
                ],
            }
        )
    except Exception:
        queue.put(
            {
                "status": "error",
                "error": traceback.format_exc(),
                "rho": float("nan"),
                "elapsed": time.time() - start,
                "history": [],
            }
        )


def recertify_on_scenario(policy, net, scenario: Scenario, timeout_s=RECERT_TIMEOUT_S):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_recert_worker,
        args=(policy.state_dict(), net.state_dict(), scenario_to_payload(scenario), queue),
    )

    start = time.time()
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        elapsed = time.time() - start
        return (
            float("nan"),
            0,
            elapsed,
            [{"epoch": -1, "rho": float("nan")}],
            "timeout",
            f"terminated after {timeout_s}s",
        )

    elapsed = time.time() - start

    if not queue.empty():
        msg = queue.get()
        status = msg.get("status", "unknown")
        if status == "certified":
            return (
                msg.get("rho", float("nan")),
                msg.get("epoch", -1),
                msg.get("elapsed", elapsed),
                msg.get("history", []),
                "certified",
                "",
            )
        return (
            msg.get("rho", float("nan")),
            0,
            msg.get("elapsed", elapsed),
            msg.get("history", []),
            status,
            msg.get("error", ""),
        )

    return (
        float("nan"),
        0,
        elapsed,
        [{"epoch": -1, "rho": float("nan")}],
        "no_result",
        "worker exited without returning a result",
    )


# ---------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------
def get_scenarios():
    return [
        Scenario(
            name="easy_recert_far_right_small",
            low=np.array([12.5, 3.8], dtype=np.float32),
            high=np.array([13.5, 4.2], dtype=np.float32),
            change_type="diffusion_scale",
            diffusion_scale=1.01,
            description=(
                "Easy recertification candidate: very small far-right box with "
                "minimal diffusion increase. Intended to produce one successful warm-start baseline."
            ),
        ),
        Scenario(
            name="saturation_case_far_right",
            low=np.array([10.0, 3.0], dtype=np.float32),
            high=np.array([15.0, 5.0], dtype=np.float32),
            change_type="diffusion_scale",
            diffusion_scale=1.2,
            description=(
                "Near-saturation case: high-value far-right region with moderate "
                "diffusion increase. Good for showing fast high reclaimed bounds."
            ),
        ),
        Scenario(
            name="tradeoff_case_mid_right_band",
            low=np.array([3.0, 1.5], dtype=np.float32),
            high=np.array([6.0, 3.0], dtype=np.float32),
            change_type="diffusion_scale",
            diffusion_scale=1.2,
            description=(
                "Tradeoff case: mid-right band, expected to have alpha < m < beta "
                "so VeRecycle gives a nonzero but noticeably reduced bound."
            ),
        ),
        Scenario(
            name="tradeoff_case_mid_right_drift",
            low=np.array([3.0, 1.5], dtype=np.float32),
            high=np.array([6.0, 3.0], dtype=np.float32),
            change_type="drift_bias",
            drift_bias=(-0.3, -0.1),
            description=(
                "Tradeoff case with mild drift degradation. Useful if diffusion-only "
                "modifications remain too easy/high."
            ),
        ),
        Scenario(
            name="zero_case_goal_side",
            low=np.array([-3.0, 1.6], dtype=np.float32),
            high=np.array([-1.0, 2.6], dtype=np.float32),
            change_type="diffusion_scale",
            diffusion_scale=1.2,
            description=(
                "Zero-case candidate near a low-certificate region. Expected m <= alpha, "
                "so VeRecycle should return zero or near-zero."
            ),
        ),
        Scenario(
            name="zero_case_goal_side_drift",
            low=np.array([-3.0, 1.6], dtype=np.float32),
            high=np.array([-1.0, 2.6], dtype=np.float32),
            change_type="drift_bias",
            drift_bias=(-0.5, -0.2),
            description=(
                "More severe zero-case candidate near a low-certificate region, "
                "using drift bias instead of diffusion scaling."
            ),
        ),
    ]


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def _plot_safe(values):
    out = []
    for v in values:
        if v is None or not np.isfinite(v):
            out.append(0.0)
        else:
            out.append(float(v))
    return out


def make_summary_plots(summary_rows, out_dir: Path):
    scenario_names = [row["scenario_name"] for row in summary_rows]
    vr_bounds = _plot_safe([row["verecycle_bound"] for row in summary_rows])
    rr_bounds = _plot_safe([row["recert_bound"] for row in summary_rows])
    mc_bounds = _plot_safe([row["mc_success"] for row in summary_rows])
    vr_times = _plot_safe([row["verecycle_time_s"] for row in summary_rows])
    rr_times = _plot_safe([row["recert_time_s"] for row in summary_rows])
    m_lb_ibp = _plot_safe([row["m_lb_ibp"] for row in summary_rows])
    speedups = _plot_safe([row["speedup"] for row in summary_rows])
    mc_ci_width = _plot_safe([
        row["mc_ci_high"] - row["mc_ci_low"] if row["mc_ci_high"] is not None and row["mc_ci_low"] is not None else float("nan")
        for row in summary_rows
    ])

    x = np.arange(len(summary_rows))
    width = 0.24

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, vr_bounds, width, label="VeRecycle")
    plt.bar(x, rr_bounds, width, label="Re-certification")
    plt.bar(x + width, mc_bounds, width, label="Monte Carlo")
    plt.xticks(x, scenario_names, rotation=15, ha="right")
    plt.ylabel("Reach-avoid probability")
    plt.title("Bound comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bounds_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, vr_times, width, label="VeRecycle")
    plt.bar(x + width / 2, rr_times, width, label="Re-certification")
    plt.xticks(x, scenario_names, rotation=15, ha="right")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width / 2, vr_times, width, label="VeRecycle")
    bars2 = plt.bar(x + width / 2, rr_times, width, label="Re-certification")
    plt.xticks(x, scenario_names, rotation=15, ha="right")
    plt.ylabel("Runtime (s, log scale)")
    plt.title("Runtime comparison (log scale)")
    plt.yscale("log")
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{h:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    plt.tight_layout()
    plt.savefig(out_dir / "runtime_comparison_log.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, m_lb_ibp, width, color="#4C72B0")
    plt.xticks(x, scenario_names, rotation=15, ha="right")
    plt.ylabel("Lower-bound m_lb_ibp")
    plt.title("Estimated local lower bound m_lb_ibp")
    plt.tight_layout()
    plt.savefig(out_dir / "m_lb_ibp_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, speedups, width, color="#55A868")
    plt.xticks(x, scenario_names, rotation=15, ha="right")
    plt.ylabel("Speedup (re-cert / VeRecycle)")
    plt.title("Measured speedup")
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, mc_ci_width, width, color="#C44E52")
    plt.xticks(x, scenario_names, rotation=15, ha="right")
    plt.ylabel("Monte Carlo CI width")
    plt.title("Monte Carlo 95% confidence interval width")
    plt.tight_layout()
    plt.savefig(out_dir / "mc_ci_width.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    gap = [rr - vr for rr, vr in zip(rr_bounds, vr_bounds)]
    plt.bar(x, gap, width, color="#8172B2")
    plt.xticks(x, scenario_names, rotation=15, ha="right")
    plt.ylabel("Recert - VeRecycle")
    plt.title("Bound gap comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "bound_gap_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(m_lb_ibp, vr_bounds, label="VeRecycle", s=80)
    plt.scatter(m_lb_ibp, rr_bounds, label="Re-certification", s=80)
    for i, name in enumerate(scenario_names):
        plt.text(m_lb_ibp[i], rr_bounds[i], name, fontsize=8, va="bottom", ha="right")
    plt.xlabel("m_lb_ibp")
    plt.ylabel("Reach-avoid probability")
    plt.title("Local lower bound vs guarantees")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "m_lb_ibp_vs_bounds.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(vr_times, vr_bounds, label="VeRecycle", s=80)
    plt.scatter(rr_times, rr_bounds, label="Re-certification", s=80)
    plt.xscale("log")
    plt.xlabel("Runtime (s, log scale)")
    plt.ylabel("Reach-avoid probability")
    plt.title("Runtime vs guarantee")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_vs_bound.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_recertification_history_plot(summary_rows, out_dir: Path):
    plt.figure(figsize=(8, 5))
    found = False

    for row in summary_rows:
        history_path = out_dir / f"{row['scenario_name']}_recert_history.csv"
        if not history_path.exists():
            continue

        epochs = []
        rhos = []
        with open(history_path, newline="") as f:
            reader = csv.DictReader(f)
            for history_row in reader:
                try:
                    epochs.append(int(history_row["epoch"]))
                    rhos.append(float(history_row["rho"]))
                except Exception:
                    continue

        if len(epochs) == 0:
            continue

        found = True
        plt.plot(epochs, rhos, marker="o", markersize=3, label=row["scenario_name"])

    if not found:
        return

    plt.xlabel("Epoch")
    plt.ylabel("Rho")
    plt.title("Re-certification convergence history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "recertification_history.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading certified.pt...")
    policy, net, base_sde, _ = load_certified_checkpoint()
    verifier = make_verifier(net)

    alpha_ra = estimate_alpha(net, initial_set)
    beta_ra = estimate_beta(net, unsafe_set)
    rho_orig = 1.0 - alpha_ra / beta_ra

    print("\n=== BASE CERTIFICATE ===")
    print(f"alpha_RA           = {alpha_ra:.6f}")
    print(f"beta_RA            = {beta_ra:.6f}")
    print(f"original bound     = {rho_orig:.6f}")

    print("\n=== ORIGINAL DYNAMICS MONTE CARLO ===")
    mc_orig, mc_orig_lo, mc_orig_hi = monte_carlo_reach_avoid(
        base_sde, n_samples=MC_SAMPLES, seed=0, show_progress=True
    )
    print(f"MC success         = {mc_orig:.6f}")
    print(f"95% CI             = [{mc_orig_lo:.6f}, {mc_orig_hi:.6f}]")

    scenarios = get_scenarios()
    summary_rows = []

    for scenario in scenarios:
        print("\n" + "#" * 80)
        print(f"SCENARIO: {scenario.name}")
        print(scenario.description)
        print("#" * 80)

        disrupted_region = make_disrupted_region(scenario.low, scenario.high)

        t0 = time.time()
        eps_vr, m_lb_ibp = run_continuous_VeRecycle(
            verifier=verifier,
            alpha_RA=alpha_ra,
            beta_RA=beta_ra,
            disrupted_region=disrupted_region,
            mesh_size=0.05,
            batch_size=1000,
        )
        t_vr = time.time() - t0

        m_grid, argmin_grid = estimate_m_on_box_grid(
            net, scenario.low, scenario.high, nx=41, ny=41
        )

        print(f"Running re-certification with timeout={RECERT_TIMEOUT_S}s ...")
        recert_bound, recert_certified_epoch, t_rr, history, recert_status, recert_note = recertify_on_scenario(
            policy, net, scenario, timeout_s=RECERT_TIMEOUT_S
        )

        print("Running modified-dynamics Monte Carlo...")
        modified_sde = LocalChangedPendulum(policy, scenario)
        mc_mod, mc_mod_lo, mc_mod_hi = monte_carlo_reach_avoid(
            modified_sde, n_samples=MC_SAMPLES, seed=0, show_progress=True
        )
        modified_sde.close()

        speedup = t_rr / t_vr if t_vr > 0 else float("inf")

        summary = {
            "scenario_name": scenario.name,
            "description": scenario.description,
            "change_type": scenario.change_type,
            "xq_low_0": float(scenario.low[0]),
            "xq_low_1": float(scenario.low[1]),
            "xq_high_0": float(scenario.high[0]),
            "xq_high_1": float(scenario.high[1]),
            "alpha_ra": alpha_ra,
            "beta_ra": beta_ra,
            "original_bound": rho_orig,
            "mc_original": mc_orig,
            "mc_original_ci_low": mc_orig_lo,
            "mc_original_ci_high": mc_orig_hi,
            "m_lb_ibp": m_lb_ibp,
            "m_grid_min_raw": m_grid,
            "m_grid_argmin_0": float(argmin_grid[0]),
            "m_grid_argmin_1": float(argmin_grid[1]),
            "verecycle_bound": eps_vr,
            "verecycle_time_s": t_vr,
            "recert_bound": recert_bound,
            "recert_certified_epoch": recert_certified_epoch,
            "recert_time_s": t_rr,
            "recert_status": recert_status,
            "recert_note": recert_note,
            "mc_success": mc_mod,
            "mc_ci_low": mc_mod_lo,
            "mc_ci_high": mc_mod_hi,
            "speedup": speedup,
        }
        summary_rows.append(summary)

        print("\nSummary")
        print(f"  m_lb_ibp          : {m_lb_ibp:.6f}")
        print(f"  m_grid_min_raw    : {m_grid:.6f}")
        print(f"  VeRecycle bound   : {eps_vr:.6f}")
        print(f"  Re-cert bound     : {recert_bound}")
        print(f"  Re-cert status    : {recert_status}")
        if recert_note:
            print(f"  Re-cert note      : {recert_note}")
        print(f"  MC success        : {mc_mod:.6f} [{mc_mod_lo:.6f}, {mc_mod_hi:.6f}]")
        print(f"  VeRecycle time    : {t_vr:.4f}s")
        print(f"  Re-cert time      : {t_rr:.4f}s")
        print(f"  Speedup           : {speedup:.2f}x")

        history_csv = out_dir / f"{scenario.name}_recert_history.csv"
        with open(history_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "rho"])
            writer.writeheader()
            writer.writerows(history)

    if summary_rows:
        summary_csv = out_dir / "summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        make_summary_plots(summary_rows, out_dir)
        make_recertification_history_plot(summary_rows, out_dir)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for row in summary_rows:
        print(f"\nScenario: {row['scenario_name']}")
        print(f"  VeRecycle bound   : {row['verecycle_bound']:.6f}")
        print(f"  Re-cert bound     : {row['recert_bound']}")
        print(f"  Re-cert status    : {row['recert_status']}")
        print(f"  MC success        : {row['mc_success']:.6f}")
        print(f"  Speedup           : {row['speedup']:.2f}x")
        print(f"  m_lb_ibp          : {row['m_lb_ibp']:.6f}")

    print(f"\nSaved results to: {out_dir}")
    base_sde.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()