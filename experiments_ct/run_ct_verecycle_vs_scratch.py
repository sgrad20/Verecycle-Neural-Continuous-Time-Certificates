import sys
import csv
import json
import re
import time
import traceback
import subprocess
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import numpy.random as npr
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pyparsing import PyparsingDeprecationWarning

    warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)
except Exception:
    pass

import controlled_sde
from rl_agent import TanhPolicy
import stochastic_rsa as rsa
from stochastic_rsa.continuous_vere_cycle import run_continuous_VeRecycle
from auto_LiRPA import BoundedModule

torch.set_default_dtype(torch.float32)
torch.use_deterministic_algorithms(True)

DEVICE = torch.device("cpu")
RESULTS_DIR = REPO_ROOT / "experiments_ct" / "results_verecycle_vs_scratch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REACH_AVOID_PROBABILITY = 0.9
STAY_PROBABILITY = 0.0
SCRATCH_REACH_AVOID_PROBABILITY = 0.8

RECERT_TIMEOUT_S = 120.0
RECERT_N_EPOCHS = 1_000_000
RECERT_VERIFY_EVERY_N = 1000
RECERT_VERIFIER_MESH = 400
RECERT_BATCH_SIZE = 128
RECERT_LR = 5e-4
RECERT_VERIFICATION_SLACK = 4.0
RECERT_REGULARIZER_LAMBDA = 1e-1
RECERT_ZETA = 1.0
RECERT_MAX_DEPTH = 4

VERECYCLE_MESH_SIZE = 0.05
VERECYCLE_BATCH_SIZE = 1000

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


def get_scenarios():
    return [
        Scenario(
            name="low_reclaim_diff_1p5",
            low=np.array([4.0, 2.0], dtype=np.float32),
            high=np.array([7.0, 3.5], dtype=np.float32),
            change_type="diffusion_scale",
            diffusion_scale=1.5,
            description="Low but nonzero reclaim region with stronger local diffusion.",
        ),
        Scenario(
            name="borderline_diff_2p0",
            low=np.array([4.5, 2.1], dtype=np.float32),
            high=np.array([6.0, 3.0], dtype=np.float32),
            change_type="diffusion_scale",
            diffusion_scale=2.0,
            description="Borderline reclaimable case with strong diffusion increase.",
        ),
        Scenario(
            name="mid_band_drift_mild",
            low=np.array([5.0, 2.0], dtype=np.float32),
            high=np.array([8.0, 3.5], dtype=np.float32),
            change_type="drift_bias",
            drift_bias=(-0.8, -0.4),
            description="Moderate drift degradation in a mid-value region.",
        ),
        Scenario(
            name="right_corridor_drift_strong",
            low=np.array([7.5, 2.0], dtype=np.float32),
            high=np.array([10.5, 4.0], dtype=np.float32),
            change_type="drift_bias",
            drift_bias=(-1.5, -0.8),
            description="Large drift-degraded corridor with still-positive VeRecycle reclaim.",
        ),
    ]


class SoftBoxGate(torch.nn.Module):
    def __init__(self, low, high, slope: float = 40.0):
        super().__init__()
        self.register_buffer("low", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.tensor(high, dtype=torch.float32))
        self.slope = float(slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_term = torch.sigmoid(self.slope * (x - self.low))
        high_term = torch.sigmoid(self.slope * (self.high - x))
        gate_per_dim = low_term * high_term
        return gate_per_dim[:, :1] * gate_per_dim[:, 1:2]


class LocalChangedDrift(torch.nn.Module):
    def __init__(self, base_drift: torch.nn.Module, scenario: Scenario):
        super().__init__()
        self.base_drift = base_drift
        self.scenario = scenario
        self.gate = SoftBoxGate(scenario.low, scenario.high)
        self.register_buffer(
            "bias",
            torch.tensor(self.scenario.drift_bias, dtype=torch.float32).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        drift = self.base_drift(x, u)
        gate = self.gate(x).to(dtype=drift.dtype, device=drift.device)

        if self.scenario.change_type == "drift_bias":
            return drift + gate * self.bias.to(dtype=drift.dtype, device=drift.device)
        if self.scenario.change_type == "diffusion_scale":
            return drift
        raise ValueError(f"Unsupported change_type: {self.scenario.change_type}")


class LocalChangedDiffusion(torch.nn.Module):
    def __init__(self, base_diffusion: torch.nn.Module, scenario: Scenario):
        super().__init__()
        self.base_diffusion = base_diffusion
        self.scenario = scenario
        self.gate = SoftBoxGate(scenario.low, scenario.high)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        diffusion = self.base_diffusion(x, u)
        gate = self.gate(x).to(dtype=diffusion.dtype, device=diffusion.device)

        if self.scenario.change_type == "diffusion_scale":
            scale = 1.0 + gate * (float(self.scenario.diffusion_scale) - 1.0)
            return diffusion * scale
        if self.scenario.change_type == "drift_bias":
            return diffusion
        raise ValueError(f"Unsupported change_type: {self.scenario.change_type}")


class LocalChangedPendulum(controlled_sde.ControlledSDE):
    def __init__(self, base_policy, scenario: Scenario):
        self.base = controlled_sde.InvertedPendulum(base_policy)
        self.scenario = scenario
        drift = LocalChangedDrift(self.base.drift, scenario)
        diffusion = LocalChangedDiffusion(self.base.diffusion, scenario)
        super().__init__(base_policy, drift, diffusion, self.base.noise_type, self.base.sde_type)

    def close(self):
        if hasattr(self.base, "close"):
            self.base.close()

    def n_dimensions(self) -> int:
        return self.base.n_dimensions()

    def __getattr__(self, name):
        return getattr(self.base, name)


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
    return policy, net, base_sde


def make_verifier(net):
    dummy_input = torch.zeros(1, 2, dtype=torch.float32, device=DEVICE)
    return BoundedModule(net, (dummy_input,), device=DEVICE)


def estimate_alpha(net, initial_set_obj, n_samples: int = 5000):
    xs = initial_set_obj.sample(n_samples)
    with torch.no_grad():
        vals = net(xs).cpu().numpy().reshape(-1)
    return float(vals.max())


def estimate_beta(alpha, probability=REACH_AVOID_PROBABILITY):
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must be strictly between 0 and 1")
    return float(alpha / (1.0 - probability))


def estimate_beta_on_unsafe(net, unsafe_set_obj, n_samples: int = 20000):
    xs = unsafe_set_obj.sample(n_samples)
    with torch.no_grad():
        vals = net(xs).cpu().numpy().reshape(-1)
    return float(vals.min())


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


def _extract_train_result(result):
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError(
            "certificate.train() is expected to return (bool, int), "
            f"got {type(result).__name__}: {result!r}"
        )
    return bool(result[0]), int(result[1])


def _extract_max_verified_prob_ra(log_text: str) -> float:
    matches = re.findall(
        r"Reach-avoid condition is satisfied with probability at least\s+([0-9]*\.?[0-9]+)",
        log_text,
    )
    if not matches:
        return float("nan")
    return max(float(match) for match in matches)


def _scratch_recert_attempt(policy_state_dict, scenario_payload, worker_seed, config):
    start = time.time()
    modified_sde = None
    try:
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)

        scenario = payload_to_scenario(scenario_payload)
        device = torch.device("cpu")

        torch.manual_seed(worker_seed)
        npr.seed(worker_seed)
        np.random.seed(worker_seed)

        local_interest_set = rsa.AABBSet(global_bounds, device)
        local_initial_set = rsa.AABBSet(initial_bounds, device)
        local_target_set = rsa.AABBSet(target_bounds, device)
        local_unsafe_set = rsa.AABBSet(unsafe_bounds, device)
        local_spec = rsa.Specification(
            local_interest_set,
            local_initial_set,
            local_unsafe_set,
            local_target_set,
            float(config.get("scratch_probability", SCRATCH_REACH_AVOID_PROBABILITY)),
            STAY_PROBABILITY,
        )

        policy = TanhPolicy(2, 1, 64, device=device)
        policy.load_state_dict(policy_state_dict)
        policy.requires_grad_(False)
        policy.eval()

        modified_sde = LocalChangedPendulum(policy, scenario)
        net = rsa.CertificateModule(device=device)
        certificate = rsa.SupermartingaleCertificate(modified_sde, local_spec, net, device)

        result = certificate.train(
            n_epochs=int(config["n_epochs"]),
            batch_size=int(config["batch_size"]),
            lr=float(config["lr"]),
            verify_every_n=int(config["verify_every_n"]),
            verifier_mesh_size=int(config["verifier_mesh_size"]),
            zeta=float(config["zeta"]),
            regularizer_lambda=float(config["regularizer_lambda"]),
            verification_slack=float(config["verification_slack"]),
            max_depth=int(config["max_depth"]),
        )

        certified, epoch = _extract_train_result(result)
        elapsed = time.time() - start

        alpha_ra = float("nan")
        beta_ra = float("nan")
        bound = float("nan")
        if certified:
            net.eval()
            alpha_ra = estimate_alpha(net, local_initial_set)
            beta_ra = estimate_beta_on_unsafe(net, local_unsafe_set)
            if np.isfinite(beta_ra) and beta_ra > 0.0:
                bound = max(1.0 - alpha_ra / beta_ra, 0.0)
            else:
                bound = 0.0

        return {
            "status": "certified" if certified else "not_certified",
            "seed": int(worker_seed),
            "epoch": epoch,
            "elapsed": elapsed,
            "bound": bound,
            "alpha_ra": alpha_ra,
            "beta_ra": beta_ra,
            "note": "",
        }
    except Exception:
        return {
            "status": "error",
            "seed": int(worker_seed),
            "epoch": -1,
            "elapsed": time.time() - start,
            "bound": float("nan"),
            "alpha_ra": float("nan"),
            "beta_ra": float("nan"),
            "note": traceback.format_exc(),
        }
    finally:
        try:
            if modified_sde is not None:
                modified_sde.close()
        except Exception:
            pass


def _worker_main(payload_path: str, result_path: str):
    payload = torch.load(payload_path, map_location="cpu")
    result = _scratch_recert_attempt(
        policy_state_dict=payload["policy_state_dict"],
        scenario_payload=payload["scenario_payload"],
        worker_seed=int(payload["worker_seed"]),
        config=payload["config"],
    )
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(result, file)


def run_scratch_attempt(policy, scenario: Scenario, worker_seed: int, config):
    with tempfile.TemporaryDirectory(prefix="ct_vr_scratch_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        payload_path = tmp_dir_path / "payload.pt"
        result_path = tmp_dir_path / "result.json"
        log_path = tmp_dir_path / "worker.log"

        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "scenario_payload": scenario_to_payload(scenario),
                "worker_seed": int(worker_seed),
                "config": config,
            },
            payload_path,
        )

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-payload",
            str(payload_path),
            "--worker-result",
            str(result_path),
        ]

        start = time.time()
        log_text = ""
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                subprocess.run(
                    command,
                    cwd=str(REPO_ROOT),
                    check=False,
                    timeout=float(config["timeout_s"]),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.TimeoutExpired:
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
            max_verified_prob_ra = _extract_max_verified_prob_ra(log_text)
            reach_only_success = (
                np.isfinite(max_verified_prob_ra)
                and max_verified_prob_ra >= float(config["scratch_probability"])
            )
            return {
                "status": "timeout",
                "seed": int(worker_seed),
                "epoch": -1,
                "elapsed": time.time() - start,
                "bound": float("nan"),
                "alpha_ra": float("nan"),
                "beta_ra": float("nan"),
                "max_verified_prob_ra": max_verified_prob_ra,
                "reach_only_success": bool(reach_only_success),
                "note": f"terminated after {config['timeout_s']}s",
            }

        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        max_verified_prob_ra = _extract_max_verified_prob_ra(log_text)
        reach_only_success = (
            np.isfinite(max_verified_prob_ra)
            and max_verified_prob_ra >= float(config["scratch_probability"])
        )

        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as file:
                result = json.load(file)
            result["max_verified_prob_ra"] = max_verified_prob_ra
            result["reach_only_success"] = bool(reach_only_success)
            return result

    return {
        "status": "no_result",
        "seed": int(worker_seed),
        "epoch": -1,
        "elapsed": time.time() - start,
        "bound": float("nan"),
        "alpha_ra": float("nan"),
        "beta_ra": float("nan"),
        "max_verified_prob_ra": max_verified_prob_ra,
        "reach_only_success": bool(reach_only_success),
        "note": "worker exited without returning a result",
    }


def best_certified_attempt(attempts):
    certified = [attempt for attempt in attempts if attempt["status"] == "certified"]
    if not certified:
        return None
    return max(
        certified,
        key=lambda attempt: (
            -float("inf") if not np.isfinite(attempt["bound"]) else float(attempt["bound"])
        ),
    )


def make_summary_plots(summary_rows, out_dir: Path):
    if not summary_rows:
        return

    names = [row["scenario_name"] for row in summary_rows]
    vr_bounds = [float(row["verecycle_bound"]) for row in summary_rows]
    scratch_bounds = [
        0.0 if not np.isfinite(row["scratch_best_bound"]) else float(row["scratch_best_bound"])
        for row in summary_rows
    ]
    vr_times = [float(row["verecycle_time_s"]) for row in summary_rows]
    scratch_times = [max(float(row["scratch_total_time_s"]), 1e-6) for row in summary_rows]

    x = np.arange(len(summary_rows))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, vr_bounds, width, label="VeRecycle")
    plt.bar(x + width / 2, scratch_bounds, width, label="Scratch re-certification")
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylabel("Certified / estimated bound")
    plt.title("VeRecycle vs scratch re-certification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bounds_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, vr_times, width, label="VeRecycle")
    plt.bar(x + width / 2, scratch_times, width, label="Scratch total wall time")
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylabel("Runtime (s, log scale)")
    plt.yscale("log")
    plt.title("Runtime comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_comparison_log.png", dpi=300, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare CT-VeRecycle against full re-certification from scratch. "
            "Each scratch attempt starts from a fresh random certificate network."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of fresh scratch attempts per scenario.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base random seed; attempts use base-seed + offset.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=RECERT_TIMEOUT_S,
        help="Per-seed wall-clock timeout for scratch re-certification.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Optional scenario name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--scratch-probability",
        type=float,
        default=SCRATCH_REACH_AVOID_PROBABILITY,
        help="Verified reach-avoid probability target for scratch re-certification.",
    )
    parser.add_argument("--worker-payload", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.worker_payload:
        _worker_main(args.worker_payload, args.worker_result)
        return

    if args.seeds <= 0:
        raise ValueError("--seeds must be positive.")
    if not (0.0 < args.scratch_probability < 1.0):
        raise ValueError("--scratch-probability must be strictly between 0 and 1.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_names = set(args.scenario)
    scenarios = get_scenarios()
    if selected_names:
        scenarios = [scenario for scenario in scenarios if scenario.name in selected_names]

    if not scenarios:
        raise ValueError("No scenarios selected.")

    print("Loading certified.pt...")
    policy, net, base_sde = load_certified_checkpoint()
    verifier = make_verifier(net)

    alpha_ra = estimate_alpha(net, initial_set)
    beta_ra = estimate_beta(alpha_ra, REACH_AVOID_PROBABILITY)
    original_bound = max(0.0, 1.0 - alpha_ra / beta_ra)

    print("\n=== BASE CERTIFICATE USED BY VERECYCLE ===")
    print(f"alpha_RA       = {alpha_ra:.6f}")
    print(f"beta_RA        = {beta_ra:.6f}")
    print(f"original bound = {original_bound:.6f}")
    print(f"scratch target = {args.scratch_probability:.3f}")

    config = {
        "timeout_s": float(args.timeout_s),
        "n_epochs": RECERT_N_EPOCHS,
        "batch_size": RECERT_BATCH_SIZE,
        "lr": RECERT_LR,
        "verify_every_n": RECERT_VERIFY_EVERY_N,
        "verifier_mesh_size": RECERT_VERIFIER_MESH,
        "zeta": RECERT_ZETA,
        "regularizer_lambda": RECERT_REGULARIZER_LAMBDA,
        "verification_slack": RECERT_VERIFICATION_SLACK,
        "max_depth": RECERT_MAX_DEPTH,
        "scratch_probability": float(args.scratch_probability),
    }

    attempt_rows = []
    summary_rows = []

    for scenario_idx, scenario in enumerate(scenarios):
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
            mesh_size=VERECYCLE_MESH_SIZE,
            batch_size=VERECYCLE_BATCH_SIZE,
        )
        t_vr = time.time() - t0

        attempts = []
        for seed_offset in range(args.seeds):
            worker_seed = int(args.base_seed + scenario_idx * 1000 + seed_offset)
            print(
                f"Scratch attempt {seed_offset + 1}/{args.seeds} "
                f"(seed={worker_seed}, timeout={args.timeout_s:.1f}s)..."
            )
            attempt = run_scratch_attempt(policy, scenario, worker_seed, config)
            attempts.append(attempt)
            attempt_rows.append(
                {
                    "scenario_name": scenario.name,
                    "seed": attempt["seed"],
                    "status": attempt["status"],
                    "epoch": attempt["epoch"],
                    "elapsed": attempt["elapsed"],
                    "bound": attempt["bound"],
                    "alpha_ra": attempt["alpha_ra"],
                    "beta_ra": attempt["beta_ra"],
                    "max_verified_prob_ra": attempt["max_verified_prob_ra"],
                    "reach_only_success": attempt["reach_only_success"],
                    "note": attempt["note"],
                }
            )
            print(
                f"  status={attempt['status']}, "
                f"bound={attempt['bound']}, "
                f"max_RA={attempt['max_verified_prob_ra']}, "
                f"time={attempt['elapsed']:.3f}s"
            )

        best_attempt = best_certified_attempt(attempts)
        n_certified = sum(1 for attempt in attempts if attempt["status"] == "certified")
        n_timeout = sum(1 for attempt in attempts if attempt["status"] == "timeout")
        n_error = sum(1 for attempt in attempts if attempt["status"] == "error")
        n_reach_only = sum(1 for attempt in attempts if attempt["reach_only_success"])
        scratch_total_time = sum(float(attempt["elapsed"]) for attempt in attempts)
        finite_verified_ra = [
            float(attempt["max_verified_prob_ra"])
            for attempt in attempts
            if np.isfinite(attempt["max_verified_prob_ra"])
        ]
        scratch_best_verified_prob_ra = (
            max(finite_verified_ra) if finite_verified_ra else float("nan")
        )

        if best_attempt is None:
            scratch_best_bound = float("nan")
            scratch_best_alpha = float("nan")
            scratch_best_beta = float("nan")
            scratch_best_epoch = -1
            scratch_best_seed = -1
        else:
            scratch_best_bound = float(best_attempt["bound"])
            scratch_best_alpha = float(best_attempt["alpha_ra"])
            scratch_best_beta = float(best_attempt["beta_ra"])
            scratch_best_epoch = int(best_attempt["epoch"])
            scratch_best_seed = int(best_attempt["seed"])

        gap_case = eps_vr > 0.0 and n_certified == 0 and n_reach_only == 0
        if n_certified > 0:
            verdict = "scratch fully certified at least once"
        elif n_reach_only > 0:
            verdict = "scratch reached the target RA threshold, but full certification did not finish"
        elif eps_vr > 0.0:
            verdict = "VeRecycle works but scratch did not reach the target RA threshold"
        else:
            verdict = "no positive VeRecycle reclaim and no scratch success"

        summary_rows.append(
            {
                "scenario_name": scenario.name,
                "change_type": scenario.change_type,
                "description": scenario.description,
                "xq_low_0": float(scenario.low[0]),
                "xq_low_1": float(scenario.low[1]),
                "xq_high_0": float(scenario.high[0]),
                "xq_high_1": float(scenario.high[1]),
                "alpha_ra": alpha_ra,
                "beta_ra": beta_ra,
                "original_bound": original_bound,
                "scratch_target_probability": float(args.scratch_probability),
                "m_lb_ibp": float(m_lb_ibp),
                "verecycle_bound": float(eps_vr),
                "verecycle_time_s": float(t_vr),
                "scratch_attempts": int(args.seeds),
                "scratch_successes": int(n_certified),
                "scratch_reach_only_passes": int(n_reach_only),
                "scratch_timeouts": int(n_timeout),
                "scratch_errors": int(n_error),
                "scratch_total_time_s": float(scratch_total_time),
                "scratch_best_verified_prob_ra": scratch_best_verified_prob_ra,
                "scratch_best_bound": scratch_best_bound,
                "scratch_best_alpha_ra": scratch_best_alpha,
                "scratch_best_beta_ra": scratch_best_beta,
                "scratch_best_epoch": int(scratch_best_epoch),
                "scratch_best_seed": int(scratch_best_seed),
                "gap_case": bool(gap_case),
                "verdict": verdict,
            }
        )

        print("\nSummary")
        print(f"  VeRecycle bound        : {eps_vr:.6f}")
        print(f"  VeRecycle time         : {t_vr:.4f}s")
        print(f"  Scratch full certs     : {n_certified}/{args.seeds}")
        print(f"  Scratch RA-only passes : {n_reach_only}/{args.seeds}")
        print(f"  Scratch total time     : {scratch_total_time:.4f}s")
        if np.isfinite(scratch_best_verified_prob_ra):
            print(f"  Scratch best verified RA: {scratch_best_verified_prob_ra:.6f}")
        if best_attempt is None:
            print("  Scratch best bound     : unavailable")
        else:
            print(f"  Scratch best bound     : {scratch_best_bound:.6f}")
            print(f"  Scratch best seed      : {scratch_best_seed}")
        print(f"  Verdict                : {verdict}")

    attempts_csv = out_dir / "scratch_attempts.csv"
    with open(attempts_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(attempt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(attempt_rows)

    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    make_summary_plots(summary_rows, out_dir)

    print("\n" + "=" * 80)
    print("CASES WHERE VERECYCLE WORKS BUT SCRATCH DID NOT CERTIFY")
    print("=" * 80)
    gap_rows = [row for row in summary_rows if row["gap_case"]]
    if not gap_rows:
        print("No such cases were found in the selected scenario set.")
    else:
        for row in gap_rows:
            print(f"\nScenario: {row['scenario_name']}")
            print(f"  VeRecycle bound    : {row['verecycle_bound']:.6f}")
            print(f"  m_lb_ibp           : {row['m_lb_ibp']:.6f}")
            print(f"  Scratch full certs : {row['scratch_successes']}/{row['scratch_attempts']}")
            print(f"  Scratch RA passes  : {row['scratch_reach_only_passes']}/{row['scratch_attempts']}")
            print(f"  Scratch total time : {row['scratch_total_time_s']:.4f}s")

    print(f"\nSaved results to: {out_dir}")
    base_sde.close()


if __name__ == "__main__":
    main()
