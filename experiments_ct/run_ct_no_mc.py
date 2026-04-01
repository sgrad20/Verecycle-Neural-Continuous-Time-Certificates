import sys
import csv
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

REACH_AVOID_PROBABILITY = 0.9
DEVICE = torch.device("cpu")
RESULTS_DIR = REPO_ROOT / "experiments_ct" / "results_try"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Experiment knobs
# ---------------------------------------------------------------------
RECERT_TIMEOUT_S = 120
RECERT_VERIFY_EVERY_N = 1000
RECERT_VERIFIER_MESH = 400
RECERT_BATCH_SIZE = 128
RECERT_LR = 5e-4
RECERT_VERIFICATION_SLACK = 4.0
RECERT_REGULARIZER_LAMBDA = 1e-1
RECERT_ZETA = 1.0
RECERT_MAX_DEPTH = 4
RECERT_N_EPOCHS = 1_000_000
STAY_PROBABILITY = 0.0

VERECYCLE_MESH_SIZE = 0.05
VERECYCLE_BATCH_SIZE = 1000
M_GRID_NX = 41
M_GRID_NY = 41

# ---------------------------------------------------------------------
# Pendulum specification
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
# Helper region classes
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


class SoftBoxGate(torch.nn.Module):
    """
    Smooth approximation of the indicator of an axis-aligned box.

    This keeps the modified drift/diffusion certifier-friendly for
    auto_LiRPA while remaining strongly localized around the scenario box.
    """

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
        if self.scenario.change_type == "absorbing":
            return (1.0 - gate) * drift
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
        if self.scenario.change_type == "absorbing":
            return (1.0 - gate) * diffusion
        raise ValueError(f"Unsupported change_type: {self.scenario.change_type}")


class LocalChangedPendulum(controlled_sde.ControlledSDE):
    """
    Certifier-friendly localized dynamics modification for re-certification.
    """

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
def estimate_alpha(net, initial_set_obj, n_samples=5000):
    xs = initial_set_obj.sample(n_samples)
    with torch.no_grad():
        vals = net(xs).cpu().numpy().reshape(-1)
    return float(vals.max())


# Keep this exactly for the ORIGINAL / VeRecycle side
def estimate_beta(alpha, probability=REACH_AVOID_PROBABILITY):
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must be strictly between 0 and 1")
    return float(alpha / (1.0 - probability))


# Use this only for the re-certification side
def estimate_beta_on_unsafe(net, unsafe_set_obj, n_samples=20000):
    xs = unsafe_set_obj.sample(n_samples)
    with torch.no_grad():
        vals = net(xs).cpu().numpy().reshape(-1)
    return float(vals.min())


def cert_value_raw(net, x_np: np.ndarray) -> float:
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        v = net(x).squeeze().item()
    return float(v)


def estimate_m_on_box_grid(net, low, high, nx=M_GRID_NX, ny=M_GRID_NY):
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

def _extract_train_result(result):
    certified = False
    epoch = -1

    if isinstance(result, dict):
        certified = bool(
            result.get("certified", result.get("success", result.get("verified", False)))
        )
        if "epoch" in result:
            try:
                epoch = int(result["epoch"])
            except Exception:
                epoch = -1
        return certified, epoch

    if isinstance(result, (tuple, list)):
        if len(result) >= 1:
            certified = bool(result[0])
        if len(result) >= 2:
            try:
                epoch = int(result[1])
            except Exception:
                epoch = -1
        return certified, epoch

    return bool(result), -1

# def _extract_train_result(result):
#     """
#     Conservative parser for certificate.train(...) output.
#     """
#     certified = False
#     epoch = -1

#     if isinstance(result, (tuple, list)):
#         if len(result) >= 1:
#             certified = bool(result[0])
#         if len(result) >= 2:
#             try:
#                 epoch = int(result[1])
#             except Exception:
#                 epoch = -1
#     else:
#         certified = bool(result)

#     return certified, epoch


# def _recert_worker(policy_state_dict, certificate_state_dict, scenario_payload, queue):
#     start = time.time()
#     try:
#         torch.set_default_dtype(torch.float32)
#         torch.use_deterministic_algorithms(True)

#         device = torch.device("cpu")
#         scenario = payload_to_scenario(scenario_payload)

#         local_interest_set = rsa.AABBSet(global_bounds, device)
#         local_initial_set = rsa.AABBSet(initial_bounds, device)
#         local_target_set = rsa.AABBSet(target_bounds, device)
#         local_unsafe_set = rsa.AABBSet(unsafe_bounds, device)
#         local_spec = rsa.Specification(
#             local_interest_set,
#             local_initial_set,
#             local_unsafe_set,
#             local_target_set,
#             REACH_AVOID_PROBABILITY,
#             0.0,
#         )

#         policy = TanhPolicy(2, 1, 64, device=device)
#         policy.load_state_dict(policy_state_dict)
#         policy.requires_grad_(False)
#         policy.eval()

#         modified_sde = LocalChangedPendulum(policy, scenario)

#         torch.manual_seed(0)
#         npr.seed(0)
#         np.random.seed(0)

#         net = rsa.CertificateModule(device=device)
#         net.load_state_dict(certificate_state_dict)
#         net.train(True)

#         certificate = rsa.SupermartingaleCertificate(modified_sde, local_spec, net, device)

#         result = certificate.train(
#             n_epochs=RECERT_N_EPOCHS,
#             batch_size=RECERT_BATCH_SIZE,
#             lr=RECERT_LR,
#             verify_every_n=RECERT_VERIFY_EVERY_N,
#             verifier_mesh_size=RECERT_VERIFIER_MESH,
#             zeta=RECERT_ZETA,
#             regularizer_lambda=RECERT_REGULARIZER_LAMBDA,
#             verification_slack=RECERT_VERIFICATION_SLACK,
#             max_depth=RECERT_MAX_DEPTH,
#         )
#         elapsed = time.time() - start
#         modified_sde.close()

#         certified, epoch = _extract_train_result(result)

#         alpha_ra = float("nan")
#         beta_ra = float("nan")
#         rho = float("nan")

#         if certified:
#             alpha_ra = estimate_alpha(net, local_initial_set)
#             beta_ra = estimate_beta_on_unsafe(net, local_unsafe_set)

#             if np.isfinite(beta_ra) and beta_ra > 0.0:
#                 rho = max(1.0 - alpha_ra / beta_ra, 0.0)
#             else:
#                 rho = 0.0

#         queue.put(
#             {
#                 "status": "certified" if certified else "not_certified",
#                 "rho": rho,
#                 "alpha_ra": alpha_ra,
#                 "beta_ra": beta_ra,
#                 "epoch": epoch,
#                 "elapsed": elapsed,
#                 "history": [{"epoch": epoch, "rho": rho}],
#                 "raw_result": repr(result),
#             }
#         )
#     except Exception:
#         queue.put(
#             {
#                 "status": "error",
#                 "error": traceback.format_exc(),
#                 "rho": float("nan"),
#                 "elapsed": time.time() - start,
#                 "history": [],
#                 "raw_result": "",
#                 "alpha_ra": float("nan"),
#                 "beta_ra": float("nan"),
#             }
#         )


# def recertify_on_scenario(policy, net, scenario: Scenario, timeout_s=RECERT_TIMEOUT_S):
#     ctx = mp.get_context("spawn")
#     queue = ctx.Queue()
#     proc = ctx.Process(
#         target=_recert_worker,
#         args=(policy.state_dict(), net.state_dict(), scenario_to_payload(scenario), queue),
#     )

#     start = time.time()
#     proc.start()
#     proc.join(timeout_s)

#     if proc.is_alive():
#         proc.terminate()
#         proc.join()
#         elapsed = time.time() - start
#         return (
#             float("nan"),
#             0,
#             elapsed,
#             [{"epoch": -1, "rho": float("nan")}],
#             "timeout",
#             f"terminated after {timeout_s}s",
#             "",
#             float("nan"),
#             float("nan"),
#         )

#     elapsed = time.time() - start

#     if not queue.empty():
#         msg = queue.get()
#         status = msg.get("status", "unknown")
#         return (
#             msg.get("rho", float("nan")),
#             msg.get("epoch", -1),
#             msg.get("elapsed", elapsed),
#             msg.get("history", []),
#             status,
#             msg.get("error", ""),
#             msg.get("raw_result", ""),
#             msg.get("alpha_ra", float("nan")),
#             msg.get("beta_ra", float("nan")),
#         )

#     return (
#         float("nan"),
#         0,
#         elapsed,
#         [{"epoch": -1, "rho": float("nan")}],
#         "no_result",
#         "worker exited without returning a result",
#         "",
#         float("nan"),
#         float("nan"),
#     )
def _recert_worker(policy_state_dict, certificate_state_dict, scenario_payload, conn):
    start = time.time()
    modified_sde = None
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
        certified, epoch = _extract_train_result(result)

        alpha_ra = float("nan")
        beta_ra = float("nan")
        rho = float("nan")

        if certified:
            net.eval()
            alpha_ra = estimate_alpha(net, local_initial_set)
            beta_ra = estimate_beta_on_unsafe(net, local_unsafe_set)

            if np.isfinite(beta_ra) and beta_ra > 0.0:
                rho = max(1.0 - alpha_ra / beta_ra, 0.0)
            else:
                rho = 0.0

        msg = {
            "status": "certified" if certified else "not_certified",
            "rho": rho,
            "alpha_ra": alpha_ra,
            "beta_ra": beta_ra,
            "epoch": epoch,
            "elapsed": elapsed,
            "history": [{"epoch": epoch, "rho": rho}],
            "raw_result": repr(result),
        }

        try:
            conn.send(msg)
        except Exception:
            pass

    except BaseException:
        try:
            conn.send(
                {
                    "status": "error",
                    "error": traceback.format_exc(),
                    "rho": float("nan"),
                    "elapsed": time.time() - start,
                    "history": [],
                    "raw_result": "",
                    "alpha_ra": float("nan"),
                    "beta_ra": float("nan"),
                }
            )
        except Exception:
            pass
    finally:
        try:
            if modified_sde is not None:
                modified_sde.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def recertify_on_scenario(policy, net, scenario: Scenario, timeout_s=RECERT_TIMEOUT_S):
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(
        target=_recert_worker,
        args=(policy.state_dict(), net.state_dict(), scenario_to_payload(scenario), child_conn),
    )

    start = time.time()
    proc.start()
    child_conn.close()

    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        elapsed = time.time() - start
        try:
            parent_conn.close()
        except Exception:
            pass
        return (
            float("nan"),
            -1,
            elapsed,
            [{"epoch": -1, "rho": float("nan")}],
            "timeout",
            f"terminated after {timeout_s}s",
            "",
            float("nan"),
            float("nan"),
        )

    elapsed = time.time() - start
    exitcode = proc.exitcode

    try:
        try:
            has_msg = parent_conn.poll(2.0)
        except (BrokenPipeError, EOFError, OSError):
            has_msg = False

        if has_msg:
            try:
                msg = parent_conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                msg = None

            if msg is not None:
                status = msg.get("status", "unknown")
                note = msg.get("error", "")
                if status != "error" and exitcode not in (0, None):
                    note = f"worker exitcode={exitcode}"
                return (
                    msg.get("rho", float("nan")),
                    msg.get("epoch", -1),
                    msg.get("elapsed", elapsed),
                    msg.get("history", []),
                    status,
                    note,
                    msg.get("raw_result", ""),
                    msg.get("alpha_ra", float("nan")),
                    msg.get("beta_ra", float("nan")),
                )
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass

    if exitcode not in (0, None):
        return (
            float("nan"),
            -1,
            elapsed,
            [{"epoch": -1, "rho": float("nan")}],
            "error",
            f"worker exited unexpectedly with exitcode={exitcode}",
            "",
            float("nan"),
            float("nan"),
        )

    return (
        float("nan"),
        -1,
        elapsed,
        [{"epoch": -1, "rho": float("nan")}],
        "no_result",
        f"worker exited without returning a result (exitcode={exitcode})",
        "",
        float("nan"),
        float("nan"),
    )

def classify_recert_outcome(status, bound):
    if status == "unsupported":
        return "unsupported"
    if status == "timeout":
        return "timeout"
    if status == "error":
        return "error"
    if status == "not_certified":
        return "not_certified"
    if status == "certified":
        return "certified"
    return "unknown"


def normalize_recert_status(raw_status, ve_recycle_bound):
    """
    When VeRecycle already proves zero reclaimed guarantee, treat solver
    crashes/timeouts as a failed re-certification attempt rather than a
    separate experimental outcome.
    """
    if ve_recycle_bound <= 0.0 and raw_status in {"error", "timeout", "no_result"}:
        return "not_certified"
    return raw_status


# ---------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------
def get_scenarios():
    return [
        Scenario(
            name="low_reclaim_diff_1p5",
            low=np.array([4.0, 2.0], dtype=np.float32),
            high=np.array([7.0, 3.5], dtype=np.float32),
            change_type="diffusion_scale",
            diffusion_scale=1.5,
            description="Low but nonzero reclaim region with stronger local diffusion."
        ),
        # Scenario(
        #     name="borderline_diff_2p0",
        #     low=np.array([4.5, 2.1], dtype=np.float32),
        #     high=np.array([6.0, 3.0], dtype=np.float32),
        #     change_type="diffusion_scale",
        #     diffusion_scale=2.0,
        #     description="Borderline reclaimable case with strong diffusion increase."
        # ),
        # Scenario(
        #     name="mid_transition_diff_2p0",
        #     low=np.array([7.0, 2.0], dtype=np.float32),
        #     high=np.array([8.0, 3.0], dtype=np.float32),
        #     change_type="diffusion_scale",
        #     diffusion_scale=2.0,
        #     description="Intermediate reclaim case in a transition region."
        # ),
        # Scenario(
        #     name="high_transition_diff_1p5",
        #     low=np.array([8.0, 2.5], dtype=np.float32),
        #     high=np.array([9.5, 3.5], dtype=np.float32),
        #     change_type="diffusion_scale",
        #     diffusion_scale=1.5,
        #     description="High reclaim case with moderate local degradation."
        # ),
        # Scenario(
        #     name="near_sat_far_right_diff_1p1",
        #     low=np.array([11.0, 3.5], dtype=np.float32),
        #     high=np.array([12.5, 4.5], dtype=np.float32),
        #     change_type="diffusion_scale",
        #     diffusion_scale=1.1,
        #     description="Near-saturation case where reclaimed guarantee should remain close to the original."
        # ),
        # Scenario(
        #     name="mid_band_drift_mild",
        #     low=np.array([5.0, 2.0], dtype=np.float32),
        #     high=np.array([8.0, 3.5], dtype=np.float32),
        #     change_type="drift_bias",
        #     drift_bias=(-0.8, -0.4),
        #     description="Mid-band drift degradation to complement diffusion-only changes."
        # ),
        # Scenario(
        #     name="right_corridor_drift_strong",
        #     low=np.array([7.5, 2.0], dtype=np.float32),
        #     high=np.array([10.5, 4.0], dtype=np.float32),
        #     change_type="drift_bias",
        #     drift_bias=(-1.5, -0.8),
        #     description="Large drift-degraded corridor intended to be visited more often."
        # ),
        Scenario(
            name="absorbing_near_init_large",
            low=np.array([-1.5, 2.0], dtype=np.float32),
            high=np.array([1.5, 3.4], dtype=np.float32),
            change_type="absorbing",
            description="Absorbing region close to the initial set; catastrophic early trapping case."
        ),
        # Scenario(
        #     name="absorbing_near_target_large",
        #     low=np.array([2.5, 0.8], dtype=np.float32),
        #     high=np.array([5.5, 2.0], dtype=np.float32),
        #     change_type="absorbing",
        #     description="Absorbing region near the target approach; sabotages success right before reaching target."
        # ),
        # Scenario(
        #     name="absorbing_central_bridge",
        #     low=np.array([1.5, 1.0], dtype=np.float32),
        #     high=np.array([4.5, 2.6], dtype=np.float32),
        #     change_type="absorbing",
        #     description="Absorbing region in the central corridor between initial and target regions."
        # ),
        # Scenario(
        #     name="absorbing_low_reclaim_region",
        #     low=np.array([4.0, 2.0], dtype=np.float32),
        #     high=np.array([7.0, 3.5], dtype=np.float32),
        #     change_type="absorbing",
        #     description="Absorbing version of the low-reclaim region."
        # ),
        # Scenario(
        #     name="absorbing_mid_transition_region",
        #     low=np.array([7.0, 2.0], dtype=np.float32),
        #     high=np.array([8.0, 3.0], dtype=np.float32),
        #     change_type="absorbing",
        #     description="Absorbing version of the mid-transition region."
        # ),
        # Scenario(
        #     name="absorbing_near_sat_far_right",
        #     low=np.array([11.0, 3.5], dtype=np.float32),
        #     high=np.array([12.5, 4.5], dtype=np.float32),
        #     change_type="absorbing",
        #     description="Absorbing region far right in a high-certificate area; expected to matter little empirically."
        # ),
        # Scenario(
        #     name="absorbing_outer_left_high_region",
        #     low=np.array([-12.5, -4.5], dtype=np.float32),
        #     high=np.array([-10.5, -3.0], dtype=np.float32),
        #     change_type="absorbing",
        #     description="Absorbing region in a distant high-certificate area used as a control scenario."
        # ),
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
    vr_times = _plot_safe([row["verecycle_time_s"] for row in summary_rows])
    rr_times = _plot_safe([row["recert_time_s"] for row in summary_rows])
    m_lb_ibp = _plot_safe([row["m_lb_ibp"] for row in summary_rows])
    speedups = _plot_safe([row["speedup"] for row in summary_rows])

    x = np.arange(len(summary_rows))
    width = 0.35
    original_bound = summary_rows[0]["original_bound"]

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, vr_bounds, width, label="VeRecycle")
    plt.bar(x + width / 2, rr_bounds, width, label="Re-certification")
    plt.axhline(original_bound, linestyle="--", linewidth=1.5, label="Original bound")
    plt.xticks(x, scenario_names, rotation=20, ha="right")
    plt.ylabel("Reach-avoid probability")
    plt.title("Bound comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bounds_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, vr_times, width, label="VeRecycle")
    plt.bar(x + width / 2, rr_times, width, label="Re-certification")
    plt.xticks(x, scenario_names, rotation=20, ha="right")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(x - width / 2, vr_times, width, label="VeRecycle")
    bars2 = plt.bar(x + width / 2, rr_times, width, label="Re-certification")
    plt.xticks(x, scenario_names, rotation=20, ha="right")
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

    plt.figure(figsize=(10, 5))
    plt.bar(x, m_lb_ibp, width, color="#4C72B0")
    plt.xticks(x, scenario_names, rotation=20, ha="right")
    plt.ylabel("Lower bound m_lb_ibp")
    plt.title("Estimated local lower bound")
    plt.tight_layout()
    plt.savefig(out_dir / "m_lb_ibp_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, speedups, width, color="#55A868")
    plt.xticks(x, scenario_names, rotation=20, ha="right")
    plt.ylabel("Speedup (re-cert / VeRecycle)")
    plt.title("Measured speedup")
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.0f}x",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    gap = [rr - vr for rr, vr in zip(rr_bounds, vr_bounds)]
    plt.bar(x, gap, width, color="#8172B2")
    plt.xticks(x, scenario_names, rotation=20, ha="right")
    plt.ylabel("Re-certification - VeRecycle")
    plt.title("Bound gap comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "bound_gap_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(m_lb_ibp, vr_bounds, label="VeRecycle", s=80)
    plt.scatter(m_lb_ibp, rr_bounds, label="Re-certification", s=80)
    for i, name in enumerate(scenario_names):
        plt.text(m_lb_ibp[i], vr_bounds[i], name, fontsize=8, va="bottom", ha="right")
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
    for i, name in enumerate(scenario_names):
        plt.annotate(name, (vr_times[i], vr_bounds[i]), fontsize=8)
    plt.xscale("log")
    plt.xlabel("Runtime (s, log scale)")
    plt.ylabel("Reach-avoid probability")
    plt.title("Runtime vs guarantee")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_vs_bound.png", dpi=300, bbox_inches="tight")
    plt.close()

    order = np.argsort(m_lb_ibp)
    ordered_names = [scenario_names[i] for i in order]
    ordered_vr = [vr_bounds[i] for i in order]
    ordered_rr = [rr_bounds[i] for i in order]
    ordered_m = [m_lb_ibp[i] for i in order]

    x_ord = np.arange(len(order))
    width_ord = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x_ord, ordered_vr)
    plt.xticks(x_ord, ordered_names, rotation=20, ha="right")
    plt.ylabel("VeRecycle bound")
    plt.title("VeRecycle bounds ordered by local lower bound")
    plt.tight_layout()
    plt.savefig(out_dir / "verecycle_ordered.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x_ord, ordered_m)
    plt.xticks(x_ord, ordered_names, rotation=20, ha="right")
    plt.ylabel("m_lb_ibp")
    plt.title("Local lower bound ordered by scenario")
    plt.tight_layout()
    plt.savefig(out_dir / "m_lb_ibp_ordered.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x_ord - width_ord, ordered_vr, width_ord, label="VeRecycle")
    plt.bar(x_ord, ordered_rr, width_ord, label="Re-certification")
    plt.axhline(original_bound, linestyle="--", linewidth=1.5, label="Original bound")
    plt.xticks(x_ord, ordered_names, rotation=20, ha="right")
    plt.ylabel("Reach-avoid probability")
    plt.title("Bounds ordered by local lower bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bounds_ordered_by_m.png", dpi=300, bbox_inches="tight")
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
        plt.close()
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

    # ORIGINAL / VERECYCLE SIDE: keep beta reconstruction from 0.9
    alpha_ra = estimate_alpha(net, initial_set)
    beta_ra = estimate_beta(alpha_ra, REACH_AVOID_PROBABILITY)
    rho_orig = 1.0 - alpha_ra / beta_ra

    print("\n=== BASE CERTIFICATE ===")
    print(f"alpha_RA           = {alpha_ra:.6f}")
    print(f"beta_RA            = {beta_ra:.6f}")
    print(f"original bound     = {rho_orig:.6f}")

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
            mesh_size=VERECYCLE_MESH_SIZE,
            batch_size=VERECYCLE_BATCH_SIZE,
        )
        t_vr = time.time() - t0

        m_grid, argmin_grid = estimate_m_on_box_grid(
            net, scenario.low, scenario.high, nx=M_GRID_NX, ny=M_GRID_NY
        )

        print(f"Running re-certification with timeout={RECERT_TIMEOUT_S}s ...")
        (
            recert_bound,
            recert_certified_epoch,
            t_rr,
            history,
            recert_status,
            recert_note,
            recert_raw_result,
            recert_alpha_ra,
            recert_beta_ra,
        ) = recertify_on_scenario(policy, net, scenario, timeout_s=RECERT_TIMEOUT_S)
        recert_status_normalized = normalize_recert_status(recert_status, eps_vr)

        speedup = t_rr / t_vr if t_vr > 0 else float("inf")

        summary = {
            "scenario_name": scenario.name,
            "scenario_family": (
                "absorbing" if scenario.change_type == "absorbing"
                else "diffusion" if scenario.change_type == "diffusion_scale"
                else "drift"
            ),
            "description": scenario.description,
            "change_type": scenario.change_type,
            "diffusion_scale": float(scenario.diffusion_scale),
            "drift_bias_0": float(scenario.drift_bias[0]),
            "drift_bias_1": float(scenario.drift_bias[1]),
            "xq_low_0": float(scenario.low[0]),
            "xq_low_1": float(scenario.low[1]),
            "xq_high_0": float(scenario.high[0]),
            "xq_high_1": float(scenario.high[1]),
            "alpha_ra": alpha_ra,
            "beta_ra": beta_ra,
            "original_bound": rho_orig,
            "m_lb_ibp": m_lb_ibp,
            "m_grid_min_raw": m_grid,
            "m_grid_argmin_0": float(argmin_grid[0]),
            "m_grid_argmin_1": float(argmin_grid[1]),
            "verecycle_bound": eps_vr,
            "verecycle_time_s": t_vr,
            "recert_bound": recert_bound,
            "recert_alpha_ra": recert_alpha_ra,
            "recert_beta_ra": recert_beta_ra,
            "recert_certified_epoch": recert_certified_epoch,
            "recert_time_s": t_rr,
            "recert_status": recert_status_normalized,
            "recert_status_raw": recert_status,
            "recert_note": recert_note,
            "recert_raw_result": recert_raw_result,
            "recert_outcome": classify_recert_outcome(recert_status_normalized, recert_bound),
            "speedup": speedup,
        }
        summary_rows.append(summary)

        print("\nSummary")
        print(f"  m_lb_ibp                  : {m_lb_ibp:.6f}")
        print(f"  m_grid_min_raw            : {m_grid:.6f}")
        print(f"  VeRecycle bound           : {eps_vr:.6f}")
        print(f"  Re-cert bound             : {recert_bound}")
        print(f"  Re-cert alpha             : {recert_alpha_ra}")
        print(f"  Re-cert beta              : {recert_beta_ra}")
        print(f"  Re-cert status            : {recert_status_normalized}")
        if recert_status_normalized != recert_status:
            print(f"  Re-cert raw status        : {recert_status}")
        if recert_note:
            print(f"  Re-cert note              : {recert_note}")
        if recert_raw_result:
            print(f"  Re-cert raw result        : {recert_raw_result}")
        print(f"  VeRecycle time            : {t_vr:.4f}s")
        print(f"  Re-cert time              : {t_rr:.4f}s")
        print(f"  Speedup                   : {speedup:.2f}x")

        history_csv = out_dir / f"{scenario.name}_recert_history.csv"
        with open(history_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "rho"])
            writer.writeheader()
            writer.writerows(history)

    summary_rows.sort(key=lambda row: row["m_lb_ibp"])
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
        print(f"  VeRecycle bound           : {row['verecycle_bound']:.6f}")
        print(f"  Re-cert bound             : {row['recert_bound']}")
        print(f"  Re-cert status            : {row['recert_status']}")
        print(f"  Speedup                   : {row['speedup']:.2f}x")
        print(f"  m_lb_ibp                  : {row['m_lb_ibp']:.6f}")

    print(f"\nSaved results to: {out_dir}")
    base_sde.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
