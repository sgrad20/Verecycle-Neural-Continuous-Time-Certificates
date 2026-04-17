import argparse
import csv
import json
import math
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import numpy.random as npr
import torch

from stochastic_rsa.continuous_vere_cycle import run_continuous_VeRecycle
from experiments_ct.run_ct_no_mc import (
    DEVICE,
    REACH_AVOID_PROBABILITY,
    Scenario,
    global_bounds,
    initial_bounds,
    initial_set,
    target_bounds,
    unsafe_bounds,
    estimate_alpha,
    estimate_beta,
    load_certified_checkpoint,
    make_disrupted_region,
    make_verifier,
)

torch.set_default_dtype(torch.float32)
torch.use_deterministic_algorithms(True)

RESULTS_DIR = REPO_ROOT / "experiments_ct" / "results_ct_vs_discretized_verecycle"


@dataclass
class GridResult:
    scenario_name: str
    grid_width: float
    n_discrete_cells: int
    m_discrete_point: float
    naive_discrete_bound: float
    missed_changed_region: bool


@dataclass
class McResult:
    scenario_name: str
    dt: float
    ct_absorbing_success: float
    ct_xq_hit_fraction: float
    sampled_xq_hit_fraction: float
    missed_xq_given_ct_hit_fraction: float


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def in_any_box(x_np: np.ndarray, bounds_np: np.ndarray) -> bool:
    x_np = np.asarray(x_np, dtype=np.float32).reshape(-1)
    for box in bounds_np:
        if np.all(x_np >= box[0]) and np.all(x_np <= box[1]):
            return True
    return False


def in_box(x_np: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    x_np = np.asarray(x_np, dtype=np.float32).reshape(-1)
    return bool(np.all(x_np >= low) and np.all(x_np <= high))


def boxes_intersect(low_a: np.ndarray, high_a: np.ndarray, low_b: np.ndarray, high_b: np.ndarray) -> bool:
    return bool(np.all(np.maximum(low_a, low_b) <= np.minimum(high_a, high_b)))


def segment_intersects_box(p0: np.ndarray, p1: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    p0 = np.asarray(p0, dtype=np.float32).reshape(-1)
    p1 = np.asarray(p1, dtype=np.float32).reshape(-1)

    if in_box(p0, low, high) or in_box(p1, low, high):
        return True

    t_min = 0.0
    t_max = 1.0
    direction = p1 - p0
    for dim in range(low.shape[0]):
        if abs(float(direction[dim])) < 1e-12:
            if p0[dim] < low[dim] or p0[dim] > high[dim]:
                return False
            continue

        inv_d = 1.0 / float(direction[dim])
        t1 = float((low[dim] - p0[dim]) * inv_d)
        t2 = float((high[dim] - p0[dim]) * inv_d)
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False

    return True


def path_from_sde(sde, x0_np: np.ndarray, ts: torch.Tensor) -> np.ndarray:
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        raw = sde.sample(x0, ts, method="srk").detach().cpu().numpy()

    if raw.ndim == 3 and raw.shape[1] == 1:
        return raw[:, 0, :]
    if raw.ndim == 3 and raw.shape[0] == 1:
        return raw[0, :, :]
    if raw.ndim == 2:
        return raw
    raise ValueError(f"Unexpected SDE sample shape: {raw.shape}")


def paths_from_sde_batch(sde, x0_np: np.ndarray, ts: torch.Tensor) -> np.ndarray:
    x0 = torch.tensor(x0_np, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        raw = sde.sample(x0, ts, method="srk").detach().cpu().numpy()

    if raw.ndim == 3 and raw.shape[1] == x0_np.shape[0]:
        return np.transpose(raw, (1, 0, 2))
    if raw.ndim == 3 and raw.shape[0] == x0_np.shape[0]:
        return raw
    raise ValueError(f"Unexpected batched SDE sample shape: {raw.shape}")


def evaluate_absorbing_path(path: np.ndarray, scenario: Scenario, sample_stride: int) -> tuple[bool, bool, bool]:
    ct_hit_xq = False

    if in_box(path[0], scenario.low, scenario.high):
        return False, True, sampled_path_hits_box(path[:1], scenario, sample_stride, include_terminal=False)

    for i in range(1, path.shape[0]):
        path_until_event = path[: i + 1]
        if segment_intersects_box(path[i - 1], path[i], scenario.low, scenario.high):
            return False, True, sampled_path_hits_box(path_until_event, scenario, sample_stride, include_terminal=False)

        if in_any_box(path[i], unsafe_bounds):
            return False, ct_hit_xq, sampled_path_hits_box(path_until_event, scenario, sample_stride, include_terminal=False)

        if in_any_box(path[i], target_bounds):
            return True, ct_hit_xq, sampled_path_hits_box(path_until_event, scenario, sample_stride, include_terminal=False)

    return False, ct_hit_xq, sampled_path_hits_box(path, scenario, sample_stride)


def sampled_path_hits_box(
    path: np.ndarray,
    scenario: Scenario,
    sample_stride: int,
    *,
    include_terminal: bool = True,
) -> bool:
    sampled = path[::sample_stride]
    if include_terminal and not np.array_equal(sampled[-1], path[-1]):
        sampled = np.vstack((sampled, path[-1:]))
    return any(in_box(x, scenario.low, scenario.high) for x in sampled)


def estimate_absorbing_mc(
    sde,
    scenario: Scenario,
    *,
    n_samples: int,
    fine_dt: float,
    coarse_dt: float,
    horizon: float,
    seed: int,
) -> McResult:
    if coarse_dt < fine_dt:
        raise ValueError("coarse_dt must be >= fine_dt")

    npr.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_steps = int(math.ceil(horizon / fine_dt))
    ts = torch.linspace(0.0, n_steps * fine_dt, n_steps + 1, device=DEVICE)
    sample_stride = max(1, int(round(coarse_dt / fine_dt)))

    successes = 0
    ct_hits = 0
    sampled_hits = 0
    missed_given_ct_hit = 0

    for _ in range(n_samples):
        x0 = np.random.uniform(initial_bounds[0, 0], initial_bounds[0, 1]).astype(np.float32)
        path = path_from_sde(sde, x0, ts)
        success, ct_hit, sampled_hit = evaluate_absorbing_path(path, scenario, sample_stride)
        successes += int(success)
        ct_hits += int(ct_hit)
        sampled_hits += int(sampled_hit)
        missed_given_ct_hit += int(ct_hit and not sampled_hit)

    return McResult(
        scenario_name=scenario.name,
        dt=float(coarse_dt),
        ct_absorbing_success=successes / n_samples,
        ct_xq_hit_fraction=ct_hits / n_samples,
        sampled_xq_hit_fraction=sampled_hits / n_samples,
        missed_xq_given_ct_hit_fraction=(missed_given_ct_hit / ct_hits if ct_hits > 0 else 0.0),
    )


def estimate_absorbing_mc_from_paths(
    paths: np.ndarray,
    scenario: Scenario,
    *,
    fine_dt: float,
    coarse_dt: float,
) -> McResult:
    if coarse_dt < fine_dt:
        raise ValueError("coarse_dt must be >= fine_dt")

    sample_stride = max(1, int(round(coarse_dt / fine_dt)))

    successes = 0
    ct_hits = 0
    sampled_hits = 0
    missed_given_ct_hit = 0

    for path in paths:
        success, ct_hit, sampled_hit = evaluate_absorbing_path(path, scenario, sample_stride)
        successes += int(success)
        ct_hits += int(ct_hit)
        sampled_hits += int(sampled_hit)
        missed_given_ct_hit += int(ct_hit and not sampled_hit)

    n_samples = paths.shape[0]
    return McResult(
        scenario_name=scenario.name,
        dt=float(coarse_dt),
        ct_absorbing_success=successes / n_samples,
        ct_xq_hit_fraction=ct_hits / n_samples,
        sampled_xq_hit_fraction=sampled_hits / n_samples,
        missed_xq_given_ct_hit_fraction=(missed_given_ct_hit / ct_hits if ct_hits > 0 else 0.0),
    )


def bound_from_m(alpha_ra: float, beta_ra: float, m_value: float) -> float:
    original = 1.0 - alpha_ra / beta_ra
    if math.isinf(m_value):
        return float(original)
    if m_value <= alpha_ra:
        return 0.0
    return float(min(original, max(0.0, 1.0 - alpha_ra / m_value)))


def grid_centers_1d(global_low: float, global_high: float, low: float, high: float, width: float) -> np.ndarray:
    first_idx = math.ceil((low - global_low - 0.5 * width) / width)
    last_idx = math.floor((high - global_low - 0.5 * width) / width)
    if last_idx < first_idx:
        return np.empty((0,), dtype=np.float32)

    centers = global_low + (np.arange(first_idx, last_idx + 1, dtype=np.float32) + 0.5) * width
    centers = centers[(centers >= low) & (centers <= high) & (centers >= global_low) & (centers <= global_high)]
    return centers.astype(np.float32)


def cert_values(net, points: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    values = []
    for start in range(0, points.shape[0], batch_size):
        end = min(start + batch_size, points.shape[0])
        x = torch.tensor(points[start:end], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            values.append(net(x).detach().cpu().numpy().reshape(-1))
    return np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)


def naive_discrete_verecycle_on_grid(
    net,
    scenario: Scenario,
    *,
    grid_width: float,
    alpha_ra: float,
    beta_ra: float,
) -> GridResult:
    global_low = global_bounds[0, 0]
    global_high = global_bounds[0, 1]

    xs = grid_centers_1d(global_low[0], global_high[0], scenario.low[0], scenario.high[0], grid_width)
    ys = grid_centers_1d(global_low[1], global_high[1], scenario.low[1], scenario.high[1], grid_width)

    if xs.size == 0 or ys.size == 0:
        return GridResult(
            scenario_name=scenario.name,
            grid_width=float(grid_width),
            n_discrete_cells=0,
            m_discrete_point=float("inf"),
            naive_discrete_bound=bound_from_m(alpha_ra, beta_ra, float("inf")),
            missed_changed_region=True,
        )

    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1).astype(np.float32)
    values = cert_values(net, points)
    m_discrete = float(np.min(values))

    return GridResult(
        scenario_name=scenario.name,
        grid_width=float(grid_width),
        n_discrete_cells=int(points.shape[0]),
        m_discrete_point=m_discrete,
        naive_discrete_bound=bound_from_m(alpha_ra, beta_ra, m_discrete),
        missed_changed_region=False,
    )


def default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="thin_corridor_absorbing_barrier",
            low=np.array([-6.0, 1.60], dtype=np.float32),
            high=np.array([6.0, 1.64], dtype=np.float32),
            change_type="absorbing",
            description=(
                "Thin absorbing band across the main corridor above the target set; "
                "it does not intersect the initial or target set, is frequently hit "
                "by CT rollouts, and is missed by coarse state grids."
            ),
        ),
        Scenario(
            name="thin_post_init_absorbing_barrier",
            low=np.array([-1.5, 2.30], dtype=np.float32),
            high=np.array([1.5, 2.34], dtype=np.float32),
            change_type="absorbing",
            description=(
                "Thin absorbing band just below the initial set; coarse state/time "
                "discretizations can miss it."
            ),
        ),
        Scenario(
            name="thin_initial_slice_grid_aliasing_diagnostic",
            low=np.array([-1.5, 3.10], dtype=np.float32),
            high=np.array([1.5, 3.30], dtype=np.float32),
            change_type="absorbing",
            description=(
                "Diagnostic aliasing case: this thin absorbing slice intersects the "
                "initial set, so it is not a clean repair-method case, but it makes "
                "the unsafe over-reclaiming behavior visible in Monte Carlo."
            ),
        ),
        Scenario(
            name="absorbing_central_bridge",
            low=np.array([1.5, 1.0], dtype=np.float32),
            high=np.array([4.5, 2.6], dtype=np.float32),
            change_type="absorbing",
            description="Wide central absorbing region used as a sanity check.",
        ),
        Scenario(
            name="absorbing_scenario_A_init_trap",
            low=np.array([-1.5, 2.0], dtype=np.float32),
            high=np.array([1.5, 3.4], dtype=np.float32),
            change_type="absorbing",
            description="Large near-initial absorbing region from earlier experiments.",
        ),
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scenario_panel_title(name: str) -> str:
    return textwrap.fill(name.replace("_", " "), width=34)


def make_panel_grid(n_panels: int, *, sharey: bool):
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.5, 3.2 * nrows + 1.0),
        sharex=True,
        sharey=sharey,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes


def add_box(ax, low: np.ndarray, high: np.ndarray, *, label: str, color: str, alpha: float = 0.18, lw: float = 1.8):
    width = float(high[0] - low[0])
    height = float(high[1] - low[1])
    patch = Rectangle(
        (float(low[0]), float(low[1])),
        width,
        height,
        facecolor=color,
        edgecolor=color,
        linewidth=lw,
        alpha=alpha,
        label=label,
    )
    ax.add_patch(patch)
    return patch


def plot_spatial_aliasing_setup(
    paths: np.ndarray,
    scenarios: list[Scenario],
    out_dir: Path,
    *,
    grid_width: float = 0.05,
    n_paths: int = 30,
) -> None:
    scenario = next((item for item in scenarios if item.name == "thin_corridor_absorbing_barrier"), None)
    if scenario is None or paths.size == 0:
        return

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for path in paths[:n_paths]:
        hit = any(
            segment_intersects_box(path[i - 1], path[i], scenario.low, scenario.high)
            for i in range(1, path.shape[0])
        )
        ax_full.plot(
            path[:, 0],
            path[:, 1],
            color="tab:red" if hit else "0.45",
            alpha=0.28 if hit else 0.18,
            linewidth=1.1,
        )

    add_box(ax_full, initial_bounds[0, 0], initial_bounds[0, 1], label="Initial set", color="tab:green", alpha=0.18)
    add_box(ax_full, target_bounds[0, 0], target_bounds[0, 1], label="Target set", color="tab:blue", alpha=0.14)
    add_box(ax_full, scenario.low, scenario.high, label="Changed region Xq", color="tab:red", alpha=0.32, lw=2.2)
    ax_full.set_xlim(-7.0, 7.0)
    ax_full.set_ylim(-1.9, 4.2)
    ax_full.set_xlabel("Angular velocity")
    ax_full.set_ylabel("Angle")
    ax_full.set_title("Thin changed region is on common trajectories")
    ax_full.grid(True, alpha=0.22)

    x_low, x_high = -6.2, 6.2
    y_low = float(scenario.low[1] - 0.08)
    y_high = float(scenario.high[1] + 0.08)
    xs = grid_centers_1d(float(global_bounds[0, 0, 0]), float(global_bounds[0, 1, 0]), x_low, x_high, grid_width)
    ys = grid_centers_1d(float(global_bounds[0, 0, 1]), float(global_bounds[0, 1, 1]), y_low, y_high, grid_width)
    if xs.size and ys.size:
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        ax_zoom.scatter(xx.reshape(-1), yy.reshape(-1), s=14, color="0.25", alpha=0.75, label="Grid centres")

    add_box(ax_zoom, scenario.low, scenario.high, label="Changed region Xq", color="tab:red", alpha=0.35, lw=2.4)
    ax_zoom.axhline(float(scenario.low[1]), color="tab:red", linewidth=1.2)
    ax_zoom.axhline(float(scenario.high[1]), color="tab:red", linewidth=1.2)
    ax_zoom.set_xlim(x_low, x_high)
    ax_zoom.set_ylim(y_low, y_high)
    ax_zoom.set_xlabel("Angular velocity")
    ax_zoom.set_ylabel("Angle")
    ax_zoom.set_title(f"Grid width {grid_width:g}: no centres lie inside Xq")
    ax_zoom.grid(True, alpha=0.22)

    handles, labels = [], []
    for ax in (ax_full, ax_zoom):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Why naive state discretisation misses the changed region", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.93))
    fig.savefig(out_dir / "thin_corridor_spatial_aliasing_setup.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_temporal_aliasing_example(
    paths: np.ndarray,
    scenario: Scenario,
    *,
    fine_dt: float,
    coarse_dt: float,
) -> tuple[int, int] | None:
    sample_stride = max(1, int(round(coarse_dt / fine_dt)))
    for path_idx, path in enumerate(paths):
        success, ct_hit, sampled_hit = evaluate_absorbing_path(path, scenario, sample_stride)
        if success or not ct_hit or sampled_hit:
            continue
        for step_idx in range(1, path.shape[0]):
            if segment_intersects_box(path[step_idx - 1], path[step_idx], scenario.low, scenario.high):
                return path_idx, step_idx
    return None


def plot_temporal_aliasing_example(
    paths: np.ndarray,
    scenarios: list[Scenario],
    out_dir: Path,
    *,
    fine_dt: float,
    coarse_dt: float = 0.5,
) -> None:
    scenario = next((item for item in scenarios if item.name == "thin_corridor_absorbing_barrier"), None)
    if scenario is None or paths.size == 0:
        return

    example = find_temporal_aliasing_example(paths, scenario, fine_dt=fine_dt, coarse_dt=coarse_dt)
    if example is None:
        return

    path_idx, hit_step = example
    path = paths[path_idx]
    sample_stride = max(1, int(round(coarse_dt / fine_dt)))
    t = np.arange(path.shape[0], dtype=np.float32) * float(fine_dt)
    sampled_idx = np.arange(0, path.shape[0], sample_stride, dtype=int)

    prev_sample = int((hit_step // sample_stride) * sample_stride)
    if prev_sample >= hit_step:
        prev_sample = max(0, prev_sample - sample_stride)
    next_sample = min(path.shape[0] - 1, prev_sample + sample_stride)

    lo = max(0, min(prev_sample, hit_step - 8))
    hi = min(path.shape[0] - 1, max(next_sample, hit_step + 8))
    local_idx = np.arange(lo, hi + 1, dtype=int)
    local_sampled = np.unique(np.asarray([prev_sample, next_sample], dtype=int))

    x_values = np.concatenate((path[local_idx, 0], path[local_sampled, 0]))
    y_values = np.concatenate((path[local_idx, 1], path[local_sampled, 1], scenario.low[1:2], scenario.high[1:2]))

    fig, (ax_phase, ax_time) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    add_box(ax_phase, scenario.low, scenario.high, label="Changed region Xq", color="tab:red", alpha=0.28, lw=2.2)
    ax_phase.plot(path[local_idx, 0], path[local_idx, 1], color="tab:orange", linewidth=2.4, label="Fine trajectory")
    ax_phase.plot(
        path[hit_step - 1 : hit_step + 1, 0],
        path[hit_step - 1 : hit_step + 1, 1],
        color="tab:red",
        linewidth=4.0,
        label="Crossing segment",
        zorder=5,
    )
    if local_sampled.size:
        ax_phase.scatter(
            path[local_sampled, 0],
            path[local_sampled, 1],
            s=70,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            label=f"Coarse samples (dt={coarse_dt:g})",
            zorder=6,
        )
    ax_phase.set_xlabel("Angular velocity")
    ax_phase.set_ylabel("Angle")
    ax_phase.set_title("Coarse samples miss the crossing in state space")
    pad_x = max(0.15, 0.12 * float(np.ptp(x_values) + 1e-6))
    pad_y = max(0.03, 0.12 * float(np.ptp(y_values) + 1e-6))
    ax_phase.set_xlim(float(np.min(x_values) - pad_x), float(np.max(x_values) + pad_x))
    ax_phase.set_ylim(float(np.min(y_values) - pad_y), float(np.max(y_values) + pad_y))
    ax_phase.grid(True, alpha=0.25)

    ax_time.axhspan(float(scenario.low[1]), float(scenario.high[1]), color="tab:red", alpha=0.22, label="Changed angle band")
    ax_time.plot(t[local_idx], path[local_idx, 1], color="tab:orange", linewidth=2.4, label="Fine trajectory")
    ax_time.plot(
        t[hit_step - 1 : hit_step + 1],
        path[hit_step - 1 : hit_step + 1, 1],
        color="tab:red",
        linewidth=4.0,
        label="Crossing between fine samples",
        zorder=5,
    )
    if local_sampled.size:
        ax_time.scatter(
            t[local_sampled],
            path[local_sampled, 1],
            s=70,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            label=f"Coarse samples (dt={coarse_dt:g})",
            zorder=6,
        )
    ax_time.set_xlabel("Time")
    ax_time.set_ylabel("Angle")
    ax_time.set_title("The changed band is crossed between sampled times")
    ax_time.set_xlim(float(t[lo]), float(t[hi]))
    ax_time.set_ylim(float(np.min(y_values) - pad_y), float(np.max(y_values) + pad_y))
    ax_time.grid(True, alpha=0.25)

    handles, labels = [], []
    for ax in (ax_phase, ax_time):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Temporal aliasing: a coarse sampled trajectory misses Xq", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.93))
    fig.savefig(out_dir / "thin_corridor_temporal_aliasing_example.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_certificate_aliasing_context(
    net,
    paths: np.ndarray,
    scenarios: list[Scenario],
    out_dir: Path,
    *,
    alpha_ra: float,
    m_ct: float,
    grid_width: float = 0.05,
    n_paths: int = 24,
) -> None:
    scenario = next((item for item in scenarios if item.name == "thin_corridor_absorbing_barrier"), None)
    if scenario is None:
        return

    x_min, x_max = -7.0, 7.0
    y_min, y_max = 1.25, 2.05
    nx, ny = 220, 140
    xs = np.linspace(x_min, x_max, nx, dtype=np.float32)
    ys = np.linspace(y_min, y_max, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1).astype(np.float32)
    values = cert_values(net, points, batch_size=8192).reshape(ny, nx)

    fig = plt.figure(figsize=(12.0, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.035])
    ax_cert = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    levels = np.linspace(float(np.nanmin(values)), float(np.nanpercentile(values, 98)), 28)
    mesh = ax_cert.contourf(xx, yy, values, levels=levels, cmap="viridis", alpha=0.92)
    ax_cert.contour(xx, yy, values, levels=[float(alpha_ra)], colors="white", linewidths=1.8)
    add_box(ax_cert, scenario.low, scenario.high, label="Changed region Xq", color="tab:red", alpha=0.32, lw=2.3)
    for path in paths[:n_paths]:
        hit = any(
            segment_intersects_box(path[i - 1], path[i], scenario.low, scenario.high)
            for i in range(1, path.shape[0])
        )
        if hit:
            ax_cert.plot(path[:, 0], path[:, 1], color="tab:red", alpha=0.25, linewidth=1.0)
    ax_cert.set_xlim(x_min, x_max)
    ax_cert.set_ylim(y_min, y_max)
    ax_cert.set_xlabel("Angular velocity")
    ax_cert.set_ylabel("Angle")
    ax_cert.set_title("Certificate values near the changed corridor")
    ax_cert.grid(True, alpha=0.18)

    zx_min, zx_max = -6.2, 6.2
    zy_min = float(scenario.low[1] - 0.08)
    zy_max = float(scenario.high[1] + 0.08)
    zoom_mask_x = (xs >= zx_min) & (xs <= zx_max)
    zoom_mask_y = (ys >= zy_min) & (ys <= zy_max)
    zxx = xx[np.ix_(zoom_mask_y, zoom_mask_x)]
    zyy = yy[np.ix_(zoom_mask_y, zoom_mask_x)]
    zvalues = values[np.ix_(zoom_mask_y, zoom_mask_x)]
    ax_zoom.contourf(zxx, zyy, zvalues, levels=levels, cmap="viridis", alpha=0.92)
    ax_zoom.contour(zxx, zyy, zvalues, levels=[float(alpha_ra)], colors="white", linewidths=1.8)

    grid_xs = grid_centers_1d(float(global_bounds[0, 0, 0]), float(global_bounds[0, 1, 0]), zx_min, zx_max, grid_width)
    grid_ys = grid_centers_1d(float(global_bounds[0, 0, 1]), float(global_bounds[0, 1, 1]), zy_min, zy_max, grid_width)
    if grid_xs.size and grid_ys.size:
        gxx, gyy = np.meshgrid(grid_xs, grid_ys, indexing="xy")
        ax_zoom.scatter(
            gxx.reshape(-1),
            gyy.reshape(-1),
            s=14,
            color="black",
            alpha=0.65,
            label="Naive grid centres",
            zorder=6,
        )
    add_box(ax_zoom, scenario.low, scenario.high, label="Changed region Xq", color="tab:red", alpha=0.32, lw=2.3)
    ax_zoom.text(
        0.02,
        0.95,
        rf"$m_{{\mathrm{{lb}}}}={m_ct:.3f}<\alpha={alpha_ra:.3f}$" + "\n" + rf"$n_{{\mathrm{{centres}}}}=0$ at $h={grid_width:g}$",
        transform=ax_zoom.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.65", "alpha": 0.86, "boxstyle": "round,pad=0.25"},
    )
    ax_zoom.set_xlim(zx_min, zx_max)
    ax_zoom.set_ylim(zy_min, zy_max)
    ax_zoom.set_xlabel("Angular velocity")
    ax_zoom.set_ylabel("Angle")
    ax_zoom.set_title("Grid centres miss the low-value band")
    ax_zoom.grid(True, alpha=0.18)

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Certificate value V")
    alpha_handle = Line2D([], [], color="white", linewidth=1.8, label=r"$V=\alpha$")
    handles = [alpha_handle]
    labels = [r"$V=\alpha$"]
    for ax in (ax_cert, ax_zoom):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Certificate context for the missed changed region", fontsize=12)
    fig.savefig(out_dir / "thin_corridor_certificate_aliasing_context.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bounds(grid_rows: list[dict], scenario_rows: list[dict], out_dir: Path) -> None:
    if not grid_rows:
        return

    ct_by_name = {row["scenario_name"]: row for row in scenario_rows}
    names = sorted({row["scenario_name"] for row in grid_rows})

    fig, axes = make_panel_grid(len(names), sharey=True)
    max_y = 0.0
    for ax, name in zip(axes, names):
        rows = sorted([row for row in grid_rows if row["scenario_name"] == name], key=lambda row: row["grid_width"])
        xs = np.asarray([float(row["grid_width"]) for row in rows])
        ys = np.asarray([float(row["naive_discrete_bound"]) for row in rows])
        max_y = max(max_y, float(np.nanmax(ys)))
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            color="tab:blue",
            label="Naive discrete",
            zorder=3,
        )
        if name in ct_by_name:
            ct_bound = float(ct_by_name[name]["ct_verecycle_bound"])
            max_y = max(max_y, ct_bound)
            ax.axhline(
                ct_bound,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="CT VeRecycle",
                zorder=4,
            )

        ax.set_xscale("log")
        ax.set_title(scenario_panel_title(name), fontsize=9)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylim(-0.04, min(1.0, max(0.95, max_y + 0.06)))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("CT VeRecycle vs naive discretize-then-VeRecycle", fontsize=12)
    fig.text(0.5, 0.055, "Naive discrete state-grid width", ha="center")
    fig.text(0.03, 0.5, "Reclaimed bound", va="center", rotation="vertical")
    fig.tight_layout(rect=(0.055, 0.095, 1.0, 0.93))
    fig.savefig(out_dir / "ct_vs_naive_discrete_bounds.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hit_rates(
    mc_rows: list[dict],
    out_dir: Path,
    *,
    filename: str = "xq_hit_rate_vs_time_discretization.png",
    title: str = "Continuous changed-region hits missed by time sampling",
) -> None:
    if not mc_rows:
        return

    names = sorted({row["scenario_name"] for row in mc_rows})
    fig, axes = make_panel_grid(len(names), sharey=True)
    for ax, name in zip(axes, names):
        rows = sorted([row for row in mc_rows if row["scenario_name"] == name], key=lambda row: row["dt"])
        xs = np.asarray([float(row["dt"]) for row in rows])
        sampled = np.asarray([float(row["sampled_xq_hit_fraction"]) for row in rows])
        ct = np.asarray([float(row["ct_xq_hit_fraction"]) for row in rows])

        ax.plot(
            xs * 0.96,
            sampled,
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            color="tab:blue",
            label="Sampled hits",
            zorder=3,
        )
        ax.plot(
            xs * 1.04,
            ct,
            marker="s",
            markersize=5.0,
            linewidth=2.0,
            linestyle="--",
            color="tab:orange",
            label="Segment hits",
            zorder=4,
        )

        if np.allclose(sampled, ct):
            ax.text(
                0.03,
                0.92,
                "same values",
                transform=ax.transAxes,
                fontsize=8,
                color="0.35",
            )

        ax.set_xscale("log")
        ax.set_ylim(-0.04, 1.04)
        ax.set_title(scenario_panel_title(name), fontsize=9)
        ax.grid(True, which="both", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.055, "Discrete sampling time step", ha="center")
    fig.text(0.03, 0.5, "Fraction of rollouts hitting Xq", va="center", rotation="vertical")
    fig.tight_layout(rect=(0.055, 0.095, 1.0, 0.93))
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CT VeRecycle with a deliberately naive discretize-then-VeRecycle baseline."
    )
    parser.add_argument("--grid-widths", default="0.01,0.02,0.05,0.1,0.25,0.5")
    parser.add_argument("--time-steps", default="0.02,0.05,0.1,0.25,0.5")
    parser.add_argument("--ct-mesh-size", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--fine-dt", type=float, default=0.01)
    parser.add_argument("--horizon", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_widths = parse_float_list(args.grid_widths)
    time_steps = parse_float_list(args.time_steps)
    scenarios = default_scenarios()

    print("Loading certified.pt...")
    policy, net, base_sde, _ = load_certified_checkpoint()
    verifier = make_verifier(net)

    alpha_ra = estimate_alpha(net, initial_set)
    beta_ra = estimate_beta(alpha_ra, REACH_AVOID_PROBABILITY)
    original_bound = 1.0 - alpha_ra / beta_ra

    print("\n=== Certificate ===")
    print(f"alpha_RA       = {alpha_ra:.6f}")
    print(f"beta_RA        = {beta_ra:.6f}")
    print(f"original bound = {original_bound:.6f}")

    n_steps = int(math.ceil(args.horizon / args.fine_dt))
    ts = torch.linspace(0.0, n_steps * args.fine_dt, n_steps + 1, device=DEVICE)
    npr.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    x0_np = np.random.uniform(
        initial_bounds[0, 0],
        initial_bounds[0, 1],
        size=(args.mc_samples, 2),
    ).astype(np.float32)
    print(f"Sampling {args.mc_samples} base trajectories once for all time-discretization diagnostics...")
    base_paths = paths_from_sde_batch(base_sde, x0_np, ts)

    scenario_rows = []
    grid_rows = []
    mc_rows = []

    for scenario in scenarios:
        print("\n" + "#" * 80)
        print(f"Scenario: {scenario.name}")
        print(scenario.description)

        t0 = time.time()
        ct_bound, m_ct = run_continuous_VeRecycle(
            verifier=verifier,
            alpha_RA=alpha_ra,
            beta_RA=beta_ra,
            disrupted_region=make_disrupted_region(scenario.low, scenario.high),
            mesh_size=args.ct_mesh_size,
            batch_size=args.batch_size,
        )
        ct_time = time.time() - t0

        scenario_row = {
            "scenario_name": scenario.name,
            "description": scenario.description,
            "xq_low_0": float(scenario.low[0]),
            "xq_low_1": float(scenario.low[1]),
            "xq_high_0": float(scenario.high[0]),
            "xq_high_1": float(scenario.high[1]),
            "intersects_initial": boxes_intersect(
                scenario.low,
                scenario.high,
                initial_bounds[0, 0],
                initial_bounds[0, 1],
            ),
            "alpha_ra": float(alpha_ra),
            "beta_ra": float(beta_ra),
            "original_bound": float(original_bound),
            "m_ct_ibp": float(m_ct),
            "ct_verecycle_bound": float(ct_bound),
            "ct_verecycle_time_s": float(ct_time),
        }
        scenario_rows.append(scenario_row)

        print(f"CT m_lb             = {m_ct:.6f}")
        print(f"CT VeRecycle bound  = {ct_bound:.6f}")

        for grid_width in grid_widths:
            result = naive_discrete_verecycle_on_grid(
                net,
                scenario,
                grid_width=grid_width,
                alpha_ra=alpha_ra,
                beta_ra=beta_ra,
            )
            row = {
                "scenario_name": result.scenario_name,
                "grid_width": result.grid_width,
                "n_discrete_cells": result.n_discrete_cells,
                "m_discrete_point": result.m_discrete_point,
                "naive_discrete_bound": result.naive_discrete_bound,
                "missed_changed_region": result.missed_changed_region,
                "ct_verecycle_bound": float(ct_bound),
                "overclaim_vs_ct_bound": float(result.naive_discrete_bound - ct_bound),
                "baseline_kind": "naive_state_grid_centers_only",
            }
            grid_rows.append(row)

        for dt in time_steps:
            result = estimate_absorbing_mc_from_paths(
                base_paths,
                scenario,
                fine_dt=args.fine_dt,
                coarse_dt=dt,
            )
            row = {
                "scenario_name": result.scenario_name,
                "dt": result.dt,
                "ct_absorbing_success": result.ct_absorbing_success,
                "ct_xq_hit_fraction": result.ct_xq_hit_fraction,
                "sampled_xq_hit_fraction": result.sampled_xq_hit_fraction,
                "missed_xq_given_ct_hit_fraction": result.missed_xq_given_ct_hit_fraction,
                "ct_verecycle_bound": float(ct_bound),
            }
            mc_rows.append(row)

        print("Naive discrete grid bounds:")
        for row in [row for row in grid_rows if row["scenario_name"] == scenario.name]:
            print(
                f"  h={row['grid_width']:<5g} cells={row['n_discrete_cells']:<5d} "
                f"bound={row['naive_discrete_bound']:.6f} "
                f"missed={row['missed_changed_region']}"
            )

    reference_dt = min(time_steps)
    mc_reference = {
        row["scenario_name"]: row
        for row in mc_rows
        if abs(float(row["dt"]) - reference_dt) < 1e-12
    }
    for row in grid_rows:
        ref = mc_reference.get(row["scenario_name"], {})
        ct_mc_success = ref.get("ct_absorbing_success", float("nan"))
        row["reference_dt"] = reference_dt
        row["ct_absorbing_success_at_reference_dt"] = ct_mc_success
        row["ct_xq_hit_fraction_at_reference_dt"] = ref.get("ct_xq_hit_fraction", float("nan"))
        row["overclaim_vs_ct_mc_success"] = (
            float(row["naive_discrete_bound"]) - float(ct_mc_success)
            if np.isfinite(ct_mc_success)
            else float("nan")
        )

    write_csv(out_dir / "scenario_summary.csv", scenario_rows)
    write_csv(out_dir / "grid_discretization_summary.csv", grid_rows)
    write_csv(out_dir / "time_sampling_summary.csv", mc_rows)

    bridge = {
        "experiment": "ct_vs_naive_discretized_verecycle_pendulum",
        "note": (
            "The naive_discrete baseline is intentionally unsafe: it represents Xq only by "
            "state-grid centers and uses point certificate values. It is included to show "
            "how discretizing the CT scenario can over-reclaim."
        ),
        "alpha_ra": float(alpha_ra),
        "beta_ra": float(beta_ra),
        "original_bound": float(original_bound),
        "scenarios": scenario_rows,
        "grid_discretization": grid_rows,
        "time_sampling": mc_rows,
    }
    with open(out_dir / "verecycle_repo_bridge.json", "w") as f:
        json.dump(bridge, f, indent=2)

    plot_bounds(grid_rows, scenario_rows, out_dir)
    plot_hit_rates(mc_rows, out_dir)
    plot_spatial_aliasing_setup(base_paths, scenarios, out_dir, grid_width=0.05)
    plot_temporal_aliasing_example(base_paths, scenarios, out_dir, fine_dt=args.fine_dt, coarse_dt=0.5)
    thin_corridor_row = next(
        (row for row in scenario_rows if row["scenario_name"] == "thin_corridor_absorbing_barrier"),
        None,
    )
    if thin_corridor_row is not None:
        plot_certificate_aliasing_context(
            net,
            base_paths,
            scenarios,
            out_dir,
            alpha_ra=alpha_ra,
            m_ct=float(thin_corridor_row["m_ct_ibp"]),
            grid_width=0.05,
        )

    base_sde.close()
    print("\nSaved results to:")
    print(out_dir)


if __name__ == "__main__":
    main()
