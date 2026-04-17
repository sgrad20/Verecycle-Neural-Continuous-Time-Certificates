import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_ct.ct_verecycle_plotting import (
    FAMILY_COLORS,
    PAPER_ABSORBING_SCENARIO,
    PAPER_LABELS,
    PAPER_MAIN_SCENARIOS,
    load_summary_rows,
)
from experiments_ct.run_ct_no_mc import (
    DEVICE,
    global_bounds,
    initial_bounds,
    load_certified_checkpoint,
    target_bounds,
    unsafe_bounds,
)


def plain_label(name: str) -> str:
    return PAPER_LABELS.get(name, name.replace("_", " ")).replace("\n", " ")


def add_box(ax, bounds, label, color, *, alpha=0.15, linewidth=1.8, linestyle="-"):
    low = np.asarray(bounds[0], dtype=np.float32)
    high = np.asarray(bounds[1], dtype=np.float32)
    rect = Rectangle(
        (float(low[0]), float(low[1])),
        float(high[0] - low[0]),
        float(high[1] - low[1]),
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    ax.add_patch(rect)
    return rect


def evaluate_certificate_grid(net, xlim, ylim, nx=220, ny=180):
    xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    values = []
    for start in range(0, points.shape[0], 4096):
        batch = torch.tensor(points[start:start + 4096], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            values.append(net(batch).detach().cpu().numpy().reshape(-1))
    zz = np.concatenate(values, axis=0).reshape(ny, nx)
    return xs, ys, zz


def sample_closed_loop_paths(sde, n_paths=16, horizon=5.0, dt=0.025, seed=11):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(initial_bounds[0, 0], initial_bounds[0, 1], size=(n_paths, 2)).astype(np.float32)
    times = torch.linspace(0.0, horizon, int(round(horizon / dt)) + 1, device=DEVICE)
    with torch.no_grad():
        raw = sde.sample(
            torch.tensor(x0, dtype=torch.float32, device=DEVICE),
            times,
            method="euler",
            dt=dt,
        ).detach().cpu().numpy()
    if raw.ndim != 3:
        raise ValueError(f"Unexpected SDE sample shape: {raw.shape}")
    if raw.shape[1] == n_paths:
        return np.transpose(raw, (1, 0, 2))
    if raw.shape[0] == n_paths:
        return raw
    raise ValueError(f"Unexpected SDE sample shape: {raw.shape}")


def plot_pendulum_setup(summary_rows, net, sde, out_dir: Path):
    scenario_names = PAPER_MAIN_SCENARIOS + [PAPER_ABSORBING_SCENARIO]
    rows_by_name = {row["scenario_name"]: row for row in summary_rows}
    selected = [rows_by_name[name] for name in scenario_names if name in rows_by_name]

    xlim = (-13.5, 13.5)
    ylim = (-5.2, 5.2)
    xs, ys, zz = evaluate_certificate_grid(net, xlim, ylim)
    cap = np.nanpercentile(zz, 95.0)
    paths = sample_closed_loop_paths(sde)

    fig, (ax_state, ax_cert) = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True, sharey=True)

    for ax in (ax_state, ax_cert):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Angular velocity")
        ax.grid(True, linestyle="--", alpha=0.22)
    ax_state.set_ylabel("Angle")

    add_box(ax_state, initial_bounds[0], "Initial set", "#2ca02c", alpha=0.20)
    add_box(ax_state, target_bounds[0], "Target set", "#1f77b4", alpha=0.16)
    for idx, ub in enumerate(unsafe_bounds):
        add_box(ax_state, ub, "Unsafe set" if idx == 0 else "_nolegend_", "#d62728", alpha=0.08)

    for path in paths:
        ax_state.plot(path[:, 0], path[:, 1], color="#333333", linewidth=1.0, alpha=0.55)

    for row in selected:
        low = np.array([row["xq_low_0"], row["xq_low_1"]], dtype=np.float32)
        high = np.array([row["xq_high_0"], row["xq_high_1"]], dtype=np.float32)
        color = FAMILY_COLORS.get(row["scenario_family"], "#ff7f0e")
        add_box(ax_state, np.stack((low, high)), plain_label(row["scenario_name"]), color, alpha=0.14, linewidth=2.2)

    ax_state.set_title("(a) Benchmark geometry and closed-loop traces")
    handles, labels = ax_state.get_legend_handles_labels()
    handles.insert(3, Line2D([0], [0], color="#333333", linewidth=1.2, alpha=0.75))
    labels.insert(3, "Closed-loop traces")
    ax_state.legend(handles, labels, frameon=False, fontsize=7, loc="lower left", ncol=1)

    contour = ax_cert.contourf(xs, ys, np.minimum(zz, cap), levels=24, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax_cert, fraction=0.046, pad=0.04)
    cbar.set_label("Certificate value V(x)")
    alpha = float(selected[0]["alpha_ra"]) if selected else float("nan")
    beta = float(selected[0]["beta_ra"]) if selected else float("nan")
    if np.isfinite(alpha):
        ax_cert.contour(xs, ys, zz, levels=[alpha], colors=["white"], linewidths=1.8)
    if np.isfinite(beta) and beta < np.nanmax(zz):
        ax_cert.contour(xs, ys, zz, levels=[beta], colors=["#f4d03f"], linewidths=1.8)

    for row in selected:
        low = np.array([row["xq_low_0"], row["xq_low_1"]], dtype=np.float32)
        high = np.array([row["xq_high_0"], row["xq_high_1"]], dtype=np.float32)
        color = FAMILY_COLORS.get(row["scenario_family"], "#ff7f0e")
        add_box(ax_cert, np.stack((low, high)), "_nolegend_", color, alpha=0.10, linewidth=2.2)
        center = 0.5 * (low + high)
        ax_cert.text(
            float(center[0]),
            float(center[1]),
            f"{float(row['m_lb_ibp']):.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.0},
        )

    ax_cert.set_title("(b) Certificate landscape; labels show local lower bounds")
    fig.suptitle("Inverted pendulum evaluation setup", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "paper_setup_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_caption(out_dir: Path):
    path = out_dir / "paper_setup_caption.txt"
    path.write_text(
        "paper_setup_overview.png\n"
        "Evaluation setup for the stochastic inverted pendulum. Left: state-space "
        "sets, representative changed regions, and sample closed-loop trajectories "
        "from the fixed policy. Right: certificate values over the same state-space "
        "window; numbers inside changed regions report the certified lower bound "
        "used by VeRecycle.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate paper-facing setup plots for CT VeRecycle.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing summary.csv.")
    args = parser.parse_args()

    summary_csv = args.results_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary.csv in {args.results_dir}")

    summary_rows = load_summary_rows(summary_csv)
    _, net, sde, _ = load_certified_checkpoint()
    plot_pendulum_setup(summary_rows, net, sde, args.results_dir)
    write_caption(args.results_dir)
    print(f"Generated setup plot in: {args.results_dir}")


if __name__ == "__main__":
    main()
