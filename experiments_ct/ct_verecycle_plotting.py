import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

# python experiments_ct\ct_verecycle_plotting.py --results-dir experiments_ct_verecycle\results_and_plots_short_run\20260402_163549
DEFAULT_GLOBAL_BOUNDS = np.array([[[-20.0, -2 * np.pi], [20.0, 2 * np.pi]]], dtype=np.float32)
DEFAULT_INITIAL_BOUNDS = np.array([[[-1.0, 3 / 4 * np.pi], [1.0, 5 / 4 * np.pi]]], dtype=np.float32)
DEFAULT_TARGET_BOUNDS = np.array([[[-4.0, -np.pi / 2], [4.0, np.pi / 2]]], dtype=np.float32)
DEFAULT_UNSAFE_BOUNDS = np.array(
    [
        [[-20.0, -2 * np.pi], [-10.0, -3 / 2 * np.pi]],
        [[10.0, 3 / 2 * np.pi], [20.0, 2 * np.pi]],
    ],
    dtype=np.float32,
)

FAMILY_PREFIX = {
    "absorbing": "A",
    "diffusion": "D",
    "drift": "R",
}

FAMILY_DISPLAY = {
    "absorbing": "Absorbing",
    "diffusion": "Diffusion",
    "drift": "Drift",
}

FAMILY_COLORS = {
    "absorbing": "#C44E52",
    "diffusion": "#4C72B0",
    "drift": "#55A868",
}

PAPER_COLORS = {
    "verecycle": "#1f77b4",
    "recert": "#d95f02",
    "reference": "#4d4d4d",
    "absorbing": "#c23b22",
    "accent": "#6c757d",
}

PAPER_MAIN_SCENARIOS = [
    "low_reclaim_diff_1p5",
    "borderline_diff_2p0",
    "right_corridor_drift_strong",
    "near_sat_far_right_diff_1p1",
]

PAPER_ABSORBING_SCENARIO = "absorbing_central_bridge"

PAPER_LABELS = {
    "low_reclaim_diff_1p5": "Low reclaim\n(diffusion)",
    "borderline_diff_2p0": "Borderline\n(diffusion)",
    "right_corridor_drift_strong": "Strong corridor\n(drift)",
    "near_sat_far_right_diff_1p1": "Near saturation\n(diffusion)",
    "absorbing_central_bridge": "Central bridge\n(absorbing)",
}

PAPER_CAPTIONS = {
    "paper_main_subset.png": (
        "Comparison of VeRecycle and re-certification on four representative non-absorbing "
        "continuous-time scenarios. Left: VeRecycle returns an immediate reclaimed guarantee, "
        "while re-certification is reported as the post-hoc sampled estimate of the retrained "
        "certificate. Right: VeRecycle completes in milliseconds to sub-second time, whereas "
        "re-certification requires tens of seconds."
    ),
    "paper_absorbing_case.png": (
        "Exact absorbing case for the central-bridge scenario. The verified lower bound "
        "m_lb on the disrupted region falls below alpha_RA, so VeRecycle cannot reclaim a "
        "positive bound. The original certificate also fails under exact absorbing semantics, "
        "and re-certification is unsupported without explicit absorbing-structure knowledge."
    ),
}


@dataclass
class ScenarioPlotSpec:
    name: str
    low: np.ndarray
    high: np.ndarray


SUMMARY_NUMERIC_FIELDS = {
    "diffusion_scale",
    "drift_bias_0",
    "drift_bias_1",
    "xq_low_0",
    "xq_low_1",
    "xq_high_0",
    "xq_high_1",
    "alpha_ra",
    "beta_ra",
    "beta_ra_actual",
    "original_bound_target",
    "original_bound",
    "original_modified_verify_time_s",
    "original_modified_prob_ra_estimate",
    "original_modified_prob_s_estimate",
    "original_modified_decrease_violations",
    "m_lb_ibp",
    "m_grid_min_raw",
    "m_grid_argmin_0",
    "m_grid_argmin_1",
    "verecycle_bound",
    "verecycle_time_s",
    "recert_bound",
    "recert_posthoc_bound",
    "recert_alpha_ra",
    "recert_beta_ra",
    "recert_certified_epoch",
    "recert_time_s",
    "runtime_ratio_recert_over_verecycle",
    "speedup",
}


def _plot_safe(values):
    out = []
    for v in values:
        if v is None or not np.isfinite(v):
            out.append(0.0)
        else:
            out.append(float(v))
    return out


def _plot_with_nan(values, min_value: float | None = None):
    out = []
    for v in values:
        if v is None:
            out.append(np.nan)
            continue
        try:
            value = float(v)
        except (TypeError, ValueError):
            out.append(np.nan)
            continue
        if not np.isfinite(value):
            out.append(np.nan)
            continue
        if min_value is not None and value < min_value:
            out.append(np.nan)
            continue
        out.append(value)
    return out


def _coerce_summary_row(row):
    parsed = dict(row)
    for key in SUMMARY_NUMERIC_FIELDS:
        if key not in parsed:
            continue
        raw = parsed[key]
        if raw in ("", None):
            parsed[key] = float("nan")
            continue
        try:
            parsed[key] = float(raw)
        except ValueError:
            parsed[key] = float("nan")

    if "original_modified_verified" in parsed:
        parsed["original_modified_verified"] = str(parsed["original_modified_verified"]).lower() == "true"

    return parsed


def load_summary_rows(summary_csv: Path):
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        return [_coerce_summary_row(row) for row in reader]


def _with_plot_codes(summary_rows):
    counters = {key: 0 for key in FAMILY_PREFIX}
    rows = []
    for row in summary_rows:
        parsed = dict(row)
        family = parsed.get("scenario_family", "")
        prefix = FAMILY_PREFIX.get(family, "S")
        counters[family] = counters.get(family, 0) + 1
        parsed["plot_code"] = f"{prefix}{counters[family]}"
        rows.append(parsed)
    return rows


def _write_scenario_codebook(summary_rows, out_dir: Path):
    codebook_path = out_dir / "scenario_codebook.csv"
    with open(codebook_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "plot_code",
                "scenario_name",
                "scenario_family",
                "change_type",
                "method_comparison",
                "recert_status",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "plot_code": row["plot_code"],
                    "scenario_name": row["scenario_name"],
                    "scenario_family": row["scenario_family"],
                    "change_type": row["change_type"],
                    "method_comparison": row.get("method_comparison", ""),
                    "recert_status": row.get("recert_status", ""),
                }
            )


def _draw_codebook(ax, summary_rows, n_cols: int = 2):
    ax.axis("off")
    ax.set_title("Scenario key", fontsize=11, loc="left")
    lines = [
        f"{row['plot_code']:<3} {row['scenario_name']}"
        for row in summary_rows
    ]
    n_cols = max(1, int(n_cols))
    col_size = int(np.ceil(len(lines) / n_cols))
    for col in range(n_cols):
        chunk = lines[col * col_size:(col + 1) * col_size]
        if not chunk:
            continue
        x = 0.0 if n_cols == 1 else 0.02 + 0.49 * col
        ax.text(
            x,
            0.98,
            "\n".join(chunk),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )


def _annotate_code(ax, x, y, code: str, idx: int):
    offsets = [(5, 5), (5, -11), (-14, 5), (-14, -11), (8, 12), (-18, 12)]
    dx, dy = offsets[idx % len(offsets)]
    ax.annotate(
        code,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )


def _comparison_legend_handles():
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7,
            label=FAMILY_DISPLAY[family],
        )
        for family, color in FAMILY_COLORS.items()
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="black",
            markerfacecolor="black",
            markersize=7,
            label="VeRecycle",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgewidth=1.5,
            markersize=7,
            label="Re-certification",
        ),
    ]
    return family_handles + method_handles


def _save_codebook_figure(summary_rows, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _draw_codebook(ax, summary_rows)
    fig.tight_layout()
    fig.savefig(out_dir / "scenario_codebook.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _paper_rc():
    return {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }


def _style_paper_axis(ax, add_y_grid: bool = True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if add_y_grid:
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.25)
    else:
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.2)


def _write_paper_captions(out_dir: Path):
    caption_path = out_dir / "paper_figure_captions.txt"
    with open(caption_path, "w", encoding="utf-8") as f:
        for filename, caption in PAPER_CAPTIONS.items():
            f.write(f"{filename}\n")
            f.write(f"{caption}\n\n")


def _find_rows_in_order(summary_rows, scenario_names):
    by_name = {row["scenario_name"]: row for row in summary_rows}
    return [by_name[name] for name in scenario_names if name in by_name]


def _paper_label(scenario_name: str) -> str:
    return PAPER_LABELS.get(scenario_name, scenario_name.replace("_", "\n"))


def _annotate_bar_values(ax, bars, fmt: str, fontsize: int = 8):
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h) and h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                format(h, fmt),
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


def make_paper_main_subset_plot(summary_rows, out_dir: Path):
    rows = _find_rows_in_order(summary_rows, PAPER_MAIN_SCENARIOS)
    if not rows:
        return

    labels = [_paper_label(row["scenario_name"]) for row in rows]
    vr_bounds = _plot_safe([row["verecycle_bound"] for row in rows])
    rr_bounds = _plot_with_nan([row["recert_posthoc_bound"] for row in rows])
    vr_times = _plot_safe([row["verecycle_time_s"] for row in rows])
    rr_times = _plot_with_nan([row["recert_time_s"] for row in rows], min_value=0.0)
    original_bound = float(rows[0]["original_bound"])

    x = np.arange(len(rows))
    width = 0.34

    with plt.rc_context(_paper_rc()):
        fig, (ax_bounds, ax_runtime) = plt.subplots(1, 2, figsize=(12.5, 4.8))

        bars1 = ax_bounds.bar(
            x - width / 2,
            vr_bounds,
            width,
            label="VeRecycle",
            color=PAPER_COLORS["verecycle"],
        )
        bars2 = ax_bounds.bar(
            x + width / 2,
            rr_bounds,
            width,
            label="Re-certification (sampled)",
            color=PAPER_COLORS["recert"],
        )
        ax_bounds.axhline(
            original_bound,
            linestyle="--",
            linewidth=1.4,
            color=PAPER_COLORS["reference"],
            label="Original bound",
        )
        ax_bounds.set_xticks(x, labels)
        ax_bounds.set_ylabel("Reach-avoid probability")
        ax_bounds.set_title("(a) Guarantees")
        ax_bounds.set_ylim(0.0, 1.02)
        _style_paper_axis(ax_bounds)
        ax_bounds.legend(frameon=False, loc="lower right")
        _annotate_bar_values(ax_bounds, bars1, ".2f")
        _annotate_bar_values(ax_bounds, bars2, ".2f")

        bars3 = ax_runtime.bar(
            x - width / 2,
            vr_times,
            width,
            label="VeRecycle",
            color=PAPER_COLORS["verecycle"],
        )
        bars4 = ax_runtime.bar(
            x + width / 2,
            rr_times,
            width,
            label="Re-certification",
            color=PAPER_COLORS["recert"],
        )
        ax_runtime.set_xticks(x, labels)
        ax_runtime.set_ylabel("Runtime (s)")
        ax_runtime.set_title("(b) Runtime")
        ax_runtime.set_yscale("log")
        _style_paper_axis(ax_runtime)
        ax_runtime.legend(frameon=False, loc="upper left")
        _annotate_bar_values(ax_runtime, bars3, ".3f")
        _annotate_bar_values(ax_runtime, bars4, ".1f")

        fig.suptitle("Representative non-absorbing scenarios", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "paper_main_subset.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_paper_absorbing_case_plot(summary_rows, out_dir: Path):
    rows = _find_rows_in_order(summary_rows, [PAPER_ABSORBING_SCENARIO])
    if not rows:
        return

    row = rows[0]
    with plt.rc_context(_paper_rc()):
        fig = plt.figure(figsize=(12.5, 4.8))
        grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.8])
        ax_left = fig.add_subplot(grid[0, 0])
        ax_right = fig.add_subplot(grid[0, 1])

        alpha = float(row["alpha_ra"])
        m_lb = float(row["m_lb_ibp"])
        bars = ax_left.bar(
            ["alpha_RA", "m_lb_ibp"],
            [alpha, m_lb],
            color=[PAPER_COLORS["accent"], PAPER_COLORS["absorbing"]],
        )
        _annotate_bar_values(ax_left, bars, ".3f")
        ax_left.set_ylabel("Certificate level")
        ax_left.set_title("(a) VeRecycle condition")
        _style_paper_axis(ax_left)
        ax_left.text(
            0.5,
            0.92,
            r"$m_{\mathrm{lb}} \leq \alpha_{\mathrm{RA}}$",
            transform=ax_left.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )

        ax_right.axis("off")
        lines = [
            f"Scenario: {_paper_label(row['scenario_name'])}",
            "",
            f"Original certificate under exact absorbing dynamics: {'failed' if not row['original_modified_verified'] else 'verified'}",
            f"Decrease-condition counterexamples: {int(row['original_modified_decrease_violations'])}",
            f"VeRecycle bound: {float(row['verecycle_bound']):.3f}",
            f"Re-certification status: {row['recert_status']}",
            "",
            "Interpretation",
            "The disrupted region enters a regime where the certified lower",
            "bound m_lb drops below alpha_RA. VeRecycle therefore cannot",
            "reclaim a positive guarantee, and retraining is unavailable",
            "without explicit absorbing-structure knowledge.",
        ]
        ax_right.text(
            0.0,
            0.98,
            "\n".join(lines),
            transform=ax_right.transAxes,
            va="top",
            ha="left",
            fontsize=11,
        )

        fig.suptitle("Exact absorbing case", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "paper_absorbing_case.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_summary_plots(summary_rows, out_dir: Path):
    summary_rows = _with_plot_codes(summary_rows)
    _write_scenario_codebook(summary_rows, out_dir)
    _save_codebook_figure(summary_rows, out_dir)
    _write_paper_captions(out_dir)
    scenario_names = [row["scenario_name"] for row in summary_rows]
    plot_labels = [row["plot_code"] for row in summary_rows]
    vr_bounds = _plot_safe([row["verecycle_bound"] for row in summary_rows])
    rr_bounds = _plot_with_nan([row["recert_posthoc_bound"] for row in summary_rows])
    vr_times = _plot_safe([row["verecycle_time_s"] for row in summary_rows])
    rr_times = _plot_with_nan([row["recert_time_s"] for row in summary_rows], min_value=0.0)
    m_lb_ibp = _plot_safe([row["m_lb_ibp"] for row in summary_rows])
    runtime_ratios = _plot_with_nan([
        row.get("runtime_ratio_recert_over_verecycle", row.get("speedup", float("nan")))
        for row in summary_rows
    ], min_value=0.0)
    method_comparison = [row.get("method_comparison", "") for row in summary_rows]

    x = np.arange(len(summary_rows))
    width = 0.35
    original_bound = float(summary_rows[0]["original_bound"])

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, vr_bounds, width, label="VeRecycle")
    plt.bar(x + width / 2, rr_bounds, width, label="Re-certification (0 when unsupported)")
    plt.axhline(original_bound, linestyle="--", linewidth=1.5, label="Original bound")
    plt.xticks(x, plot_labels)
    plt.ylabel("Reach-avoid probability")
    plt.title("Bound comparison by scenario code")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bounds_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    colors = [
        "#C44E52" if tag == "verecycle_only" else "#4C72B0"
        for tag in method_comparison
    ]
    bars = plt.bar(x, vr_bounds, width=0.6, color=colors)
    recert_failed = [
        np.isfinite(vr) and vr > 0.0 and tag == "verecycle_only"
        for vr, tag in zip(vr_bounds, method_comparison)
    ]
    failed_y = [vr_bounds[i] if recert_failed[i] else np.nan for i in range(len(summary_rows))]
    plt.scatter(
        x,
        failed_y,
        marker="x",
        s=80,
        color="black",
        label="VeRecycle works, re-certification does not",
        zorder=3,
    )
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if np.isfinite(h) and h > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    plt.xticks(x, scenario_names, rotation=20, ha="right")
    plt.ylabel("VeRecycle bound")
    plt.title("Cases where VeRecycle succeeds but re-certification does not")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "verecycle_only_cases.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, vr_times, width, label="VeRecycle")
    plt.bar(x + width / 2, rr_times, width, label="Re-certification")
    plt.xticks(x, plot_labels)
    plt.ylabel("Runtime (s)")
    plt.title("Runtime comparison by scenario code")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(x - width / 2, vr_times, width, label="VeRecycle")
    bars2 = plt.bar(x + width / 2, rr_times, width, label="Re-certification")
    plt.xticks(x, plot_labels)
    plt.ylabel("Runtime (s, log scale)")
    plt.title("Runtime comparison by scenario code (log scale)")
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
    plt.savefig(out_dir / "runtime_comparison_log.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, m_lb_ibp, width, color="#4C72B0")
    plt.xticks(x, plot_labels)
    plt.ylabel("Lower bound m_lb_ibp")
    plt.title("Estimated local lower bound by scenario code")
    plt.tight_layout()
    plt.savefig(out_dir / "m_lb_ibp_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, runtime_ratios, width, color="#55A868")
    plt.xticks(x, plot_labels)
    plt.ylabel("Runtime ratio (re-cert / VeRecycle)")
    plt.title("Runtime ratio by scenario code")
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h) and h > 0:
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
    plt.savefig(out_dir / "runtime_ratio_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "speedup_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    gap = [
        rr - vr if np.isfinite(rr) and np.isfinite(vr) else np.nan
        for rr, vr in zip(rr_bounds, vr_bounds)
    ]
    plt.bar(x, gap, width, color="#8172B2")
    plt.xticks(x, plot_labels)
    plt.ylabel("Post-hoc re-certification - VeRecycle")
    plt.title("Bound gap comparison by scenario code")
    plt.tight_layout()
    plt.savefig(out_dir / "bound_gap_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(14.5, 7.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.9, 1.5])
    ax = fig.add_subplot(grid[0, 0])
    key_ax = fig.add_subplot(grid[0, 1])
    for i, row in enumerate(summary_rows):
        family = row["scenario_family"]
        color = FAMILY_COLORS.get(family, "#4C72B0")
        x = m_lb_ibp[i]
        y_vr = vr_bounds[i]
        y_rr = rr_bounds[i]
        if np.isfinite(y_rr):
            ax.plot([x, x], [y_vr, y_rr], color=color, alpha=0.25, linewidth=1.5, zorder=1)
        ax.scatter(x, y_vr, color=color, s=90, marker="o", zorder=3)
        if np.isfinite(y_rr):
            ax.scatter(x, y_rr, facecolors="white", edgecolors=color, linewidths=1.8, s=90, marker="s", zorder=4)
    ax.set_xlabel("m_lb_ibp")
    ax.set_ylabel("Reach-avoid probability")
    ax.set_title("Local lower bound vs guarantees")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(handles=_comparison_legend_handles(), loc="lower right", fontsize=9)
    _draw_codebook(key_ax, summary_rows, n_cols=1)
    fig.tight_layout()
    fig.savefig(out_dir / "m_lb_ibp_vs_bounds.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.6, 1.6])
    ax = fig.add_subplot(grid[0, 0])
    key_ax = fig.add_subplot(grid[0, 1])
    for i, row in enumerate(summary_rows):
        family = row["scenario_family"]
        color = FAMILY_COLORS.get(family, "#4C72B0")
        x_vr = vr_times[i]
        y_vr = vr_bounds[i]
        x_rr = rr_times[i]
        y_rr = rr_bounds[i]
        ax.scatter(x_vr, y_vr, color=color, s=90, marker="o", zorder=3)
        if np.isfinite(x_rr) and np.isfinite(y_rr):
            ax.plot([x_vr, x_rr], [y_vr, y_rr], color=color, alpha=0.25, linewidth=1.5, zorder=1)
            ax.scatter(x_rr, y_rr, facecolors="white", edgecolors=color, linewidths=1.8, s=90, marker="s", zorder=4)
        _annotate_code(ax, x_vr, y_vr, row["plot_code"], i)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (s, log scale)")
    ax.set_ylabel("Reach-avoid probability")
    ax.set_title("Runtime vs guarantee")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(handles=_comparison_legend_handles(), loc="lower right", fontsize=9)
    _draw_codebook(key_ax, summary_rows)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_vs_bound.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    order = np.argsort(m_lb_ibp)
    ordered_names = [plot_labels[i] for i in order]
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
    plt.bar(x_ord, ordered_rr, width_ord, label="Re-certification (0 when unsupported)")
    plt.axhline(original_bound, linestyle="--", linewidth=1.5, label="Original bound")
    plt.xticks(x_ord, ordered_names, rotation=20, ha="right")
    plt.ylabel("Reach-avoid probability")
    plt.title("Bounds ordered by local lower bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bounds_ordered_by_m.png", dpi=300, bbox_inches="tight")
    plt.close()

    make_paper_main_subset_plot(summary_rows, out_dir)
    make_paper_absorbing_case_plot(summary_rows, out_dir)


def plot_scenario_space(
    scenario,
    out_dir: Path,
    global_bounds,
    initial_bounds,
    target_bounds,
    unsafe_bounds,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    def add_box(bounds, label, edgecolor, facecolor, alpha=0.15):
        low = bounds[0]
        high = bounds[1]
        width = float(high[0] - low[0])
        height = float(high[1] - low[1])
        rect = Rectangle(
            (float(low[0]), float(low[1])),
            width,
            height,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            linewidth=2,
            label=label,
        )
        ax.add_patch(rect)

    add_box(initial_bounds[0], "Initial set", "green", "green", alpha=0.12)
    add_box(target_bounds[0], "Target set", "blue", "blue", alpha=0.12)
    for ub in unsafe_bounds:
        add_box(ub, "Unsafe set", "red", "none", alpha=0.0)
    add_box(np.stack([scenario.low, scenario.high]), "Disrupted X_q", "orange", "orange", alpha=0.18)

    ax.set_title(f"Scenario space: {scenario.name}")
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Angle")
    ax.set_xlim(float(global_bounds[0, 0, 0]) - 0.5, float(global_bounds[0, 1, 0]) + 0.5)
    ax.set_ylim(float(global_bounds[0, 0, 1]) - 0.5, float(global_bounds[0, 1, 1]) + 0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"scenario_space_{scenario.name}.png", dpi=300)
    plt.close(fig)


def plot_certificate_value_histograms(
    net,
    initial_set,
    unsafe_set,
    out_dir: Path,
    alpha_ra: float,
    beta_ra: float,
    beta_ra_actual: float,
    n_initial: int = 5000,
    n_unsafe: int = 20000,
):
    xs_init = initial_set.sample(n_initial)
    xs_unsafe = unsafe_set.sample(n_unsafe)
    with torch.no_grad():
        vals_init = net(xs_init).cpu().numpy().reshape(-1)
        vals_unsafe = net(xs_unsafe).cpu().numpy().reshape(-1)

    plt.figure(figsize=(10, 5))
    bins = 80
    plt.hist(vals_init, bins=bins, alpha=0.6, label="Initial set", density=True, color="#4C72B0")
    plt.hist(vals_unsafe, bins=bins, alpha=0.6, label="Unsafe set", density=True, color="#C44E52")
    plt.axvline(alpha_ra, color="green", linestyle="--", linewidth=2, label="alpha_RA")
    plt.axvline(beta_ra, color="orange", linestyle="--", linewidth=2, label="beta_RA nominal")
    plt.axvline(beta_ra_actual, color="red", linestyle=":", linewidth=2, label="beta_RA actual")
    plt.xlabel("Certificate value V(x)")
    plt.ylabel("Density")
    plt.title("Certificate value distribution on initial and unsafe sets")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "certificate_value_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_certificate_baseline_comparison(alpha_ra: float, beta_ra: float, beta_ra_actual: float, out_dir: Path):
    rho_nominal = 1.0 - alpha_ra / beta_ra
    rho_actual = 0.0 if beta_ra_actual <= 0.0 else max(0.0, 1.0 - alpha_ra / beta_ra_actual)
    names = ["nominal", "actual"]
    betas = [beta_ra, beta_ra_actual]
    bounds = [rho_nominal, rho_actual]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    bars = plt.bar(names, betas, color=["#FFB000", "#D62728"])
    plt.ylabel("Beta value")
    plt.title("Nominal vs actual beta")
    for bar, value in zip(bars, betas):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    plt.subplot(1, 2, 2)
    bars = plt.bar(names, bounds, color=["#4C72B0", "#C44E52"])
    plt.ylabel("Original bound")
    plt.title("Nominal vs actual original bound")
    for bar, value in zip(bars, bounds):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "certificate_baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_recertification_history_plot(summary_rows, out_dir: Path):
    summary_rows = _with_plot_codes(summary_rows)
    code_by_name = {row["scenario_name"]: row["plot_code"] for row in summary_rows}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
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
        ax.plot(epochs, rhos, marker="o", markersize=3, linewidth=1.6, label=code_by_name[row["scenario_name"]])

    if not found:
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rho")
    ax.set_title("Re-certification convergence history")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.14)
    fig.savefig(out_dir / "recertification_history.png", dpi=180)
    plt.close(fig)


def regenerate_plots_from_results_dir(
    results_dir: Path,
    global_bounds=DEFAULT_GLOBAL_BOUNDS,
    initial_bounds=DEFAULT_INITIAL_BOUNDS,
    target_bounds=DEFAULT_TARGET_BOUNDS,
    unsafe_bounds=DEFAULT_UNSAFE_BOUNDS,
):
    results_dir = Path(results_dir)
    summary_csv = results_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary.csv in {results_dir}")

    summary_rows = load_summary_rows(summary_csv)
    if not summary_rows:
        raise ValueError(f"No rows found in {summary_csv}")

    make_summary_plots(summary_rows, results_dir)
    make_recertification_history_plot(summary_rows, results_dir)

    first_row = summary_rows[0]
    plot_certificate_baseline_comparison(
        float(first_row["alpha_ra"]),
        float(first_row["beta_ra"]),
        float(first_row["beta_ra_actual"]),
        results_dir,
    )

    for row in summary_rows:
        plot_scenario_space(
            ScenarioPlotSpec(
                name=row["scenario_name"],
                low=np.array([row["xq_low_0"], row["xq_low_1"]], dtype=np.float32),
                high=np.array([row["xq_high_0"], row["xq_high_1"]], dtype=np.float32),
            ),
            results_dir,
            global_bounds=global_bounds,
            initial_bounds=initial_bounds,
            target_bounds=target_bounds,
            unsafe_bounds=unsafe_bounds,
        )


def main():
    parser = argparse.ArgumentParser(description="Regenerate VeRecycle result plots from a saved run directory.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing summary.csv and recert history CSVs.")
    args = parser.parse_args()

    regenerate_plots_from_results_dir(args.results_dir)
    print(f"Regenerated plots in: {args.results_dir}")


if __name__ == "__main__":
    main()
