import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NUMERIC_FIELDS = {
    "alpha_ra",
    "beta_ra",
    "beta_ra_actual",
    "original_bound_target",
    "original_bound",
    "m_lb_ibp",
    "verecycle_bound",
    "recert_bound",
    "recert_posthoc_bound",
    "recert_alpha_ra",
    "recert_beta_ra",
    "recert_time_s",
    "verecycle_time_s",
    "runtime_ratio_recert_over_verecycle",
}


def load_summary(summary_csv: Path):
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = dict(row)
            for key in NUMERIC_FIELDS:
                raw = parsed.get(key, "")
                if raw in ("", None):
                    parsed[key] = float("nan")
                    continue
                try:
                    parsed[key] = float(raw)
                except ValueError:
                    parsed[key] = float("nan")
            rows.append(parsed)
    return rows


def short_label(name: str) -> str:
    mapping = {
        "low_reclaim_diff_1p5": "Low certificate margin",
        "borderline_diff_2p0": "Borderline margin",
        "mid_band_drift_mild": "Mild drift change",
        "mid_transition_diff_2p0": "Transition region",
        "right_corridor_drift_strong": "Strong drift corridor",
        "high_transition_diff_1p5": "High-margin transition",
        "near_sat_far_right_diff_1p1": "Near original bound",
    }
    return mapping.get(name, name.replace("_", " "))


def certified_recert_rows(rows):
    return [
        row for row in rows
        if row.get("recert_status") == "certified"
        and np.isfinite(row["recert_alpha_ra"])
        and np.isfinite(row["recert_beta_ra"])
    ]


def recert_bound(row):
    bound = row.get("recert_bound", float("nan"))
    if np.isfinite(bound):
        return bound
    return row["recert_posthoc_bound"]


def original_certified_bound(row):
    bound = row.get("original_bound_target", float("nan"))
    if np.isfinite(bound):
        return bound
    return row["original_bound"]


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.8, alpha=0.25)


def annotate_bars(ax, bars, fmt=".2f", fontsize=8):
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


def plot_certificate_bound_ladder(rows, out_dir: Path):
    labels = [short_label(row["scenario_name"]) for row in rows]
    original = [original_certified_bound(row) for row in rows]
    verecycle = [row["verecycle_bound"] for row in rows]
    recert = [recert_bound(row) for row in rows]

    x = np.arange(len(rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 4.8))
    bars1 = ax.bar(x - width, original, width, label="Original bound", color="#7A7A7A")
    bars2 = ax.bar(x, verecycle, width, label="VeRecycle", color="#1f77b4")
    bars3 = ax.bar(x + width, recert, width, label="Re-certification", color="#d95f02")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Reach-avoid probability")
    ax.set_title("Certificate-derived guarantees after local change")
    ax.set_ylim(0.0, 1.02)
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    annotate_bars(ax, bars2)
    annotate_bars(ax, bars3)
    fig.tight_layout()
    fig.savefig(out_dir / "certificate_bound_ladder.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_certificate_level_shift(rows, out_dir: Path):
    labels = [short_label(row["scenario_name"]) for row in rows]
    alpha_orig = [row["alpha_ra"] for row in rows]
    alpha_recert = [row["recert_alpha_ra"] for row in rows]
    beta_orig = [row["beta_ra_actual"] for row in rows]
    beta_recert = [row["recert_beta_ra"] for row in rows]

    x = np.arange(len(rows))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)

    bars1 = ax1.bar(x - width / 2, alpha_orig, width, label="Original initial-set level", color="#7A7A7A")
    bars2 = ax1.bar(x + width / 2, alpha_recert, width, label="Re-certified initial-set level", color="#1f77b4")
    ax1.set_ylabel("Certificate level")
    ax1.set_title("How retraining changes the certificate levels")
    style_axis(ax1)
    ax1.legend(frameon=False, loc="upper right")
    annotate_bars(ax1, bars2, ".3f")

    bars3 = ax2.bar(x - width / 2, beta_orig, width, label="Original unsafe-set level", color="#7A7A7A")
    bars4 = ax2.bar(x + width / 2, beta_recert, width, label="Re-certified unsafe-set level", color="#d95f02")
    ax2.set_ylabel("Certificate level")
    ax2.set_xticks(x, labels)
    style_axis(ax2)
    ax2.legend(frameon=False, loc="upper left")
    annotate_bars(ax2, bars4, ".2f")

    fig.tight_layout()
    fig.savefig(out_dir / "certificate_level_shift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_certificate_margin_gain(rows, out_dir: Path):
    labels = [short_label(row["scenario_name"]) for row in rows]
    gains_vs_original = [recert_bound(row) - original_certified_bound(row) for row in rows]
    gains_vs_verecycle = [recert_bound(row) - row["verecycle_bound"] for row in rows]

    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 4.8))
    bars1 = ax.bar(x - width / 2, gains_vs_original, width, label="Re-cert minus original", color="#55A868")
    bars2 = ax.bar(x + width / 2, gains_vs_verecycle, width, label="Re-cert minus VeRecycle", color="#8172B2")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Bound difference")
    ax.set_title("Re-certification gain relative to original and VeRecycle")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    annotate_bars(ax, bars1, ".2f")
    annotate_bars(ax, bars2, ".2f")
    fig.tight_layout()
    fig.savefig(out_dir / "certificate_margin_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_caption_file(out_dir: Path):
    text = (
        "certificate_bound_ladder.png\n"
        "Comparison of the original bound, the VeRecycle reclaimed bound, and the verified bound of the retrained certificate for scenarios where re-certification succeeded.\n\n"
        "certificate_level_shift.png\n"
        "Comparison of the original and re-certified certificate levels. Retraining typically reduces the initial-set level while keeping the unsafe-set level high, which explains the larger verified bound.\n\n"
        "certificate_margin_gain.png\n"
        "Difference between the re-certification bound and the original / VeRecycle bounds, highlighting how much extra performance retraining recovers when it succeeds.\n"
    )
    (out_dir / "certificate_plot_captions.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate certificate-focused plots from a CT VeRecycle results folder.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing summary.csv.")
    args = parser.parse_args()

    summary_csv = args.results_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary.csv in {args.results_dir}")

    rows = load_summary(summary_csv)
    cert_rows = certified_recert_rows(rows)
    if not cert_rows:
        raise ValueError("No certified re-certification rows found in summary.csv")

    plot_certificate_bound_ladder(cert_rows, args.results_dir)
    plot_certificate_level_shift(cert_rows, args.results_dir)
    plot_certificate_margin_gain(cert_rows, args.results_dir)
    write_caption_file(args.results_dir)
    print(f"Generated certificate plots in: {args.results_dir}")


if __name__ == "__main__":
    main()
