# make_ct_verecycle_plots.py
# # Usage:
# python auto_LiRPA/experiments/plots.py --mesh_csv auto_LiRPA/experiments/ct_verecycle_mesh_sweep_20260113_105506.csv --random_csv auto_LiRPA/experiments/ct_verecycle_randomXq_20260113_105512.csv --out_dir auto_LiRPA/experiments/figures
# Assumes your CSVs contain (at least) columns like:
#   mesh sweep: mesh_size, m, alpha_RA, beta_RA, eps_reclaimed, verecycle_time_s (or runtime_s)
#   random Xq:  m, alpha_RA, beta_RA, eps_reclaimed, (optional) mc_success_rate / mc_estimate / empirical_success
#
# If a column name differs, edit the COL_* constants below.

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Column name configuration
# ---------------------------
COL_MESH_SIZE = "mesh_size"
COL_M = "m"
COL_ALPHA = "alpha_RA"
COL_BETA = "beta_RA"
COL_EPS_RECLAIMED = "eps_reclaimed"

# runtime column (script will look for the first existing name)
RUNTIME_CANDIDATES = ["verecycle_time_s", "runtime_s", "time_s", "seconds"]

# optional Monte Carlo / empirical success columns (script will pick first that exists)
MC_CANDIDATES = ["mc_success_rate", "mc_estimate", "empirical_success", "empirical_rate", "mc_rate"]


def first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def savefig(out_dir: str, name: str):
    png = os.path.join(out_dir, f"{name}.png")
    pdf = os.path.join(out_dir, f"{name}.pdf")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.savefig(pdf, bbox_inches="tight")


def compute_eps_orig(df: pd.DataFrame) -> pd.Series:
    return (1.0 - df[COL_ALPHA] / df[COL_BETA]).clip(lower=0.0, upper=1.0)


def compute_eps_from_xq(df: pd.DataFrame) -> pd.Series:
    # eps_from_Xq = 1 - alpha/m (clip to [0,1])
    eps = 1.0 - df[COL_ALPHA] / df[COL_M]
    return eps.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_csv", type=str, required=True)
    ap.add_argument("--random_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="figures")
    args = ap.parse_args()

    ensure_outdir(args.out_dir)

    mesh = pd.read_csv(args.mesh_csv)
    rand = pd.read_csv(args.random_csv)

    # Derived columns (won't overwrite if already present)
    if "eps_orig" not in mesh.columns:
        mesh["eps_orig"] = compute_eps_orig(mesh)
    if "eps_from_Xq" not in mesh.columns:
        mesh["eps_from_Xq"] = compute_eps_from_xq(mesh)

    if "eps_orig" not in rand.columns:
        rand["eps_orig"] = compute_eps_orig(rand)
    if "eps_from_Xq" not in rand.columns:
        rand["eps_from_Xq"] = compute_eps_from_xq(rand)

    # ---------------------------
    # Plot 1: Mesh refinement — m vs mesh_size (expect: m increases as mesh refines)
    # ---------------------------
    if COL_MESH_SIZE in mesh.columns and COL_M in mesh.columns:
        mesh_sorted = mesh.sort_values(COL_MESH_SIZE, ascending=False)  # coarse -> fine
        plt.figure()
        plt.plot(mesh_sorted[COL_MESH_SIZE], mesh_sorted[COL_M], marker="o")
        plt.xscale("log")
        plt.xlabel("mesh_size")
        plt.ylabel(r"$m \;=\;\mathrm{LB}\;\inf_{x\in X_q} V(x)$")
        plt.title("Mesh refinement: lower bound on certificate infimum")
        savefig(args.out_dir, "mesh_m_vs_meshsize")
        plt.close()

    # ---------------------------
    # Plot 2: Mesh refinement — eps_reclaimed vs mesh_size, with eps_from_Xq and eps_orig
    # ---------------------------
    if COL_MESH_SIZE in mesh.columns and COL_EPS_RECLAIMED in mesh.columns:
        mesh_sorted = mesh.sort_values(COL_MESH_SIZE, ascending=False)
        eps_orig_val = float(mesh_sorted["eps_orig"].iloc[0]) if "eps_orig" in mesh_sorted.columns else None

        plt.figure()
        plt.plot(mesh_sorted[COL_MESH_SIZE], mesh_sorted[COL_EPS_RECLAIMED],
                 marker="o", label=r"$\varepsilon_{\mathrm{reclaimed}}$")
        if "eps_from_Xq" in mesh_sorted.columns:
            plt.plot(mesh_sorted[COL_MESH_SIZE], mesh_sorted["eps_from_Xq"],
                     marker="o", linestyle="--", label=r"$\varepsilon_{X_q}=1-\alpha/m$")
        if eps_orig_val is not None:
            plt.axhline(eps_orig_val, linestyle=":", label=r"$\varepsilon_{\mathrm{orig}}=1-\alpha/\beta$")

        plt.xscale("log")
        plt.xlabel("mesh_size")
        plt.ylabel("probability lower bound")
        plt.title("Mesh refinement: reclaimed probability bound")
        plt.legend()
        savefig(args.out_dir, "mesh_eps_vs_meshsize")
        plt.close()

    # ---------------------------
    # Plot 3: Runtime vs mesh_size (expect: runtime increases with refinement)
    # ---------------------------
    runtime_col = first_existing_col(mesh, RUNTIME_CANDIDATES)
    if runtime_col is not None and COL_MESH_SIZE in mesh.columns:
        mesh_sorted = mesh.sort_values(COL_MESH_SIZE, ascending=False)
        plt.figure()
        plt.plot(mesh_sorted[COL_MESH_SIZE], mesh_sorted[runtime_col], marker="o")
        plt.xscale("log")
        plt.xlabel("mesh_size")
        plt.ylabel("VeRecycle runtime (s)")
        plt.title("Mesh refinement: runtime vs mesh_size")
        savefig(args.out_dir, "mesh_runtime_vs_meshsize")
        plt.close()

    # ---------------------------
    # Plot 4: Random Xq — eps_reclaimed should equal min(eps_orig, eps_from_Xq)
    # This directly “proves” the implementation is applying the VeRecycle logic.
    # ---------------------------
    if COL_EPS_RECLAIMED in rand.columns:
        rand_calc = rand.copy()
        rand_calc["eps_min_expected"] = np.minimum(rand_calc["eps_orig"], rand_calc["eps_from_Xq"])
        # scatter: actual vs expected
        plt.figure()
        plt.scatter(rand_calc["eps_min_expected"], rand_calc[COL_EPS_RECLAIMED], s=18)
        lo = np.nanmin([rand_calc["eps_min_expected"].min(), rand_calc[COL_EPS_RECLAIMED].min()])
        hi = np.nanmax([rand_calc["eps_min_expected"].max(), rand_calc[COL_EPS_RECLAIMED].max()])
        plt.plot([lo, hi], [lo, hi])  # y=x reference
        plt.xlabel(r"expected $\min(\varepsilon_{\mathrm{orig}},\,\varepsilon_{X_q})$")
        plt.ylabel(r"logged $\varepsilon_{\mathrm{reclaimed}}$")
        plt.title("Random disrupted regions: implementation matches min-rule")
        savefig(args.out_dir, "random_eps_reclaimed_matches_minrule")
        plt.close()

    # ---------------------------
    # Plot 5: Random Xq — reclaimed bound vs m (monotone increasing after alpha)
    # ---------------------------
    if COL_M in rand.columns and COL_EPS_RECLAIMED in rand.columns:
        plt.figure()
        plt.scatter(rand[COL_M], rand[COL_EPS_RECLAIMED], s=18)
        # visualize the "zero guarantee" threshold m <= alpha
        if COL_ALPHA in rand.columns:
            alpha_val = float(np.nanmedian(rand[COL_ALPHA].values))
            plt.axvline(alpha_val, linestyle=":", label=r"$m=\alpha$ threshold (median)")
            plt.legend()
        plt.xlabel(r"$m \;=\;\mathrm{LB}\;\inf_{x\in X_q} V(x)$")
        plt.ylabel(r"$\varepsilon_{\mathrm{reclaimed}}$")
        plt.title("Random disrupted regions: reclaimed bound depends on m")
        savefig(args.out_dir, "random_eps_vs_m")
        plt.close()

    # ---------------------------
    # Plot 6 (optional): Empirical MC success vs reclaimed bound (soundness check)
    # Only runs if an MC column exists in the random CSV.
    # ---------------------------
    mc_col = first_existing_col(rand, MC_CANDIDATES)
    if mc_col is not None and COL_EPS_RECLAIMED in rand.columns:
        plt.figure()
        plt.scatter(rand[COL_EPS_RECLAIMED], rand[mc_col], s=18)
        lo = np.nanmin([rand[COL_EPS_RECLAIMED].min(), rand[mc_col].min()])
        hi = np.nanmax([rand[COL_EPS_RECLAIMED].max(), rand[mc_col].max()])
        plt.plot([lo, hi], [lo, hi])  # y=x reference
        plt.xlabel(r"$\varepsilon_{\mathrm{reclaimed}}$ (certified lower bound)")
        plt.ylabel(f"{mc_col} (empirical success)")
        plt.title("Empirical validation: success should lie above the bound")
        savefig(args.out_dir, "mc_success_vs_reclaimed_bound")
        plt.close()

    print(f"Saved figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
