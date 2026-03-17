import numpy as np
from stochastic_rsa.cells import define_grid_jax, batched_forward_pass_ibp


def run_continuous_VeRecycle(
    verifier,
    alpha_RA: float,
    beta_RA: float,
    disrupted_region,
    mesh_size: float = 0.01,
    batch_size: int = 1000,
):
    """
    Continuous-Time VeRecycle.

    Reclaims a reach-avoid probability guarantee after a local dynamics change
    supported on the disrupted region X_q.

    The implementation computes a sound lower bound m_lb on
        inf_{x in X_q} V(x)
    using IBP over a mesh of cells, and then returns the reclaimed threshold

        eps_rec = min(1 - alpha / beta, max(0, 1 - alpha / m_lb)).

    Returns
    -------
    eps_reclaimed : float
        Reclaimed certified reach-avoid probability lower bound.
    m_lb : float
        Sound lower bound on inf_{x in X_q} V(x).
    """
    if alpha_RA < 0:
        raise ValueError("alpha_RA must be non-negative")
    if beta_RA <= 0:
        raise ValueError("beta_RA must be > 0")
    if mesh_size <= 0:
        raise ValueError("mesh_size must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not hasattr(disrupted_region, "sets") or len(disrupted_region.sets) == 0:
        raise ValueError("disrupted_region must have a non-empty .sets list")

    # ------------------------------------------------------------
    # Step 1 — Create mesh inside disrupted region X_q
    # ------------------------------------------------------------
    cell_widths = [float(mesh_size) for _ in disrupted_region.sets]

    num_per_dimensions = [
        np.maximum(
            np.array(np.ceil((region.high - region.low) / width), dtype=int),
            1,
        )
        for region, width in zip(disrupted_region.sets, cell_widths)
    ]

    centers_all = []
    eps_all = []

    for region, width, num_dim in zip(disrupted_region.sets, cell_widths, num_per_dimensions):
        centers = define_grid_jax(
            region.low + 0.5 * width,
            region.high - 0.5 * width,
            size=num_dim,
        )

        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim == 1:
            centers = centers.reshape(1, -1)

        centers_all.append(centers)
        eps_all.append(
            np.full((centers.shape[0],), 0.5 * width, dtype=np.float32)
        )

    centers_all = np.vstack(centers_all).astype(np.float32)
    eps_all = np.concatenate(eps_all, axis=0).astype(np.float32)

    # ------------------------------------------------------------
    # Step 2 — IBP lower bound on V(x) for x ∈ X_q
    # ------------------------------------------------------------
    lb, _ = batched_forward_pass_ibp(
        verifier=verifier,
        centers=centers_all,
        epsilon=eps_all,
        batch_size=batch_size,
    )

    m_lb = float(np.min(lb))
    print(f"[VeRecycle debug] m_lb = {m_lb:.6f}, alpha_RA = {alpha_RA:.6f}")

    # ------------------------------------------------------------
    # Step 3 — Degenerate case
    # ------------------------------------------------------------
    if m_lb <= alpha_RA:
        return 0.0, m_lb

    # ------------------------------------------------------------
    # Step 4 — Compute reclaimed probability
    # ------------------------------------------------------------
    eps_orig = 1.0 - (alpha_RA / beta_RA)
    eps_from_Xq = 1.0 - (alpha_RA / m_lb)

    eps_reclaimed = min(eps_orig, eps_from_Xq)
    eps_reclaimed = max(min(eps_reclaimed, 1.0), 0.0)

    return float(eps_reclaimed), float(m_lb)
# from pathlib import Path
# import argparse
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# HERE = Path(__file__).resolve().parent
# FIG_DIR = HERE / "figures"

# # .\.venv\Scripts\activate.bat then  python auto_LiRPA/experiments/run_ct_verecycle_pendulum.py                                                                                    
# def save_png(name: str):
#     FIG_DIR.mkdir(exist_ok=True)
#     path = FIG_DIR / f"{name}.png"
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"Saved: {path}")


# def plot_mesh_sweep(csv_path: Path):
#     df = pd.read_csv(csv_path)

#     mesh = df["mesh_size"]
#     eps = df["eps_reclaimed"]
#     time = df["verecycle_time_s"]

#     # eps vs mesh
#     plt.figure()
#     plt.plot(mesh, eps, marker="o")
#     plt.xlabel("mesh size")
#     plt.ylabel(r"$\varepsilon_{\mathrm{rec}}$")
#     plt.grid(True)
#     save_png(f"ct_verecycle_mesh_eps__{csv_path.stem}")

#     # runtime vs mesh
#     plt.figure()
#     plt.plot(mesh, time, marker="o")
#     plt.xlabel("mesh size")
#     plt.ylabel("runtime (s)")
#     plt.grid(True)
#     save_png(f"ct_verecycle_mesh_runtime__{csv_path.stem}")


# def plot_random_xq(csv_path: Path):
#     df = pd.read_csv(csv_path)

#     m = df["m"].to_numpy()
#     eps = df["eps_reclaimed"].to_numpy()

#     alpha = float(df["alpha_RA"].iloc[0])
#     beta = float(df["beta_RA"].iloc[0])

#     # theory curve
#     m_line = np.linspace(max(1e-6, m.min()), m.max(), 400)
#     eps_line = np.maximum(1.0 - alpha / np.minimum(beta, m_line), 0.0)

#     plt.figure()
#     plt.scatter(m, eps, label="measured (VeRecycle-CT)")
#     plt.plot(m_line, eps_line, label="theory: max(1 - α / min(β, m), 0)")
#     plt.xlabel(r"$m = \inf_{x \in X_q} V(x)$ (IBP lower bound)")
#     plt.ylabel(r"$\varepsilon_{\mathrm{rec}}$")
#     plt.legend()
#     plt.grid(True)

#     save_png(f"ct_verecycle_randomXq_epsrec_vs_m__{csv_path.stem}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mesh_csv", type=Path, required=True)
#     parser.add_argument("--random_csv", type=Path, required=True)
#     args = parser.parse_args()

#     plot_mesh_sweep(args.mesh_csv)
#     plot_random_xq(args.random_csv)


# if __name__ == "__main__":
#     main()