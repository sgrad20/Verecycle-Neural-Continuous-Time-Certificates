import numpy as np
import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm, BoundedModule

# --------------------------------------------------------------------------------
# Optional JAX acceleration (Windows jaxlib can fail to load DLLs)
# --------------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None


# --------------------------------------------------------------------------------
# Grid utilities (JAX if available, NumPy fallback otherwise)
# --------------------------------------------------------------------------------
if jax is not None:
    @jax.jit
    def meshgrid_jax(points, size):
        """
        Set rectangular grid over state space (JAX version).
        points: list of 1D arrays
        size: list/array of ints
        """
        meshgrid = jnp.asarray(jnp.meshgrid(*points))
        grid = jnp.reshape(meshgrid, (len(size), -1)).T
        return grid
else:
    def meshgrid_jax(points, size):
        """
        Set rectangular grid over state space (NumPy fallback).
        points: list of 1D arrays
        size: list/array of ints
        """
        mesh = np.meshgrid(*points, indexing="xy")
        grid = np.stack([m.reshape(-1) for m in mesh], axis=1)
        return grid


def define_grid_jax(low, high, size, mode="linspace"):
    """
    Define a grid (JAX if available, otherwise NumPy).
    Returns a NumPy array.
    """
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    size = np.asarray(size, dtype=int)

    if mode == "linspace":
        points = [np.linspace(low[i], high[i], int(size[i])) for i in range(len(size))]
    else:
        step = (high - low) / (size - 1)
        points = [np.arange(low[i], high[i] + step[i] / 2, step[i]) for i in range(len(size))]

    grid = meshgrid_jax(points, size)

    try:
        grid = np.asarray(grid, dtype=np.float32)
    except Exception:
        pass

    return grid


def mesh2cell_width(mesh, dim, Linfty):
    """Convert mesh size in L1 norm to cell width in a rectangular gridding."""
    return mesh * 2 if Linfty else mesh * (2 / dim)


def cell_width2mesh(cell_width, dim, Linfty):
    """Convert cell width in a rectangular gridding to mesh size in L1 norm."""
    return cell_width / 2 if Linfty else cell_width * (dim / 2)


# --------------------------------------------------------------------------------
# IBP utilities (auto_LiRPA)
# --------------------------------------------------------------------------------
def batched_forward_pass_ibp(
    verifier: BoundedModule,
    centers: np.ndarray,
    epsilon: np.ndarray,
    batch_size: int = 1000,
):
    """
    IBP lower/upper bounds over L_inf boxes centered at `centers`
    with radius `epsilon`.
    Returns (lb, ub) as numpy arrays of shape (N,).
    """
    centers = np.asarray(centers, dtype=np.float32)
    epsilon = np.asarray(epsilon, dtype=np.float32).reshape(-1, 1)

    lbs, ubs = [], []
    n = centers.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        x0 = torch.tensor(centers[start:end], dtype=torch.float32)
        e0 = torch.tensor(epsilon[start:end], dtype=torch.float32)

        bounded_x = BoundedTensor(
            x0,
            PerturbationLpNorm(x_L=x0 - e0, x_U=x0 + e0),
        )

        lb, ub = verifier.compute_bounds(bounded_x, method="IBP")
        lbs.append(lb.squeeze(-1).detach().cpu().numpy())
        ubs.append(ub.squeeze(-1).detach().cpu().numpy())

    return np.concatenate(lbs, axis=0), np.concatenate(ubs, axis=0)


# --------------------------------------------------------------------------------
# Cell splitting verifier
# --------------------------------------------------------------------------------
class CellVerificationSystem:
    def __init__(self, max_depth=10):
        super().__init__()
        self.max_depth = max_depth
        self.corners = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    def verify(
        self,
        verifier: BoundedModule,
        locations: torch.Tensor,
        magnitude: torch.Tensor,
        depth: int = 0,
    ):
        bounded_cells = BoundedTensor(
            locations,
            PerturbationLpNorm(
                x_L=locations - magnitude,
                x_U=locations + magnitude,
            )
        )
        _, ub = verifier.compute_bounds(
            bounded_cells,
            bound_lower=False,
            method="IBP",
        )
        mask = ub.squeeze() >= 0.0
        counterexamples = locations[mask]

        if torch.numel(counterexamples) > 0:
            print(
                f"Could not verify decrease at {counterexamples.shape[0]} cells. "
                "Splitting further"
            )
            if depth < self.max_depth:
                new_cells = torch.empty((0, locations.shape[1]))
                half_m = 0.5 * magnitude
                for _, loc in enumerate(counterexamples):
                    new_cells = torch.cat(
                        (new_cells, loc + half_m * self.corners),
                        dim=0,
                    )
                counterexamples = self.verify(
                    verifier, new_cells, half_m, depth + 1
                )
        return counterexamples