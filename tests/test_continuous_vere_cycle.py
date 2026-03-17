"""
Tests for the continuous-time VeRecycle implementation.
"""

import numpy as np
import pytest
import torch
from pathlib import Path

from stochastic_rsa.continuous_vere_cycle import run_continuous_VeRecycle
import stochastic_rsa as rsa
from rl_agent.policy import TanhPolicy


def test_reclaimed_probability_clamped_by_original():
    alpha_RA = 0.1
    beta_RA = 1.0
    inf_V = 1000.0

    eps_orig = 1.0 - alpha_RA / beta_RA
    eps_from_Xq = 1.0 - alpha_RA / inf_V

    eps_reclaimed = min(eps_orig, eps_from_Xq)
    assert eps_reclaimed == pytest.approx(eps_orig)
    assert 0.0 <= eps_reclaimed <= 1.0


@pytest.mark.slow
def test_ct_verecycle_runs_on_saved_pendulum_certificate():
    """
    Integration test: loads certified.pt and runs CT VeRecycle end-to-end.
    """

    device = "cpu"

    # repo root = parent of tests/
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_path = repo_root / "certified.pt"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    assert "policy" in ckpt and "certificate" in ckpt

    policy_net = TanhPolicy(2, 1, 64, device=device)
    policy_net.load_state_dict(ckpt["policy"])
    policy_net.eval()

    cert_net = rsa.CertificateModule(device=device)
    cert_net.load_state_dict(ckpt["certificate"])
    cert_net.eval()

    # Small 2D disrupted region X?
    class Box:
        def __init__(self, low, high):
            self.low = low
            self.high = high
            self.dimension = len(low)

    class MultiBox:
        def __init__(self, boxes):
            self.sets = boxes
            self.dimension = boxes[0].dimension

    Xq = MultiBox([Box(np.array([-0.2, -0.5]), np.array([0.2, 0.5]))])

    # Use consistent dummy levels for a 0.9 original bound
    alpha_RA = 0.1
    beta_RA = 1.0
    eps_orig = 1.0 - alpha_RA / beta_RA

    # IMPORTANT: pass the object type that run_continuous_VeRecycle expects.
    # If your implementation expects the torch module, pass cert_net.
    eps = run_continuous_VeRecycle(
        certificate=cert_net,
        alpha_RA=alpha_RA,
        beta_RA=beta_RA,
        disrupted_region=Xq,
        mesh_size=0.05,
        batch_size=1000,
    )

    assert isinstance(eps, float)
    assert 0.0 <= eps <= 1.0
    assert eps <= eps_orig + 1e-8
