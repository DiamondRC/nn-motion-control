"""Reliability probes.

Recover a linear map's operator norm, quantisation trend.
"""

import torch
import torch.nn as nn

from nn_motion_control.eval.reliability import (
    jacobian_norms,
    local_lipschitz,
    probe_reliability,
    quant_sensitivity,
)


class _Linear(nn.Module):
    """f(x) = x @ W — a map whose Jacobian is exactly W^T everywhere."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1) @ self.weight


def test_jacobian_recovers_linear_operator_norm():
    torch.manual_seed(0)
    d, a, n = 12, 3, 16
    w = torch.randn(d, a)
    m = _Linear(w)
    j = jacobian_norms(m, torch.randn(n, d))
    op_norm = torch.linalg.svdvals(w).amax()
    assert torch.allclose(j["spectral"], op_norm.expand(n), atol=1e-4)
    assert torch.allclose(j["frobenius"], w.norm().expand(n), atol=1e-4)


def test_local_lipschitz_bounded_by_spectral_norm():
    torch.manual_seed(1)
    w = torch.randn(12, 3)
    m = _Linear(w)
    op_norm = float(torch.linalg.svdvals(w).amax())
    sens = local_lipschitz(m, torch.randn(16, 12), n_directions=16, eps=1e-3)
    # random unit directions can only realise up to the largest singular value.
    assert (sens <= op_norm + 1e-3).all()
    assert (
        float(sens.max()) > 0.3 * op_norm
    )  # and they do probe a real fraction of it


def test_quant_sensitivity_grows_as_bits_drop_and_restores_weights():
    torch.manual_seed(2)
    m = _Linear(torch.randn(12, 3))
    x = torch.randn(8, 12)
    before = m.weight.detach().clone()
    hi = float(quant_sensitivity(m, x, 16).mean())
    lo = float(quant_sensitivity(m, x, 3).mean())
    assert hi < lo  # coarser quantisation perturbs the output more
    assert hi < 0.05  # 16-bit barely changes anything
    assert torch.equal(m.weight.detach(), before)  # weights restored


def test_probe_reliability_report_shape():
    m = _Linear(torch.randn(12, 3))
    r = probe_reliability(m, torch.randn(16, 12), quant_bits=(8, 4))
    assert r.n_samples == 16
    assert r.jacobian_spectral["max"] >= r.jacobian_spectral["p50"]
    assert set(r.quant) == {8, 4}
