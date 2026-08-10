"""
Reliability probes for a learned one-step map.

A plant is only trustworthy if its map is *smooth* and *non-amplifying*: small changes
to the input window should produce proportionally small, bounded changes to the output.
That single property underwrites three things at once —

  * rollout stability: a large local sensitivity (Jacobian norm) amplifies fed-back
    errors step to step — the expansive-map failure the error-vs-horizon curve measures;
  * quantisation robustness: a jagged, high-sensitivity map is fragile under the
    fixed-point rounding an FPGA deployment imposes — an early warning long before M5;
  * training health: a sane sensitivity is a quick check that a small model learned a
    real function rather than a brittle interpolation.

The probes here are deliberately lightweight and model-agnostic (they take any callable
and a batch of inputs). Heavier discontinuity / functional analysis can build on them.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(os.path.basename(__file__))

Model = Callable[[torch.Tensor], torch.Tensor]


def _summary(values: torch.Tensor) -> dict[str, float]:
    """
    Distribution summary (mean / p50 / p99 / max) of a 1-D tensor of per-sample values.
    """

    v = values.detach().float().cpu().numpy()
    return {
        "mean": float(v.mean()),
        "p50": float(np.percentile(v, 50)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
    }


def jacobian_norms(model: Model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Per-sample norms of the exact input-output Jacobian ``J = d model(x) / d x``.

    ``x`` is ``[N, ...]`` and ``model(x)`` is ``[N, A]``; samples are independent, one
    backward per output channel yields every sample's Jacobian row at once. Returns the
    spectral norm (largest singular value — the worst-case local gain) and the Frobenius
    norm, each ``[N]``.
    """

    x = x.detach().clone().requires_grad_(True)
    out = model(x)
    if out.dim() != 2:
        raise ValueError(f"Expected model output [N, A], got {tuple(out.shape)}")
    n, a = out.shape
    rows = []
    for j in range(a):
        (grad,) = torch.autograd.grad(out[:, j].sum(), x, retain_graph=j < a - 1)
        rows.append(grad.reshape(n, -1))
    jac = torch.stack(rows, dim=1)  # [N, A, D]
    spectral = torch.linalg.svdvals(jac).amax(dim=1)  # [N]
    frobenius = jac.reshape(n, -1).norm(dim=1)  # [N]
    return {"spectral": spectral.detach(), "frobenius": frobenius.detach()}


@torch.no_grad()
def local_lipschitz(
    model: Model,
    x: torch.Tensor,
    n_directions: int = 8,
    eps: float = 1e-3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Finite-difference directional sensitivity ``||f(x+eps*u) - f(x)|| / eps``.

    A model-agnostic smoothness cross-check: random unit perturbations ``u`` probe the
    local gain the Jacobian describes, without differentiating (so it also catches
    non-smoothness the analytic Jacobian would miss). Returns all ``n_directions * N``
    sensitivities flattened; a heavy P99/max tail flags fragile input directions.
    """

    base = model(x)
    n = x.shape[0]
    view = (-1,) + (1,) * (x.dim() - 1)
    sens = []
    for _ in range(n_directions):
        d = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        d = d / d.reshape(n, -1).norm(dim=1).clamp_min(1e-12).reshape(view)
        delta = model(x + eps * d) - base
        sens.append(delta.reshape(n, -1).norm(dim=1) / eps)
    return torch.stack(sens).reshape(-1)


@torch.no_grad()
def quant_sensitivity(
    model: torch.nn.Module, x: torch.Tensor, bits: int
) -> torch.Tensor:
    """
    Relative output change when weights are rounded to ``bits``-bit fixed point.

    Symmetric per-tensor quantisation of every parameter, restored afterwards. A model
    that is stable at bf16 but swings under 8-bit rounding is quantisation-fragile — the
    FPGA-deployment early warning. Returns per-sample ``||f_q(x) - f(x)|| / ||f(x)||``.
    """

    base = model(x)
    qmax = 2 ** (bits - 1) - 1
    saved = {name: p.detach().clone() for name, p in model.named_parameters()}
    try:
        for p in model.parameters():
            scale = p.detach().abs().max().clamp_min(1e-12) / qmax
            p.copy_((p / scale).round().clamp(-qmax - 1, qmax) * scale)
        q_out = model(x)
    finally:
        for name, p in model.named_parameters():
            p.copy_(saved[name])
    n = x.shape[0]
    denom = base.reshape(n, -1).norm(dim=1).clamp_min(1e-12)
    return (q_out - base).reshape(n, -1).norm(dim=1) / denom


@dataclass
class ReliabilityReport:
    """
    Summary of the reliability probes on a sample of inputs.
    """

    n_samples: int
    jacobian_spectral: dict[str, float]
    jacobian_frobenius: dict[str, float]
    lipschitz: dict[str, float]
    quant: dict[int, dict[str, float]]  # bit-width -> relative-change summary

    def log(self) -> None:
        logger.info("Reliability probes over %d input windows:", self.n_samples)
        logger.info(
            "  Jacobian spectral norm  mean %.3f  p99 %.3f  max %.3f",
            self.jacobian_spectral["mean"],
            self.jacobian_spectral["p99"],
            self.jacobian_spectral["max"],
        )
        logger.info(
            "  Local Lipschitz (f-d)   mean %.3f  p99 %.3f  max %.3f",
            self.lipschitz["mean"],
            self.lipschitz["p99"],
            self.lipschitz["max"],
        )
        for bits in sorted(self.quant):
            q = self.quant[bits]
            logger.info(
                "  Weight quant %2d-bit      rel-change mean %.4f  p99 %.4f",
                bits,
                q["mean"],
                q["p99"],
            )


def probe_reliability(
    model: torch.nn.Module,
    x: torch.Tensor,
    quant_bits: tuple[int, ...] = (8, 6, 4),
    n_directions: int = 8,
    eps: float = 1e-3,
    seed: int = 42,
) -> ReliabilityReport:
    """
    Run all probes on ``x`` (a batch of input windows) and summarise.
    """

    model.eval()
    gen = torch.Generator(device=x.device).manual_seed(seed)
    jac = jacobian_norms(model, x)
    lip = local_lipschitz(model, x, n_directions, eps, gen)
    quant = {b: _summary(quant_sensitivity(model, x, b)) for b in quant_bits}
    return ReliabilityReport(
        n_samples=x.shape[0],
        jacobian_spectral=_summary(jac["spectral"]),
        jacobian_frobenius=_summary(jac["frobenius"]),
        lipschitz=_summary(lip),
        quant=quant,
    )


def run_reliability(
    config_path: str,
    ckpt_path: str,
    device: str = "cpu",
    batch_size: int = 2048,
    quant_bits: tuple[int, ...] = (8, 6, 4),
    seed: int = 42,
) -> ReliabilityReport:
    """
    Build a plant from a checkpoint and run the reliability probes on held-out windows.
    """

    from nn_motion_control.core.config import RunConfiguration
    from nn_motion_control.data import build_rollout_splits
    from nn_motion_control.plant.plant import Plant, RolloutLayout

    config = RunConfiguration(config_path)
    layout = RolloutLayout.from_config(config)
    data = build_rollout_splits(
        h5_path=config.datafile_dir,
        allowed_inputs=config.input_params,
        allowed_targets=config.target_params,
        window_size=config.window_size,
        max_horizon=1,
        pos_cols=layout.pos_cols,
        dac_cols=layout.dac_cols,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        training_dtype=torch.float32,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    plant = Plant.from_checkpoint(config, ckpt_path, device)
    batch = next(iter(data.tst_loader))
    warmup = batch[0].float()  # [N, F, W] input windows
    report = probe_reliability(plant.model, warmup, quant_bits=quant_bits, seed=seed)
    report.log()
    return report
