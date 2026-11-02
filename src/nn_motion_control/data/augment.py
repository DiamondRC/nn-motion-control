"""
Linearity augmentation for plant rollout training.

A precision stage is linear about an equilibrium at the count level (its
count-level response matches its large-signal FRF), yet the data barely
shows that regime: count-level motion is a sliver of it and is buried in
ambient vibration near the resonance. Scaling a recorded window about an
equilibrium gives a valid small-signal example of the same linear stage,
with the ambient part shrunk alike, so the model sees the regime it must
get right.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch

from nn_motion_control.data.device_loader import RolloutWindowLoader


class LinearityAugment:
    """
    Wrap a rollout loader, replacing 'fraction' of each batch's rows with
    copies scaled by alpha (log-uniform over 'scale') about an equilibrium.

    For a row with command u and position x, the equilibrium is the
    window's mean command u_eq and its static position x_eq = gain * u_eq
    (the rest frame puts zero command at zero position). The copy carries
    u_eq + alpha * (u - u_eq), x_eq + alpha * (x - x_eq) and alpha times the
    velocity, which a linear stage would follow exactly. Batches are the
    loader's own, normalised; the scaling is done in physical units.
    """

    def __init__(
        self,
        loader: RolloutWindowLoader,
        pos_cols: Sequence[int],
        vel_cols: Sequence[int],
        dac_cols: Sequence[int],
        in_mean: torch.Tensor,
        in_std: torch.Tensor,
        gain: torch.Tensor,
        spec: Mapping[str, Any],
        seed: int = 0,
    ):
        self.loader = loader
        self.dataset = loader.dataset
        self.fraction = float(spec["fraction"])
        lo, hi = (float(v) for v in spec["scale"])
        if not 0.0 < self.fraction <= 1.0 or not 0.0 < lo <= hi <= 1.0:
            raise ValueError(
                "Linearity_augment needs 0 < fraction <= 1 and "
                "0 < scale lo <= hi <= 1"
            )
        self.log_scale = (math.log(lo), math.log(hi))
        # With 'weighted', a scaled row's loss is weighted 1 / alpha^2 so
        # it counts as much as the window it came from: its errors shrink
        # with alpha, and unweighted it would teach almost nothing.
        self.weighted = bool(spec.get("weighted", False))
        dev = loader.device
        self.pos = torch.as_tensor(list(pos_cols), device=dev)
        self.vel = torch.as_tensor(list(vel_cols), device=dev)
        self.dac = torch.as_tensor(list(dac_cols), device=dev)
        self.mean = in_mean.to(dev).float()
        self.std = in_std.to(dev).float()
        self.gain = gain.to(dev).float()
        self.generator = torch.Generator(device=dev).manual_seed(seed)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        for warmup, dac_future, gt_pos in self.loader:
            batch = self._augment(warmup, dac_future, gt_pos)
            yield batch if self.weighted else batch[:3]

    def _augment(self, warmup, dac_future, gt_pos):
        b = warmup.shape[0]
        dev = warmup.device
        pick = torch.rand(b, generator=self.generator, device=dev)
        rows = (pick < self.fraction).nonzero().flatten()
        row_weights = torch.ones(b, device=dev)
        if rows.numel() == 0:
            return warmup, dac_future, gt_pos, row_weights
        lo, hi = self.log_scale
        alpha = torch.exp(
            lo
            + (hi - lo)
            * torch.rand(rows.numel(), generator=self.generator, device=dev)
        )
        warmup, dac_future, gt_pos = (
            warmup.clone(),
            dac_future.clone(),
            gt_pos.clone(),
        )
        m, s = self.mean, self.std
        dac_m, dac_s = m[self.dac], s[self.dac]
        pos_m, pos_s = m[self.pos], s[self.pos]
        a = alpha[:, None, None]

        # Physical warmup command and the row's equilibrium, [R, A].
        u = warmup[rows][:, self.dac, :].float() * dac_s[None, :, None]
        u = u + dac_m[None, :, None]
        u_eq = u.mean(dim=2)
        x_eq = self.gain[None, :] * u_eq

        def scaled(values, mean, std, centre):
            # values [R, A, T] normalised -> scaled about centre, normalised
            phys = values.float() * std[None, :, None] + mean[None, :, None]
            out = centre[:, :, None] + a * (phys - centre[:, :, None])

            return ((out - mean[None, :, None]) / std[None, :, None]).to(
                values.dtype
            )

        w = warmup[rows]
        w[:, self.dac, :] = scaled(w[:, self.dac, :], dac_m, dac_s, u_eq)
        w[:, self.pos, :] = scaled(w[:, self.pos, :], pos_m, pos_s, x_eq)
        if self.vel.numel():
            vel_m, vel_s = m[self.vel], s[self.vel]
            zero = torch.zeros_like(u_eq)
            w[:, self.vel, :] = scaled(w[:, self.vel, :], vel_m, vel_s, zero)
        warmup[rows] = w
        # The future tensors are [R, H, A]: scale along the same axes.
        dac_future[rows] = scaled(
            dac_future[rows].transpose(1, 2), dac_m, dac_s, u_eq
        ).transpose(1, 2)
        gt_pos[rows] = scaled(
            gt_pos[rows].transpose(1, 2), pos_m, pos_s, x_eq
        ).transpose(1, 2)

        row_weights[rows] = 1.0 / alpha**2

        return warmup, dac_future, gt_pos, row_weights
