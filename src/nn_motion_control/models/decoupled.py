"""
Decoupled-dynamics plant: a recurrent stack on the bounded channels plus a
static path on position.

Physics motivation: the stage's response to a command is near
translation-invariant (how it accelerates does not depend on where it
absolutely is), so the memory/dynamic part should be driven only by bounded,
stationary signals (velocity, command). The one genuine position dependence
is the flexure restoring force, a static memoryless term. Feeding absolute
position into a recurrent SSM instead inflates its state on long scans (a
sustained large input drives the state to a large DC value via the
high-gain long-memory poles), which is what makes the plain recurrent SSM
diverge in free-run.

So this model routes velocity+command into a DiagSSM stack (the dynamics)
and current position into a small static path (the flexure spring), merging
before the head. Position stays a declared model input, so the rollout
reconstruction, layout and dataset are unchanged; only the model splits it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.models.builder import ModelComponents


def _build_sequential(
    specs: list, components: ModelComponents
) -> nn.Sequential:
    """Build a plain Sequential from a JSON layer spec list."""

    layers: list[nn.Module] = []

    for spec in specs:
        if isinstance(spec, str):
            name, args = spec, []
        elif isinstance(spec, dict):
            name, raw = next(iter(spec.items()))
            args = list(raw) if raw else []
        else:
            raise TypeError(f"Unsupported layer spec {spec!r}")
        layers.append(components.get(name)(*args))

    return nn.Sequential(*layers)


class DecoupledSSMPlant(nn.Module):
    """
    Route velocity+command through a recurrent stack and position through a
    static path.

    Input is [B, F, W] with F the axis-major (position, velocity, command)
    columns; output is [B, A] (the per-axis target). The recurrent stack and
    static path are read from the config's 'decoupled' sub-spec.
    """

    def __init__(self, config: RunConfiguration):
        super().__init__()
        if config.decoupled is None:
            raise ValueError("DecoupledSSMPlant needs a 'decoupled' arch spec")
        comp = ModelComponents()
        in_names = config.input_channels
        n_ch = len(in_names)
        axes = len(config.system.axes)
        target = set(config.target_channels)
        static_names = [c for c in in_names if c in target]
        dyn_names = [c for c in in_names if c not in target]

        def cols(names: list[str]) -> list[int]:
            idxs = [in_names.index(c) for c in names]

            return sorted(a * n_ch + j for a in range(axes) for j in idxs)

        self.dyn_cols = cols(dyn_names)  # velocity + command (bounded)
        self.static_cols = cols(static_names)  # position (flexure spring)

        spec = config.decoupled
        self.dynamic = _build_sequential(spec["dynamic"], comp)
        static_spec = spec.get("static") or []
        self.static = (
            _build_sequential(static_spec, comp) if static_spec else None
        )
        self.head = _build_sequential(spec["head"], comp)
        # A Sequential named 'network' so the shared training-pipeline hooks
        # (channels-last probe) can inspect it; the SSM stack is not
        # channels-last-able, so the probe simply skips it.
        self.network = self.dynamic
        self.use_channels_last = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def enable_channels_last(self) -> None:
        raise ValueError("Decoupled SSM plant is not channels-last-able")

    def _readout(
        self, dyn_feat: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        """Merge the last dynamic feature with the static position term."""

        if self.static is not None:
            return self.head(torch.cat([dyn_feat, self.static(pos)], dim=-1))

        return self.head(dyn_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, F, W], got {tuple(x.shape)}")
        u = x[:, self.dyn_cols, :].transpose(1, 2)  # [B, W, d_dyn]
        u = self.dynamic(u)
        pos = x[:, self.static_cols, -1]  # [B, A] current position

        return self._readout(u[:, -1, :], pos)
