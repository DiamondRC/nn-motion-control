from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_KINDS = ("measured", "derived", "command")


def _as_per_axis(
    value: Any, axes: list[str], *, channel: str, field: str
) -> dict[str, Any]:
    """
    Normalise a broadcast-or-per-axis value to {axis: value}.
    """

    if isinstance(value, dict):
        if set(value) != set(axes):
            raise ValueError(
                f"Channel '{channel}' field '{field}': per-axis keys "
                f"{sorted(value)} must match system axes {axes}"
            )
        return {axis: value[axis] for axis in axes}

    return dict.fromkeys(axes, value)


@dataclass(frozen=True)
class ChannelSpec:
    """
    One signal of the system (a measured state, a derived state, or a command).
    """

    name: str
    kind: str
    per_axis: bool = True
    unit: str | None = None
    label_template: str = "{axis}_{name}"

    # measured
    limits: dict[str, list[float]] | None = None
    resolution: dict[str, float] | None = None
    noise_rms: dict[str, float] | None = None

    # derived
    source: str | None = None
    order: int | None = None

    # command
    range: dict[str, list[float]] | None = None
    safe_range: dict[str, list[float]] | None = None

    def label(self, axis: str | None = None) -> str:
        if not self.per_axis:
            return self.name
        if axis is None:
            raise ValueError(f"channel '{self.name}' is per-axis; an axis is required")
        return self.label_template.format(axis=axis, name=self.name)


@dataclass(frozen=True)
class SystemSpec:
    """
    The full description of a motion system.
    """

    name: str
    axes: list[str]
    servo_rate_hz: float | None
    data_rate_hz: float | None
    board: str | None
    clock_hz: float | None
    channels: dict[str, ChannelSpec]

    @classmethod
    def from_toml(cls, path: str | Path) -> SystemSpec:
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SystemSpec:
        axes = list(raw["axes"])
        if not axes:
            raise ValueError("System must define at least one axis")
        if len(set(axes)) != len(axes):
            raise ValueError(f"Axes must be unique, got {axes}")

        channels = {
            name: _build_channel(name, spec, axes)
            for name, spec in raw.get("channels", {}).items()
        }
        for ch in channels.values():
            if ch.kind == "derived" and ch.source not in channels:
                raise ValueError(
                    f"Derived channel '{ch.name}' references unknown source "
                    f"'{ch.source}'"
                )

        target = raw.get("target", {})
        return cls(
            name=raw["name"],
            axes=axes,
            servo_rate_hz=raw.get("servo_rate_hz") or None,
            data_rate_hz=raw.get("data_rate_hz") or None,
            board=target.get("board"),
            clock_hz=target.get("clock_hz"),
            channels=channels,
        )

    # Accessors
    def channel(self, name: str) -> ChannelSpec:
        try:
            return self.channels[name]
        except KeyError:
            raise KeyError(
                f"Unknown channel '{name}'. Have {list(self.channels)}"
            ) from None

    def labels(self, channel_names: list[str]) -> list[str]:
        """
        Expand channel names into concrete dataset labels (axis-major).

        Axis-major means each axis's selected channels are contiguous.
        """
        out: list[str] = []
        for axis in self.axes:
            for name in channel_names:
                ch = self.channel(name)
                out.append(ch.label(axis if ch.per_axis else None))
        return out

    def clocks_per_step(self) -> float | None:
        """
        FPGA fabric clocks available per control step, if both rates are known.
        """

        if self.clock_hz and self.servo_rate_hz:
            return self.clock_hz / self.servo_rate_hz
        return None

    def control_substeps(self) -> float | None:
        """
        Control steps per data step, if both rates are known.

        The plant is identified at ``data_rate_hz`` (the native rate of the logs),
        while the controller runs at ``servo_rate_hz``. The ratio is how many
        control decisions occur within a single plant-observable transition — the
        deployment-time reconciliation of the two rates is an M2 concern.
        """

        if self.servo_rate_hz and self.data_rate_hz:
            return self.servo_rate_hz / self.data_rate_hz
        return None


def _build_channel(name: str, spec: dict[str, Any], axes: list[str]) -> ChannelSpec:
    kind = spec.get("kind")
    if kind not in _KINDS:
        raise ValueError(
            f"Channel '{name}': kind must be one of {_KINDS}, got {kind!r}"
        )

    def per_axis_field(field: str) -> dict[str, Any] | None:
        value = spec.get(field)
        if value is None:
            return None
        return _as_per_axis(value, axes, channel=name, field=field)

    return ChannelSpec(
        name=name,
        kind=kind,
        per_axis=spec.get("per_axis", True),
        unit=spec.get("unit"),
        label_template=spec.get("label", "{axis}_{name}"),
        limits=per_axis_field("limits"),
        resolution=per_axis_field("resolution"),
        noise_rms=per_axis_field("noise_rms"),
        source=spec.get("from"),
        order=spec.get("order"),
        range=per_axis_field("range"),
        safe_range=per_axis_field("safe_range"),
    )
