"""
Command-line interface for 'python -m nn_motion_control'.
"""

from datetime import datetime

import typer
from typer.core import TyperGroup

from . import __version__
from .training.run import CompleteRun

__all__ = ["main"]


class NaturalOrderGroup(TyperGroup):
    """
    Typer group that lists commands in declaration order rather than
    alphabetically.
    """

    def list_commands(self, ctx):
        return list(self.commands)


cli = typer.Typer(cls=NaturalOrderGroup, pretty_exceptions_enable=False)


def version_callback(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def _resolve_cfg(path: str | None, purpose: str) -> str:
    """
    Return the given config path, or prompt for one when omitted on an
    interactive TTY.

    Passing the argument keeps every command scriptable; omitting it on a
    terminal opens the picker, while omitting it with a piped stdin is an
    error (deterministic in CI).
    """

    if path:
        return path
    from .cli.select import is_interactive, pick_config

    if not is_interactive():
        raise typer.BadParameter(
            "No config path given and stdin is not a terminal; pass one "
            "explicitly."
        )

    return pick_config(purpose)


def _interactive_setup(
    command: str, purpose: str, options: list[tuple]
) -> dict:
    """
    Interactive config + options flow (with a reuse-last-settings shortcut)
    for a bare command on a terminal; an error off a TTY so scripted use
    stays deterministic.
    """

    from .cli.select import interactive_setup, is_interactive

    if not is_interactive():
        raise typer.BadParameter(
            "No config path given and stdin is not a terminal; pass one "
            "explicitly."
        )

    return interactive_setup(command, purpose, options)


@cli.command()
def model(
    model_cfg_pth: str | None = typer.Argument(
        None,
        help="Path to the artifact run-config JSON (omit to pick).",
    ),
) -> None:
    """
    Train and test a model from its artifact config.
    """

    from .cli.select import offer_promotion

    cfg = _resolve_cfg(model_cfg_pth, "plant/model config")
    CompleteRun(cfg)
    offer_promotion(cfg, "plant", datetime.now().strftime("%Y-%m-%d"))


@cli.command()
def horizon(
    model_cfg_pth: str | None = typer.Argument(
        None,
        help="Path to the artifact run-config JSON (omit to pick).",
    ),
    checkpoint: str = typer.Option(
        "", help="Plant checkpoint (defaults to the config's save path)."
    ),
    steps: int = typer.Option(256, help="Rollout horizon to evaluate."),
    max_batches: int = typer.Option(
        32,
        help="Test batches, strided across the full set (more = stabler "
        "P99 tail).",
    ),
    example_step: int = typer.Option(
        0,
        help="Compounded-regime step for worked examples (0 = auto: "
        "mid-rollout).",
    ),
    early_step: int = typer.Option(
        1, help="Early (noise-floor) step for worked examples."
    ),
) -> None:
    """
    Error-vs-horizon evaluation: free-run a trained plant and report
    position drift.
    """

    import logging

    import torch

    from .core.config import RunConfiguration
    from .eval.horizon import run_horizon_eval

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_cfg_pth = _resolve_cfg(model_cfg_pth, "plant/model config")
    config = RunConfiguration(model_cfg_pth)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_horizon_eval(
        model_cfg_pth,
        checkpoint or config.m_save_dir,
        horizon=steps,
        device=device,
        max_batches=max_batches,
        plot_path=f"{config.eval_dir}/error_vs_horizon.svg",
        example_step=example_step or None,
        early_step=early_step,
    )


@cli.command()
def reliability(
    model_cfg_pth: str | None = typer.Argument(
        None,
        help="Path to the artifact run-config JSON (omit to pick).",
    ),
    checkpoint: str = typer.Option(
        "", help="Plant checkpoint (defaults to the config's save path)."
    ),
    samples: int = typer.Option(
        2048, help="Number of held-out windows to probe."
    ),
) -> None:
    """
    Reliability probes: Jacobian/Lipschitz sensitivity + weight-quantisation
    fragility.
    """

    import logging

    import torch

    from .core.config import RunConfiguration
    from .eval.reliability import run_reliability

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if model_cfg_pth is None:
        setup = _interactive_setup(
            "reliability",
            "plant/model config",
            [
                (
                    "checkpoint",
                    "Plant checkpoint (blank = config's path)",
                    str,
                    checkpoint,
                ),
                (
                    "samples",
                    "Held-out windows to probe (samples)",
                    int,
                    samples,
                ),
            ],
        )
        model_cfg_pth = str(setup["config"])
        checkpoint = str(setup["checkpoint"])
        samples = int(setup["samples"])
    config = RunConfiguration(model_cfg_pth)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_reliability(
        model_cfg_pth,
        checkpoint or config.m_save_dir,
        device=device,
        batch_size=samples,
    )


@cli.command()
def controller(
    controller_cfg_pth: str | None = typer.Argument(
        None,
        help="Path to the controller artifact config JSON (omit to pick).",
    ),
) -> None:
    """
    Train a quantisation-configurable controller by policy gradient through
    a plant.
    """

    from .cli.select import offer_promotion
    from .training.control_run import ControlRun

    cfg = _resolve_cfg(controller_cfg_pth, "controller config")
    ControlRun(cfg)
    offer_promotion(cfg, "controller", datetime.now().strftime("%Y-%m-%d"))


@cli.command()
def resource(
    controller_cfg_pth: str | None = typer.Argument(
        None,
        help="Path to the controller artifact config JSON (omit to pick).",
    ),
    servo_rate: float = typer.Option(
        0.0, help="Servo rate to score at (0 = the config's servo_rate_hz)."
    ),
) -> None:
    """
    Score a controller config's FPGA cost (DSP/BRAM/cycles/max rate). No
    training.
    """

    from .control.config import ControllerConfig
    from .control.resource import dense_layers, score_controller

    config = ControllerConfig(
        _resolve_cfg(controller_cfg_pth, "controller config")
    )
    dims = [config.in_features, *config.hidden, config.out_features]
    layers = dense_layers(
        dims,
        [q.weight_bits for q in config.quant],
        [q.act_bits for q in config.quant],
    )
    report = score_controller(layers, servo_rate or config.servo_rate_hz)
    typer.echo(
        f"Controller '{config.model_name}' at {report.servo_rate_hz:.0f} Hz:"
    )
    typer.echo(
        f"  params={report.params}  BRAM bits={report.bram_bits}  "
        f"DSP-cycles={report.dsp_cycles}"
    )
    typer.echo(
        f"  cycles: need {report.cycles_needed} of "
        f"{report.cycles_available:.0f}  "
        f"(max rate {report.max_servo_rate_hz:.0f} Hz)"
    )

    for reason in report.reasons:
        typer.echo(f"  {'OK' if report.fits else 'X'}: {reason}")


@cli.command()
def track(
    controller_cfg_pth: str | None = typer.Argument(
        None,
        help="Path to the controller artifact config JSON (omit to pick).",
    ),
    samples: int = typer.Option(
        4096, help="Quiescent-seed rollouts to evaluate."
    ),
    viz_steps: int = typer.Option(
        512, help="Free-run length for the 3D animation / interactive window."
    ),
    reanchor: int = typer.Option(
        0,
        help="Correct position onto the reference every N steps "
        "(feedback); the honest long-setpoint metric. 0 = raw free-run "
        "(conflates control with plant drift).",
    ),
    no_animate: bool = typer.Option(
        False,
        "--no-animate",
        help="Skip the GIF animation (plot + metrics only).",
    ),
    no_show: bool = typer.Option(
        False,
        "--no-show",
        help="Skip the native interactive 3D window (files only).",
    ),
    shape: str = typer.Option(
        "config",
        "--shape",
        help="Reference trajectory: config (the config's own), spiral, "
        "helix, line, step, smooth, mixed, morph or sequence. spiral and "
        "step are deterministic; the rest vary with --seed.",
    ),
    seed: int = typer.Option(
        42, "--seed", help="Seed for the randomised reference shapes."
    ),
) -> None:
    """
    Evaluate a trained controller closed-loop: per-axis tracking metrics +
    motion plots.
    """

    import logging

    import torch

    from .control.track import run_track
    from .training.control_run import TRACK_SHAPES

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if shape not in TRACK_SHAPES:
        raise typer.BadParameter(
            f"shape must be one of {', '.join(TRACK_SHAPES)}"
        )
    # A bare "track" (no config) lets you choose the config and its options
    # in the terminal (offering to reuse the last run); an explicit config
    # path keeps the given flags, so scripts and CI are unaffected.
    if controller_cfg_pth is None:
        setup = _interactive_setup(
            "track",
            "controller config",
            [
                (
                    "reanchor",
                    "Re-anchor every N steps (0 = free-run)",
                    int,
                    reanchor,
                ),
                ("samples", "Quiescent-seed rollouts (samples)", int, samples),
                (
                    "viz_steps",
                    "Free-run length for 3D animation (viz-steps)",
                    int,
                    viz_steps,
                ),
                ("animate", "Render the GIF animation?", bool, not no_animate),
                ("show", "Open the interactive 3D window?", bool, not no_show),
                (
                    "shape",
                    "Trajectory shape",
                    str,
                    shape,
                    list(TRACK_SHAPES),
                ),
                ("seed", "Reference seed", int, seed),
            ],
        )
        controller_cfg_pth = str(setup["config"])
        reanchor = int(setup["reanchor"])
        samples = int(setup["samples"])
        viz_steps = int(setup["viz_steps"])
        no_animate, no_show = not setup["animate"], not setup["show"]
        shape = str(setup["shape"])
        seed = int(setup["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_track(
        controller_cfg_pth,
        device=device,
        samples=samples,
        animate=not no_animate,
        viz_steps=viz_steps,
        show=not no_show,
        reanchor=reanchor,
        shape=shape,
        seed=seed,
    )


@cli.command()
def champions(
    system: str | None = typer.Option(
        None,
        help="SystemSpec whose registry to list (default: scan examples/*/).",
    ),
) -> None:
    """
    List the champion registry (best model per role) and flag any missing
    checkpoints.
    """

    from pathlib import Path

    from .core.champions import load_champions, registry_path

    if system:
        registries = [registry_path(system)]
    else:
        registries = sorted(Path().glob("examples/*/champions.json"))
    if not registries:
        typer.echo("No champion registries found under examples/*/.")

        return

    for registry in registries:
        champs = load_champions(registry)
        typer.echo(f"{registry}:")
        if not champs:
            typer.echo("  (no champions registered)")
            continue
        base = registry.parent

        for role, champ in champs.items():
            ok = (base / champ.checkpoint).exists()
            typer.echo(
                f"  [{'OK' if ok else 'MISSING'}] {role}: {champ.model} "
                f"({champ.promoted})"
            )
            typer.echo(f"      config:     {champ.config}")
            typer.echo(f"      checkpoint: {champ.checkpoint}")
            if champ.metric:
                typer.echo(f"      metric:     {champ.metric}")


@cli.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Print the program version and exit.",
    ),
):
    """
    Neural-network motion-controller builder.
    """


if __name__ == "__main__":
    cli()
