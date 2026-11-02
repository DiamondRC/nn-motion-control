"""
Command-line interface for 'python -m nn_motion_control'.
"""

from datetime import datetime
from typing import Annotated

import typer
from typer.core import TyperGroup

from . import __version__
from .core.precision import strict_float32
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
            "No config path given and stdin is not a terminal: "
            "pass one explicitly."
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
            "No config path given and stdin is not a terminal: "
            "pass one explicitly."
        )

    return interactive_setup(command, purpose, options)


def _resolve_dataset(source: str | None, system_path: str):
    """
    Resolve a dataset source to a (SystemSpec, data path) pair.

    A '.json' source (or an interactive pick) is an artifact config: its
    referenced dataset and SystemSpec are used. Any other source is a dataset
    file, paired with 'system_path' or the single examples/*/system.toml.
    """

    from pathlib import Path

    from .core.system import SystemSpec

    if source is None:
        source = _resolve_cfg(None, "dataset config")
    if source.endswith(".json"):
        from .core.config import RunConfiguration

        config = RunConfiguration(source)

        return config.system, config.datafile_dir
    if system_path:
        return SystemSpec.from_toml(system_path), source

    found = sorted(Path().glob("examples/*/system.toml"))
    if len(found) != 1:
        raise typer.BadParameter(
            f"Pass --system: found {len(found)} examples/*/system.toml files"
        )

    return SystemSpec.from_toml(found[0]), source


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
        help="Test batches, strided across the full set.",
    ),
    example_step: int = typer.Option(
        0,
        help="Compounded-regime step for worked examples.",
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
def analyse(
    source: str | None = typer.Argument(
        None,
        help="Dataset HDF5 to analyse, or a config that references one "
        "(omit to pick a config).",
    ),
    system: str = typer.Option(
        "",
        help="SystemSpec toml (default: the single examples/*/system.toml). "
        "Ignored when 'source' is a config.",
    ),
    out: str = typer.Option(
        "", help="Output dir (defaults to data_analysis/<dataset>)."
    ),
    per_axis: bool = typer.Option(
        True,
        "--per-axis/--no-per-axis",
        help="Also write full-size per-axis figures under per_axis/.",
    ),
) -> None:
    """
    Descriptive analysis of a dataset: envelope, statistics, per-channel
    distributions and demand-to-position linearity.
    """

    import logging

    from .data_analysis.analyse import run_analysis

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec, data_path = _resolve_dataset(source, system)
    run_analysis(spec, data_path, out or None, per_axis=per_axis)


@cli.command()
def characterise(
    source: str | None = typer.Argument(
        None,
        help="Dataset HDF5 to characterise, or a config that references one "
        "(omit to pick a config).",
    ),
    system: str = typer.Option(
        "",
        help="SystemSpec toml (default: the single examples/*/system.toml). "
        "Ignored when 'source' is a config.",
    ),
    out: str = typer.Option(
        "",
        help="Output dir (default: data_analysis/<dataset>/characterisation).",
    ),
) -> None:
    """
    Signal-characterisation battery: size, complexity (dimension, spectral,
    entropy, dynamics) and physics/model-correctness axes.
    """

    import logging

    from .data_analysis.characterise import run_characterisation

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec, data_path = _resolve_dataset(source, system)
    run_characterisation(spec, data_path, out or None)


@cli.command()
def compare(
    model_cfg_pth: str | None = typer.Argument(
        None,
        help="Plant config to compare against the data (omit to pick).",
    ),
    horizon: int = typer.Option(
        20000, help="Window length (steps) driven through the plant."
    ),
    out: str = typer.Option(
        "", help="Output dir (default: data_analysis/<model>/comparison)."
    ),
) -> None:
    """
    Compare the characterisation battery on data vs a plant (teacher-forced
    one-step and free-run generation).
    """

    import logging

    from .data_analysis.compare import run_comparison

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_cfg_pth = _resolve_cfg(model_cfg_pth, "plant config")
    run_comparison(model_cfg_pth, out or None, horizon=horizon)


@cli.command()
def compare_controllers(
    ssm_cfg: str = typer.Option(
        "examples/deltabot/configs/controller_ssm_integral.json",
        help="SSM controller config.",
    ),
    tcn_cfg: str = typer.Option(
        "examples/deltabot/configs/controller_tcn_integral.json",
        help="TCN controller config.",
    ),
    horizon: int = typer.Option(
        20000, help="Closed-loop rollout length (steps)."
    ),
    out: str = typer.Option(
        "", help="Output dir (default: data_analysis/controller_comparison)."
    ),
) -> None:
    """
    Compare the characterisation battery on the SSM vs TCN controllers,
    closed-loop (controlled position and tracking error, both re-anchor
    regimes), against the data.
    """

    import logging

    from .data_analysis.compare import run_controller_comparison

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_controller_comparison(ssm_cfg, tcn_cfg, out or None, horizon=horizon)


@cli.command("bench-controllers")
def bench_controllers(
    configs: Annotated[
        list[str], typer.Argument(help="Controller configs to benchmark.")
    ],
    bias: float = typer.Option(
        10.0, help="DAC bias for the step-rejection test (DAC units)."
    ),
    out: str = typer.Option(
        "", help="Output dir (default: data_analysis/controller_bench)."
    ),
) -> None:
    """
    Benchmark trained controllers head-to-head: long hold, tracking, step
    response, DAC-bias rejection and FPGA cost, from shared seeds and
    references.
    """

    import logging

    from .control.bench import run_controller_bench

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_controller_bench(configs, out or None, bias=bias)


@cli.command("plant-frf")
def plant_frf(
    model_cfg_pth: str | None = typer.Argument(
        None,
        help="Path to the plant run-config JSON (omit to pick).",
    ),
    checkpoint: str = typer.Option(
        "", help="Plant checkpoint (defaults to the config's save path)."
    ),
) -> None:
    """
    Frequency-response check: free-run a trained plant on the held-out
    multisine records and compare its response to the recorded one per
    band.
    """

    import logging

    from .eval.frf import format_report, run_plant_frf

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_cfg_pth = _resolve_cfg(model_cfg_pth, "plant/model config")
    report = run_plant_frf(model_cfg_pth, checkpoint or None)
    typer.echo(format_report(report))


@cli.command("build-dataset")
def build_dataset(
    spec: str = typer.Argument(
        help="Build spec JSON (e.g. examples/deltabot/datasets/panda_20k.json)."
    ),
    overwrite: bool = typer.Option(
        True, "--overwrite/--no-overwrite", help="Replace an existing output."
    ),
) -> None:
    """
    Build a schema-v3 training dataset from open-loop PandA captures.
    """

    import json
    import logging

    from .data.capture_ingest import build_capture_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = build_capture_dataset(spec, overwrite=overwrite)
    typer.echo(json.dumps(summary, indent=2))


@cli.command("preflight")
def preflight(
    config: Annotated[str, typer.Argument(help="Controller config to check.")],
) -> None:
    """
    Build a controller run and run its preflight checks without training.
    """

    from .training.control_run import ControlRun

    ControlRun(config, run_training=False)


@cli.command("plant-small-signal")
def plant_small_signal(
    config: Annotated[str, typer.Argument(help="Plant config to test.")],
    out: str = typer.Option("", help="Output dir (default: the plant's eval)."),
) -> None:
    """
    Gate a plant on its small-signal response: gain against the measured
    FRF under count-level command noise.
    """

    import logging

    from .eval.small_signal import run_small_signal

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_small_signal(config, out or None)


@cli.command("evaluate-controllers")
def evaluate_controllers(
    controllers: Annotated[
        list[str],
        typer.Argument(
            help="Controller configs (.json) or checkpoints (.pth)."
        ),
    ],
    suite_config: str = typer.Option(
        "examples/deltabot/configs/controller_panda_10k_tcn_workspace.json",
        help="Controller config whose system, plant and seeds set the suite.",
    ),
    pid_capture: str = typer.Option(
        "data/runs/campaign/20261016-142922/closed_loop/drive_x/"
        "20261016-151422",
        help="PandA PID capture run: its gains and trajectory set the PID.",
    ),
    out: str = typer.Option(
        "", help="Output dir (default: data_analysis/controller_suite)."
    ),
) -> None:
    """
    Score controllers and the PandA PID baseline on the fixed suite: holds
    near and far, every reference family, endurance and the hardware
    trajectory, with tracking error, divergence and command thrash.
    """

    import logging

    from .eval.controller_suite import run_suite

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_suite(controllers, suite_config, pid_capture, out or None)


@cli.command("endurance-controllers")
def endurance_controllers(
    configs: Annotated[
        list[str], typer.Argument(help="Controller configs to test.")
    ],
    bias: float = typer.Option(
        10.0, help="DAC bias for the biased hold (DAC units)."
    ),
    out: str = typer.Option(
        "", help="Output dir (default: data_analysis/controller_endurance)."
    ),
) -> None:
    """
    Hold each controller for 30 s on a biased hold, a step and bounded
    trajectories, reporting error at 0.5, 1, 5, 10 and 30 s.
    """

    import logging

    from .control.bench import run_controller_endurance

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_controller_endurance(configs, out or None, bias=bias)


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
        "helix, line, step, smooth, mixed, morph or sequence. step and "
        "helix are deterministic; the rest (spiral's tilt included) vary "
        "with --seed.",
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

    from .core.champions import REGISTRY_GLOB, load_champions, registry_path

    if system:
        registries = [registry_path(system)]
    else:
        registries = sorted(Path().glob(f"examples/*/{REGISTRY_GLOB}"))
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

    # Every command starts at full float32 precision whatever the
    # environment enables, a training config may opt in to TF32.
    strict_float32()


if __name__ == "__main__":
    cli()
