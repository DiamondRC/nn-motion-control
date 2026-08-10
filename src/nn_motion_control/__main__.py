"""
Command-line interface for ``python -m nn_motion_control``.
"""

import typer
from typer.core import TyperGroup

from . import __version__
from .training.run import CompleteRun

__all__ = ["main"]


class NaturalOrderGroup(TyperGroup):
    """
    Typer group that lists commands in declaration order rather than alphabetically.
    """

    def list_commands(self, ctx):
        return list(self.commands)


cli = typer.Typer(cls=NaturalOrderGroup, pretty_exceptions_enable=False)


def version_callback(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@cli.command()
def model(
    model_cfg_pth: str = typer.Argument(
        default="examples/deltabot/configs/plant_tcn.json",
        help="Path to the artifact run-config JSON.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
) -> None:
    """
    Train and test a model from its artifact config.
    """

    CompleteRun(model_cfg_pth)


@cli.command()
def horizon(
    model_cfg_pth: str = typer.Argument(
        ...,
        help="Path to the artifact run-config JSON.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    checkpoint: str = typer.Option(
        "", help="Plant checkpoint (defaults to the config's save path)."
    ),
    steps: int = typer.Option(256, help="Rollout horizon to evaluate."),
    max_batches: int = typer.Option(
        32, help="Test batches, strided across the full set (more = stabler P99 tail)."
    ),
    example_step: int = typer.Option(
        0, help="Compounded-regime step for worked examples (0 = auto: mid-rollout)."
    ),
    early_step: int = typer.Option(
        1, help="Early (noise-floor) step for worked examples."
    ),
) -> None:
    """
    Error-vs-horizon evaluation: free-run a trained plant and report position drift.
    """

    import logging

    import torch

    from .core.config import RunConfiguration
    from .eval.horizon import run_horizon_eval

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = RunConfiguration(model_cfg_pth)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_horizon_eval(
        model_cfg_pth,
        checkpoint or config.m_save_dir,
        horizon=steps,
        device=device,
        max_batches=max_batches,
        plot_path=f"{config.logging_dir}/error_vs_horizon.svg",
        example_step=example_step or None,
        early_step=early_step,
    )


@cli.command()
def reliability(
    model_cfg_pth: str = typer.Argument(
        ...,
        help="Path to the artifact run-config JSON.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    checkpoint: str = typer.Option(
        "", help="Plant checkpoint (defaults to the config's save path)."
    ),
    samples: int = typer.Option(2048, help="Number of held-out windows to probe."),
) -> None:
    """
    Reliability probes: Jacobian/Lipschitz sensitivity + weight-quantisation fragility.
    """

    import logging

    import torch

    from .core.config import RunConfiguration
    from .eval.reliability import run_reliability

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = RunConfiguration(model_cfg_pth)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_reliability(
        model_cfg_pth,
        checkpoint or config.m_save_dir,
        device=device,
        batch_size=samples,
    )


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
